//! A PUCT tree with a Gumbel root, ported line-for-line from
//! `js/normal-duel-puct-search.mjs`, with the leaf evaluation deferred.
//!
//! What is different from the JavaScript, and what is not
//! ------------------------------------------------------
//! The *algorithm* is not different. Same Gumbel draws in the same order, same
//! sequential-halving schedule, same PUCT selection rule, same FPU, same
//! tie-breaks, same budget accounting. The only change is control flow: where
//! the JavaScript calls `evaluate(config, state)` in the middle of `simulate`,
//! this implementation stops, hands the caller a feature vector, and resumes
//! when the caller supplies the policy and value.
//!
//! That is what makes batching possible without changing a single search
//! decision. A [`PuctTreeSearch`] has **at most one outstanding evaluation**;
//! the batch is formed across independent games, never within one. There is no
//! virtual loss, no tree parallelism, and nothing that would make the search a
//! different algorithm from the reference. Batching across games is a pure
//! scheduling change; batching within a game would not be.
//!
//! Recursion becomes an explicit path stack ([`PuctTreeSearch::path`]) so a
//! descent can pause at an unexpanded leaf and be resumed. The stack holds
//! `(node, edge)` pairs exactly matching the JavaScript call frames, and the
//! unwind negates once per frame, which is the same "negate when crossing a
//! ply" rule the reference documents.
//!
//! Float reproducibility
//! ---------------------
//! Every arithmetic expression is written with the same association as the
//! JavaScript, because IEEE-754 double arithmetic is only reproducible
//! operation-by-operation. `Math.log` is not an IEEE operation; see
//! [`crate::js_math`]. `Math.sqrt` is, so [`f64::sqrt`] is used directly.
//!
//! Repetition and terminality
//! --------------------------
//! The reference builds a full [`crate::GameState`] per node, whose cost is
//! dominated by `validateState`. Here a node carries a [`SearchPosition`] plus
//! the two facts adjudication actually needs: its ply, and how many times its
//! position has occurred in the current repetition window. A wall placement
//! resets that window (`applyTrustedAction` sets `historyStartPly = ply` and
//! `counts = {key: 1}` for wall moves), and inside a window the walls and the
//! stock are constant, so a position is fully identified there by the two pawn
//! squares and the side to move — that is [`compact_key`]. Terminal states are
//! scored from the engine's adjudication rules and never expanded.

use thiserror::Error;

use crate::js_math::{js_log, Lcg32};
use crate::{
    Config, Coord, GameState, NormalDuelError, Orientation, Player, SearchPosition, Wall,
    MAX_POLICY_CODES, NN_INPUT_PLANES,
};

/// Frozen identifier for this search + self-play record format.
///
/// `v2` is the completed-Q policy target: the recorded `policyTarget` is the
/// Gumbel improved policy over every legal root action
/// ([`PuctResult::improved_policy`]) instead of the normalised visit counts of
/// the considered set. Nothing about which move the search *plays* changed, so
/// the version bump names a record-format change, not a search change.
pub const PUCT_SEARCH_VERSION: &str = "puct-az-tree-v2";

/// The version the JavaScript reference search is frozen at.
///
/// `js/normal-duel-puct-search.mjs` deliberately stays on `v1`. It is the parity
/// oracle for the *search decisions* — visit counts, chosen action, root value,
/// simulations spent, considered set — and every one of those is unchanged by
/// the improved policy, so `tests/js_puct_parity.rs` still compares exactly what
/// it always compared, at full strength.
///
/// Porting the improved policy to the JavaScript would mean comparing
/// `Math.exp` against [`f64::exp`] bit for bit. `exp` is not an IEEE-754
/// operation, which is why [`crate::js_math::js_log`] exists at all; there is no
/// `js_exp`, and writing one to cross-check a training target would be a far
/// larger correctness surface than the target itself. The production self-play
/// driver is the Rust/wasm [`crate::selfplay::SelfPlayBatch`], not the
/// JavaScript reference, so the reference is frozen and the divergence is named
/// here rather than left to be discovered.
///
/// `tests/js_puct_parity.rs` asserts the JavaScript still reports this string,
/// so the freeze cannot drift silently in either direction.
pub const JS_REFERENCE_SEARCH_VERSION: &str = "puct-az-tree-v1";

/// Floor applied inside the logit so a legal action the policy assigns exactly
/// zero probability is merely very unlikely, not `-Infinity`.
const POLICY_FLOOR: f64 = 1e-9;

/// Gumbel-MuZero sigma constants.
const C_VISIT: u32 = 50;
const C_SCALE: f64 = 1.0;

/// Default exploration constant if the caller does not supply one.
pub const DEFAULT_C_PUCT: f64 = 1.25;

/// First-play-urgency reduction. See the reference for the full rationale.
pub const FPU_REDUCTION: f64 = 0.25;

const NO_CHILD: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PuctError {
    #[error("unsupported_board")]
    UnsupportedBoard,
    #[error("terminal_state")]
    TerminalState,
    #[error("no_legal_actions")]
    NoLegalActions,
    #[error("invalid_evaluation")]
    InvalidEvaluation,
    #[error("invalid_max_considered")]
    InvalidMaxConsidered,
    #[error("invalid_simulations")]
    InvalidSimulations,
    #[error("invalid_ply_cap")]
    InvalidPlyCap,
    #[error("invalid_games")]
    InvalidGames,
    #[error("contradictory_exploration")]
    ContradictoryExploration,
    #[error("invalid_c_puct")]
    InvalidCPuct,
    #[error("invalid_state")]
    InvalidState,
    #[error("invalid_action_code")]
    InvalidActionCode,
    /// `submit` was called with no evaluation outstanding, or `next_leaf` was
    /// called with one still outstanding.
    #[error("out_of_order_evaluation")]
    OutOfOrderEvaluation,
    #[error("invalid_buffer_length")]
    InvalidBufferLength,
    #[error("{0}")]
    Engine(#[from] NormalDuelError),
}

impl PuctError {
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            Self::UnsupportedBoard => "unsupported_board",
            Self::TerminalState => "terminal_state",
            Self::NoLegalActions => "no_legal_actions",
            Self::InvalidEvaluation => "invalid_evaluation",
            Self::InvalidMaxConsidered => "invalid_max_considered",
            Self::InvalidSimulations => "invalid_simulations",
            Self::InvalidPlyCap => "invalid_ply_cap",
            Self::InvalidGames => "invalid_games",
            Self::ContradictoryExploration => "contradictory_exploration",
            Self::InvalidCPuct => "invalid_c_puct",
            Self::InvalidState => "invalid_state",
            Self::InvalidActionCode => "invalid_action_code",
            Self::OutOfOrderEvaluation => "out_of_order_evaluation",
            Self::InvalidBufferLength => "invalid_buffer_length",
            Self::Engine(error) => error.code(),
        }
    }
}

type Result<T> = std::result::Result<T, PuctError>;

/// `clampValue` from the reference: NaN passes through unchanged, as in JS.
fn clamp_value(value: f64) -> f64 {
    if value > 1.0 {
        return 1.0;
    }
    if value < -1.0 {
        return -1.0;
    }
    value
}

/// Strictly increasing in `q`, so it never reorders two candidates by value.
fn sigma(q: f64, max_visits: u32) -> f64 {
    f64::from(C_VISIT + max_visits) * C_SCALE * q
}

/// The paper's `completedQ`: a visited root action is worth what the tree
/// measured, an unvisited one is worth the root's own value.
///
/// This is the completion the halving already used to rank survivors, lifted
/// into one function so the improved policy and the schedule cannot drift apart.
///
/// Danihelka et al. refine the unvisited case to `v_mix`, a prior-weighted blend
/// of the root value and the visited children's Q. That is a possible future
/// swap and this function is the single place it would happen; it is
/// deliberately not implemented here, because `v_mix` changes the halving
/// ranking too and therefore the search's decisions, which this change does not
/// touch.
fn completed_q(edge: &Edge, root_value: f64) -> f64 {
    if edge.visits > 0 {
        edge.value_sum / f64::from(edge.visits)
    } else {
        root_value
    }
}

/// The Gumbel improved policy over `edges`, which must be one node's whole edge
/// list: `pi'(a) ∝ exp(logit(a) + sigma(completedQ(a)))`.
///
/// Four details are load-bearing.
///
/// `logit(a)` is `js_log(prior.max(POLICY_FLOOR))`, the same expression
/// `seed_candidates` uses to rank the Gumbel draws — the improved policy and the
/// considered set read the prior through the same floor.
///
/// `max_visits` is the maximum over *all* the node's edges, not over the
/// halving's surviving set, because the policy covers actions halving discarded.
///
/// The softmax subtracts the maximum score before exponentiating. That is not
/// tidiness: `sigma` spans roughly `±(C_VISIT + max_visits)`, so at a
/// four-figure simulation budget the raw exponentials would overflow to
/// infinity and the normalisation would return NaN. After the subtraction the
/// largest term is exactly `1.0`, so the total is in `[1, edges.len()]` and can
/// neither overflow nor be zero.
///
/// And `sigma`'s scale now does a second job.
///
/// In sequential halving `sigma` is only ever a RANKING term: it is strictly
/// increasing in `q`, so its magnitude is irrelevant there and only the induced
/// order matters. Here it is inside a softmax, where the magnitude IS the
/// temperature. With `C_VISIT = 50`, `C_SCALE = 1.0` and completed-Q left in
/// `[-1, 1]`, the score span is `2 * (50 + max_visits)` -- about 160 at 128
/// simulations over 16 candidates -- so a 0.1 difference in Q is a factor of
/// e^8 in probability. The published implementations reach a much gentler
/// distribution by min-max normalising completed-Q to `[0, 1]` and using
/// `c_scale ~ 0.1`.
///
/// Reusing this project's own `sigma` unchanged is deliberate: it is what keeps
/// the halving ranking and the recorded target derived from ONE expression, so
/// they cannot drift apart. But it means the target's entropy is set by a
/// constant that was only ever tuned for an ordering. If the first long run
/// wants a softer target, THIS is the knob -- normalise the Q range or scale
/// `C_SCALE` for the policy only -- not the simulation budget.
fn improved_policy(edges: &[Edge], root_value: f64) -> Vec<(u16, f64)> {
    let max_visits = edges.iter().map(|edge| edge.visits).max().unwrap_or(0);

    let mut scored: Vec<(u16, f64)> = Vec::with_capacity(edges.len());
    let mut highest = f64::NEG_INFINITY;
    for edge in edges {
        let logit = js_log(edge.prior.max(POLICY_FLOOR));
        let score = logit + sigma(completed_q(edge, root_value), max_visits);
        if score > highest {
            highest = score;
        }
        scored.push((edge.code, score));
    }
    if scored.is_empty() {
        return scored;
    }

    let mut total = 0.0_f64;
    for (_, score) in &mut scored {
        *score = (*score - highest).exp();
        total += *score;
    }
    for (_, weight) in &mut scored {
        *weight /= total;
    }
    scored
}

/// `Math.ceil(Math.log2(Math.max(m, 2)))` without a logarithm: the smallest
/// `r` with `2^r >= max(m, 2)`. Verified equal to the JavaScript expression for
/// every `m` in `1..=209`, the whole reachable candidate range — an integer
/// computation cannot land on the wrong side of a `ceil` boundary the way a
/// 1-ULP `log2` could.
fn halving_rounds(candidates: usize) -> u32 {
    let target = candidates.max(2);
    let mut rounds = 0_u32;
    let mut reach = 1_usize;
    while reach < target {
        reach *= 2;
        rounds += 1;
    }
    rounds.max(1)
}

/// Identity of a position *within one repetition window*.
///
/// Only valid for comparing positions that share a window: a window is delimited
/// by wall placements, and neither the walls nor the stock can change without
/// one, so the two pawn squares and the side to move are the whole of what
/// varies. Packing them into a `u32` avoids the reference's JSON position key.
#[must_use]
pub fn compact_key(config: &Config, pawn_a: Coord, pawn_b: Coord, turn: Player) -> u32 {
    let a = crate::square(config, pawn_a) as u32;
    let b = crate::square(config, pawn_b) as u32;
    let turn_bit = u32::from(turn == Player::B);
    a | (b << 8) | (turn_bit << 16)
}

fn position_key(config: &Config, position: &SearchPosition) -> u32 {
    compact_key(config, position.pawns.a, position.pawns.b, position.turn)
}

/// The repetition window a search starts from: how many times each position has
/// already occurred since the last wall placement, including the root itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepetitionWindow {
    counts: Vec<(u32, u32)>,
}

impl RepetitionWindow {
    /// A window holding only the given position, which is what a wall placement
    /// produces.
    #[must_use]
    pub fn fresh(key: u32) -> Self {
        Self {
            counts: vec![(key, 1)],
        }
    }

    #[must_use]
    pub fn get(&self, key: u32) -> u32 {
        self.counts
            .iter()
            .find(|(entry, _)| *entry == key)
            .map_or(0, |(_, count)| *count)
    }

    /// Record one more occurrence of `key` and return its new count.
    pub fn push(&mut self, key: u32) -> u32 {
        if let Some(entry) = self.counts.iter_mut().find(|(entry, _)| *entry == key) {
            entry.1 += 1;
            return entry.1;
        }
        self.counts.push((key, 1));
        1
    }

    pub fn reset(&mut self, key: u32) {
        self.counts.clear();
        self.counts.push((key, 1));
    }

    /// The window's `(key, count)` pairs, in whatever order they were inserted.
    ///
    /// Exposed for the shard writer, which has to put the window on disk so a
    /// later re-search can start from the state the game was actually in rather
    /// than one synthesised around the position. Insertion order is deterministic
    /// but it is an implementation detail of [`Self::push`], so the writer sorts
    /// rather than trusting it; the keys are unique, so the sort is total.
    #[must_use]
    pub fn entries(&self) -> &[(u32, u32)] {
        &self.counts
    }

    /// Rebuild the window from a validated [`GameState`].
    ///
    /// `validateState` guarantees every entry shares the current walls and
    /// stock, which is exactly the precondition [`compact_key`] needs.
    pub fn from_state(config: &Config, state: &GameState) -> Result<Self> {
        let mut counts = Vec::with_capacity(state.repetition_counts.len());
        for entry in &state.repetition_counts {
            let position = crate::parse_position_key(config, &entry.position_key)
                .map_err(|_| PuctError::InvalidState)?;
            let count = u32::try_from(entry.count).map_err(|_| PuctError::InvalidState)?;
            counts.push((
                compact_key(config, position.pawns.a, position.pawns.b, position.turn),
                count,
            ));
        }
        Ok(Self { counts })
    }
}

/// Everything a search needs about its root beyond the board itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootContext {
    pub position: SearchPosition,
    pub ply: u64,
    pub window: RepetitionWindow,
}

impl RootContext {
    pub fn from_state(config: &Config, state: &GameState) -> Result<Self> {
        Ok(Self {
            position: SearchPosition::from_position(config, &state.position)?,
            ply: state.ply,
            window: RepetitionWindow::from_state(config, state)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PuctParams {
    pub simulations: u32,
    pub max_considered: u32,
    pub c_puct: f64,
}

impl Default for PuctParams {
    fn default() -> Self {
        Self {
            simulations: 32,
            max_considered: 8,
            c_puct: DEFAULT_C_PUCT,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    code: u16,
    prior: f64,
    visits: u32,
    /// Visits accrued since the last [`PuctTreeSearch::restart`].
    ///
    /// Equal to `visits` for a search that never resumed, so nothing changes for
    /// one. On a RESUMED search the two diverge, and the difference is the whole
    /// point: `visits` carries the inherited value estimate, which is what reuse is
    /// for, while the halving schedule must rank on what THIS search has explored.
    /// Ranking on the total let a candidate that was good before the opponent moved
    /// arrive with a visit count no fresh candidate could match.
    visits_since: u32,
    value_sum: f64,
    child: u32,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    position: SearchPosition,
    ply: u64,
    /// Occurrences of this node's position in its own repetition window.
    rep_count: u32,
    /// True when the move that produced this node placed a wall, which starts a
    /// fresh repetition window at this node.
    resets_window: bool,
    terminal: bool,
    /// Terminal score in this node's own frame; meaningless unless `terminal`.
    terminal_value: f64,
    expanded: bool,
    edges_start: u32,
    edges_len: u32,
    visits: u32,
    value_sum: f64,
    /// Node value in its OWN frame: network value, refined to `W/N` once visited.
    value: f64,
    visited_prior: f64,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    code: u16,
    /// Index into `edges` of the root edge this candidate owns.
    edge: u32,
    score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The root has not been expanded; its features have not been handed out.
    RootPending,
    /// The root's features are with the caller.
    RootAwaiting,
    /// No evaluation outstanding; the schedule may advance.
    Ready,
    /// A descent is paused at `pending_leaf`, whose features are with the caller.
    LeafAwaiting,
    /// Several descents are paused at once and their features are with the
    /// caller. Only reachable through [`PuctTreeSearch::collect_leaves`].
    BatchAwaiting,
    Done,
}

/// A descent paused at a leaf while the rest of its batch is collected.
///
/// Only `path` and `leaf` are needed: the repetition-window state
/// (`descent_keys`, `window_from`, `root_window_active`) is consumed during the
/// descent and reset by `begin_visit`, and `submit` reads neither.
#[derive(Clone, Debug)]
struct PendingLeaf {
    path: Vec<(u32, u32)>,
    leaf: u32,
}

/// One game's search. Holds at most one outstanding leaf evaluation.
#[derive(Debug)]
pub struct PuctTreeSearch {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    params: PuctParams,

    /// The game's shared draw stream. A whole self-play game runs off one
    /// `createLcg32(seed)` in the reference, so the owner hands its stream in
    /// at construction and takes the advanced stream back with [`Self::rng`].
    rng: Lcg32,

    root_window: RepetitionWindow,
    root_value: f64,
    max_depth: u32,
    used: u32,
    budget: i64,

    /// The considered set, ascending by action code (`considered` in the JS).
    candidates: Vec<Candidate>,
    /// Indices into `candidates`, ascending by action code.
    survivors: Vec<usize>,
    rounds: u32,
    per_candidate: u32,
    pass: u32,
    next_survivor: usize,
    round_fresh: bool,
    draining_single: bool,

    /// Descents collected for the current batch, in collection order.
    pending: Vec<PendingLeaf>,
    /// Penalty applied to every node on an in-flight path so a later descent in
    /// the same batch is steered away from it.
    ///
    /// `0.0` means no penalty, which is the ONLY setting that keeps the search
    /// bit-identical to the sequential reference: with no penalty a batch may
    /// only hold descents into DISTINCT root candidates, whose subtrees are
    /// disjoint, so batching them changes nothing. Any non-zero value permits
    /// several descents into one subtree and makes this a different algorithm.
    virtual_loss: f64,

    /// `(node, edge)` frames of the paused descent, root-first.
    path: Vec<(u32, u32)>,
    /// Compact keys of the path's nodes below the root, in descent order.
    descent_keys: Vec<u32>,
    /// Index into `descent_keys` where the current repetition window starts.
    window_from: usize,
    /// True while the root's own window is still the live one.
    root_window_active: bool,
    pending_leaf: u32,

    phase: Phase,
    codes: [u16; MAX_POLICY_CODES],
    ranking: Vec<(usize, f64)>,
}

/// What a completed search decided.
#[derive(Debug, Clone, PartialEq)]
pub struct PuctResult {
    pub action_code: u16,
    /// `(code, visits)` for every considered candidate, ascending by code.
    pub visit_counts: Vec<(u16, u32)>,
    /// `(code, pi')` for every **legal root action**, ascending by code: the
    /// Gumbel improved policy, and what self-play records as `policyTarget`.
    ///
    /// This is what the visit counts above cannot be. Under sequential halving
    /// the counts are the *schedule*: at 128 simulations over 16 candidates both
    /// finalists take exactly 30 visits whichever one won and by whatever
    /// margin, and every legal action outside the considered set takes zero. The
    /// improved policy reads each action's completed Q instead, so it separates
    /// the finalists and it covers actions the Gumbel draw never considered.
    ///
    /// Ascending by code, and normalised in that order, so the float rounding a
    /// consumer sees does not depend on iteration order.
    pub improved_policy: Vec<(u16, f64)>,
    pub root_value: f64,
    pub simulations_used: u32,
    pub max_depth_reached: u32,
    pub considered: Vec<u16>,
}

impl PuctResult {
    /// `effectiveVisitCounts`: a search with nothing to decide spends no
    /// simulations, so the raw counts sum to zero and cannot be normalised; the
    /// played action then carries the whole target. That is correct only because
    /// there was one action to choose.
    ///
    /// **No longer the self-play target.** Since the record format moved to
    /// `puct-az-tree-v2` the policy target is [`Self::improved_policy`], which
    /// needs no such fallback: a root with one legal action has one edge, and
    /// the softmax over one element is exactly `1.0`. This is kept because it
    /// mirrors `effectiveVisitCounts` in the frozen JavaScript reference, which
    /// still uses it — see [`JS_REFERENCE_SEARCH_VERSION`].
    ///
    /// A zero *budget* used to reach this fallback too, which made the same
    /// one-hot the policy target at a position with ~130 legal codes. It can no
    /// longer: `PuctTreeSearch::new` rejects `simulations < 1`, which is the
    /// constructor every entry point funnels through -- the wasm boundary and
    /// `SelfPlayBatch` included -- so the fallback is reachable only for the
    /// nothing-to-decide case it is written for.
    #[must_use]
    pub fn effective_visit_counts(&self) -> Vec<(u16, u32)> {
        let total: u32 = self.visit_counts.iter().map(|(_, count)| *count).sum();
        if total > 0 {
            return self.visit_counts.clone();
        }
        vec![(self.action_code, 1)]
    }
}

/// The board change one action code makes, without building a `GameState`.
#[derive(Debug, Clone, Copy)]
pub struct AppliedAction {
    pub position: SearchPosition,
    pub placed_wall: bool,
    /// `Some(mover)` when the move landed the mover on its goal row, which is
    /// the only way `adjudicate` can produce a win.
    pub goal_winner: Option<Player>,
}

/// Apply one already-legal action code to a [`SearchPosition`].
///
/// The caller must have taken `code` from
/// [`SearchPosition::legal_action_codes_fast`]; the checks here are the cheap
/// structural ones that fail closed, not a legality generator.
pub fn apply_action_code(
    config: &Config,
    parent: &SearchPosition,
    code: u16,
) -> Result<AppliedAction> {
    let columns = config.columns as usize;
    let cells = config.cells();
    let mut position = *parent;
    let mover = position.turn;
    let code = usize::from(code);

    if code < cells {
        let to = Coord {
            r: (code / columns) as u8,
            c: (code % columns) as u8,
        };
        *position.pawns.get_mut(mover) = to;
        position.turn = mover.other();
        let goal_winner = (to.r == *config.goal_rows.get(mover)).then_some(mover);
        return Ok(AppliedAction {
            position,
            placed_wall: false,
            goal_winner,
        });
    }

    let anchors = config.anchors_per_axis();
    let offset = code
        .checked_sub(cells)
        .ok_or(PuctError::InvalidActionCode)?;
    if offset >= 2 * anchors {
        return Err(PuctError::InvalidActionCode);
    }
    let orientation = if offset < anchors {
        Orientation::Horizontal
    } else {
        Orientation::Vertical
    };
    let anchor = offset % anchors;
    let wall = Wall {
        orientation,
        r: (anchor / (columns - 1)) as u8,
        c: (anchor % (columns - 1)) as u8,
    };
    if *position.stock.get(mover) == 0 || !position.walls.insert_valid(wall, config) {
        return Err(PuctError::InvalidActionCode);
    }
    let mut board = position.board();
    board.add_wall_edges(config, wall);
    position.blocked_down = board.blocked_down;
    position.blocked_right = board.blocked_right;
    *position.stock.get_mut(mover) -= 1;
    position.turn = mover.other();
    Ok(AppliedAction {
        position,
        placed_wall: true,
        goal_winner: None,
    })
}

impl PuctTreeSearch {
    /// Start a search at `root`. The caller must have checked that the root is
    /// not already terminal; the reference fails with `terminal_state`.
    pub fn new(config: &Config, root: RootContext, params: PuctParams, rng: Lcg32) -> Result<Self> {
        config.validate()?;
        if config.rows != 9 || config.columns != 9 {
            return Err(PuctError::UnsupportedBoard);
        }
        if params.simulations < 1 {
            return Err(PuctError::InvalidSimulations);
        }
        if params.max_considered < 1 {
            return Err(PuctError::InvalidMaxConsidered);
        }
        if !params.c_puct.is_finite() || params.c_puct <= 0.0 {
            return Err(PuctError::InvalidCPuct);
        }

        let node = Node {
            position: root.position,
            ply: root.ply,
            rep_count: root.window.get(position_key(config, &root.position)),
            resets_window: false,
            terminal: false,
            terminal_value: 0.0,
            expanded: false,
            edges_start: 0,
            edges_len: 0,
            visits: 0,
            value_sum: 0.0,
            value: 0.0,
            visited_prior: 0.0,
        };

        // Pre-size the arenas for the whole budget so the common case does no
        // reallocation: a search performs at most `simulations` expansions plus
        // the root, and each expansion adds one node and its edge list. This is
        // a throughput measure, not a correctness one — see the wasm memory
        // note in `crate::selfplay`.
        let expansions = params.simulations as usize + 1;
        Ok(Self {
            nodes: {
                let mut nodes = Vec::with_capacity(expansions + 1);
                nodes.push(node);
                nodes
            },
            edges: Vec::with_capacity(expansions.min(64) * 32),
            params,
            rng,
            root_window: root.window,
            root_value: 0.0,
            max_depth: 0,
            used: 0,
            budget: i64::from(params.simulations),
            candidates: Vec::new(),
            survivors: Vec::new(),
            rounds: 1,
            per_candidate: 0,
            pass: 0,
            next_survivor: 0,
            round_fresh: true,
            draining_single: false,
            path: Vec::new(),
            pending: Vec::new(),
            virtual_loss: 0.0,
            descent_keys: Vec::new(),
            window_from: 0,
            root_window_active: true,
            pending_leaf: NO_CHILD,
            phase: Phase::RootPending,
            codes: [0; MAX_POLICY_CODES],
            ranking: Vec::new(),
        })
    }

    pub fn from_state(
        config: &Config,
        state: &GameState,
        params: PuctParams,
        rng: Lcg32,
    ) -> Result<Self> {
        if !state.outcome.is_ongoing() {
            return Err(PuctError::TerminalState);
        }
        Self::new(config, RootContext::from_state(config, state)?, params, rng)
    }

    /// The draw stream as this search left it, for the caller continuing a game.
    #[must_use]
    pub fn rng(&self) -> Lcg32 {
        self.rng
    }

    #[must_use]
    pub fn is_done(&self) -> bool {
        self.phase == Phase::Done
    }

    #[must_use]
    pub fn awaiting_evaluation(&self) -> bool {
        matches!(
            self.phase,
            Phase::RootAwaiting | Phase::LeafAwaiting | Phase::BatchAwaiting
        )
    }

    /// Advance the search to its next leaf evaluation.
    ///
    /// Writes `NN_INPUT_PLANES * cells` features for that leaf into `features`
    /// and returns `true`; returns `false` once the search is complete, leaving
    /// `features` untouched. Terminal leaves are scored by the engine and cost
    /// no evaluation, so they are consumed inside this call.
    pub fn next_leaf(&mut self, config: &Config, features: &mut [f32]) -> Result<bool> {
        if features.len() != NN_INPUT_PLANES * config.cells() {
            return Err(PuctError::InvalidBufferLength);
        }
        if self.awaiting_evaluation() {
            return Err(PuctError::OutOfOrderEvaluation);
        }
        loop {
            match self.phase {
                Phase::Done => return Ok(false),
                Phase::RootPending => {
                    self.encode(config, 0, features);
                    self.phase = Phase::RootAwaiting;
                    return Ok(true);
                }
                Phase::Ready => {
                    let Some(candidate) = self.next_candidate() else {
                        self.phase = Phase::Done;
                        return Ok(false);
                    };
                    if self.begin_visit(config, candidate)? {
                        self.encode(config, self.pending_leaf, features);
                        self.phase = Phase::LeafAwaiting;
                        return Ok(true);
                    }
                }
                Phase::RootAwaiting | Phase::LeafAwaiting | Phase::BatchAwaiting => {
                    return Err(PuctError::OutOfOrderEvaluation)
                }
            }
        }
    }

    /// Supply the evaluation for the leaf [`Self::next_leaf`] handed out.
    ///
    /// `policy` is the raw policy head over all `policy_size` codes; it is
    /// masked to the legal actions and renormalised here. `value` is in the
    /// leaf's own frame and must lie in `[-1, 1]`.
    /// Legal-mask floats (`1.0` legal, `0.0` illegal) for the leaf the last
    /// [`Self::next_leaf`] produced.
    ///
    /// The batched driver needs this because the network's masked softmax runs
    /// outside the search: masking before the softmax and skipping illegal
    /// codes — rather than softmaxing all 209 logits and letting `expand`
    /// renormalise over the legal ones — is what keeps the batched path
    /// bit-identical to the serial JS `evaluate`. The two are algebraically the
    /// same and differ only in f32 rounding, which is exactly the difference
    /// the parity suite is built to catch.
    /// The legal-action mask for the `index`-th leaf of the current batch.
    ///
    /// A batch has several leaves outstanding, so [`Self::pending_leaf_mask`]'s
    /// "the pending leaf" has no meaning here. Each leaf needs its OWN mask: the
    /// caller uses it to turn raw logits into probabilities, and a mask from the
    /// wrong position would produce a well-formed distribution over the wrong
    /// moves -- which `submit_batch` would accept without complaint.
    pub fn batch_leaf_mask(
        &mut self,
        config: &Config,
        index: usize,
        mask: &mut [f32],
    ) -> Result<()> {
        if mask.len() != config.policy_size() {
            return Err(PuctError::InvalidBufferLength);
        }
        // The root is handed out as a batch of one, so a caller driving the batch
        // API sees it through this accessor too. Without this it has to special-case
        // the very first collection, which is exactly the kind of asymmetry that
        // gets it wrong.
        let leaf = match self.phase {
            Phase::RootAwaiting if index == 0 => 0,
            Phase::BatchAwaiting => {
                self.pending
                    .get(index)
                    .ok_or(PuctError::InvalidBufferLength)?
                    .leaf
            }
            _ => return Err(PuctError::OutOfOrderEvaluation),
        };
        let position = self.nodes[leaf as usize].position;
        let count = position.legal_action_codes_fast(config, &mut self.codes);
        mask.fill(0.0);
        for code in &self.codes[..count] {
            mask[usize::from(*code)] = 1.0;
        }
        Ok(())
    }

    pub fn pending_leaf_mask(&mut self, config: &Config, mask: &mut [f32]) -> Result<()> {
        if mask.len() != config.policy_size() {
            return Err(PuctError::InvalidBufferLength);
        }
        let node = match self.phase {
            Phase::RootAwaiting => 0,
            Phase::LeafAwaiting => self.pending_leaf,
            _ => return Err(PuctError::OutOfOrderEvaluation),
        };
        let position = self.nodes[node as usize].position;
        let count = position.legal_action_codes_fast(config, &mut self.codes);
        mask.fill(0.0);
        for code in &self.codes[..count] {
            mask[usize::from(*code)] = 1.0;
        }
        Ok(())
    }

    /// `(visits, visits_since)` for each root edge, in code order.
    ///
    /// Exposed so the separation between inherited exploration and this search's own
    /// can be asserted rather than assumed -- it is the difference between reuse
    /// helping and reuse handing halving a stale ranking.
    #[must_use]
    pub fn root_edge_visits(&self) -> Vec<(u32, u32)> {
        let root = self.nodes[0];
        (root.edges_start..root.edges_start + root.edges_len)
            .map(|index| {
                let edge = self.edges[index as usize];
                (edge.visits, edge.visits_since)
            })
            .collect()
    }

    /// The root's repetition window. Exposed so a resumed search can be checked
    /// against a fresh one at the same position -- the rebase is the part of
    /// extraction most likely to be wrong, and wrong silently.
    #[must_use]
    pub fn root_window(&self) -> &RepetitionWindow {
        &self.root_window
    }

    /// The root's occurrence count within its own window.
    #[must_use]
    pub fn root_rep_count(&self) -> u32 {
        self.nodes[0].rep_count
    }

    /// How many nodes the tree holds. Reuse is only worth anything if this is
    /// large after extraction.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Start a search on an inherited tree from [`Self::into_subtree`].
    ///
    /// THE LOAD-BEARING DECISION: the Gumbel sequential-halving schedule is re-run
    /// from scratch -- fresh draws, fresh candidates/survivors/rounds, `used = 0`,
    /// budget reset. Only `visits`/`value_sum`/`value`/`prior` on inherited nodes
    /// and edges are kept.
    ///
    /// Counting inherited visits toward the plan would break Gumbel's
    /// policy-improvement guarantee, which rests on each candidate receiving a
    /// PLANNED number of visits, and it would bias the target: a candidate with 40
    /// inherited visits and 10 planned ones would dominate the visit-count policy
    /// for reasons that have nothing to do with this search.
    ///
    /// The honest consequence: inherited statistics make each simulation BETTER
    /// INFORMED, they do not make simulations unnecessary. The payoff shows up as
    /// strength at equal simulations, not as fewer simulations for equal strength.
    pub fn resume(
        config: &Config,
        inherited: Self,
        params: PuctParams,
        rng: Lcg32,
    ) -> Result<Self> {
        config.validate()?;
        if config.rows != 9 || config.columns != 9 {
            return Err(PuctError::UnsupportedBoard);
        }
        if params.simulations < 1 {
            return Err(PuctError::InvalidSimulations);
        }
        if params.max_considered < 1 {
            return Err(PuctError::InvalidMaxConsidered);
        }
        if !params.c_puct.is_finite() || params.c_puct <= 0.0 {
            return Err(PuctError::InvalidCPuct);
        }
        if inherited.nodes.is_empty() {
            return Err(PuctError::InvalidState);
        }

        let expanded = inherited.nodes[0].expanded;
        // An expanded root has no network evaluation pending, so its raw value is
        // gone -- what survives is the node's refined estimate. That is the best
        // available and it is NOT the same number a fresh search would hold here,
        // which is a real difference between a resumed root and a fresh one.
        let root_value = if expanded { inherited.nodes[0].value } else { 0.0 };

        let mut search = Self {
            params,
            rng,
            root_value,
            max_depth: 0,
            used: 0,
            budget: i64::from(params.simulations),
            candidates: Vec::new(),
            survivors: Vec::new(),
            rounds: 1,
            per_candidate: 0,
            pass: 0,
            next_survivor: 0,
            round_fresh: true,
            draining_single: false,
            path: Vec::new(),
            pending: Vec::new(),
            virtual_loss: 0.0,
            descent_keys: Vec::new(),
            window_from: 0,
            root_window_active: true,
            pending_leaf: NO_CHILD,
            phase: if expanded { Phase::Ready } else { Phase::RootPending },
            ranking: Vec::new(),
            ..inherited
        };
        if expanded {
            search.seed_candidates();
        }
        Ok(search)
    }

    /// The tree rooted at the node reached by following `codes` from this root,
    /// or `None` when that path was never expanded.
    ///
    /// This is the extraction half of tree reuse: after we move and the opponent
    /// replies, the position we now face was a grandchild of the tree we are about
    /// to throw away, and the visits under it are still valid statistics about the
    /// same game.
    ///
    /// Only the arena and the repetition window come across. The Gumbel schedule is
    /// NOT inherited -- see [`Self::resume`] for why that is load-bearing rather
    /// than incidental.
    #[must_use]
    pub fn into_subtree(self, config: &Config, codes: &[u16]) -> Option<Self> {
        let (nodes, edges, window) = self.extract_parts(config, codes)?;
        Some(Self { nodes, edges, root_window: window, ..self })
    }

    /// Re-root IN PLACE, keeping the statistics; `false` when the path was not in
    /// the tree and the caller must start fresh.
    ///
    /// Same extraction as [`Self::into_subtree`], in the shape a wasm binding can
    /// actually use: consuming `self` across that boundary would mean holding the
    /// search in an `Option` and unwrapping it in every other method.
    pub fn reroot(&mut self, config: &Config, codes: &[u16]) -> bool {
        match self.extract_parts(config, codes) {
            Some((nodes, edges, window)) => {
                self.nodes = nodes;
                self.edges = edges;
                self.root_window = window;
                true
            }
            None => false,
        }
    }

    /// Reset the schedule for a new search on the tree this already holds.
    ///
    /// The statistics stay; everything the Gumbel plan owns is rebuilt. See
    /// [`Self::resume`] for why that separation is load-bearing.
    pub fn restart(&mut self, params: PuctParams, rng: Lcg32) -> Result<()> {
        if params.simulations < 1 {
            return Err(PuctError::InvalidSimulations);
        }
        if params.max_considered < 1 {
            return Err(PuctError::InvalidMaxConsidered);
        }
        if !params.c_puct.is_finite() || params.c_puct <= 0.0 {
            return Err(PuctError::InvalidCPuct);
        }
        let expanded = self.nodes[0].expanded;
        self.params = params;
        self.rng = rng;
        self.root_value = if expanded { self.nodes[0].value } else { 0.0 };
        self.max_depth = 0;
        self.used = 0;
        self.budget = i64::from(params.simulations);
        self.candidates.clear();
        self.survivors.clear();
        self.rounds = 1;
        self.per_candidate = 0;
        self.pass = 0;
        self.next_survivor = 0;
        self.round_fresh = true;
        self.draining_single = false;
        self.path.clear();
        self.pending.clear();
        self.descent_keys.clear();
        self.window_from = 0;
        self.root_window_active = true;
        self.pending_leaf = NO_CHILD;
        self.ranking.clear();
        // The inherited VALUE estimates stay; the inherited exploration counts do
        // not get to drive this search's halving.
        for edge in &mut self.edges {
            edge.visits_since = 0;
        }
        self.phase = if expanded { Phase::Ready } else { Phase::RootPending };
        if expanded {
            self.seed_candidates();
        }
        Ok(())
    }

    fn extract_parts(
        &self,
        config: &Config,
        codes: &[u16],
    ) -> Option<(Vec<Node>, Vec<Edge>, RepetitionWindow)> {
        // 1. Walk to the new root, rebasing the repetition window as we go.
        //
        // The new root's window is NOT the old root's. Each step either extends the
        // window or, at a wall placement, starts a fresh one -- `resets_window`
        // records which. Getting this wrong changes threefold adjudication silently:
        // the search would see draws that are not there, or miss ones that are.
        let mut window = self.root_window.clone();
        let mut current = 0_u32;
        for code in codes {
            let node = self.nodes[current as usize];
            if !node.expanded {
                return None;
            }
            let start = node.edges_start as usize;
            let end = start + node.edges_len as usize;
            let edge = (start..end).find(|index| self.edges[*index].code == *code)?;
            let child = self.edges[edge].child;
            if child == NO_CHILD {
                return None;
            }
            let key = position_key(config, &self.nodes[child as usize].position);
            if self.nodes[child as usize].resets_window {
                window.reset(key);
            } else {
                window.push(key);
            }
            current = child;
        }

        // 2. Reachability copy. The arena has no free list, so compacting in place
        //    would leave a partial remap nobody can review; this builds fresh
        //    vectors and remaps as it goes.
        let mut mapping = vec![NO_CHILD; self.nodes.len()];
        let mut nodes: Vec<Node> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        mapping[current as usize] = 0;
        nodes.push(self.nodes[current as usize]);
        let mut queue = vec![current];
        let mut head = 0_usize;
        while head < queue.len() {
            let old = queue[head];
            head += 1;
            let new_index = mapping[old as usize] as usize;
            let node = self.nodes[old as usize];
            if !node.expanded {
                nodes[new_index].edges_start = 0;
                nodes[new_index].edges_len = 0;
                continue;
            }
            let start = node.edges_start as usize;
            let end = start + node.edges_len as usize;
            let new_start = u32::try_from(edges.len()).ok()?;
            for index in start..end {
                let mut edge = self.edges[index];
                if edge.child != NO_CHILD {
                    let child = edge.child;
                    if mapping[child as usize] == NO_CHILD {
                        mapping[child as usize] = u32::try_from(nodes.len()).ok()?;
                        nodes.push(self.nodes[child as usize]);
                        queue.push(child);
                    }
                    edge.child = mapping[child as usize];
                }
                edges.push(edge);
            }
            nodes[new_index].edges_start = new_start;
        }

        // 3. `rep_count` is relative to a node's OWN window, and the window just
        //    moved, so every copied node's count has to be recomputed rather than
        //    carried over. Walk down from the new root carrying the window.
        let root_key = position_key(config, &nodes[0].position);
        nodes[0].rep_count = window.get(root_key);
        let mut stack: Vec<(u32, RepetitionWindow)> = vec![(0, window.clone())];
        while let Some((index, at)) = stack.pop() {
            let node = nodes[index as usize];
            let start = node.edges_start as usize;
            let end = start + node.edges_len as usize;
            for edge in start..end {
                let child = edges[edge].child;
                if child == NO_CHILD {
                    continue;
                }
                let key = position_key(config, &nodes[child as usize].position);
                let mut next = at.clone();
                if nodes[child as usize].resets_window {
                    next.reset(key);
                } else {
                    next.push(key);
                }
                nodes[child as usize].rep_count = next.get(key);
                stack.push((child, next));
            }
        }

        Some((nodes, edges, window))
    }

    /// The node ids currently awaiting evaluation, in collection order.
    ///
    /// Exposed so a caller (and the tests) can assert the batch holds DISTINCT
    /// leaves: nothing in the descent guarantees it, and a repeat would be
    /// expanded twice.
    #[must_use]
    pub fn pending_leaf_ids(&self) -> Vec<u32> {
        self.pending.iter().map(|slot| slot.leaf).collect()
    }

    /// Set the in-flight penalty used by [`Self::collect_leaves`].
    ///
    /// Leave it at `0.0` for a search that is bit-identical to the sequential
    /// one. See the field's documentation for why anything else is a different
    /// algorithm rather than a scheduling change.
    pub fn set_virtual_loss(&mut self, virtual_loss: f64) -> Result<()> {
        if !virtual_loss.is_finite() || virtual_loss < 0.0 {
            return Err(PuctError::InvalidEvaluation);
        }
        self.virtual_loss = virtual_loss;
        Ok(())
    }

    /// Collect up to `max_n` leaves at once, writing each one's features into
    /// its own `NN_INPUT_PLANES * cells` slice of `features`.
    ///
    /// Returns how many were collected; `0` means the search is finished. The
    /// caller evaluates all of them together and hands the results back through
    /// [`Self::submit_batch`] in the SAME order.
    ///
    /// The root is always a batch of one: nothing else can be selected until it
    /// has been expanded.
    ///
    /// With `virtual_loss == 0.0` a batch never holds two descents into the same
    /// root candidate. Sequential halving visits survivors on a fixed schedule
    /// that no evaluation influences, and distinct survivors own disjoint
    /// subtrees, so those descents cannot interact and batching them is a pure
    /// scheduling change. That also bounds the batch: the last halving round
    /// drains a single survivor, and that phase stays serial.
    pub fn collect_leaves(
        &mut self,
        config: &Config,
        features: &mut [f32],
        max_n: usize,
    ) -> Result<usize> {
        let stride = NN_INPUT_PLANES * config.cells();
        if max_n == 0 || features.len() < stride * max_n {
            return Err(PuctError::InvalidBufferLength);
        }
        if self.awaiting_evaluation() {
            return Err(PuctError::OutOfOrderEvaluation);
        }
        self.pending.clear();
        match self.phase {
            Phase::Done => return Ok(0),
            Phase::RootPending => {
                self.encode(config, 0, &mut features[..stride]);
                self.phase = Phase::RootAwaiting;
                return Ok(1);
            }
            _ => {}
        }

        let mut seen: Vec<usize> = Vec::new();
        while self.pending.len() < max_n {
            // Without a penalty, stop rather than repeat a candidate: a second
            // descent into a subtree that already has an unevaluated leaf would
            // see stale statistics and could pick a different edge than the
            // sequential search did.
            let before = self.scheduler_state();
            let Some(candidate) = self.next_candidate() else { break };
            // A batch must not span a halving boundary. `halve()` ranks the
            // survivors on their accumulated statistics, and a leaf still in
            // flight has not contributed its value yet -- so halving mid-batch
            // decides on stale numbers and keeps a different candidate than the
            // sequential search would. Caught by the equivalence test: two
            // candidates swapped visit counts while the total stayed right.
            // ...but only when something is actually in flight. With an empty
            // batch the halving reads complete statistics and is exactly what the
            // sequential search does here, so deferring it would end the search a
            // round early instead of protecting anything.
            // BOTH guards require something to actually be in flight. A visit that
            // lands on a terminal node scores and unwinds immediately, adding to
            // `seen` without adding to `pending` -- so a later duplicate could break
            // the loop with an EMPTY batch, which the caller reads as "search
            // finished". That silently truncated the search: 241 simulations used of
            // 256, with the lost visits coming off the two best candidates. An empty
            // batch has nothing to conflict with, so neither guard applies to it.
            let in_flight = !self.pending.is_empty();
            let halved = self.survivors != before.0 && in_flight;
            let repeat = self.virtual_loss == 0.0 && in_flight && seen.contains(&candidate);
            if halved || repeat {
                self.restore_scheduler(before);
                break;
            }
            seen.push(candidate);
            // Everything begin_visit/descend can mutate, so a descent that lands on a
            // leaf already in this batch can be undone completely.
            let nodes_before = self.nodes.len();
            let depth_before = self.max_depth;
            let used_before = self.used;
            if self.begin_visit(config, candidate)? {
                // A leaf stays UNEXPANDED until its evaluation is submitted, so a later
                // descent in the same batch can land on it again -- the penalty is the
                // only thing steering away, and one too small to change the argmax does
                // not steer. Expanding a node twice allocates a second edge list,
                // orphans the first, and unwinds the same path twice, inflating the
                // statistics of exactly the line the search likes most.
                if self.pending.iter().any(|slot| slot.leaf == self.pending_leaf) {
                    // Undo the descent and end the batch; the visit happens next time,
                    // against statistics that include this batch's results.
                    for (_, edge) in &self.path {
                        if self.edges[*edge as usize].child >= nodes_before as u32 {
                            self.edges[*edge as usize].child = NO_CHILD;
                        }
                    }
                    self.nodes.truncate(nodes_before);
                    self.max_depth = depth_before;
                    self.used = used_before;
                    self.path.clear();
                    self.pending_leaf = NO_CHILD;
                    self.restore_scheduler(before);
                    break;
                }
                let slot = self.pending.len();
                self.encode(
                    config,
                    self.pending_leaf,
                    &mut features[slot * stride..(slot + 1) * stride],
                );
                let path = core::mem::take(&mut self.path);
                let leaf = self.pending_leaf;
                if self.virtual_loss != 0.0 {
                    self.apply_virtual_loss(&path, leaf);
                }
                self.pending.push(PendingLeaf { path, leaf });
                self.pending_leaf = NO_CHILD;
            }
        }

        if self.pending.is_empty() {
            self.phase = Phase::Done;
            return Ok(0);
        }
        self.phase = Phase::BatchAwaiting;
        Ok(self.pending.len())
    }

    /// Hand back `n` evaluations for the leaves [`Self::collect_leaves`] produced,
    /// in the order it produced them.
    ///
    /// `policies` is `n` consecutive `policy_size` vectors. Applying them in
    /// collection order is what keeps the zero-penalty case bit-identical: every
    /// node accumulates the same values in the same sequence as it would have
    /// sequentially, so even the floating-point sums match.
    pub fn submit_batch(
        &mut self,
        config: &Config,
        policies: &[f32],
        values: &[f64],
        n: usize,
    ) -> Result<()> {
        let width = config.policy_size();
        if policies.len() < width * n || values.len() < n {
            return Err(PuctError::InvalidBufferLength);
        }
        if self.phase == Phase::RootAwaiting {
            if n != 1 {
                return Err(PuctError::OutOfOrderEvaluation);
            }
            return self.submit(config, &policies[..width], values[0]);
        }
        if self.phase != Phase::BatchAwaiting || n != self.pending.len() {
            return Err(PuctError::OutOfOrderEvaluation);
        }
        // Validate the WHOLE batch before mutating anything: a half-applied batch
        // would leave virtual loss on the paths that were never submitted.
        for (index, value) in values[..n].iter().enumerate() {
            if !value.is_finite() || !(-1.0..=1.0).contains(value) {
                return Err(PuctError::InvalidEvaluation);
            }
            for probability in &policies[index * width..(index + 1) * width] {
                if !probability.is_finite() || *probability < 0.0 {
                    return Err(PuctError::InvalidEvaluation);
                }
            }
        }

        let pending = core::mem::take(&mut self.pending);
        for (index, slot) in pending.into_iter().enumerate() {
            if self.virtual_loss != 0.0 {
                self.undo_virtual_loss(&slot.path, slot.leaf);
            }
            let value = values[index];
            self.expand(config, slot.leaf, &policies[index * width..(index + 1) * width], value)?;
            let node = &mut self.nodes[slot.leaf as usize];
            node.visits += 1;
            node.value_sum += value;
            self.path = slot.path;
            self.unwind(value);
        }
        self.path.clear();
        self.phase = Phase::Ready;
        Ok(())
    }

    /// The scheduler's whole mutable position, so a candidate can be looked at
    /// and then put back.
    fn scheduler_state(&self) -> (Vec<usize>, u32, u32, usize, bool, bool, i64) {
        (
            self.survivors.clone(),
            self.per_candidate,
            self.pass,
            self.next_survivor,
            self.round_fresh,
            self.draining_single,
            self.budget,
        )
    }

    fn restore_scheduler(&mut self, state: (Vec<usize>, u32, u32, usize, bool, bool, i64)) {
        let (survivors, per_candidate, pass, next_survivor, round_fresh, draining_single, budget) =
            state;
        self.survivors = survivors;
        self.per_candidate = per_candidate;
        self.pass = pass;
        self.next_survivor = next_survivor;
        self.round_fresh = round_fresh;
        self.draining_single = draining_single;
        self.budget = budget;
    }

    /// Charge an in-flight visit along `path` and its leaf, as a loss for the
    /// side to move at each node. Mirrors `unwind`'s alternation exactly so the
    /// undo is its inverse.
    fn apply_virtual_loss(&mut self, path: &[(u32, u32)], leaf: u32) {
        let vl = self.virtual_loss;
        let node = &mut self.nodes[leaf as usize];
        node.visits += 1;
        node.value_sum -= vl;
        let mut value = -vl;
        for (node_index, edge_index) in path.iter().rev() {
            value = -value;
            let entry = &mut self.edges[*edge_index as usize];
            entry.visits += 1;
            entry.visits_since += 1;
            entry.value_sum += value;
            let entry = &mut self.nodes[*node_index as usize];
            entry.visits += 1;
            entry.value_sum += value;
        }
    }

    fn undo_virtual_loss(&mut self, path: &[(u32, u32)], leaf: u32) {
        let vl = self.virtual_loss;
        let node = &mut self.nodes[leaf as usize];
        node.visits -= 1;
        node.value_sum += vl;
        let mut value = -vl;
        for (node_index, edge_index) in path.iter().rev() {
            value = -value;
            let entry = &mut self.edges[*edge_index as usize];
            entry.visits -= 1;
            entry.visits_since -= 1;
            entry.value_sum -= value;
            let entry = &mut self.nodes[*node_index as usize];
            entry.visits -= 1;
            entry.value_sum -= value;
        }
    }

    pub fn submit(&mut self, config: &Config, policy: &[f32], value: f64) -> Result<()> {
        if policy.len() != config.policy_size() {
            return Err(PuctError::InvalidBufferLength);
        }
        // `readEvaluation` checks the whole policy vector, not just the legal
        // entries, so an out-of-range probability behind an illegal code is a
        // rejected evaluation in both engines.
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(PuctError::InvalidEvaluation);
        }
        for probability in policy {
            if !probability.is_finite() || *probability < 0.0 {
                return Err(PuctError::InvalidEvaluation);
            }
        }
        match self.phase {
            Phase::RootAwaiting => {
                self.expand(config, 0, policy, value)?;
                self.root_value = value;
                self.seed_candidates();
                self.phase = Phase::Ready;
                Ok(())
            }
            Phase::LeafAwaiting => {
                let leaf = self.pending_leaf;
                self.expand(config, leaf, policy, value)?;
                let node = &mut self.nodes[leaf as usize];
                node.visits += 1;
                node.value_sum += value;
                self.unwind(value);
                self.pending_leaf = NO_CHILD;
                self.phase = Phase::Ready;
                Ok(())
            }
            _ => Err(PuctError::OutOfOrderEvaluation),
        }
    }

    /// The completed search's decision. Callable once `next_leaf` returns
    /// `false`; before that the counts are partial and `action_code` is not yet
    /// decided.
    #[must_use]
    pub fn result(&self) -> PuctResult {
        let winner = self
            .survivors
            .first()
            .map_or(0, |index| self.candidates[*index].code);
        // The root's edge list is every legal root action, in the ascending code
        // order `legal_action_codes_fast` produced, so slicing it is already the
        // improved policy's domain and ordering. An unexpanded root has no edges
        // and yields an empty policy.
        let root = self.nodes[0];
        let root_edges =
            &self.edges[root.edges_start as usize..(root.edges_start + root.edges_len) as usize];
        PuctResult {
            action_code: winner,
            visit_counts: self
                .candidates
                .iter()
                .map(|candidate| (candidate.code, self.edges[candidate.edge as usize].visits))
                .collect(),
            improved_policy: improved_policy(root_edges, self.root_value),
            root_value: self.root_value,
            simulations_used: self.used,
            max_depth_reached: self.max_depth,
            considered: self.candidates.iter().map(|c| c.code).collect(),
        }
    }

    /// The root position, for the caller that needs to encode or record it.
    #[must_use]
    pub fn root_position(&self) -> &SearchPosition {
        &self.nodes[0].position
    }

    /// Visits sitting under the node reached by walking `codes` from the root, or
    /// `None` when that path was never expanded.
    ///
    /// READ-ONLY DIAGNOSTIC. It exists to answer one question before any tree-reuse
    /// machinery is built: if the next search inherited this subtree instead of
    /// starting from zero, how many visits would it inherit? Playing our move and the
    /// opponent's reply gives a two-code path, and the child's visit count is exactly
    /// the inheritance.
    ///
    /// This measures the CEILING on what reuse could save, not what it would save. An
    /// inherited visit is worth less than a fresh one: Gumbel's schedule would still
    /// have to be re-run at the new root, so the statistics inform the search rather
    /// than replacing simulations. A small number here therefore rules reuse out; a
    /// large number does not rule it in.
    ///
    /// See `docs/tree-reuse-plan.md` for the decision rule this feeds.
    #[must_use]
    pub fn subtree_visits_after(&self, codes: &[u16]) -> Option<u32> {
        let mut node = 0u32;
        for code in codes {
            let n = self.nodes[node as usize];
            if !n.expanded {
                return None;
            }
            let start = n.edges_start as usize;
            let edge =
                (start..start + n.edges_len as usize).find(|i| self.edges[*i].code == *code)?;
            let child = self.edges[edge].child;
            if child == NO_CHILD {
                // The move is legal and was scored, but never actually visited, so
                // there is no subtree. Zero is the honest answer, not None -- the
                // distinction matters when reading the distribution.
                return Some(0);
            }
            node = child;
        }
        Some(self.nodes[node as usize].visits)
    }

    /* -------------------------------------------------------------- *
     * Root setup
     * -------------------------------------------------------------- */

    /// One Gumbel per legal root action, drawn in ascending code order — the
    /// draw order is part of the contract, because the same seed has to
    /// reproduce the same game in both engines.
    fn seed_candidates(&mut self) {
        let root = self.nodes[0];
        self.candidates.clear();
        self.candidates.reserve(root.edges_len as usize);
        for index in root.edges_start..root.edges_start + root.edges_len {
            let edge = self.edges[index as usize];
            let logit = js_log(edge.prior.max(POLICY_FLOOR));
            self.candidates.push(Candidate {
                code: edge.code,
                edge: index,
                score: self.rng.gumbel() + logit,
            });
        }
        self.select_considered();
    }

    /// Keep the top `m = min(maxConsidered, legalCount)` by `g + logit`, then
    /// restore ascending code order. Ties break to the lowest code, matching
    /// `(right.score - left.score) || (left.code - right.code)`.
    fn select_considered(&mut self) {
        let keep = (self.params.max_considered as usize).min(self.candidates.len());
        self.candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.code.cmp(&right.code))
        });
        self.candidates.truncate(keep);
        self.candidates.sort_by_key(|candidate| candidate.code);

        self.survivors = (0..self.candidates.len()).collect();
        self.rounds = halving_rounds(self.candidates.len());
        self.round_fresh = true;
        self.draining_single = false;
    }

    /* -------------------------------------------------------------- *
     * Sequential halving schedule
     * -------------------------------------------------------------- */

    /// The next candidate to visit, or `None` when the budget and the schedule
    /// are both exhausted.
    ///
    /// This is the reference's nested loop turned inside out. Note that halving
    /// still runs when the budget is gone: the JavaScript's
    /// `while (survivors.length > 1)` keeps cutting the field with zero visits
    /// per round, and the winner it lands on is part of the contract.
    fn next_candidate(&mut self) -> Option<usize> {
        loop {
            if self.draining_single {
                // A single candidate still deserves the budget: it deepens the
                // tree and refines `rootValue`, and the action is forced anyway.
                if self.budget <= 0 {
                    return None;
                }
                return Some(*self.survivors.first()?);
            }
            if self.survivors.len() <= 1 {
                self.draining_single = true;
                continue;
            }
            if self.round_fresh {
                let divisor = self.rounds as u64 * self.survivors.len() as u64;
                let per = (u64::from(self.params.simulations) / divisor) as u32;
                self.per_candidate = per.max(1);
                self.pass = 0;
                self.next_survivor = 0;
                self.round_fresh = false;
            }
            if self.budget <= 0 || self.pass >= self.per_candidate {
                self.halve();
                self.round_fresh = true;
                continue;
            }
            if self.next_survivor >= self.survivors.len() {
                self.pass += 1;
                self.next_survivor = 0;
                continue;
            }
            let candidate = self.survivors[self.next_survivor];
            self.next_survivor += 1;
            return Some(candidate);
        }
    }

    /// Keep the top `ceil(k / 2)` by `g + logit + sigma(qhat)`.
    fn halve(&mut self) {
        // `visits_since`, not `visits`: sigma scales the value term by how far THIS
        // search has got, and inherited visits made it large from the first round,
        // drowning the Gumbel noise and turning halving greedy on a stale estimate.
        // For a search that never resumed the two are identical, so this changes
        // nothing for one.
        let max_visits = self
            .survivors
            .iter()
            .map(|index| self.edges[self.candidates[*index].edge as usize].visits_since)
            .max()
            .unwrap_or(0);

        self.ranking.clear();
        for index in &self.survivors {
            let candidate = self.candidates[*index];
            let edge = self.edges[candidate.edge as usize];
            let qhat = completed_q(&edge, self.root_value);
            self.ranking
                .push((*index, candidate.score + sigma(qhat, max_visits)));
        }
        let codes = &self.candidates;
        self.ranking.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(codes[left.0].code.cmp(&codes[right.0].code))
        });
        let keep = self.survivors.len().div_ceil(2);
        self.ranking.truncate(keep);
        self.survivors.clear();
        self.survivors.extend(self.ranking.iter().map(|(i, _)| *i));
        self.survivors
            .sort_by_key(|index| self.candidates[*index].code);
    }

    /* -------------------------------------------------------------- *
     * Descent
     * -------------------------------------------------------------- */

    /// Start one simulation through `candidate`. Returns `true` when the
    /// descent paused at an unexpanded leaf, `false` when it completed against
    /// a terminal node and has already been backed up.
    fn begin_visit(&mut self, config: &Config, candidate: usize) -> Result<bool> {
        self.budget -= 1;
        self.used += 1;

        self.path.clear();
        self.descent_keys.clear();
        self.window_from = 0;
        self.root_window_active = true;

        let edge = self.candidates[candidate].edge;
        let child = self.step_into(config, 0, edge)?;
        self.descend(config, child)
    }

    /// Take `edge` out of `node`: create the child if this is its first visit,
    /// charge the parent's visited prior, and push the frame.
    fn step_into(&mut self, config: &Config, node: u32, edge: u32) -> Result<u32> {
        let mut child = self.edges[edge as usize].child;
        if child == NO_CHILD {
            child = self.create_child(config, node, edge)?;
            self.edges[edge as usize].child = child;
        } else {
            // Existing child: only the window bookkeeping has to be replayed.
            // The stored count is re-derived under `debug_assertions` because a
            // repetition bug would otherwise be silent — it would just quietly
            // stop scoring some draws.
            let existing = self.nodes[child as usize];
            let key = position_key(config, &existing.position);
            debug_assert_eq!(
                if existing.resets_window {
                    1
                } else {
                    self.window_count(key) + 1
                },
                existing.rep_count,
                "repetition bookkeeping diverged when re-descending an existing node"
            );
            self.note_descent(key, existing.resets_window);
        }
        if self.edges[edge as usize].visits == 0 {
            self.nodes[node as usize].visited_prior += self.edges[edge as usize].prior;
        }
        self.path.push((node, edge));
        Ok(child)
    }

    /// How many times `key` has already occurred in the live repetition window:
    /// the root's own counts while no wall has been placed on this descent, plus
    /// the path positions since the most recent wall.
    fn window_count(&self, key: u32) -> u32 {
        let base = if self.root_window_active {
            self.root_window.get(key)
        } else {
            0
        };
        let seen = self.descent_keys[self.window_from..]
            .iter()
            .filter(|entry| **entry == key)
            .count() as u32;
        base + seen
    }

    /// Extend the descent's repetition bookkeeping by one ply.
    fn note_descent(&mut self, key: u32, resets_window: bool) {
        self.descent_keys.push(key);
        if resets_window {
            self.root_window_active = false;
            self.window_from = self.descent_keys.len() - 1;
        }
    }

    // Repetition and ply cap both yield a draw, so clippy sees identical arms in
    // the terminal chain below. They stay separate because it mirrors
    // `adjudicate`'s documented order -- goal, then repetition, then ply cap --
    // and those are three distinct reasons a game ended. Collapsing them would
    // save a line and lose the correspondence.
    #[allow(clippy::if_same_then_else)]
    fn create_child(&mut self, config: &Config, node: u32, edge: u32) -> Result<u32> {
        let parent = self.nodes[node as usize];
        let code = self.edges[edge as usize].code;
        let applied = apply_action_code(config, &parent.position, code)?;
        let key = position_key(config, &applied.position);
        let ply = parent.ply + 1;

        let rep_count = if applied.placed_wall {
            1
        } else {
            self.window_count(key) + 1
        };

        // `adjudicate`, in its order: goal, then repetition, then ply cap.
        let (terminal, terminal_value) = if let Some(winner) = applied.goal_winner {
            let value = if winner == applied.position.turn {
                1.0
            } else {
                -1.0
            };
            (true, value)
        } else if u64::from(rep_count) >= u64::from(config.repetition_threshold) {
            (true, 0.0)
        } else if ply >= config.ply_cap {
            (true, 0.0)
        } else {
            (false, 0.0)
        };

        self.note_descent(key, applied.placed_wall);
        self.nodes.push(Node {
            position: applied.position,
            ply,
            rep_count,
            resets_window: applied.placed_wall,
            terminal,
            terminal_value,
            expanded: false,
            edges_start: 0,
            edges_len: 0,
            visits: 0,
            value_sum: 0.0,
            value: 0.0,
            visited_prior: 0.0,
        });
        Ok((self.nodes.len() - 1) as u32)
    }

    /// Walk down from `node` until a terminal node (scored and backed up here)
    /// or an unexpanded leaf (paused). Returns `true` when paused.
    fn descend(&mut self, config: &Config, mut node: u32) -> Result<bool> {
        loop {
            self.max_depth = self.max_depth.max(self.path.len() as u32);
            let current = self.nodes[node as usize];

            if current.terminal {
                // Terminal: scored by the engine, never expanded, no evaluation.
                let value = current.terminal_value;
                let entry = &mut self.nodes[node as usize];
                entry.visits += 1;
                entry.value_sum += value;
                self.unwind(value);
                return Ok(false);
            }
            if !current.expanded {
                self.pending_leaf = node;
                return Ok(true);
            }
            let edge = self.select_edge(node);
            node = self.step_into(config, node, edge)?;
        }
    }

    /// `argmax(Q + cPuct * P * sqrt(sumN) / (1 + N))`, ties to the lowest code.
    ///
    /// Edges are stored ascending by code and the comparison is strict, so the
    /// tie-break needs no extra work. The FPU fallback is hoisted out of the
    /// loop: it depends only on the node, and hoisting a value out of a loop
    /// changes no float operation.
    fn select_edge(&self, node: u32) -> u32 {
        let current = self.nodes[node as usize];
        let sqrt_total = f64::from(current.visits).sqrt();
        let node_value = if current.visits > 0 {
            current.value_sum / f64::from(current.visits)
        } else {
            current.value
        };
        let fpu = clamp_value(node_value - FPU_REDUCTION * current.visited_prior.sqrt());

        let mut best = current.edges_start;
        let mut best_score = f64::NEG_INFINITY;
        for index in current.edges_start..current.edges_start + current.edges_len {
            let edge = self.edges[index as usize];
            let q = if edge.visits > 0 {
                edge.value_sum / f64::from(edge.visits)
            } else {
                fpu
            };
            let score =
                q + self.params.c_puct * edge.prior * sqrt_total / f64::from(1 + edge.visits);
            if score > best_score {
                best_score = score;
                best = index;
            }
        }
        best
    }

    /// Back a leaf value up the paused path, negating once per ply boundary.
    fn unwind(&mut self, mut value: f64) {
        while let Some((node, edge)) = self.path.pop() {
            value = -value;
            let entry = &mut self.edges[edge as usize];
            entry.visits += 1;
            entry.visits_since += 1;
            entry.value_sum += value;
            let entry = &mut self.nodes[node as usize];
            entry.visits += 1;
            entry.value_sum += value;
        }
    }

    /* -------------------------------------------------------------- *
     * Expansion
     * -------------------------------------------------------------- */

    fn encode(&self, config: &Config, node: u32, features: &mut [f32]) {
        let position = self.nodes[node as usize].position;
        crate::encode_board_into(
            config,
            &position.board(),
            position.pawns,
            position.stock,
            position.turn,
            features,
        );
    }

    /// Install `node`'s edges from the network's policy, masked to the legal
    /// actions and renormalised. A policy with no mass on any legal action
    /// falls back to uniform, so the search never divides by zero.
    fn expand(&mut self, config: &Config, node: u32, policy: &[f32], value: f64) -> Result<()> {
        let position = self.nodes[node as usize].position;
        let count = position.legal_action_codes_fast(config, &mut self.codes);
        if count == 0 {
            return Err(PuctError::NoLegalActions);
        }

        let mut mass = 0.0_f64;
        for code in &self.codes[..count] {
            let probability = f64::from(policy[usize::from(*code)]);
            if !probability.is_finite() || probability < 0.0 {
                return Err(PuctError::InvalidEvaluation);
            }
            mass += probability;
        }

        let start = self.edges.len() as u32;
        self.edges.reserve(count);
        for code in &self.codes[..count] {
            let prior = if mass > 0.0 {
                f64::from(policy[usize::from(*code)]) / mass
            } else {
                1.0 / count as f64
            };
            self.edges.push(Edge {
                code: *code,
                prior,
                visits: 0,
                visits_since: 0,
                value_sum: 0.0,
                child: NO_CHILD,
            });
        }

        let entry = &mut self.nodes[node as usize];
        entry.edges_start = start;
        entry.edges_len = count as u32;
        entry.expanded = true;
        entry.value = value;
        entry.visited_prior = 0.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_rounds_matches_the_javascript_expression() {
        // `Math.ceil(Math.log2(Math.max(m, 2)))`, cross-checked against node for
        // the whole reachable range.
        let expected = [
            (1_usize, 1_u32),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (9, 4),
            (16, 4),
            (17, 5),
            (128, 7),
            (129, 8),
            (209, 8),
        ];
        for (candidates, rounds) in expected {
            assert_eq!(halving_rounds(candidates), rounds, "m = {candidates}");
        }
    }

    #[test]
    fn sigma_is_strictly_increasing_in_q() {
        assert!(sigma(0.1, 4) < sigma(0.2, 4));
        assert_eq!(sigma(0.5, 0), 25.0);
    }

    /// An edge with `visits` visits averaging `q`, so `completed_q` reads back
    /// exactly `q` when `visits > 0` and the root value when it is 0.
    fn edge(code: u16, prior: f64, visits: u32, q: f64) -> Edge {
        Edge {
            code,
            prior,
            visits,
            // These helpers build edges for a search that never resumed, where the
            // two counters are by definition equal.
            visits_since: visits,
            value_sum: f64::from(visits) * q,
            child: NO_CHILD,
        }
    }

    fn mass(policy: &[(u16, f64)], code: u16) -> f64 {
        policy
            .iter()
            .find(|(candidate, _)| *candidate == code)
            .unwrap_or_else(|| panic!("code {code} is missing from the improved policy"))
            .1
    }

    #[test]
    fn improved_policy_is_a_distribution_over_exactly_the_edges_given() {
        let edges = [
            edge(3, 0.5, 12, 0.2),
            edge(11, 0.25, 6, -0.1),
            edge(40, 0.2, 0, 0.0),
            // A legal action the network gave literally zero prior: floored, so
            // it is very unlikely rather than impossible.
            edge(97, 0.0, 0, 0.0),
        ];
        let policy = improved_policy(&edges, 0.05);

        assert_eq!(
            policy.iter().map(|(code, _)| *code).collect::<Vec<_>>(),
            vec![3, 11, 40, 97],
            "the improved policy must cover exactly the edges, ascending by code"
        );
        let total: f64 = policy.iter().map(|(_, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "improved policy sums to {total}, not 1"
        );
        for (code, probability) in &policy {
            assert!(
                probability.is_finite() && *probability > 0.0,
                "code {code} carries {probability}"
            );
        }
    }

    /// The property the visit distribution could not express.
    ///
    /// Both finalists of a sequential-halving round hold *identical* visit
    /// counts — that is the schedule, not a judgement — so the old target gave
    /// them identical mass however far apart their values were. The improved
    /// policy separates them by exactly `exp(sigma(dq))`.
    #[test]
    fn a_better_completed_q_takes_strictly_more_mass_at_equal_visits() {
        let good = edge(5, 0.25, 30, 0.4);
        let bad = edge(9, 0.25, 30, -0.2);
        assert_eq!(good.visits, bad.visits);
        assert_eq!(good.prior, bad.prior);

        let policy = improved_policy(&[good, bad], 0.0);
        let (good_mass, bad_mass) = (mass(&policy, 5), mass(&policy, 9));
        assert!(
            good_mass > bad_mass,
            "Q = 0.4 took {good_mass}, Q = -0.2 took {bad_mass}"
        );
        // Equal priors cancel, so the ratio is the sigma gap alone.
        let expected = sigma(0.6, 30).exp();
        assert!(
            ((good_mass / bad_mass) / expected - 1.0).abs() < 1e-9,
            "mass ratio {} is not exp(sigma(0.6, 30)) = {expected}",
            good_mass / bad_mass
        );
    }

    /// An action the halving never visited is completed with the root value, so
    /// two unvisited actions are separated by their priors alone — which is how
    /// the improved policy covers moves the considered set skipped instead of
    /// targeting them to zero.
    #[test]
    fn unvisited_actions_are_completed_with_the_root_value() {
        let policy = improved_policy(&[edge(2, 0.6, 0, 0.0), edge(8, 0.15, 0, 0.0)], -0.3);
        assert!(
            (mass(&policy, 2) / mass(&policy, 8) - 4.0).abs() < 1e-9,
            "equal completions must leave the prior ratio intact"
        );
        // And a visited action beating the root value outranks a better-priored
        // unvisited one, which no visit-count target could say either.
        let policy = improved_policy(&[edge(2, 0.9, 0, 0.0), edge(8, 0.1, 4, 0.5)], -0.3);
        assert!(mass(&policy, 8) > mass(&policy, 2));
    }

    #[test]
    fn a_single_legal_action_gives_a_one_hot() {
        let policy = improved_policy(&[edge(17, 1.0, 3, -0.75)], -0.75);
        assert_eq!(policy, vec![(17, 1.0)]);
        // Exactly 1.0 in f32 too: this is the target a forced move records.
        assert_eq!(policy[0].1 as f32, 1.0_f32);
    }

    /// `sigma` grows with the visit budget, so without the max-subtraction the
    /// exponentials overflow to infinity and every probability comes back NaN.
    /// 100k visits puts the raw exponent past 100050, far beyond `f64::MAX`.
    #[test]
    fn the_softmax_does_not_overflow_at_a_large_visit_budget() {
        let policy = improved_policy(
            &[edge(1, 0.5, 100_000, 1.0), edge(2, 0.5, 100_000, -1.0)],
            0.0,
        );
        let total: f64 = policy.iter().map(|(_, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "improved policy sums to {total}, not 1"
        );
        assert_eq!(mass(&policy, 1), 1.0, "the winner should hold all the mass");
        assert_eq!(mass(&policy, 2), 0.0, "the loser underflows, not NaNs");
    }

    #[test]
    fn an_unexpanded_root_yields_an_empty_improved_policy() {
        assert!(improved_policy(&[], 0.0).is_empty());
    }

    #[test]
    fn clamp_value_passes_nan_through_like_javascript() {
        assert_eq!(clamp_value(2.0), 1.0);
        assert_eq!(clamp_value(-2.0), -1.0);
        assert_eq!(clamp_value(0.25), 0.25);
        assert!(clamp_value(f64::NAN).is_nan());
    }

    #[test]
    fn repetition_window_counts_and_resets() {
        let mut window = RepetitionWindow::fresh(7);
        assert_eq!(window.get(7), 1);
        assert_eq!(window.push(7), 2);
        assert_eq!(window.push(9), 1);
        window.reset(9);
        assert_eq!(window.get(7), 0);
        assert_eq!(window.get(9), 1);
    }
}
