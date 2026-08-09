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
/// `v2` was the completed-Q policy target: the recorded `policyTarget` became
/// the Gumbel improved policy over every legal root action
/// ([`PuctResult::improved_policy`]) instead of the normalised visit counts of
/// the considered set. Nothing about which move the search *played* changed, so
/// that bump named a record-format change only.
///
/// `v3` is different in kind. It replaces this project's own `sigma` — raw
/// completed-Q scaled by `(50 + maxN) * 1.0` — with the reference qtransform,
/// `mctx`'s [`qtransform_completed_by_mix_value`]: completed-Q is min-max
/// rescaled to `[0, 1]` per node and scaled by `(50 + maxN) * 0.1`, and the
/// completion for an unvisited action is `v_mix` rather than the raw root
/// value. That expression is used in BOTH places the old one was — the
/// sequential-halving ranking and the improved policy — so **`v3` changes
/// search decisions, not just the recorded target**. There is no bit-parity
/// bridge from `v2`, by design; the correctness anchor is
/// `tests/qtransform_goldens.rs` (identical to `mctx`) plus
/// `tests/qtransform_properties.rs`, not identity with the old behaviour.
pub const PUCT_SEARCH_VERSION: &str = "puct-az-tree-v3";

/// The version the JavaScript reference search is frozen at.
///
/// `js/normal-duel-puct-search.mjs` deliberately stays on `v1`. Through `v2` it
/// was the parity oracle for every *search decision* — visit counts, chosen
/// action, root value, simulations spent, considered set — because the improved
/// policy was a read-out of a finished tree and moved nothing.
///
/// `v3` ends that. The qtransform sits inside the sequential-halving ranking, so
/// a `v1` tree and a `v3` tree visit different children and can finish on
/// different actions. `tests/js_puct_parity.rs` therefore splits: the quantities
/// the qtransform provably cannot reach are still compared exactly across both
/// engines (the Gumbel considered set, the root value, the budget accounting,
/// and — at `max_considered = 1`, where halving never runs — the whole descent,
/// backup, repetition and terminality path at full strength), while the
/// halving-dependent quantities are compared separately and reported as the
/// by-design divergence they are. See that file's module docs for the split.
///
/// Porting `v3` to the JavaScript would not fix this: the reference is the
/// oracle precisely because it did not change, and the production self-play
/// driver is the Rust/wasm [`crate::selfplay::SelfPlayBatch`] anyway. The
/// improved policy could never be compared there in the first place — it needs
/// `Math.exp` against [`f64::exp`] bit for bit, and `exp` is not an IEEE-754
/// operation any more than [`crate::js_math::js_log`]'s `Math.log` is.
///
/// `tests/js_puct_parity.rs` asserts the JavaScript still reports this string
/// and that the Rust side is on neither `v1` nor `v2`, so neither a silent JS
/// bump nor a silent revert of the qtransform can pass.
pub const JS_REFERENCE_SEARCH_VERSION: &str = "puct-az-tree-v1";

/// The record format `v3` replaced, named so the freeze test can assert the
/// Rust side has not silently reverted to it.
pub const SUPERSEDED_SEARCH_VERSION: &str = "puct-az-tree-v2";

/// Floor applied inside the logit so a legal action the policy assigns exactly
/// zero probability is merely very unlikely, not `-Infinity`.
const POLICY_FLOOR: f64 = 1e-9;

/// Gumbel-MuZero qtransform constants: the defaults of `mctx`'s
/// `qtransform_completed_by_mix_value`, which
/// `tests/fixtures/qtransform-mctx-goldens.json` was generated with.
///
/// `MAXVISIT_INIT` is the paper's `c_visit`, `VALUE_SCALE` its `c_scale`, and
/// `RESCALE_EPSILON` the floor on the min-max denominator. The pairing matters
/// more than either constant: `value_scale = 0.1` is only gentle *because* the
/// completed Q-values it multiplies have been rescaled into `[0, 1]` first. Used
/// on raw completed-Q in `[-1, 1]` it would be a 20x sharper target than `v2`'s,
/// not 20x softer.
const MAXVISIT_INIT: f64 = 50.0;
const VALUE_SCALE: f64 = 0.1;
const RESCALE_EPSILON: f64 = 1e-8;

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

/// The prior as the qtransform reads it: [`POLICY_FLOOR`] applied, so a legal
/// action the network priced at exactly zero contributes a vanishing weight to
/// `v_mix` instead of turning its denominator into `0 / 0`, and a finite logit
/// to the improved policy instead of `-Infinity`.
///
/// `mctx` floors in the same two places, with `finfo(dtype).tiny` rather than
/// `1e-9`. The two agree for every prior at or above `1e-9`, which is every
/// prior this engine produces that is not exactly zero after the legal-mask
/// renormalisation. Where they could differ, the goldens are generated *through
/// this floor* (`scripts/gen-qtransform-goldens.py` hands `mctx` the logits
/// `log(max(p, 1e-9))`), so the fixtures pin the engine's own convention. That
/// is sound because both consumers are invariant to a positive rescaling of the
/// prior: the improved policy is a softmax, where a constant logit shift
/// cancels, and `v_mix` divides by the sum of the visited priors, where a
/// constant factor cancels.
fn effective_prior(prior: f64) -> f64 {
    prior.max(POLICY_FLOOR)
}

/// `v_mix` from Appendix D of *Policy improvement by planning with Gumbel*: the
/// node's own value interpolated with the prior-weighted mean of the Q-values
/// the tree actually measured.
///
/// ```text
/// v_mix = (v_raw + N * (sum_visited p(a) q(a) / sum_visited p(a))) / (1 + N)
/// ```
///
/// `N` is the total number of CHILD visits, so `N = 0` returns `v_raw`
/// unchanged, and a deep search converges on the weighted mean. `v_raw` is the
/// network's own value for the node — not `value_sum / visits`, which is the
/// search's refinement of it; `mctx` reads `tree.raw_values` here and the
/// distinction is pinned by a fixture whose `node_values` deliberately differs.
///
/// The denominator sums only the VISITED actions' priors, which is where
/// [`effective_prior`] earns its keep: a search that only ever visited actions
/// the network priced at exactly zero would otherwise divide zero by zero.
///
/// Each term is divided by that denominator before being summed, rather than
/// the sum being divided once, because that is the association `mctx` uses and
/// the goldens are compared at 1e-6 against it.
fn mixed_value(edges: &[Edge], raw_value: f64) -> f64 {
    let mut visits = 0_u64;
    let mut prior_mass = 0.0_f64;
    for edge in edges {
        if edge.visits > 0 {
            visits += u64::from(edge.visits);
            prior_mass += effective_prior(edge.prior);
        }
    }
    if visits == 0 {
        return raw_value;
    }
    // `mctx` guards the same division; with the floor above, `prior_mass` is
    // only ever zero when nothing was visited, which returned already.
    let denominator = if prior_mass > 0.0 { prior_mass } else { 1.0 };
    let mut weighted_q = 0.0_f64;
    for edge in edges {
        if edge.visits > 0 {
            let q = edge.value_sum / f64::from(edge.visits);
            weighted_q += effective_prior(edge.prior) * q / denominator;
        }
    }
    let visits = visits as f64;
    (raw_value + visits * weighted_q) / (visits + 1.0)
}

/// `completedQ`: a visited root action is worth what the tree measured, an
/// unvisited one is worth `mixed` — [`mixed_value`]'s `v_mix`.
///
/// Through `v2` the unvisited case was the raw root value. That biased the
/// completion for every action the search never touched — at 128 simulations
/// over 16 candidates on a 40-move board, most of them — in whichever direction
/// the value head happened to run relative to the Q-values the tree had
/// actually measured. `v_mix` interpolates between the two instead, and this is
/// the one place the substitution happens.
fn completed_q(edge: &Edge, mixed: f64) -> f64 {
    if edge.visits > 0 {
        edge.value_sum / f64::from(edge.visits)
    } else {
        mixed
    }
}

/// `mctx`'s `qtransform_completed_by_mix_value` over one node's whole edge list:
/// the completed Q-values, min-max rescaled to `[0, 1]` across the node, times
/// `(50 + maxN) * 0.1`.
///
/// This is the ONE expression both the sequential-halving ranking
/// ([`PuctTreeSearch::halve`]) and the improved policy ([`improved_policy`])
/// score with, so the schedule and the recorded target cannot drift apart — the
/// property `v2` had and `v3` keeps. What changed is what the expression *is*.
///
/// Three details are load-bearing.
///
/// The rescale is over the COMPLETED values, `v_mix` included, not over the
/// measured ones — an unvisited action can therefore set the minimum or the
/// maximum. And it is over the node's whole edge list, not the halving's
/// surviving subset, because the improved policy has to cover the actions
/// halving discarded and both readers must see the same numbers.
///
/// `RESCALE_EPSILON` floors the denominator. Its job is not overflow, it is
/// meaning: when every completed value is equal the search has learned nothing
/// about the ordering, `(c - min)` is exactly `0` for every action, and dividing
/// by the floor keeps it exactly `0` — a flat boost, and therefore an improved
/// policy that is exactly the renormalised prior. A search that learned nothing
/// sharpens nothing. `tests/qtransform_properties.rs` holds this to it.
///
/// The output is bounded: every entry lies in `[0, (50 + maxN) * 0.1]`. That is
/// the whole point of D1. The `v2` expression put it in `[-(50 + maxN), (50 +
/// maxN)]`, a span twenty times wider and centred differently, which is a
/// factor of `e^160` rather than `e^8` between the best and worst action at 128
/// simulations.
fn qtransform_completed_by_mix_value(edges: &[Edge], raw_value: f64) -> Vec<f64> {
    let mixed = mixed_value(edges, raw_value);
    let mut values: Vec<f64> = edges.iter().map(|edge| completed_q(edge, mixed)).collect();
    if values.is_empty() {
        return values;
    }

    let mut lowest = f64::INFINITY;
    let mut highest = f64::NEG_INFINITY;
    for value in &values {
        if *value < lowest {
            lowest = *value;
        }
        if *value > highest {
            highest = *value;
        }
    }
    let span = (highest - lowest).max(RESCALE_EPSILON);
    let max_visits = edges.iter().map(|edge| edge.visits).max().unwrap_or(0);
    let scale = (MAXVISIT_INIT + f64::from(max_visits)) * VALUE_SCALE;
    for value in &mut values {
        *value = scale * ((*value - lowest) / span);
    }
    values
}

/// The Gumbel improved policy over `edges`, which must be one node's whole edge
/// list: `pi'(a) ∝ exp(logit(a) + qtransform(a))`.
///
/// This is `mctx`'s `gumbel_muzero_policy` `action_weights`, restricted to the
/// legal actions — which is exactly the domain of an edge list, so there is no
/// `invalid_actions` mask to apply; illegal codes never enter and the recorder
/// writes them as exact `0.0`.
///
/// `logit(a)` is `js_log(effective_prior(a))`, the same expression
/// `seed_candidates` uses to rank the Gumbel draws, so the improved policy and
/// the considered set read the prior through one floor.
///
/// The softmax subtracts the maximum score before exponentiating. The
/// qtransform's `[0, (50 + maxN) * 0.1]` range makes overflow far less likely
/// than it was under `v2`, but not impossible — at a six-figure visit count the
/// scale alone is five figures — and the subtraction also keeps the largest
/// term exactly `1.0`, so the total is in `[1, edges.len()]` and can neither
/// overflow nor be zero.
fn improved_policy(edges: &[Edge], root_value: f64) -> Vec<(u16, f64)> {
    let boosts = qtransform_completed_by_mix_value(edges, root_value);

    let mut scored: Vec<(u16, f64)> = Vec::with_capacity(edges.len());
    let mut highest = f64::NEG_INFINITY;
    for (edge, boost) in edges.iter().zip(boosts) {
        let logit = js_log(effective_prior(edge.prior));
        let score = logit + boost;
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

/// One root action's search statistics, as the qtransform reads them.
///
/// The engine's own [`Edge`] carries a code and a child pointer the transform
/// has no use for, and a `value_sum` rather than a mean; this is the same data
/// in the shape `mctx` states it in, so a fixture can be fed to both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionStats {
    /// The network's prior for this action, masked to the legal set and
    /// renormalised: a probability, not a logit.
    pub prior: f64,
    pub visits: u32,
    /// The action's mean value from the tree. Undefined when `visits == 0`, and
    /// never read in that case — the completion replaces it.
    pub qvalue: f64,
}

/// Every stage of the Gumbel-MuZero qtransform, kept separate.
///
/// A cross-implementation fixture that only compared the final weights could
/// say two implementations disagree but not where; these four fields are
/// exactly what `tests/fixtures/qtransform-mctx-goldens.json` records from
/// `mctx`, so a mismatch names its own stage.
#[derive(Debug, Clone, PartialEq)]
pub struct RootQtransform {
    /// `v_mix`: the completion used for every unvisited action.
    pub mixed_value: f64,
    /// Completed Q-values, before the rescale.
    pub completed: Vec<f64>,
    /// Completed, min-max rescaled and visit-scaled: the boost added to the
    /// logit in both the halving ranking and the improved policy.
    pub transformed: Vec<f64>,
    /// The improved policy `pi'`, in the order `stats` was given.
    pub action_weights: Vec<f64>,
}

/// Run the qtransform the search itself uses over raw statistics.
///
/// This exists so `tests/qtransform_goldens.rs` and
/// `tests/qtransform_properties.rs` can address the transform directly instead
/// of reaching it through a whole 9x9 search, and it deliberately calls the
/// same private functions the search does: there is no second implementation
/// here to drift from the first.
#[must_use]
pub fn root_qtransform(stats: &[ActionStats], raw_value: f64) -> RootQtransform {
    let edges: Vec<Edge> = stats
        .iter()
        .enumerate()
        .map(|(index, stat)| Edge {
            // Synthetic, and only ever passed back out again: the transform
            // reads the prior, the visits and the value sum.
            code: index as u16,
            prior: stat.prior,
            visits: stat.visits,
            value_sum: f64::from(stat.visits) * stat.qvalue,
            child: NO_CHILD,
        })
        .collect();
    let mixed = mixed_value(&edges, raw_value);
    RootQtransform {
        mixed_value: mixed,
        completed: edges.iter().map(|edge| completed_q(edge, mixed)).collect(),
        transformed: qtransform_completed_by_mix_value(&edges, raw_value),
        action_weights: improved_policy(&edges, raw_value)
            .into_iter()
            .map(|(_, weight)| weight)
            .collect(),
    }
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
    Done,
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
        matches!(self.phase, Phase::RootAwaiting | Phase::LeafAwaiting)
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
                Phase::RootAwaiting | Phase::LeafAwaiting => {
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

    /// Keep the top `ceil(k / 2)` by `g + logit + qtransform(a)`.
    ///
    /// The qtransform is evaluated over the ROOT's whole edge list, not over the
    /// surviving subset, for two reasons. It is what `mctx` does — its
    /// `seq_halving.score_considered` is handed the same `completed_qvalues`
    /// vector `action_weights` is built from, computed once over the root. And
    /// it is what makes "one shared expression" true rather than nearly true:
    /// under `v2` this ranking took `max_visits` over the survivors while the
    /// improved policy took it over every edge, so the two already disagreed on
    /// the scale factor. Now they cannot.
    fn halve(&mut self) {
        let root = self.nodes[0];
        let start = root.edges_start as usize;
        let end = start + root.edges_len as usize;
        let boosts = qtransform_completed_by_mix_value(&self.edges[start..end], self.root_value);

        self.ranking.clear();
        for index in &self.survivors {
            let candidate = self.candidates[*index];
            let boost = boosts[candidate.edge as usize - start];
            self.ranking.push((*index, candidate.score + boost));
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

    /// The ranking property `sigma` was originally chosen for, restated for the
    /// transform that replaced it: it is still strictly increasing in `q`, so it
    /// still never reorders two candidates by value. What changed is the range.
    #[test]
    fn the_qtransform_is_strictly_increasing_in_q_and_bounded() {
        let boosts = qtransform_completed_by_mix_value(
            &[
                edge(1, 0.25, 4, -0.5),
                edge(2, 0.25, 4, 0.1),
                edge(3, 0.25, 4, 0.2),
                edge(4, 0.25, 4, 0.9),
            ],
            0.0,
        );
        for pair in boosts.windows(2) {
            assert!(
                pair[0] < pair[1],
                "boosts are not increasing in q: {boosts:?}"
            );
        }
        // Min-max rescaling pins the ends, whatever the Q values were.
        assert_eq!(boosts[0], 0.0);
        assert_eq!(boosts[3], (MAXVISIT_INIT + 4.0) * VALUE_SCALE);
        // The `v2` expression would have put the span at 2 * (50 + 4) = 108.
        assert!((boosts[3] - boosts[0] - 5.4).abs() < 1e-12);
    }

    /// An edge with `visits` visits averaging `q`, so `completed_q` reads back
    /// exactly `q` when `visits > 0` and the root value when it is 0.
    fn edge(code: u16, prior: f64, visits: u32, q: f64) -> Edge {
        Edge {
            code,
            prior,
            visits,
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
        // Equal priors cancel, so the ratio is the qtransform gap alone. Under
        // the min-max rescale that gap is the full scale factor whatever the Q
        // values were -- `exp(8)`, where `v2` charged `exp(sigma(0.6, 30))`,
        // which is `exp(48)`.
        let expected = ((MAXVISIT_INIT + 30.0) * VALUE_SCALE).exp();
        assert!(
            ((good_mass / bad_mass) / expected - 1.0).abs() < 1e-9,
            "mass ratio {} is not exp((50 + 30) * 0.1) = {expected}",
            good_mass / bad_mass
        );
        assert!(
            expected < (0.6_f64 * 80.0).exp(),
            "the v3 ratio must be far below the v2 one"
        );
    }

    /// An action the halving never visited is completed with `v_mix`, so two
    /// unvisited actions are separated by their priors alone — which is how the
    /// improved policy covers moves the considered set skipped instead of
    /// targeting them to zero.
    #[test]
    fn unvisited_actions_are_completed_with_the_mixed_value() {
        let policy = improved_policy(&[edge(2, 0.6, 0, 0.0), edge(8, 0.15, 0, 0.0)], -0.3);
        assert!(
            (mass(&policy, 2) / mass(&policy, 8) - 4.0).abs() < 1e-9,
            "equal completions must leave the prior ratio intact"
        );
        // And a visited action beating the completion outranks a better-priored
        // unvisited one, which no visit-count target could say either.
        let policy = improved_policy(&[edge(2, 0.9, 0, 0.0), edge(8, 0.1, 4, 0.5)], -0.3);
        assert!(mass(&policy, 8) > mass(&policy, 2));
    }

    /// The `v2` completion and the `v3` one, on the configuration that made the
    /// difference matter: a value head running cold against a tree that has
    /// found something.
    ///
    /// `v_mix = (-1.0 + 4 * 0.9) / 5 = 0.52`, so the unvisited action is
    /// completed near the visited one rather than a full point below it. Under
    /// `v2`'s raw root value the unvisited action would have been completed at
    /// `-1.0` — the bottom of the range — and crushed.
    #[test]
    fn mixed_value_lifts_the_unvisited_set_off_a_cold_root_value() {
        let edges = [edge(1, 0.5, 4, 0.9), edge(2, 0.5, 0, 0.0)];
        assert!((mixed_value(&edges, -1.0) - 0.52).abs() < 1e-12);

        let boosts = qtransform_completed_by_mix_value(&edges, -1.0);
        let span = boosts[0] - boosts[1];
        assert!((span - (MAXVISIT_INIT + 4.0) * VALUE_SCALE).abs() < 1e-12);

        // The unvisited action keeps a real share of the mass: `exp(-5.4)`
        // relative to the visited one, not `exp(-108)`.
        let policy = improved_policy(&edges, -1.0);
        assert!(
            mass(&policy, 2) > 4e-3,
            "the unvisited action took {}",
            mass(&policy, 2)
        );
    }

    /// `v_mix` interpolates: `N = 0` is the raw value, and a large `N` converges
    /// on the prior-weighted mean of the visited Q-values.
    #[test]
    fn mixed_value_interpolates_between_the_raw_value_and_the_visited_mean() {
        let unvisited = [edge(1, 0.7, 0, 0.0), edge(2, 0.3, 0, 0.0)];
        assert_eq!(mixed_value(&unvisited, 0.25), 0.25);

        // Weighted mean of the visited pair: (0.2 * 0.5 + 0.6 * -0.5) / 0.8.
        let weighted = (0.2 * 0.5 + 0.6 * -0.5) / 0.8;
        let mut previous = f64::INFINITY;
        for visits in [1_u32, 4, 64, 4096] {
            let edges = [
                edge(1, 0.2, visits, 0.5),
                edge(2, 0.6, visits, -0.5),
                edge(3, 0.2, 0, 0.0),
            ];
            let mixed = mixed_value(&edges, 1.0);
            assert!(
                mixed > weighted && mixed < 1.0,
                "v_mix {mixed} left the interval [{weighted}, 1.0]"
            );
            assert!(mixed < previous, "v_mix must approach the weighted mean");
            previous = mixed;
        }
        assert!((previous - weighted).abs() < 1e-3);
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
