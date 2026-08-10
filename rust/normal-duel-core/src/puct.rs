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
/// and — at `max_considered = 1`, where halving never runs — every field,
/// though with one candidate the only descent-sensitive one left is the depth
/// reached), while the halving-dependent quantities are compared separately and
/// reported as the by-design divergence they are. See that file's module docs
/// for the split and for what it does and does not buy.
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

/// The Dirichlet root floor's mixing weight, as `TRAINING-DESIGN-FIX.md` (D3)
/// specifies it: `P'(a) = (1 - eps) * P(a) + eps * eta_a`, root only. This is
/// AlphaZero's own 0.25 and it is **not** the default any option struct uses —
/// [`PuctParams::dirichlet_epsilon`] defaults to `0.0`, which is off. It is the
/// value the b2 arm is specified to run.
pub const DEFAULT_DIRICHLET_EPSILON: f64 = 0.25;

/// The Dirichlet concentration D3 specifies, `10 / mean_legal`.
///
/// The plan writes that as "~0.15 on this board" and this constant is that
/// number, but the derivation deserves recording because it does not reproduce
/// from anything measured here: `0.15` implies a mean legal-action count of
/// about 67, while the mean this engine actually plays at is **44.5** over a
/// mock-network self-play run (369 recorded positions, min 1, max 131) and
/// **40.4** on the live shard `crate::selfplay`'s module docs quote. The formula
/// on either of those gives `0.22`-`0.25`.
///
/// The plan's number is kept as the default anyway, deliberately: it is what the
/// b2 arm is pre-committed to and a default that silently disagreed with the
/// design doc would be worse than one that is documented as approximate. A
/// smaller alpha is the more aggressive floor (more mass on fewer actions), so
/// this errs toward the sparser noise. The value is an option precisely so the
/// range can be swept.
pub const DEFAULT_DIRICHLET_ALPHA: f64 = 0.15;

/// Domain separator for the Dirichlet's own draw stream, so its key cannot
/// collide with any other keyed stream this codebase grows later. The bytes are
/// `"DIR1"`.
const DIRICHLET_TAG: u32 = 0x4449_5231;

/// Cap on the gamma sampler's rejection loop. See [`sample_gamma_below_one`]
/// for the acceptance rate; the cap exists so that a wasm self-play worker can
/// never hang instead of finishing a shard, not because it is expected to bind.
const GAMMA_MAX_ATTEMPTS: u32 = 128;

/// Murmur3's 32-bit finalizer: the avalanche step, used here to key one stream
/// from several small integers.
///
/// Needed rather than an addition or an xor because *both* inputs to the key
/// vary by one between neighbours — game `i` seeds off `seed_base + i` and plies
/// run `0, 1, 2, ...` — and an LCG32 seeded with adjacent values produces
/// visibly related streams, especially in the low bits, which is where a
/// `u32 / 2^32` uniform's leading digits do NOT come from but where the
/// rejection sampler's accept/reject boundary can still feel it.
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// `mix(game_seed, ply, DIRICHLET_TAG)`: the seed of the Dirichlet's own
/// [`Lcg32`], and the whole of D3's determinism contract in one function.
///
/// The root noise MUST NOT be drawn from the game's main stream. Gamma sampling
/// for `alpha < 1` is a rejection method, so it consumes a variable number of
/// words — a number that depends on the priors, the legal count, and (through
/// the accept test's `exp`) on the target's libm. The main stream's offset is
/// contractually a pure function of the ply index; see
/// `crate::selfplay::Game::complete_move`, where the temperature draw is taken
/// unconditionally inside the phase for exactly that reason. Drawing the noise
/// from that stream would make the offset a function of the noise instead, and
/// two builds that disagreed by one accepted sample would fork every subsequent
/// game.
///
/// Keying off `(game_seed, ply)` also means the noise at a given ply is
/// reproducible without replaying the game, and that a search run in isolation —
/// a test, a diagnostic — sees the same noise the shard worker saw.
#[must_use]
pub fn dirichlet_stream_seed(game_seed: u32, ply: u64) -> u32 {
    // Fold the ply's 64 bits into 32 before mixing; no board reaches 2^32 plies,
    // so this is a formality that keeps the key defined for every input.
    let ply = (ply as u32) ^ ((ply >> 32) as u32);
    fmix32(fmix32(DIRICHLET_TAG ^ game_seed) ^ ply)
}

/// One `Gamma(alpha, 1)` draw for `0 < alpha < 1`, by Ahrens and Dieter's GS
/// algorithm (Devroye, *Non-Uniform Random Variate Generation*, IX.3).
///
/// ```text
/// b = 1 + alpha / e
/// loop:
///   p = b * U1
///   if p <= 1:  x = p^(1/alpha);        accept if U2 <= exp(-x)
///   else:       x = -log((b - p) / alpha);  accept if U2 <= x^(alpha - 1)
/// ```
///
/// Chosen over Marsaglia-Tsang because that method needs a standard normal, and
/// every cheap normal generator needs either a trigonometric pair or a second
/// rejection loop; GS needs only uniforms, a logarithm, an exponential and a
/// power. The restriction to `alpha < 1` is not a limitation in this codebase —
/// [`PuctTreeSearch::new`] rejects anything else, and `10 / mean_legal` is
/// `0.25` at its largest here — but it IS a restriction, and adding `alpha >= 1`
/// later means adding a second sampler, not relaxing a bound.
///
/// **Portability.** [`js_log`] is bit-portable, so it is used for the logarithm.
/// `exp` and `powf` are libm and are not; a target whose `exp` differs by an ULP
/// can therefore accept a different sample and consume a different number of
/// words. That is a real hazard and it is why [`dirichlet_stream_seed`] exists:
/// the divergence is confined to one root's noise vector on that platform and
/// cannot move the main stream by a single word. The same caveat already applies
/// to the recorded policy target, which goes through `f64::exp`, and to
/// `crate::selfplay::sample_visit_temperature`, which goes through `powf`.
fn sample_gamma_below_one(rng: &mut Lcg32, alpha: f64) -> f64 {
    let b = 1.0 + alpha / std::f64::consts::E;
    let mut last = 0.0_f64;
    for _ in 0..GAMMA_MAX_ATTEMPTS {
        let p = b * rng.unit_interval();
        let u = rng.unit_interval();
        if p <= 1.0 {
            // `x` is in (0, 1], so `exp(-x)` is in [1/e, 1) and the test accepts
            // with probability at least 1/e.
            let x = p.powf(1.0 / alpha);
            last = x;
            if u <= (-x).exp() {
                return x;
            }
        } else {
            // `(b - p) / alpha` is in (0, 1/e], so `x >= 1` and `x^(alpha - 1)`
            // is in (0, 1].
            let x = -js_log((b - p) / alpha);
            last = x;
            if u <= x.powf(alpha - 1.0) {
                return x;
            }
        }
    }
    // Unreachable in practice. GS's expected number of attempts is
    // `(e + alpha) / (e * gamma(1 + alpha))`, which over `alpha` in (0, 1) peaks
    // at about 1.39 -- an acceptance rate of ~0.72 -- so 128 consecutive
    // rejections is below 1e-70. Falling through with the last candidate keeps
    // the draw count deterministic and the value inside the distribution's
    // support, which neither a panic nor a zero would.
    last
}

/// `eta ~ Dir(alpha, ..., alpha)` over `count` components, drawn from the
/// stream [`dirichlet_stream_seed`] keys and from nothing else.
///
/// Public for the same reason [`root_qtransform`] is: the tests address the
/// distribution directly rather than inferring it from a whole 9x9 search, and
/// they address the *same code the search runs*, so there is no second
/// implementation to drift.
///
/// The components are drawn in index order and normalised by their sum, which is
/// the standard construction. The guard on that sum is not decorative at
/// `alpha = 0.15`: the gammas are heavily skewed and most of them are tiny, so a
/// degenerate draw is worth handling explicitly. A non-finite or non-positive
/// total falls back to the uniform vector — the noise that changes nothing about
/// the prior's *shape* — rather than to a division that would poison every
/// prior with NaN.
#[must_use]
pub fn root_dirichlet(game_seed: u32, ply: u64, alpha: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let mut rng = Lcg32::new(dirichlet_stream_seed(game_seed, ply));
    let mut draws: Vec<f64> = (0..count)
        .map(|_| sample_gamma_below_one(&mut rng, alpha))
        .collect();
    let total: f64 = draws.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        let uniform = 1.0 / count as f64;
        return vec![uniform; count];
    }
    for draw in &mut draws {
        *draw /= total;
    }
    draws
}

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
    #[error("invalid_dirichlet")]
    InvalidDirichlet,
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
            Self::InvalidDirichlet => "invalid_dirichlet",
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
/// That statement is exact for the TARGET and only approximate for the RANKING,
/// and the difference is worth naming. When a root mixes visited and unvisited
/// actions on one Q value, `v_mix` reproduces that Q to within an ULP rather
/// than exactly, so `(max - min)` is a crumb of order `1e-16`; the floor
/// replaces it with `1e-8` and the rescale therefore multiplies the crumb by
/// `1e8` before the visit scale multiplies it again. Against a target consumed
/// at `1e-3` that is nothing — 8.1e-5 worst case over the property sweep's
/// widest shapes, 2.9e-6 at the production `maxN <= 128`. But the same vector is
/// added to `g + logit` in [`PuctTreeSearch::halve`], where the tightest gap the
/// parity grid observes is `4.1e-7`, so on a degenerate root the halving cut can
/// be decided by that noise where `v2` produced an exact tie. It is
/// deterministic, it is what `mctx` does, and a root where every completed value
/// is equal has no better answer available — but it is a tie broken by float
/// noise rather than by the code order, and it is not a rounding detail that
/// stays inside the target.
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
/// list: `pi'(a) ∝ p(a) * exp(qtransform(a))`.
///
/// This is `mctx`'s `gumbel_muzero_policy` `action_weights`, restricted to the
/// legal actions — which is exactly the domain of an edge list, so there is no
/// `invalid_actions` mask to apply; illegal codes never enter and the recorder
/// writes them as exact `0.0`.
///
/// Written as `p * exp(boost)` rather than `exp(log(p) + boost)`
///
/// The two are the same function and `mctx` writes the second one, because its
/// input IS a logit vector. Ours is a probability vector — the network's policy
/// masked to the legal actions and renormalised — so taking a logarithm only to
/// exponentiate it again costs a round trip of one or two ULP for nothing. It
/// costs one specific thing, in fact: the plan's degenerate case says that when
/// the search has separated nothing, `pi'` must be EXACTLY the renormalised
/// prior. With a flat boost this form gives `p(a) * 1.0 / sum(p)`, which is that
/// statement bit for bit; the logarithmic form gives it to about `1e-14`.
/// `tests/qtransform_properties.rs` asserts the exact version.
///
/// [`effective_prior`] applies the same floor `seed_candidates` reads the prior
/// through, so the improved policy and the considered set still agree about
/// what a zero-prior legal action is worth.
///
/// Subtracting the largest boost before exponentiating keeps every term in
/// `(0, p(a)]`, so the total is in `(0, 1]` and can neither overflow nor be
/// zero — the qtransform's range makes overflow far less likely than it was
/// under `v2`, but at a six-figure visit count the scale alone is five figures,
/// and `exp` of that is infinity.
fn improved_policy(edges: &[Edge], root_value: f64) -> Vec<(u16, f64)> {
    let boosts = qtransform_completed_by_mix_value(edges, root_value);
    if edges.is_empty() {
        return Vec::new();
    }

    let mut highest = f64::NEG_INFINITY;
    for boost in &boosts {
        if *boost > highest {
            highest = *boost;
        }
    }

    let mut scored: Vec<(u16, f64)> = Vec::with_capacity(edges.len());
    let mut total = 0.0_f64;
    for (edge, boost) in edges.iter().zip(&boosts) {
        let weight = effective_prior(edge.prior) * (*boost - highest).exp();
        total += weight;
        scored.push((edge.code, weight));
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

/// The AlphaZero policy target: `N(a) / sum_b N(b)` over one node's whole edge
/// list, ascending by code — [`RootMode::Classic`]'s recorded target.
///
/// The properties this has and [`improved_policy`] does not: its concentration
/// is bounded by 1.0 whatever the value head says, it is invariant to any
/// monotone rescaling of Q because it never reads Q at all, and every unit of
/// mass it moves off the prior was paid for by a real tree descent. What it
/// gives up is coverage — an action the search never selected is targeted to
/// exactly zero, which is only defensible at a budget large enough for the
/// visits to be a judgement rather than a schedule.
///
/// Under sequential halving that condition fails badly (both finalists take the
/// same count whoever won), which is why this is *not* offered as an option on
/// the Gumbel root: it is the target of a different root algorithm, not a
/// different target for the same one.
fn visit_count_policy(edges: &[Edge]) -> Vec<(u16, f64)> {
    let visits: u32 = edges.iter().map(|edge| edge.visits).sum();
    if visits > 0 {
        let total = f64::from(visits);
        return edges
            .iter()
            .map(|edge| (edge.code, f64::from(edge.visits) / total))
            .collect();
    }
    // No root action has been visited yet. `PuctTreeSearch::new` rejects
    // `simulations < 1` and every simulation backs up through exactly one root
    // edge, so a *completed* classic search cannot land here; `result()` is
    // callable mid-search, though, and the honest answer before the first backup
    // is the network's own prior. `expand` leaves those normalised over the
    // legal actions, so they are already a distribution — and, unlike a one-hot
    // fallback, one that teaches the policy head nothing rather than something
    // false.
    edges.iter().map(|edge| (edge.code, edge.prior)).collect()
}

/// The most-visited action, ties to the lowest code — [`RootMode::Classic`]'s
/// played move once the temperature phase is over.
///
/// Edges are stored ascending by code and the comparison is strict, so the
/// tie-break needs no extra work, exactly as in [`PuctTreeSearch::select_edge`].
fn most_visited(edges: &[Edge]) -> u16 {
    let mut chosen = 0_u16;
    let mut best = 0_u32;
    let mut found = false;
    for edge in edges {
        if !found || edge.visits > best {
            chosen = edge.code;
            best = edge.visits;
            found = true;
        }
    }
    chosen
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

/// Which algorithm drives the **root** of the search. Everything below the root
/// is the same tree either way: same [`PuctTreeSearch::select_edge`], same FPU,
/// same backup, same adjudication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RootMode {
    /// Gumbel-MuZero: one Gumbel draw per legal root action, a considered set of
    /// `max_considered` survivors, sequential halving over them, and the
    /// completed-Q improved policy as the recorded target.
    ///
    /// The default, and the production recipe. Nothing in this file may make a
    /// Gumbel search behave differently than it did before [`RootMode`] existed;
    /// `tests/root_mode_classic.rs` pins that to digests taken from the earlier
    /// build.
    #[default]
    Gumbel,
    /// AlphaZero-classic: plain PUCT at the root, no Gumbel draws, no considered
    /// set, no halving, and `N(a) / sum N` over the root's visits as the
    /// recorded target.
    ///
    /// This is a *control arm*, not a replacement. The Gumbel target is a
    /// softmax whose sharpness a constant sets and no tree descent bounds; the
    /// visit distribution is bounded at 1.0 concentration by construction and
    /// every unit of sharpness is paid for by a simulation that could have
    /// contradicted the prior. Running both from the same seed is what
    /// distinguishes "the design is wrong" from "the design is
    /// mis-parameterised", and that question cannot be answered by tuning the
    /// Gumbel arm alone.
    ///
    /// [`PuctParams::max_considered`] is not read in this mode — there is no
    /// candidate set to bound — but it is still validated, so a driver that
    /// keeps sending the production value is not rejected for it.
    ///
    /// Costs more per position for the same quality of target: the visit
    /// distribution over ~40 legal actions is only informative at a budget that
    /// can actually spread over them, which is why the arm is specified at 512
    /// simulations rather than the Gumbel arm's 128.
    Classic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PuctParams {
    pub simulations: u32,
    /// Size of the Gumbel considered set. Read only by [`RootMode::Gumbel`].
    pub max_considered: u32,
    pub c_puct: f64,
    /// Which root algorithm to run. Defaults to [`RootMode::Gumbel`], so every
    /// caller that predates this field gets exactly the search it had.
    pub root_mode: RootMode,
    /// The game this search belongs to, for [`dirichlet_stream_seed`]. Together
    /// with `RootContext::ply` it is the whole key of the Dirichlet's stream.
    ///
    /// It is a *game* seed, not a search seed: the same number the game's main
    /// [`Lcg32`] was constructed from, unchanged for the whole game. Read only
    /// when [`Self::dirichlet_epsilon`] is positive, so a caller that does not
    /// use the floor may leave it at `0`.
    pub game_seed: u32,
    /// D3's mixing weight `eps` in `P'(a) = (1 - eps) * P(a) + eps * eta_a`,
    /// applied to the ROOT's priors only, before the Gumbel draws.
    ///
    /// **`0.0` means byte-for-byte absent** and that is the default: no stream is
    /// created, no draw is taken from anywhere, and every prior the search reads
    /// is the identical `f64` it read before D3 existed. `tests/dirichlet_root.rs`
    /// pins that against digests taken from the pre-D3 build.
    ///
    /// See [`PuctTreeSearch::setup_root`] for the one thing this deliberately
    /// does NOT touch: the recorded policy target.
    pub dirichlet_epsilon: f64,
    /// D3's concentration. Read only when [`Self::dirichlet_epsilon`] is
    /// positive, and then required to lie in `(0, 1)` — see
    /// [`sample_gamma_below_one`] for why the upper bound is real.
    pub dirichlet_alpha: f64,
}

impl Default for PuctParams {
    fn default() -> Self {
        Self {
            simulations: 32,
            max_considered: 8,
            c_puct: DEFAULT_C_PUCT,
            root_mode: RootMode::Gumbel,
            game_seed: 0,
            // Off. The floor is an arm of the experiment, not the baseline: b1
            // runs without it and b2 with it, and a default of 0.25 would have
            // made every caller that never heard of D3 into a b2 run.
            dirichlet_epsilon: 0.0,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
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
    /// D3's noised root priors, parallel to the ROOT's edge list, or **empty**
    /// when the floor is off.
    ///
    /// Empty is the load-bearing state: every read goes through
    /// [`Self::selection_priors`], which returns this slice and whose callers
    /// fall back to `edge.prior` when it is empty, so `eps = 0` takes the
    /// identical arithmetic path it took before D3 and not a `(1 - 0) * p + 0`
    /// rewrite of it.
    ///
    /// Separate from `edges` rather than written into it because the recorded
    /// training target reads `edge.prior`; see [`Self::setup_root`].
    root_selection_priors: Vec<f64>,
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
    ///
    /// Under [`RootMode::Classic`] this field carries that root's own target
    /// instead — `N(a) / sum N`, see [`visit_count_policy`]. The name is kept
    /// because the *contract* is unchanged and it is the contract self-play
    /// depends on: a normalised distribution over exactly the legal root
    /// actions, ascending by code. What the search recorded as its opinion is
    /// read from here whichever root produced it.
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
        // `epsilon` is checked always -- a NaN or an out-of-range weight is a
        // caller bug whether or not it would be read -- and `alpha` only when
        // the floor is actually on, matching how `SelfPlayBatch` treats the
        // temperature it only reads inside the sampling phase.
        if !params.dirichlet_epsilon.is_finite() || !(0.0..=1.0).contains(&params.dirichlet_epsilon)
        {
            return Err(PuctError::InvalidDirichlet);
        }
        if params.dirichlet_epsilon > 0.0
            && !(params.dirichlet_alpha.is_finite()
                && params.dirichlet_alpha > 0.0
                && params.dirichlet_alpha < 1.0)
        {
            return Err(PuctError::InvalidDirichlet);
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
            root_selection_priors: Vec::new(),
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
                    // The only place the two root algorithms differ: which root
                    // edge the next simulation starts down, and when to stop.
                    let edge = match self.params.root_mode {
                        RootMode::Gumbel => {
                            let Some(candidate) = self.next_candidate() else {
                                self.phase = Phase::Done;
                                return Ok(false);
                            };
                            self.candidates[candidate].edge
                        }
                        // Plain PUCT: no schedule to consult, so the budget is
                        // the whole stopping rule, and the root is selected by
                        // the same `select_edge` every other node uses.
                        RootMode::Classic => {
                            if self.budget <= 0 {
                                self.phase = Phase::Done;
                                return Ok(false);
                            }
                            self.select_edge(0)
                        }
                    };
                    if self.begin_visit(config, edge)? {
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
                self.setup_root(value);
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
        // The root's edge list is every legal root action, in the ascending code
        // order `legal_action_codes_fast` produced, so slicing it is already the
        // policy target's domain and ordering. An unexpanded root has no edges
        // and yields an empty policy.
        let root = self.nodes[0];
        let root_edges =
            &self.edges[root.edges_start as usize..(root.edges_start + root.edges_len) as usize];
        if self.params.root_mode == RootMode::Classic {
            // No candidate set: the classic root considers every legal action,
            // so `visit_counts` and `considered` cover the whole edge list. Most
            // of those counts are zero at a realistic budget, which is exactly
            // what makes the target's support a measurement rather than a
            // schedule.
            return PuctResult {
                action_code: most_visited(root_edges),
                visit_counts: root_edges
                    .iter()
                    .map(|edge| (edge.code, edge.visits))
                    .collect(),
                improved_policy: visit_count_policy(root_edges),
                root_value: self.root_value,
                simulations_used: self.used,
                max_depth_reached: self.max_depth,
                considered: root_edges.iter().map(|edge| edge.code).collect(),
            };
        }
        let winner = self
            .survivors
            .first()
            .map_or(0, |index| self.candidates[*index].code);
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

    /// Everything that happens once, at the root, after the network has spoken.
    ///
    /// Both root modes funnel through here, which is what makes it the right
    /// place for D3's Dirichlet floor — `P'(a) = (1 - eps) * P(a) + eps * eta_a`,
    /// root only, **before** anything reads a root prior. The Gumbel draws in
    /// [`Self::seed_candidates`] take one through [`js_log`]; the classic root
    /// takes one per edge in [`Self::select_edge`]; both happen after this point.
    ///
    /// Two properties of how it is applied are the whole of D3's correctness,
    /// and neither is visible from the arithmetic alone.
    ///
    /// **The noise never touches `edge.prior`.** [`improved_policy`] reads
    /// `edge.prior` as its logit source and [`mixed_value`] reads it as `v_mix`'s
    /// weight, so writing the mixture back into the edge list would make the
    /// RECORDED TRAINING TARGET a softmax over noised logits rather than over the
    /// network's prior. Dirichlet is meant to perturb what the search explores,
    /// not what the network is asked to imitate — a b2 arm that fitted the net to
    /// its own noise would be measuring something nobody designed. The mixture
    /// therefore lives in [`Self::root_selection_priors`], which only the
    /// selection path reads, and `tests/dirichlet_root.rs` holds the target to
    /// being a function of `(prior, visits, Q, root_value)` alone.
    ///
    /// **The draws come from a stream of this root's own.** See
    /// [`dirichlet_stream_seed`]: `alpha < 1` gamma sampling is a rejection
    /// method, and the main stream's offset is contractually a pure function of
    /// the ply index. `self.rng` is not touched here, at any `eps`.
    ///
    /// With `eps = 0` this function does exactly what it did before D3: no
    /// stream, no draw, no vector, and every later prior read resolves to the
    /// same `edge.prior` load it always was.
    fn setup_root(&mut self, value: f64) {
        self.root_value = value;
        self.apply_root_dirichlet();
        if self.params.root_mode == RootMode::Gumbel {
            self.seed_candidates();
        }
    }

    /// Fill [`Self::root_selection_priors`] with the D3 mixture, or leave it
    /// empty when the floor is off.
    fn apply_root_dirichlet(&mut self) {
        let epsilon = self.params.dirichlet_epsilon;
        if epsilon <= 0.0 {
            return;
        }
        let root = self.nodes[0];
        let count = root.edges_len as usize;
        if count == 0 {
            return;
        }
        let noise = root_dirichlet(
            self.params.game_seed,
            root.ply,
            self.params.dirichlet_alpha,
            count,
        );
        let start = root.edges_start as usize;
        self.root_selection_priors.clear();
        self.root_selection_priors.reserve(count);
        for (edge, eta) in self.edges[start..start + count].iter().zip(&noise) {
            self.root_selection_priors
                .push((1.0 - epsilon) * edge.prior + epsilon * eta);
        }
    }

    /// The priors `node`'s selection reads: the D3 mixture at the root when the
    /// floor is on, and an empty slice everywhere else, meaning "use
    /// `edge.prior`".
    ///
    /// Returning a slice rather than a per-edge value keeps the emptiness check
    /// out of the inner loop in [`Self::select_edge`], and keeps the un-noised
    /// path a plain `edge.prior` load.
    fn selection_priors(&self, node: u32) -> &[f64] {
        if node == 0 {
            &self.root_selection_priors
        } else {
            &[]
        }
    }

    /// One edge's selection prior, by index into [`Self::edges`].
    fn selection_prior(&self, node: u32, index: u32) -> f64 {
        let priors = self.selection_priors(node);
        if priors.is_empty() {
            return self.edges[index as usize].prior;
        }
        priors[(index - self.nodes[node as usize].edges_start) as usize]
    }

    /// One Gumbel per legal root action, drawn in ascending code order — the
    /// draw order is part of the contract, because the same seed has to
    /// reproduce the same game in both engines.
    fn seed_candidates(&mut self) {
        let root = self.nodes[0];
        self.candidates.clear();
        self.candidates.reserve(root.edges_len as usize);
        for index in root.edges_start..root.edges_start + root.edges_len {
            let edge = self.edges[index as usize];
            // `effective_prior` is `prior.max(POLICY_FLOOR)` and nothing else,
            // so this is the identical expression it always was -- named, now
            // that the improved policy reads the prior through the same floor
            // without taking its logarithm at all.
            //
            // `selection_prior` is `edge.prior` unless D3's floor is on, in
            // which case it is `(1 - eps) * prior + eps * eta`. This is the
            // "before the Gumbel draws" of the plan's sentence: the noise enters
            // the logit the Gumbel is added to, not the target the tree records.
            let logit = js_log(effective_prior(self.selection_prior(0, index)));
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

    /// Start one simulation down root `edge` (an index into [`Self::edges`]).
    /// Returns `true` when the descent paused at an unexpanded leaf, `false`
    /// when it completed against a terminal node and has already been backed up.
    ///
    /// Takes the edge rather than a candidate index because the classic root has
    /// no candidates; the budget accounting and the repetition bookkeeping below
    /// are identical for both root modes and are deliberately not duplicated.
    fn begin_visit(&mut self, config: &Config, edge: u32) -> Result<bool> {
        self.budget -= 1;
        self.used += 1;

        self.path.clear();
        self.descent_keys.clear();
        self.window_from = 0;
        self.root_window_active = true;

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
            // The FPU's visited-prior accounting is part of selection, so it
            // reads the same prior selection does -- the D3 mixture at the root,
            // `edge.prior` everywhere else. It is only ever *read* at the root by
            // `RootMode::Classic` (a Gumbel root's children are chosen by the
            // schedule, not by `select_edge`), but charging one prior here and
            // selecting on another would be a silent inconsistency either way.
            self.nodes[node as usize].visited_prior += self.selection_prior(node, edge);
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
    ///
    /// `P` is the D3 mixture at the root when the floor is on — that is where
    /// Dirichlet reaches [`RootMode::Classic`], which has no Gumbel logit to
    /// perturb — and `edge.prior` everywhere else. The empty-slice test is
    /// hoisted for the same reason the FPU is, and when it is empty the
    /// expression below is the identical `edge.prior` load it was before D3.
    fn select_edge(&self, node: u32) -> u32 {
        let current = self.nodes[node as usize];
        let sqrt_total = f64::from(current.visits).sqrt();
        let node_value = if current.visits > 0 {
            current.value_sum / f64::from(current.visits)
        } else {
            current.value
        };
        let fpu = clamp_value(node_value - FPU_REDUCTION * current.visited_prior.sqrt());
        let selection = self.selection_priors(node);

        let mut best = current.edges_start;
        let mut best_score = f64::NEG_INFINITY;
        for index in current.edges_start..current.edges_start + current.edges_len {
            let edge = self.edges[index as usize];
            let q = if edge.visits > 0 {
                edge.value_sum / f64::from(edge.visits)
            } else {
                fpu
            };
            let prior = if selection.is_empty() {
                edge.prior
            } else {
                selection[(index - current.edges_start) as usize]
            };
            let score = q + self.params.c_puct * prior * sqrt_total / f64::from(1 + edge.visits);
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
    /// policy separates them by the qtransform gap.
    ///
    /// **Three actions, not two.** With two the extremes are the only two
    /// actions, min-max rescaling pins them to `0` and `1` whatever the Q values
    /// were, and the asserted ratio `exp((50 + maxN) * 0.1)` is a tautology of
    /// the rescale that survives swapping `0.4 / -0.2` for `0.9 / -0.9`. The
    /// middle action is what makes the assertion depend on the Q values again:
    /// its share is `exp(scale * (q - min) / (max - min))`, so moving it moves
    /// the number.
    #[test]
    fn a_better_completed_q_takes_strictly_more_mass_at_equal_visits() {
        let good = edge(5, 0.25, 30, 0.4);
        let middling = edge(7, 0.25, 30, 0.1);
        let bad = edge(9, 0.25, 30, -0.2);
        assert_eq!(good.visits, bad.visits);
        assert_eq!(good.prior, bad.prior);

        let scale = (MAXVISIT_INIT + 30.0) * VALUE_SCALE;
        let policy = improved_policy(&[good, middling, bad], 0.0);
        let (good_mass, middling_mass, bad_mass) =
            (mass(&policy, 5), mass(&policy, 7), mass(&policy, 9));
        assert!(
            good_mass > middling_mass && middling_mass > bad_mass,
            "Q = 0.4 took {good_mass}, Q = 0.1 took {middling_mass}, Q = -0.2 took {bad_mass}"
        );

        // Equal priors cancel, so every ratio is a qtransform gap alone. The
        // extremes are pinned by the rescale; the middle one is not.
        let ends = scale.exp();
        assert!(
            ((good_mass / bad_mass) / ends - 1.0).abs() < 1e-9,
            "mass ratio {} is not exp((50 + 30) * 0.1) = {ends}",
            good_mass / bad_mass
        );
        let middle = (scale * (0.1 - -0.2) / (0.4 - -0.2)).exp();
        assert!(
            ((middling_mass / bad_mass) / middle - 1.0).abs() < 1e-9,
            "mass ratio {} is not exp(scale * (0.1 + 0.2) / 0.6) = {middle}",
            middling_mass / bad_mass
        );
        // Moving the middle action's Q must move its mass, or the assertion
        // above is measuring the rescale rather than the value.
        let moved = improved_policy(&[good, edge(7, 0.25, 30, 0.3), bad], 0.0);
        assert!(mass(&moved, 7) > middling_mass * 1.5);

        assert!(
            ends < (0.6_f64 * 80.0).exp(),
            "the v3 ratio must be far below the v2 one"
        );
    }

    /// An action the halving never visited is completed with `v_mix`, so two
    /// unvisited actions are separated by their priors alone — which is how the
    /// improved policy covers moves the considered set skipped instead of
    /// targeting them to zero.
    ///
    /// **Two DISTINCT visited Q-values, for a reason.** Nothing weaker
    /// discriminates `v_mix` from `v2`'s raw root value. With no visited action
    /// at all `v_mix` collapses to the raw value by definition; with exactly one
    /// visited action there are only two distinct completions and the rescale
    /// pins them to `0` and the full scale whichever completion was used. Both
    /// pass with D2 reverted. Two visited actions plus the unvisited pair give
    /// three distinct completed values, and then WHICH action sits at the bottom
    /// of the range is decided by the completion.
    #[test]
    fn unvisited_actions_are_completed_with_the_mixed_value() {
        // v_mix = (-0.9 + 8 * 0.32) / 9, where 0.32 is the prior-weighted mean
        // of the two visited Q-values: (0.3 * 0.8 + 0.2 * -0.4) / 0.5.
        let raw_value = -0.9;
        let edges = [
            edge(2, 0.4, 0, 0.0),
            edge(8, 0.1, 0, 0.0),
            edge(11, 0.3, 4, 0.8),
            edge(14, 0.2, 4, -0.4),
        ];
        let weighted = (0.3 * 0.8 + 0.2 * -0.4) / 0.5;
        let expected_mix = (raw_value + 8.0 * weighted) / 9.0;
        let mixed = mixed_value(&edges, raw_value);
        assert!((mixed - expected_mix).abs() < 1e-12, "v_mix is {mixed}");
        assert!(
            mixed > -0.4,
            "v_mix {mixed} must land inside the visited range, or this test \
             cannot tell it from the raw root value"
        );

        let boosts = qtransform_completed_by_mix_value(&edges, raw_value);
        // The two unvisited actions share the completion, so they share a boost
        // and are separated by their priors alone.
        assert_eq!(boosts[0], boosts[1]);
        let policy = improved_policy(&edges, raw_value);
        assert!(
            (mass(&policy, 2) / mass(&policy, 8) - 4.0).abs() < 1e-9,
            "equal completions must leave the prior ratio intact"
        );

        // The discriminator. Under `v3` the WEAKEST VISITED action is the
        // minimum and the unvisited pair sits strictly inside the range. Under
        // `v2`'s raw root value of -0.9 the unvisited pair would BE the minimum
        // at exactly 0, and the weak visited action would be lifted off the
        // floor to 1.59.
        let ceiling = (MAXVISIT_INIT + 4.0) * VALUE_SCALE;
        assert_eq!(
            boosts[3], 0.0,
            "the weakest visited action must be the floor"
        );
        assert_eq!(boosts[2], ceiling);
        let expected_boost = ceiling * (expected_mix - -0.4) / (0.8 - -0.4);
        assert!(
            (boosts[0] - expected_boost).abs() < 1e-12,
            "the unvisited pair took {}, not v_mix's {expected_boost}",
            boosts[0]
        );
        assert!(boosts[0] > 1.0 && boosts[0] < ceiling);

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

    /// The classic target is the visit distribution, exactly, and the tie-break
    /// on the played move is the lowest code — the same rule `select_edge` uses.
    ///
    /// A real search almost never produces a tie at the maximum (PUCT keeps
    /// pushing the leader), so this is where the rule can be exercised at all: a
    /// sweep of 1..260 simulations from the opening produced no tied maximum,
    /// which is precisely why the integration test cannot cover it and why a
    /// `>=` here would otherwise be invisible.
    #[test]
    fn most_visited_breaks_ties_to_the_lowest_code() {
        let tied = [
            edge(3, 0.2, 7, 0.0),
            edge(11, 0.5, 7, 0.0),
            edge(40, 0.3, 2, 0.0),
        ];
        // Anti-vacuity: this only tests a tie-break if there is a tie, and only
        // tests the *lowest* code if the tie is not already won by position.
        assert_eq!(tied[0].visits, tied[1].visits, "the fixture has no tie");
        assert!(
            tied[1].prior > tied[0].prior,
            "the tie must not be breakable by prior"
        );
        assert_eq!(most_visited(&tied), 3);

        // A strictly greater count wins wherever it sits in the list.
        assert_eq!(
            most_visited(&[edge(3, 0.2, 7, 0.0), edge(11, 0.2, 8, 0.0)]),
            11
        );
        assert_eq!(
            most_visited(&[edge(3, 0.2, 9, 0.0), edge(11, 0.2, 8, 0.0)]),
            3
        );
        // An all-unvisited root is a tie at zero: still the lowest code, never
        // an arbitrary one.
        assert_eq!(
            most_visited(&[edge(5, 0.9, 0, 0.0), edge(9, 0.1, 0, 0.0)]),
            5
        );
    }

    #[test]
    fn the_classic_target_is_exactly_the_visit_share() {
        let policy = visit_count_policy(&[
            edge(3, 0.7, 6, 0.0),
            edge(11, 0.2, 2, 0.0),
            edge(40, 0.1, 0, 0.0),
        ]);
        // 8 visits: 6/8, 2/8, and an unvisited action at exactly zero. Exact
        // equality, not a tolerance -- these are dyadic rationals.
        assert_eq!(policy, vec![(3, 0.75), (11, 0.25), (40, 0.0)]);
        // The prior is not consulted: 0.7 against 0.2 did not move the split.
        let reprioritised = visit_count_policy(&[
            edge(3, 0.1, 6, 0.0),
            edge(11, 0.8, 2, 0.0),
            edge(40, 0.1, 0, 0.0),
        ]);
        assert_eq!(policy, reprioritised);
    }

    /// The zero-visit fallback, pinned rather than merely argued for.
    ///
    /// This codebase lost 114 iterations to a silent one-hot policy target
    /// (`effective_visit_counts` under the v1 format), so the shape of this
    /// fallback is not a detail: a one-hot here would be that same defect
    /// wearing the new target's name. It returns the network's own prior, which
    /// teaches the policy head nothing rather than something false.
    #[test]
    fn a_classic_target_with_no_visits_is_the_prior_and_never_a_one_hot() {
        let edges = [
            edge(2, 0.5, 0, 0.0),
            edge(8, 0.3, 0, 0.0),
            edge(40, 0.2, 0, 0.0),
        ];
        let policy = visit_count_policy(&edges);
        assert_eq!(policy, vec![(2, 0.5), (8, 0.3), (40, 0.2)]);
        let support = policy.iter().filter(|(_, p)| *p > 0.0).count();
        assert_eq!(
            support, 3,
            "an unvisited classic root must not collapse onto one action"
        );
        // And an empty root is empty, not a one-hot on code 0.
        assert!(visit_count_policy(&[]).is_empty());
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

    /* -------------------------------------------------------------- *
     * The halving ranking's use of the shared expression
     * -------------------------------------------------------------- */

    fn canonical_config() -> Config {
        Config {
            ruleset: crate::RULESET.into(),
            rows: 9,
            columns: 9,
            start: crate::Players {
                a: Coord { r: 8, c: 4 },
                b: Coord { r: 0, c: 4 },
            },
            goal_rows: crate::Players { a: 0, b: 8 },
            initial_stock: crate::Players { a: 10, b: 10 },
            jump_rule: crate::JUMP_RULE.into(),
            repetition_threshold: crate::REPETITION_THRESHOLD,
            ply_cap: 200,
            first_player: Player::A,
        }
    }

    /// A search whose root has been expanded from a uniform prior, so the
    /// candidates and their Gumbel scores exist and the root edges can be
    /// written to directly.
    fn expanded_root(
        config: &Config,
        max_considered: u32,
        seed: u32,
        root_value: f64,
    ) -> PuctTreeSearch {
        expanded_root_with(
            config,
            PuctParams {
                simulations: 64,
                max_considered,
                c_puct: DEFAULT_C_PUCT,
                // Explicit, not defaulted: the whole point of these two tests is
                // the sequential-halving ranking, which `RootMode::Classic` does
                // not run at all. D3's floor is off, from `Default`.
                root_mode: RootMode::Gumbel,
                ..PuctParams::default()
            },
            seed,
            &vec![1.0_f32; config.policy_size()],
            root_value,
        )
    }

    /// The same, with the params and the network's raw policy under the caller's
    /// control — the shape D3's tests need, since the whole question there is
    /// what the search does with a prior it was given.
    fn expanded_root_with(
        config: &Config,
        params: PuctParams,
        seed: u32,
        policy: &[f32],
        root_value: f64,
    ) -> PuctTreeSearch {
        let state = crate::create_initial_state(config).expect("the 9x9 start is valid");
        let mut search = PuctTreeSearch::from_state(config, &state, params, Lcg32::new(seed))
            .expect("the start position begins a search");

        let mut features = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];
        assert!(search
            .next_leaf(config, &mut features)
            .expect("the root is handed out first"));
        search
            .submit(config, policy, root_value)
            .expect("a positive policy expands the root");
        search
    }

    /// **The halving ranking reads the shared expression, and its result depends
    /// on it.**
    ///
    /// This is the half of D1 that changes search decisions, and it is the half
    /// the goldens cannot see: they address
    /// [`qtransform_completed_by_mix_value`] directly, so they would still pass
    /// if [`PuctTreeSearch::halve`] stopped calling it. The cross-engine grid
    /// cannot see it either — its `v1` oracle disagrees with `v3` by design, so
    /// its divergence counters move but nothing there says which way is right.
    ///
    /// So: plant known statistics on the root edges, run one `halve`, and
    /// require the survivors to be exactly the top half by
    /// `candidate.score + root_qtransform(root edges).transformed[edge]` —
    /// recomputed here through the PUBLIC entry point, so the assertion is
    /// against the same expression the improved policy is built from rather than
    /// against a number copied out of a previous run.
    ///
    /// The Q-values are assigned in the inverse of the Gumbel order on purpose,
    /// and the test asserts that the boosted ranking and the bare
    /// `g + logit` ranking disagree before it checks which one `halve` followed.
    /// Without that, a `halve` that ignored the boost entirely would pass.
    #[test]
    fn halving_ranks_by_the_gumbel_score_plus_the_shared_qtransform() {
        let config = canonical_config();
        let mut search = expanded_root(&config, 4, 20_260_809, 0.25);
        let root = search.nodes[0];
        let start = root.edges_start as usize;
        let end = start + root.edges_len as usize;
        assert_eq!(search.candidates.len(), 4);
        assert_eq!(search.survivors.len(), 4);

        // Rank the four candidates by their Gumbel score and hand the BEST
        // Q-value to the WORST-scoring one, so the boost has something to
        // overturn. 64 visits put the boost span at 11.4, comfortably wider than
        // the spread of four Gumbel draws.
        let mut by_score: Vec<usize> = (0..search.candidates.len()).collect();
        by_score.sort_by(|left, right| {
            search.candidates[*right]
                .score
                .partial_cmp(&search.candidates[*left].score)
                .expect("Gumbel scores are finite")
        });
        for (rank, index) in by_score.iter().enumerate() {
            let q = -1.0 + 2.0 * (rank as f64) / 3.0;
            let edge = search.candidates[*index].edge as usize;
            search.edges[edge].visits = 64;
            search.edges[edge].value_sum = 64.0 * q;
        }

        // The expected ranking, from the public entry point over the whole root
        // edge list -- every legal action, not just the four candidates.
        let stats: Vec<ActionStats> = search.edges[start..end]
            .iter()
            .map(|edge| ActionStats {
                prior: edge.prior,
                visits: edge.visits,
                qvalue: if edge.visits > 0 {
                    edge.value_sum / f64::from(edge.visits)
                } else {
                    f64::NAN
                },
            })
            .collect();
        let boosts = root_qtransform(&stats, search.root_value).transformed;

        let keep = |scored: &mut Vec<(u16, f64)>| -> Vec<u16> {
            scored.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .expect("scores are finite")
                    .then(left.0.cmp(&right.0))
            });
            let mut kept: Vec<u16> = scored.iter().take(2).map(|(code, _)| *code).collect();
            kept.sort_unstable();
            kept
        };
        let mut boosted: Vec<(u16, f64)> = search
            .candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.code,
                    candidate.score + boosts[candidate.edge as usize - start],
                )
            })
            .collect();
        let mut bare: Vec<(u16, f64)> = search
            .candidates
            .iter()
            .map(|candidate| (candidate.code, candidate.score))
            .collect();
        let expected = keep(&mut boosted);
        let without_the_boost = keep(&mut bare);
        assert_ne!(
            expected, without_the_boost,
            "the case does not discriminate: `g + logit` alone would keep the same two \
             survivors, so a halve() that never consulted the qtransform would pass"
        );

        search.halve();
        let survivors: Vec<u16> = search
            .survivors
            .iter()
            .map(|index| search.candidates[*index].code)
            .collect();
        assert_eq!(
            survivors, expected,
            "halve() kept {survivors:?}; ranking by score + qtransform keeps {expected:?} and \
             ranking by the bare Gumbel score keeps {without_the_boost:?}"
        );
    }

    /* -------------------------------------------------------------- *
     * D3: the Dirichlet floor's two-sided contract
     * -------------------------------------------------------------- */

    /// A non-uniform, strictly positive root policy, so "the noise moved the
    /// prior" cannot be confused with "the prior was flat anyway".
    ///
    /// The values are arbitrary but deterministic and they differ across codes
    /// by more than the noise does at some indices and less at others, which is
    /// what makes the mixture's arithmetic worth checking pointwise.
    fn sloped_policy(config: &Config) -> Vec<f32> {
        (0..config.policy_size())
            .map(|code| 0.25 + ((code * 37) % 101) as f32)
            .collect()
    }

    fn floored_params(epsilon: f64, game_seed: u32) -> PuctParams {
        PuctParams {
            simulations: 64,
            // Above the legal count, so `select_considered` keeps every
            // candidate in edge order and `candidates[i]` is root edge `i`.
            max_considered: 256,
            c_puct: DEFAULT_C_PUCT,
            root_mode: RootMode::Gumbel,
            game_seed,
            dirichlet_epsilon: epsilon,
            dirichlet_alpha: DEFAULT_DIRICHLET_ALPHA,
        }
    }

    /// The mixture is exactly `(1 - eps) * P + eps * eta`, it is a distribution,
    /// and — the point of the whole design — the edge list the target is read
    /// from is untouched.
    #[test]
    fn the_floor_mixes_into_a_copy_and_leaves_the_networks_prior_in_the_edges() {
        let config = canonical_config();
        let policy = sloped_policy(&config);
        let plain = expanded_root_with(&config, floored_params(0.0, 77), 5, &policy, 0.25);
        let floored = expanded_root_with(
            &config,
            floored_params(DEFAULT_DIRICHLET_EPSILON, 77),
            5,
            &policy,
            0.25,
        );

        let root = floored.nodes[0];
        let count = root.edges_len as usize;
        assert!(count > 100, "the opening root has {count} edges");
        assert!(
            plain.root_selection_priors.is_empty(),
            "eps = 0 must allocate no mixture at all"
        );
        assert_eq!(floored.root_selection_priors.len(), count);

        // The network's priors are byte-identical between the two searches: the
        // floor wrote nowhere near them.
        let priors: Vec<f64> = plain.edges[..count].iter().map(|edge| edge.prior).collect();
        let after: Vec<f64> = floored.edges[..count]
            .iter()
            .map(|edge| edge.prior)
            .collect();
        assert_eq!(priors, after, "the floor rewrote the root's edge priors");

        // And the mixture is the formula, term by term, over the noise the
        // public entry point produces from this root's key.
        let eta = root_dirichlet(77, root.ply, DEFAULT_DIRICHLET_ALPHA, count);
        let epsilon = DEFAULT_DIRICHLET_EPSILON;
        for index in 0..count {
            assert_eq!(
                floored.root_selection_priors[index],
                (1.0 - epsilon) * priors[index] + epsilon * eta[index],
                "edge {index}"
            );
        }
        let total: f64 = floored.root_selection_priors.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "the noised priors sum to {total}"
        );
        // Anti-vacuity: the noise has to have MOVED something, or every
        // assertion above holds for a mixture with weight zero.
        let moved = (0..count).filter(|i| floored.root_selection_priors[*i] != priors[*i]);
        assert!(moved.count() > count / 2);
    }

    /// The noise enters the Gumbel logit — `g + log P'` and not `g + log P`.
    ///
    /// The draws themselves are reproduced here from a fresh stream, so the
    /// assertion separates the two halves of the score: the Gumbel is the same
    /// as it would have been (the floor took no word from this stream) and the
    /// logit is the noised one.
    #[test]
    fn the_gumbel_score_is_drawn_against_the_noised_logit() {
        let config = canonical_config();
        let policy = sloped_policy(&config);
        let search = expanded_root_with(
            &config,
            floored_params(DEFAULT_DIRICHLET_EPSILON, 909),
            2024,
            &policy,
            0.0,
        );
        let count = search.nodes[0].edges_len as usize;
        assert_eq!(
            search.candidates.len(),
            count,
            "max_considered was supposed to keep every candidate"
        );

        let mut stream = Lcg32::new(2024);
        let mut differed = 0_usize;
        for index in 0..count {
            let candidate = search.candidates[index];
            assert_eq!(candidate.edge as usize, index, "candidates lost edge order");
            let gumbel = stream.gumbel();
            let noised = js_log(effective_prior(search.root_selection_priors[index]));
            let bare = js_log(effective_prior(search.edges[index].prior));
            assert_eq!(
                candidate.score,
                gumbel + noised,
                "candidate {index} scored against the bare logit {bare} rather than the \
                 noised {noised}"
            );
            if noised != bare {
                differed += 1;
            }
        }
        assert!(
            differed > count / 2,
            "only {differed} of {count} logits moved; the assertion above cannot tell the two \
             apart"
        );
    }

    /// **The recorded target is a function of the network's prior, the visits,
    /// the Q-values and the root value — and of nothing else.**
    ///
    /// This is the constraint the whole design exists for. Two searches are run
    /// from the same seed and the same policy, one floored and one not, and then
    /// given IDENTICAL root statistics by hand. Their recorded targets must be
    /// bit-identical, even though their candidate sets are not, because the
    /// target reads `edge.prior` and the noise lives elsewhere.
    ///
    /// The last assertion is what stops this passing for the wrong reason: it
    /// computes the target the other way — over the noised priors — and requires
    /// it to be a genuinely different distribution. If the noise were too small
    /// to matter, or absent, that would fail.
    #[test]
    fn the_recorded_target_ignores_the_floor_at_identical_tree_statistics() {
        let config = canonical_config();
        let policy = sloped_policy(&config);
        let mut plain =
            expanded_root_with(&config, floored_params(0.0, 31), 8_675_309, &policy, -0.4);
        let mut floored = expanded_root_with(
            &config,
            floored_params(DEFAULT_DIRICHLET_EPSILON, 31),
            8_675_309,
            &policy,
            -0.4,
        );
        assert_ne!(
            plain.candidates.iter().map(|c| c.score).collect::<Vec<_>>(),
            floored
                .candidates
                .iter()
                .map(|c| c.score)
                .collect::<Vec<_>>(),
            "the two searches are supposed to differ in what they would explore"
        );

        // Plant the same tree on both: a visited prefix with spread-out
        // Q-values, an unvisited tail completed by `v_mix`.
        let count = plain.nodes[0].edges_len as usize;
        for index in 0..count {
            let (visits, q) = if index % 5 == 0 {
                (3 + (index as u32 % 7), -0.6 + 0.03 * (index as f64 % 11.0))
            } else {
                (0, 0.0)
            };
            for search in [&mut plain, &mut floored] {
                search.edges[index].visits = visits;
                search.edges[index].value_sum = f64::from(visits) * q;
            }
        }

        let expected = plain.result().improved_policy;
        assert_eq!(
            floored.result().improved_policy,
            expected,
            "the floor reached the recorded policy target"
        );

        // The discriminator: the same target computed over the NOISED priors is
        // a different distribution, so the equality above is a statement about
        // which prior was used and not about the noise being negligible.
        let noised_edges: Vec<Edge> = (0..count)
            .map(|index| Edge {
                prior: floored.root_selection_priors[index],
                ..floored.edges[index]
            })
            .collect();
        let over_noised = improved_policy(&noised_edges, floored.root_value);
        assert_ne!(over_noised, expected);
        let divergence: f64 = over_noised
            .iter()
            .zip(&expected)
            .map(|((_, noised), (_, clean))| (noised - clean).abs())
            .sum();
        assert!(
            divergence > 0.1,
            "the two targets differ by only {divergence} in total variation; this case cannot \
             tell them apart"
        );
    }

    /// **The classic root's PUCT term reads `P'`, not `P`.**
    ///
    /// This is where Dirichlet enters [`RootMode::Classic`] — there is no Gumbel
    /// logit for it to perturb — and it needs an assertion of its own, because
    /// the obvious end-to-end test does not discriminate. "A floored classic
    /// shard differs from an unfloored one" is satisfied by the FPU's
    /// visited-prior accounting ALONE: a `select_edge` that ignored the mixture
    /// entirely would still produce different games, because the priors it
    /// charges to `visited_prior` moved. A mutation that did exactly that
    /// survived every other test in this file and in `tests/dirichlet_root.rs`.
    ///
    /// So this plants known statistics, computes the argmax both ways, requires
    /// them to disagree, and only then asks which one `select_edge` followed.
    #[test]
    fn the_classic_roots_puct_term_ranks_by_the_noised_prior() {
        let config = canonical_config();
        let policy = sloped_policy(&config);
        let mut search = expanded_root_with(
            &config,
            PuctParams {
                root_mode: RootMode::Classic,
                ..floored_params(DEFAULT_DIRICHLET_EPSILON, 2718)
            },
            281,
            &policy,
            0.2,
        );
        let count = search.nodes[0].edges_len as usize;

        // A root part-way through its budget: 100 visits spread over a handful
        // of edges, the rest untouched, so the exploration term is live
        // (`sqrt(100) = 10`) and the FPU is the same for every unvisited edge.
        search.nodes[0].visits = 100;
        search.nodes[0].value_sum = 12.0;
        search.nodes[0].visited_prior = 0.3;
        for index in 0..count {
            let (visits, q) = match index % 23 {
                0 => (9_u32, 0.42),
                7 => (5, -0.11),
                _ => (0, 0.0),
            };
            search.edges[index].visits = visits;
            search.edges[index].value_sum = f64::from(visits) * q;
        }

        let node = search.nodes[0];
        let fpu = clamp_value(
            node.value_sum / f64::from(node.visits) - FPU_REDUCTION * node.visited_prior.sqrt(),
        );
        let sqrt_total = f64::from(node.visits).sqrt();
        let argmax = |priors: &[f64]| -> usize {
            let mut best = 0_usize;
            let mut best_score = f64::NEG_INFINITY;
            for (index, prior) in priors.iter().enumerate().take(count) {
                let edge = search.edges[index];
                let q = if edge.visits > 0 {
                    edge.value_sum / f64::from(edge.visits)
                } else {
                    fpu
                };
                let score = q + DEFAULT_C_PUCT * prior * sqrt_total / f64::from(1 + edge.visits);
                if score > best_score {
                    best_score = score;
                    best = index;
                }
            }
            best
        };
        let bare: Vec<f64> = search.edges[..count]
            .iter()
            .map(|edge| edge.prior)
            .collect();
        let noised = argmax(&search.root_selection_priors);
        let unnoised = argmax(&bare);
        assert_ne!(
            noised, unnoised,
            "the planted root ranks the same edge first either way, so it cannot tell a \
             noised PUCT term from a bare one"
        );

        assert_eq!(
            search.select_edge(0) as usize,
            noised,
            "the classic root selected edge {} -- the bare-prior argmax is {unnoised} and the \
             noised one is {noised}",
            search.select_edge(0)
        );
    }

    /// The FPU's visited-prior accounting reads the mixture too, so the classic
    /// root selects and charges on ONE prior rather than two.
    ///
    /// Nothing above would notice if it did not: every other test compares a
    /// floored search against an unfloored one, and charging `edge.prior` while
    /// selecting on `P'` is wrong in a way that is invisible to that comparison.
    /// This reads the accumulator directly.
    #[test]
    fn the_classic_roots_fpu_charges_the_same_prior_it_selects_on() {
        let config = canonical_config();
        let policy = sloped_policy(&config);
        let mut search = expanded_root_with(
            &config,
            PuctParams {
                simulations: 48,
                root_mode: RootMode::Classic,
                ..floored_params(DEFAULT_DIRICHLET_EPSILON, 5150)
            },
            17,
            &policy,
            0.1,
        );
        let mut features = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];
        let mut leaf_policy = vec![0.0_f32; config.policy_size()];
        while search
            .next_leaf(&config, &mut features)
            .expect("the classic root runs to completion")
        {
            let value = crate::mock_evaluator::evaluate(&features, &mut leaf_policy);
            search
                .submit(&config, &leaf_policy, value)
                .expect("the mock evaluation is well formed");
        }

        let count = search.nodes[0].edges_len as usize;
        let visited: Vec<usize> = (0..count)
            .filter(|index| search.edges[*index].visits > 0)
            .collect();
        assert!(
            visited.len() > 3 && visited.len() < count,
            "{} of {count} root edges were visited; this case cannot discriminate",
            visited.len()
        );

        let charged: f64 = visited
            .iter()
            .map(|index| search.root_selection_priors[*index])
            .sum();
        let bare: f64 = visited.iter().map(|index| search.edges[*index].prior).sum();
        assert!(
            (search.nodes[0].visited_prior - charged).abs() < 1e-12,
            "the root charged {} against the noised {charged}",
            search.nodes[0].visited_prior
        );
        assert!(
            (charged - bare).abs() > 1e-6,
            "the noised and bare visited priors agree to {}, so this case cannot tell them apart",
            (charged - bare).abs()
        );
    }

    /// `Gamma(alpha, 1)` really is `Gamma(alpha, 1)`: mean `alpha`, variance
    /// `alpha`. The Dirichlet is only correct if this is, and the closed form is
    /// a stronger check than any ordering property of the normalised vector.
    #[test]
    fn the_gamma_sampler_matches_the_distributions_moments() {
        for alpha in [0.02_f64, 0.15, 0.5, 0.9] {
            let mut rng = Lcg32::new(0x5eed_1234);
            const DRAWS: usize = 40_000;
            let mut sum = 0.0_f64;
            let mut square_sum = 0.0_f64;
            for _ in 0..DRAWS {
                let x = sample_gamma_below_one(&mut rng, alpha);
                assert!(x.is_finite() && x >= 0.0, "Gamma({alpha}) returned {x}");
                sum += x;
                square_sum += x * x;
            }
            let mean = sum / DRAWS as f64;
            let variance = square_sum / DRAWS as f64 - mean * mean;
            // Standard error of the mean is sqrt(alpha / DRAWS) <= 0.005 here;
            // the variance converges more slowly, hence the looser band.
            assert!(
                (mean - alpha).abs() < 0.02 * alpha.max(0.15) + 0.005,
                "Gamma({alpha}) has mean {mean}"
            );
            assert!(
                (variance - alpha).abs() < 0.15 * alpha.max(0.15),
                "Gamma({alpha}) has variance {variance}"
            );
        }
    }

    /// The halving's ranking VALUES are `g + logit + qtransform`, and the
    /// qtransform is taken over the ROOT's whole edge list.
    ///
    /// The test above pins which candidates survive, which is the observable
    /// consequence; this one pins the arithmetic that produced them. `halve`
    /// leaves the scores it kept in `self.ranking`, so they can be read back and
    /// compared against [`root_qtransform`] — the public entry point the
    /// improved policy and the goldens go through. Agreement to 1e-12 says the
    /// two readers are looking at one expression, which is the property `v2`
    /// claimed and only nearly had: it scoped `max_visits` to the survivors
    /// while the improved policy scoped it to every edge.
    ///
    /// Scoping is what the second assertion is for. `mctx` computes
    /// `completed_qvalues` once over the root and hands the same vector to
    /// `seq_halving.score_considered` and to `action_weights`. A survivor-scoped
    /// rescale would divide by a different span — the unconsidered actions are
    /// completed with `v_mix` and here it sits below every candidate, so it owns
    /// the minimum — and the test asserts the two vectors really do differ
    /// before requiring `halve` to have used the root-scoped one.
    #[test]
    fn halvings_ranking_values_are_the_root_scoped_shared_expression() {
        let config = canonical_config();
        let mut search = expanded_root(&config, 8, 4_242, 1.0);
        let root = search.nodes[0];
        let start = root.edges_start as usize;
        let end = start + root.edges_len as usize;
        assert_eq!(search.candidates.len(), 8);

        // Getting the two scopes to disagree takes a specific root, and the
        // constraint is worth writing down. `v_mix` is a weighted average of the
        // raw root value and the visited actions' prior-weighted mean, so it
        // only escapes the visited Q range when the raw value is far outside it
        // AND the total visit count is small: `v_mix > qmax` needs
        // `raw > qmax + N * (qmax - qbar)`. And EVERY candidate has to be
        // visited, because an unvisited candidate would put `v_mix` into the
        // survivor-scoped vector too and the ranges would match again.
        //
        // So: eight candidates, one visit each, Q-values clustered in
        // [0.2000, 0.2175] and a raw root value of 1.0. `v_mix` lands at 0.2967,
        // above every measured Q, which makes the unconsidered actions the top
        // of the root-scoped range -- a span of 0.09667 against the survivors'
        // own 0.0175.
        for index in 0..search.candidates.len() {
            let q = 0.2 + 0.0025 * (index as f64);
            let edge = search.candidates[index].edge as usize;
            search.edges[edge].visits = 1;
            search.edges[edge].value_sum = q;
        }

        let stats_of = |edges: &[Edge]| -> Vec<ActionStats> {
            edges
                .iter()
                .map(|edge| ActionStats {
                    prior: edge.prior,
                    visits: edge.visits,
                    qvalue: if edge.visits > 0 {
                        edge.value_sum / f64::from(edge.visits)
                    } else {
                        f64::NAN
                    },
                })
                .collect()
        };
        let whole_root =
            root_qtransform(&stats_of(&search.edges[start..end]), search.root_value).transformed;
        let candidate_edges: Vec<Edge> = search
            .candidates
            .iter()
            .map(|candidate| search.edges[candidate.edge as usize])
            .collect();
        let survivor_scoped =
            root_qtransform(&stats_of(&candidate_edges), search.root_value).transformed;

        // The unconsidered actions must actually be moving the range, or the
        // scope assertion below is comparing a vector with itself.
        let scoped_differently = search
            .candidates
            .iter()
            .enumerate()
            .any(|(index, candidate)| {
                (whole_root[candidate.edge as usize - start] - survivor_scoped[index]).abs() > 1e-6
            });
        assert!(
            scoped_differently,
            "the root-scoped and survivor-scoped transforms agree here, so this case cannot \
             tell them apart"
        );

        search.halve();
        assert_eq!(search.ranking.len(), 4, "8 survivors halve to 4");
        for (index, score) in &search.ranking {
            let candidate = search.candidates[*index];
            let expected = candidate.score + whole_root[candidate.edge as usize - start];
            assert!(
                (score - expected).abs() < 1e-12,
                "halve() ranked code {} at {score}, not `g + logit + qtransform` = {expected} \
                 (the survivor-scoped transform would have given {})",
                candidate.code,
                candidate.score + survivor_scoped[*index]
            );
        }
    }
}
