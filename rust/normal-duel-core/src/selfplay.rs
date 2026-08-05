//! Batched self-play: many independent games, one shared evaluation batch.
//!
//! The batching rule
//! -----------------
//! Every game holds **at most one outstanding leaf evaluation**. A batch is
//! formed by taking that one leaf from each of `games` concurrent games, so the
//! per-game search is bit-identical to a serial run — no virtual loss, no tree
//! parallelism, nothing that changes a search decision. What changes is only
//! how many feature vectors the network sees per call.
//!
//! One qualification on "bit-identical": the value channel is `f32`, so a leaf
//! value is rounded to single precision on the way in. For the production path
//! that is lossless, and measurably so rather than by assumption. The GPU
//! inference server runs `WW_DTYPE=bf16` (the default, and what the live
//! deployment sets), so the network computes in bfloat16 and only widens to
//! `f32` for the wire. Probing the live server with a 16-position batch, all
//! 16 values and all 3344 policy logits came back with the low 16 mantissa bits
//! zero — every number it returns is a bf16 value in an `f32` box. bf16 carries
//! 8 mantissa bits; `f32` carries 24. The buffer holds 16 bits more than the
//! data ever contains.
//!
//! Where it would not be lossless is a caller computing values in `f64` —
//! `createNetworkEvaluator` returns an `f64` `tanh` — which hands over ~29 bits
//! fewer than it holds; two PUCT edges within ~6e-8 could then order differently
//! than in an `f64` serial run. That path does not feed this type in production.
//! `NormalDuelSearch`, the matchplay path, takes `f64` and has no such step, and
//! the mock evaluator is deliberately `f32`-exact, so the driver parity check
//! cannot observe the narrowing either way.
//!
//! The loop is:
//!
//! ```text
//! while !batch.done() {
//!     let n = batch.collect()?;   // advance every game to its next leaf
//!     if n == 0 { break }
//!     evaluate(features[..n], policy[..n], value[..n]);
//!     batch.submit(n)?;           // expand, back up, continue
//! }
//! let records = batch.take_records();
//! ```
//!
//! Games finish at different plies, so `n` shrinks towards the end of a batch;
//! that is the cost of never batching within a game, and it is the price of
//! exact parity.
//!
//! Record format
//! -------------
//! One record per position actually played, laid out as a flat `f32` run of
//! [`RECORD_FLOATS`]: `features` (810), `policyTarget` (209), `legalMask`
//! (209), `z` (1).
//!
//! `policyTarget` is the **visit distribution** from `effectiveVisitCounts`,
//! never a one-hot — a one-hot would throw away everything the tree computed
//! and is only correct for a search that had nothing to decide, which is the
//! case `effectiveVisitCounts` already handles.
//!
//! Exploration records the *search's* target for the state actually visited
//! even when the played move is not the search's argmax. The label answers
//! "what did search think here", and that question does not change because a
//! different move was played afterwards. This holds for both [`Exploration`]
//! modes and is the invariant `meanTargetEntropy` guards.

use crate::js_math::Lcg32;
use crate::puct::{
    apply_action_code, compact_key, PuctError, PuctParams, PuctTreeSearch, RepetitionWindow,
    RootContext,
};
use crate::{Config, Player, SearchPosition, MAX_POLICY_CODES, NN_INPUT_PLANES};

type Result<T> = std::result::Result<T, PuctError>;

/// Feature floats per position on the canonical 9x9 board.
pub const RECORD_FEATURES: usize = NN_INPUT_PLANES * 81;
/// Policy-target and legal-mask floats per position.
pub const RECORD_POLICY: usize = MAX_POLICY_CODES;
/// `features | policyTarget | legalMask | z`.
pub const RECORD_FLOATS: usize = RECORD_FEATURES + 2 * RECORD_POLICY + 1;
/// `game | ply | turn (0 = A, 1 = B) | actionCode` per record.
pub const RECORD_META_FIELDS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameOutcome {
    Ongoing,
    Win(Player),
    Draw,
}

/// How the played move is chosen once a search finishes.
///
/// Both modes leave the recorded policy target alone: it is the search's full
/// visit distribution for the state actually visited, always.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Exploration {
    /// AlphaZero's recipe. For the first [`SelfPlayOptions::temperature_moves`]
    /// plies, sample the played move from `visits^(1/T)` over the search's own
    /// visit counts; after that play the argmax.
    ///
    /// This explores only among moves the search already rated plausible, and
    /// it tapers by construction: the sampling stops when the opening does, so
    /// no exploration move can throw away a decided endgame.
    #[default]
    VisitTemperature,
    /// The legacy path: play a uniformly random legal move with probability
    /// [`SelfPlayOptions::epsilon`], otherwise the argmax.
    ///
    /// Uniform over ~130 legal codes is mostly wall placements the search had
    /// already dismissed, and it fires at every ply rather than only in the
    /// opening. Kept reachable so the two recipes can be run head to head.
    UniformEpsilon,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfPlayOptions {
    pub games: usize,
    pub simulations: u32,
    pub max_considered: u32,
    pub c_puct: f64,
    /// Which exploration recipe drives the played move.
    pub exploration: Exploration,
    /// Probability of playing a uniformly random legal move instead of the
    /// search's choice. Read only by [`Exploration::UniformEpsilon`]. The
    /// recorded target is unaffected.
    pub epsilon: f64,
    /// Softmax temperature for [`Exploration::VisitTemperature`]. `1.0` samples
    /// straight from the visit distribution; `-> 0` approaches argmax; `> 1`
    /// flattens it.
    pub temperature: f64,
    /// Number of **absolute** plies sampled from the visit distribution before
    /// switching to argmax. Absolute for the same reason [`Self::ply_cap`] is:
    /// `ply` counts the forced opening, whose plies no search produced.
    ///
    /// `0` disables temperature sampling entirely (pure argmax), which is what
    /// [`Default`] does so existing callers see no behaviour change.
    pub temperature_moves: u64,
    /// Cap on a game's **absolute** ply count, matching the incumbent shard
    /// worker's `while (state.outcome.kind === 'ongoing' && state.ply < PLY_CAP)`.
    ///
    /// Absolute, not recorded: `state.ply` counts the forced opening's plies,
    /// so a game opened from a 5-ply book line and capped at 60 records 55
    /// positions, not 60. Counting recorded plies here instead ran every opened
    /// game past the cap by the opening's length, which forked the shard at the
    /// first game to reach it. Adjudication still uses `config.ply_cap`.
    pub ply_cap: u64,
    /// Game `i` runs off `seed_base + i`, so a batch reproduces exactly.
    pub seed_base: u32,
    /// Optional forced openings as action-code sequences; game `i` uses
    /// `openings[i % openings.len()]`. Opening plies are played but not
    /// recorded, because no search produced them.
    pub openings: Vec<Vec<u16>>,
}

impl Default for SelfPlayOptions {
    fn default() -> Self {
        Self {
            games: 8,
            simulations: 32,
            max_considered: 8,
            c_puct: crate::puct::DEFAULT_C_PUCT,
            exploration: Exploration::VisitTemperature,
            epsilon: 0.0,
            temperature: 1.0,
            temperature_moves: 0,
            ply_cap: 200,
            seed_base: 0,
            openings: Vec::new(),
        }
    }
}

/// Sample an action code from `target^(1/temperature)`, where `target` is the
/// normalised visit distribution already written as the policy target.
///
/// Sampling from the *target* rather than from the raw `(code, visits)` pairs is
/// deliberate on two counts. It reuses the exact f32 values that went to disk,
/// so the sampled move can never be drawn from a distribution the record does
/// not describe; and it fixes the traversal to ascending code order, so the f64
/// accumulator sees the same addition sequence on every platform.
///
/// `roll` must be in `[0, 1)`. `word / 2^32` is the intended source: the divisor
/// is a power of two, so that quotient is exact in f64 and identical in JS,
/// which is what lets `cluster-selfplay.mjs` reproduce this draw for draw.
///
/// `temperature == 1.0` short-circuits the `powf`: `powf` is not guaranteed
/// bit-identical across targets, and `T = 1` is the default, so the default path
/// stays free of it. Any other `T` is reproducible per platform but should not
/// be assumed bit-exact between the wasm and native builds.
///
/// Returns `None` when `target` carries no positive mass, which leaves the
/// caller on the search's argmax.
#[must_use]
pub fn sample_visit_temperature(target: &[f32], temperature: f64, roll: f64) -> Option<u16> {
    let exponent = 1.0 / temperature;
    let weight = |p: f32| -> f64 {
        let p = f64::from(p);
        if temperature == 1.0 {
            p
        } else {
            p.powf(exponent)
        }
    };

    let mut total = 0.0_f64;
    for &p in target.iter() {
        if p > 0.0 {
            total += weight(p);
        }
    }
    // `is_finite` first, so a NaN total is rejected here rather than falling
    // through a negated comparison.
    if !total.is_finite() || total <= 0.0 {
        return None;
    }

    let threshold = roll * total;
    let mut acc = 0.0_f64;
    let mut last = None;
    for (code, &p) in target.iter().enumerate() {
        if p <= 0.0 {
            continue;
        }
        acc += weight(p);
        last = Some(code as u16);
        if acc > threshold {
            return last;
        }
    }
    // Only reachable when float error leaves `acc` a hair under `threshold` on
    // the final code. Falling through to the last positive code keeps the draw
    // inside the support instead of reporting failure.
    last
}

#[derive(Debug, Clone)]
struct PendingPly {
    features: Vec<f32>,
    policy_target: Vec<f32>,
    legal_mask: Vec<f32>,
    turn: Player,
    ply: u64,
    action_code: u16,
}

#[derive(Debug)]
struct Game {
    index: usize,
    position: SearchPosition,
    ply: u64,
    window: RepetitionWindow,
    rng: Lcg32,
    search: Option<PuctTreeSearch>,
    plies: Vec<PendingPly>,
    outcome: GameOutcome,
    finished: bool,
    records: Vec<f32>,
    meta: Vec<i32>,
    codes: [u16; MAX_POLICY_CODES],
}

impl Game {
    fn start(config: &Config, options: &SelfPlayOptions, index: usize) -> Result<Self> {
        let state = crate::create_initial_state(config)?;
        let position = SearchPosition::from_position(config, &state.position)?;
        let mut game = Self {
            index,
            position,
            ply: 0,
            window: RepetitionWindow::fresh(compact_key(
                config,
                position.pawns.a,
                position.pawns.b,
                position.turn,
            )),
            rng: Lcg32::new(options.seed_base.wrapping_add(index as u32)),
            search: None,
            plies: Vec::new(),
            outcome: GameOutcome::Ongoing,
            finished: false,
            records: Vec::new(),
            meta: Vec::new(),
            codes: [0; MAX_POLICY_CODES],
        };
        if !options.openings.is_empty() {
            let opening = &options.openings[index % options.openings.len()];
            for code in opening {
                if game.outcome != GameOutcome::Ongoing {
                    break;
                }
                let count = game
                    .position
                    .legal_action_codes_fast(config, &mut game.codes);
                if !game.codes[..count].contains(code) {
                    return Err(PuctError::InvalidActionCode);
                }
                game.play(config, *code)?;
            }
        }
        Ok(game)
    }

    /// Apply one legal action code and re-adjudicate, without building a
    /// `GameState`. Mirrors `applyTrustedAction`: a wall placement restarts the
    /// repetition window, a pawn move extends it.
    // Repetition and ply cap both yield a draw, so clippy sees identical arms in
    // the terminal chain below. They stay separate because it mirrors
    // `adjudicate`'s documented order -- goal, then repetition, then ply cap --
    // and those are three distinct reasons a game ended. Collapsing them would
    // save a line and lose the correspondence.
    #[allow(clippy::if_same_then_else)]
    fn play(&mut self, config: &Config, code: u16) -> Result<()> {
        let applied = apply_action_code(config, &self.position, code)?;
        self.position = applied.position;
        self.ply += 1;
        let key = compact_key(
            config,
            self.position.pawns.a,
            self.position.pawns.b,
            self.position.turn,
        );
        let repetitions = if applied.placed_wall {
            self.window.reset(key);
            1
        } else {
            self.window.push(key)
        };
        self.outcome = if let Some(winner) = applied.goal_winner {
            GameOutcome::Win(winner)
        } else if u64::from(repetitions) >= u64::from(config.repetition_threshold) {
            GameOutcome::Draw
        } else if self.ply >= config.ply_cap {
            GameOutcome::Draw
        } else {
            GameOutcome::Ongoing
        };
        Ok(())
    }

    /// Advance to this game's next leaf evaluation, writing its features into
    /// `features`. Returns `false` once the game is over.
    fn advance(
        &mut self,
        config: &Config,
        options: &SelfPlayOptions,
        features: &mut [f32],
        mask: &mut [f32],
    ) -> Result<bool> {
        loop {
            if self.finished {
                return Ok(false);
            }
            if self.search.is_none() {
                if self.outcome != GameOutcome::Ongoing || self.ply >= options.ply_cap {
                    self.finish();
                    return Ok(false);
                }
                self.search = Some(PuctTreeSearch::new(
                    config,
                    RootContext {
                        position: self.position,
                        ply: self.ply,
                        window: self.window.clone(),
                    },
                    PuctParams {
                        simulations: options.simulations,
                        max_considered: options.max_considered,
                        c_puct: options.c_puct,
                    },
                    self.rng,
                )?);
            }
            let search = self
                .search
                .as_mut()
                .ok_or(PuctError::OutOfOrderEvaluation)?;
            if search.next_leaf(config, features)? {
                search.pending_leaf_mask(config, mask)?;
                return Ok(true);
            }
            self.complete_move(config, options)?;
        }
    }

    fn submit(&mut self, config: &Config, policy: &[f32], value: f64) -> Result<()> {
        let search = self
            .search
            .as_mut()
            .ok_or(PuctError::OutOfOrderEvaluation)?;
        search.submit(config, policy, value)
    }

    /// The search finished: record the position, choose the move, play it.
    fn complete_move(&mut self, config: &Config, options: &SelfPlayOptions) -> Result<()> {
        let search = self.search.take().ok_or(PuctError::OutOfOrderEvaluation)?;
        self.rng = search.rng();
        let result = search.result();
        let turn = self.position.turn;

        let mut features = vec![0.0_f32; RECORD_FEATURES];
        crate::encode_board_into(
            config,
            &self.position.board(),
            self.position.pawns,
            self.position.stock,
            self.position.turn,
            &mut features,
        );

        let count = self
            .position
            .legal_action_codes_fast(config, &mut self.codes);
        if count == 0 {
            return Err(PuctError::NoLegalActions);
        }
        let mut legal_mask = vec![0.0_f32; RECORD_POLICY];
        for code in &self.codes[..count] {
            legal_mask[usize::from(*code)] = 1.0;
        }

        // `encodePolicyTarget`: normalise the visit distribution over the codes
        // it names, writing in ascending code order so the float rounding does
        // not depend on iteration order.
        let effective = result.effective_visit_counts();
        let total: u64 = effective.iter().map(|(_, visits)| u64::from(*visits)).sum();
        let mut policy_target = vec![0.0_f32; RECORD_POLICY];
        for (code, visits) in &effective {
            if legal_mask[usize::from(*code)] != 1.0 {
                return Err(PuctError::InvalidActionCode);
            }
            policy_target[usize::from(*code)] = (f64::from(*visits) / total as f64) as f32;
        }

        // Exploration. Every draw below comes from the same per-game `Lcg32`
        // that drives the search, taken *after* `search.rng()` was folded back
        // in above and *after* the policy target is computed -- i.e. at exactly
        // the stream position the incumbent epsilon probe occupied. A seed
        // therefore still reproduces the whole game.
        let mut played = result.action_code;
        match options.exploration {
            // The probe is `(word % 10000) / 10000`, NOT `word / 2^32`. Both
            // are uniform on [0, 1) and the two are not interchangeable here:
            // they map the *same* word to different sides of the epsilon
            // threshold for a large fraction of words, so the second form
            // accepts a different subset of plies. Once one probe disagrees,
            // the accepted ply consumes an extra draw for the random-move pick,
            // the two streams are offset by one word forever after, and the
            // games fork. This form is the incumbent shard worker's, and it is
            // the one the RNG stream is defined by.
            Exploration::UniformEpsilon => {
                if options.epsilon > 0.0 {
                    let roll = f64::from(self.rng.next_u32() % 10_000) / 10_000.0;
                    if roll < options.epsilon {
                        let pick = self.rng.next_u32() as usize % count;
                        played = self.codes[pick];
                    }
                }
            }
            // Exactly ONE word per ply while the temperature phase is live, and
            // ZERO once it is over. Unconditional within the phase: not skipped
            // when the search had a single candidate, not skipped when the
            // sample lands on the argmax anyway. That makes the stream offset a
            // pure function of the ply index, so a fork cannot be introduced by
            // a search-shape change that happens to alter how many plies would
            // have "needed" a draw.
            Exploration::VisitTemperature => {
                if self.ply < options.temperature_moves {
                    let roll = f64::from(self.rng.next_u32()) / 4_294_967_296.0;
                    if let Some(code) =
                        sample_visit_temperature(&policy_target, options.temperature, roll)
                    {
                        played = code;
                    }
                }
            }
        }
        if !self.codes[..count].contains(&played) {
            return Err(PuctError::InvalidActionCode);
        }

        self.plies.push(PendingPly {
            features,
            policy_target,
            legal_mask,
            turn,
            ply: self.ply,
            action_code: played,
        });
        self.play(config, played)
    }

    /// Stamp `z` onto every recorded ply and flatten them into the record
    /// buffer. `z` is from that ply's mover's perspective, as the trainer wants.
    fn finish(&mut self) {
        self.finished = true;
        self.records.reserve(self.plies.len() * RECORD_FLOATS);
        self.meta.reserve(self.plies.len() * RECORD_META_FIELDS);
        for ply in &self.plies {
            let z = match self.outcome {
                GameOutcome::Win(winner) => {
                    if winner == ply.turn {
                        1.0
                    } else {
                        -1.0
                    }
                }
                _ => 0.0,
            };
            self.records.extend_from_slice(&ply.features);
            self.records.extend_from_slice(&ply.policy_target);
            self.records.extend_from_slice(&ply.legal_mask);
            self.records.push(z);
            self.meta.push(self.index as i32);
            self.meta.push(ply.ply as i32);
            self.meta.push(i32::from(ply.turn == Player::B));
            self.meta.push(i32::from(ply.action_code));
        }
        self.plies.clear();
        self.plies.shrink_to_fit();
    }
}

/// A fixed set of concurrent games sharing one evaluation batch.
///
/// # Buffer lifetime (the wasm-memory-growth hazard)
///
/// [`Self::features`], [`Self::policy_mut`] and [`Self::value_mut`] hand out
/// slices into buffers allocated once at construction and **never resized**, so
/// on the native side their addresses are stable for the batch's lifetime.
///
/// That is *not* enough for the wasm boundary. Growing the wasm heap
/// reallocates its backing `ArrayBuffer` and detaches every JS typed-array view
/// over it; a detached view reads as zeros or throws, silently or otherwise. A
/// batch cannot promise never to grow the heap — the tree arenas and the record
/// buffer both allocate as a run proceeds — so the wasm wrapper documents the
/// only safe rule: **rebuild every view after every call into wasm.** See
/// `rust/normal-duel-wasm/src/lib.rs`, and `scripts/diagnose-selfplay-driver-parity.mjs`
/// for a driver that follows the rule.
#[derive(Debug)]
pub struct SelfPlayBatch {
    config: Config,
    options: SelfPlayOptions,
    games: Vec<Game>,
    features: Vec<f32>,
    policy: Vec<f32>,
    value: Vec<f32>,
    legal: Vec<f32>,
    slots: Vec<usize>,
    records: Vec<f32>,
    meta: Vec<i32>,
}

/// Bounds on the options that drive the up-front record reservation.
///
/// The per-factor caps are cheap early rejections and are NOT the real bound:
/// 512 games x 2048 plies x 1229 floats x 4 B is about 5.15 GB, past wasm32's
/// entire address space, so at the top of the range those two alone still let
/// `Vec::with_capacity` abort instead of returning a readable error -- exactly
/// what the bound exists to prevent. [`MAX_RECORD_BYTES`] is what holds the line.
const MAX_GAMES: usize = 512;
const MAX_PLY_CAP: u64 = 2048;
/// Ceiling on the up-front `records` reservation, in bytes. 256 MiB sits under
/// the documented 512 MiB wasm memory profile with room for the trunk, the meta
/// buffer and the JS heap. The cluster's shard shape -- 32 games at `ply_cap`
/// 200 -- reserves about 31 MiB, so this is roughly 8x real use.
const MAX_RECORD_BYTES: usize = 256 * 1024 * 1024;
/// Width of a record float, for the byte bound above.
const BYTES_PER_F32: usize = 4;

impl SelfPlayBatch {
    pub fn new(config: &Config, options: SelfPlayOptions) -> Result<Self> {
        config.validate()?;
        if config.rows != 9 || config.columns != 9 {
            return Err(PuctError::UnsupportedBoard);
        }
        if options.games == 0 {
            return Err(PuctError::InvalidBufferLength);
        }
        // `games` and `ply_cap` arrive from JSON and are multiplied into an
        // up-front `Vec::with_capacity` below. Unbounded, they abort rather than
        // erroring: on wasm32 `usize` is 32 bits, so a large `ply_cap` saturates
        // to `usize::MAX` and `with_capacity` raises a capacity-overflow panic
        // across the boundary instead of returning `invalid_options`. Even
        // in-range values bite -- `games: 64, ply_cap: 4000` reserves
        // 64 * 4000 * 1229 * 4 B, about 1.26 GB, against a 512 MiB budget.
        // Bound them here so the failure is a string the caller can read.
        if options.games > MAX_GAMES {
            return Err(PuctError::InvalidGames);
        }
        if options.ply_cap == 0 || options.ply_cap > MAX_PLY_CAP {
            return Err(PuctError::InvalidPlyCap);
        }
        // `simulations == 0` must be rejected, not tolerated. With no
        // simulations every search returns empty visit counts, so
        // `effective_visit_counts` falls back to a one-hot of the chosen action
        // and the recorded policy target becomes one-hot at a position with ~130
        // legal codes -- silently, at full record count. That is precisely the
        // degenerate target that removes AlphaZero's improvement ratchet and cost
        // an earlier run 114 flat iterations. `max_considered == 0` was already
        // rejected, so this was an asymmetry rather than a decision, and the
        // trigger is mundane: `Number('')` is 0, so any driver resolving its sim
        // count from an unset environment variable sends it.
        if options.simulations == 0 {
            return Err(PuctError::InvalidSimulations);
        }
        // Reject option combinations that contradict themselves, rather than
        // silently honouring one and ignoring the other.
        //
        // `exploration` defaults to VisitTemperature and `temperature_moves` to
        // 0, so an options object in the PRE-CHANGE wire format -- carrying
        // `epsilon` and no `exploration` key -- parsed cleanly and ran pure
        // argmax with NO exploration at all, while `epsilon` was range-checked
        // and then never read. An old driver must fail loudly, not quietly stop
        // exploring.
        if options.exploration == Exploration::VisitTemperature && options.epsilon != 0.0 {
            return Err(PuctError::ContradictoryExploration);
        }
        if options.exploration == Exploration::UniformEpsilon && options.temperature_moves != 0 {
            return Err(PuctError::ContradictoryExploration);
        }
        if !(0.0..=1.0).contains(&options.epsilon) || !options.epsilon.is_finite() {
            return Err(PuctError::InvalidEvaluation);
        }
        // A non-positive or non-finite temperature would make `1/T` infinite or
        // NaN and every weight garbage. Only checked when it can actually be
        // read, so a caller on the epsilon path need not supply a sane T.
        if options.exploration == Exploration::VisitTemperature
            && options.temperature_moves > 0
            && !(options.temperature.is_finite() && options.temperature > 0.0)
        {
            return Err(PuctError::InvalidEvaluation);
        }
        let games = (0..options.games)
            .map(|index| Game::start(config, &options, index))
            .collect::<Result<Vec<_>>>()?;
        // Reserve against the cap a game can actually reach. Adjudication stops
        // at `config.ply_cap`, so an option cap above it can never be reached and
        // reserving for it wastes memory a caller cannot use -- today by up to
        // 10x. The bound below is then in bytes, because the per-factor caps
        // multiply to ~5.15 GB and would abort rather than error.
        let effective_ply_cap = options.ply_cap.min(config.ply_cap);
        let positions = options
            .games
            .saturating_mul(usize::try_from(effective_ply_cap).unwrap_or(usize::MAX));
        let record_capacity = positions.saturating_mul(RECORD_FLOATS);
        if record_capacity.saturating_mul(BYTES_PER_F32) > MAX_RECORD_BYTES {
            return Err(PuctError::InvalidBufferLength);
        }
        let meta_capacity = positions.saturating_mul(RECORD_META_FIELDS);
        Ok(Self {
            config: config.clone(),
            features: vec![0.0; options.games * RECORD_FEATURES],
            policy: vec![0.0; options.games * RECORD_POLICY],
            value: vec![0.0; options.games],
            legal: vec![0.0; options.games * RECORD_POLICY],
            slots: Vec::with_capacity(options.games),
            // Reserved to the worst case — every game recording a full
            // `ply_cap` of plies — so the sinks the wasm boundary hands out
            // pointers to never reallocate, and so `take_records` cannot grow
            // the wasm heap and detach a live JS view. See the type docs.
            records: Vec::with_capacity(record_capacity),
            meta: Vec::with_capacity(meta_capacity),
            options,
            games,
        })
    }

    /// Advance every unfinished game to its next leaf, filling the first `n`
    /// feature slots. Returns `n`; `0` means every game is over.
    pub fn collect(&mut self) -> Result<usize> {
        self.slots.clear();
        for index in 0..self.games.len() {
            if self.games[index].finished {
                continue;
            }
            let slot = self.slots.len();
            let window = &mut self.features[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES];
            let mask = &mut self.legal[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY];
            if self.games[index].advance(&self.config, &self.options, window, mask)? {
                self.slots.push(index);
            }
        }
        Ok(self.slots.len())
    }

    /// Feed back `n` evaluations, in the slot order [`Self::collect`] produced.
    pub fn submit(&mut self, n: usize) -> Result<()> {
        if n != self.slots.len() {
            return Err(PuctError::InvalidBufferLength);
        }
        for slot in 0..n {
            let game = self.slots[slot];
            let policy = &self.policy[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY];
            let value = f64::from(self.value[slot]);
            self.games[game].submit(&self.config, policy, value)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.games.iter().all(|game| game.finished)
    }

    /// `n * RECORD_FEATURES` floats, slot-major.
    #[must_use]
    pub fn features(&self) -> &[f32] {
        &self.features
    }

    /// `n * RECORD_POLICY` legal-mask floats, slot-major, parallel to
    /// [`Self::features`]. The evaluator masks before its softmax with these.
    #[must_use]
    pub fn legal_mask(&self) -> &[f32] {
        &self.legal
    }

    /// Write `n * RECORD_POLICY` policy logits here before [`Self::submit`].
    pub fn policy_mut(&mut self) -> &mut [f32] {
        &mut self.policy
    }

    /// Write `n` values, each in `[-1, 1]`, here before [`Self::submit`].
    pub fn value_mut(&mut self) -> &mut [f32] {
        &mut self.value
    }

    /// Drain every finished game's records into the batch's own buffer and
    /// return how many records are now available from [`Self::records`].
    pub fn take_records(&mut self) -> usize {
        self.records.clear();
        self.meta.clear();
        for game in &mut self.games {
            self.records.append(&mut game.records);
            self.meta.append(&mut game.meta);
        }
        self.records.len() / RECORD_FLOATS
    }

    /// Valid until the next [`Self::take_records`].
    #[must_use]
    pub fn records(&self) -> &[f32] {
        &self.records
    }

    /// `RECORD_META_FIELDS` ints per record, parallel to [`Self::records`].
    #[must_use]
    pub fn record_meta(&self) -> &[i32] {
        &self.meta
    }

    #[must_use]
    pub fn outcomes(&self) -> Vec<GameOutcome> {
        self.games.iter().map(|game| game.outcome).collect()
    }

    #[must_use]
    pub fn plies_played(&self) -> Vec<u64> {
        self.games.iter().map(|game| game.ply).collect()
    }
}
