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
//! [`RECORD_FLOATS`]: `features` (648), `policyTarget` (209), `legalMask`
//! (209), `z` (1).
//!
//! `policyTarget` is the **visit distribution** from `effectiveVisitCounts`,
//! never a one-hot — a one-hot would throw away everything the tree computed
//! and is only correct for a search that had nothing to decide, which is the
//! case `effectiveVisitCounts` already handles.
//!
//! Epsilon exploration records the *search's* target for the state actually
//! visited even when the played move is the random one. The label answers
//! "what did search think here", and that question does not change because a
//! different move was played afterwards.

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

#[derive(Debug, Clone, PartialEq)]
pub struct SelfPlayOptions {
    pub games: usize,
    pub simulations: u32,
    pub max_considered: u32,
    pub c_puct: f64,
    /// Probability of playing a uniformly random legal move instead of the
    /// search's choice. The recorded target is unaffected.
    pub epsilon: f64,
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
            epsilon: 0.0,
            ply_cap: 200,
            seed_base: 0,
            openings: Vec::new(),
        }
    }
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

        // Epsilon exploration. The draw is taken from the same stream that
        // drives the search, so a seed still reproduces the whole game.
        //
        // The probe is `(word % 10000) / 10000`, NOT `word / 2^32`. Both are
        // uniform on [0, 1) and the two are not interchangeable here: they map
        // the *same* word to different sides of the epsilon threshold for a
        // large fraction of words, so the second form accepts a different
        // subset of plies. Once one probe disagrees, the accepted ply consumes
        // an extra draw for the random-move pick, the two streams are offset by
        // one word forever after, and the games fork. This form is the
        // incumbent shard worker's, and it is the one the RNG stream is
        // defined by.
        let mut played = result.action_code;
        if options.epsilon > 0.0 {
            let roll = f64::from(self.rng.next_u32() % 10_000) / 10_000.0;
            if roll < options.epsilon {
                let pick = self.rng.next_u32() as usize % count;
                played = self.codes[pick];
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
/// `rust/normal-duel-wasm/src/lib.rs` and `js/normal-duel-selfplay-batch.mjs`.
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

impl SelfPlayBatch {
    pub fn new(config: &Config, options: SelfPlayOptions) -> Result<Self> {
        config.validate()?;
        if config.rows != 9 || config.columns != 9 {
            return Err(PuctError::UnsupportedBoard);
        }
        if options.games == 0 {
            return Err(PuctError::InvalidBufferLength);
        }
        if !(0.0..=1.0).contains(&options.epsilon) || !options.epsilon.is_finite() {
            return Err(PuctError::InvalidEvaluation);
        }
        let games = (0..options.games)
            .map(|index| Game::start(config, &options, index))
            .collect::<Result<Vec<_>>>()?;
        let positions = options
            .games
            .saturating_mul(usize::try_from(options.ply_cap).unwrap_or(usize::MAX));
        let record_capacity = positions.saturating_mul(RECORD_FLOATS);
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
