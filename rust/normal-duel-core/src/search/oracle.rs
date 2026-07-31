use std::collections::{HashMap, HashSet};

use std::time::Duration;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now_millis() -> f64;
}

use crate::{
    compact_legal_codes, Config, Outcome, PreparedGameState, PreparedOracleIdentity, Result,
};

use super::budget::{CanonicalOracleLimits, SearchBudget};
use super::eval::MATE_SCORE;
use super::tt::mirror_code;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Wdl {
    Loss = -1,
    Draw = 0,
    Win = 1,
}

impl Wdl {
    const fn negate(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::Draw => Self::Draw,
            Self::Win => Self::Loss,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleValue {
    wdl: Wdl,
    distance: u64,
    best_action: Option<usize>,
}

#[derive(Debug, Clone)]
struct MemoEntry {
    identity: PreparedOracleIdentity,
    value: OracleValue,
}

/// Exact zero-stock solver over the complete augmented game state.
///
/// The search is finite because `ply` strictly increases and the configured
/// ply cap is part of the identity. Memoization includes pawns, side to move,
/// fixed wall layout, absolute ply, outcome, and the complete active
/// repetition-count multiset. This is intentionally more expensive than a
/// wall-layout-only pawn graph, which would be unsound under threefold draws.
#[derive(Debug, Default)]
pub(crate) struct ExactZeroWallOracle {
    memo: HashMap<u64, Vec<MemoEntry>>,
    /// Cheap canonical search-key prefilter for memo-only probes after a
    /// canonical oracle backoff. A hit here is only a hint: the complete
    /// repetition-aware oracle identity still verifies the exact memo entry.
    memo_search_keys: HashSet<u64>,
    quota_backoff_active: bool,
}

impl ExactZeroWallOracle {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) const fn quota_backoff_active(&self) -> bool {
        self.quota_backoff_active
    }

    fn terminal(state: &PreparedGameState) -> Option<OracleValue> {
        match state.outcome {
            Outcome::Ongoing => None,
            Outcome::Draw { .. } => Some(OracleValue {
                wdl: Wdl::Draw,
                distance: 0,
                best_action: None,
            }),
            Outcome::Win { winner, .. } => Some(OracleValue {
                wdl: if winner == state.position.turn {
                    Wdl::Win
                } else {
                    Wdl::Loss
                },
                distance: 0,
                best_action: None,
            }),
        }
    }

    fn lookup(&self, config: &Config, identity: &PreparedOracleIdentity) -> Option<OracleValue> {
        let mut value = self
            .memo
            .get(&identity.key())?
            .iter()
            .find(|entry| entry.identity == *identity)?
            .value;
        if identity.mirrored() {
            value.best_action = value.best_action.map(|code| mirror_code(config, code));
        }
        Some(value)
    }

    fn store(
        &mut self,
        config: &Config,
        identity: PreparedOracleIdentity,
        search_key: u64,
        mut value: OracleValue,
    ) {
        if identity.mirrored() {
            value.best_action = value.best_action.map(|code| mirror_code(config, code));
        }
        let bucket = self.memo.entry(identity.key()).or_default();
        if let Some(existing) = bucket.iter_mut().find(|entry| entry.identity == identity) {
            existing.value = value;
        } else {
            bucket.push(MemoEntry { identity, value });
        }
        self.memo_search_keys.insert(search_key);
    }

    fn better(candidate: OracleValue, incumbent: OracleValue) -> bool {
        if candidate.wdl != incumbent.wdl {
            return candidate.wdl > incumbent.wdl;
        }
        match candidate.wdl {
            Wdl::Win => {
                candidate.distance < incumbent.distance
                    || (candidate.distance == incumbent.distance
                        && candidate.best_action < incumbent.best_action)
            }
            Wdl::Loss => {
                candidate.distance > incumbent.distance
                    || (candidate.distance == incumbent.distance
                        && candidate.best_action < incumbent.best_action)
            }
            Wdl::Draw => candidate.best_action < incumbent.best_action,
        }
    }

    fn visit<B: SearchBudget>(budget: &mut B, nodes: &mut u64) -> bool {
        if budget.exhausted(*nodes) {
            false
        } else {
            *nodes += 1;
            true
        }
    }

    fn solve_value<B: SearchBudget>(
        &mut self,
        config: &Config,
        state: &mut PreparedGameState,
        budget: &mut B,
        nodes: &mut u64,
    ) -> Result<Option<OracleValue>> {
        if !Self::visit(budget, nodes) {
            return Ok(None);
        }
        if let Some(value) = Self::terminal(state) {
            return if budget.exhausted(*nodes) {
                Ok(None)
            } else {
                Ok(Some(value))
            };
        }
        debug_assert_eq!(state.position.stock.a, 0);
        debug_assert_eq!(state.position.stock.b, 0);
        let identity = state.oracle_identity(config);
        if let Some(value) = self.lookup(config, &identity) {
            return if budget.exhausted(*nodes) {
                Ok(None)
            } else {
                Ok(Some(value))
            };
        }

        let actions = compact_legal_codes(config, state);
        let mut best: Option<OracleValue> = None;
        for code in actions.iter() {
            if budget.exhausted(*nodes) {
                return Ok(None);
            }
            let undo = state.apply_generated_code(config, code)?;
            let child = self.solve_value(config, state, budget, nodes);
            if !state.undo_generated_code(undo) {
                return Err(crate::NormalDuelError::InvalidState);
            }
            let Some(child) = child? else {
                return Ok(None);
            };
            let candidate = OracleValue {
                wdl: child.wdl.negate(),
                distance: child.distance.saturating_add(1),
                best_action: Some(code),
            };
            if best.map_or(true, |incumbent| Self::better(candidate, incumbent)) {
                best = Some(candidate);
            }
            // A one-ply win is the strongest possible exact result.
            if candidate.wdl == Wdl::Win && candidate.distance == 1 {
                break;
            }
        }
        let Some(best) = best else {
            return Err(crate::NormalDuelError::InvalidState);
        };
        if budget.exhausted(*nodes) {
            return Ok(None);
        }
        self.store(config, identity, state.search_identity(config).key(), best);
        Ok(Some(best))
    }

    fn encode_hit(state: &PreparedGameState, root_ply: u64, value: OracleValue) -> OracleHit {
        let root_distance = state
            .ply
            .saturating_sub(root_ply)
            .saturating_add(value.distance);
        let encoded_distance =
            i32::try_from(root_distance.min((MATE_SCORE - 1) as u64)).unwrap_or(MATE_SCORE - 1);
        let score = match value.wdl {
            Wdl::Win => MATE_SCORE - encoded_distance,
            Wdl::Draw => 0,
            Wdl::Loss => -MATE_SCORE + encoded_distance,
        };
        OracleHit {
            score,
            best_action: value.best_action,
            wdl: value.wdl,
            distance: value.distance,
        }
    }

    pub(crate) fn solve<B: SearchBudget>(
        &mut self,
        config: &Config,
        state: &mut PreparedGameState,
        budget: &mut B,
        nodes: &mut u64,
        root_ply: u64,
    ) -> Result<Option<OracleHit>> {
        if state.position.stock.a != 0 || state.position.stock.b != 0 {
            return Ok(None);
        }
        let Some(value) = self.solve_value(config, state, budget, nodes)? else {
            return Ok(None);
        };
        if budget.exhausted(*nodes) {
            return Ok(None);
        }
        Ok(Some(Self::encode_hit(state, root_ply, value)))
    }

    /// Try one exact solve while charging nodes to the real search budget and
    /// independently limiting this attempt's oracle work.
    ///
    /// The proxy always polls the parent first. If both limits become
    /// exhausted at the same poll, the parent exhaustion therefore preserves
    /// the search's existing atomic interruption semantics. A local-slice or
    /// node-quota-only stop disables later oracle attempts for this search;
    /// the caller may then resume ordinary negamax from the exactly restored
    /// state.
    pub(crate) fn solve_with_canonical_limits<B: SearchBudget>(
        &mut self,
        config: &Config,
        state: &mut PreparedGameState,
        budget: &mut B,
        nodes: &mut u64,
        root_ply: u64,
        limits: CanonicalOracleLimits,
    ) -> Result<QuotaOracleResult> {
        debug_assert!(!self.quota_backoff_active);
        let node_limit = nodes.saturating_add(limits.node_quota);
        let mut quota_budget = OracleQuotaBudget::new(budget, node_limit, limits.wall_clock_slice);
        let hit = self.solve(config, state, &mut quota_budget, nodes, root_ply)?;
        let parent_exhausted = quota_budget.parent_exhausted;
        let quota_exhausted = quota_budget.quota_exhausted;
        if let Some(hit) = hit {
            return Ok(QuotaOracleResult::Exact(hit));
        }
        if parent_exhausted {
            return Ok(QuotaOracleResult::ParentExhausted);
        }
        if quota_exhausted {
            self.quota_backoff_active = true;
            return Ok(QuotaOracleResult::QuotaBackoff);
        }
        // An eligible exact solve returns `None` only after its budget reports
        // exhaustion. Treat any other shape as an invalid internal state.
        Err(crate::NormalDuelError::InvalidState)
    }

    /// Probe only exact values already proven before a quota backoff.
    ///
    /// This deliberately performs the same charged visit and post-visit
    /// parent-budget poll as an ordinary memo hit, but it never generates an
    /// action or descends recursively. With a fixed-node parent, one remaining
    /// node is therefore an atomic parent exhaustion while two remaining
    /// nodes can return an exact cached value.
    pub(crate) fn probe_memo_after_backoff<B: SearchBudget>(
        &self,
        config: &Config,
        state: &PreparedGameState,
        budget: &mut B,
        nodes: &mut u64,
        root_ply: u64,
    ) -> PostBackoffMemoResult {
        if !Self::visit(budget, nodes) {
            return PostBackoffMemoResult::ParentExhausted;
        }
        let value = if state.position.stock.a == 0 && state.position.stock.b == 0 {
            let search_key = state.search_identity(config).key();
            self.memo_search_keys
                .contains(&search_key)
                .then(|| self.lookup(config, &state.oracle_identity(config)))
                .flatten()
        } else {
            None
        };
        if budget.exhausted(*nodes) {
            return PostBackoffMemoResult::ParentExhausted;
        }
        match value {
            Some(value) => PostBackoffMemoResult::Exact(Self::encode_hit(state, root_ply, value)),
            None => PostBackoffMemoResult::Miss,
        }
    }
}

struct OracleQuotaBudget<'a, B: SearchBudget> {
    parent: &'a mut B,
    node_limit: u64,
    local_slice: LocalOracleSlice,
    parent_exhausted: bool,
    quota_exhausted: bool,
}

impl<'a, B: SearchBudget> OracleQuotaBudget<'a, B> {
    fn new(parent: &'a mut B, node_limit: u64, wall_clock_slice: Option<Duration>) -> Self {
        Self {
            parent,
            node_limit,
            local_slice: LocalOracleSlice::new(wall_clock_slice),
            parent_exhausted: false,
            quota_exhausted: false,
        }
    }

    #[cfg(test)]
    fn with_test_local_expiry(parent: &'a mut B, node_limit: u64, allowed_polls: u64) -> Self {
        Self {
            parent,
            node_limit,
            local_slice: LocalOracleSlice::TestPolls(allowed_polls),
            parent_exhausted: false,
            quota_exhausted: false,
        }
    }
}

impl<B: SearchBudget> SearchBudget for OracleQuotaBudget<'_, B> {
    fn exhausted(&mut self, visited_nodes: u64) -> bool {
        if self.parent.exhausted(visited_nodes) {
            self.parent_exhausted = true;
            return true;
        }
        if self.local_slice.exhausted() {
            self.quota_exhausted = true;
            return true;
        }
        if visited_nodes >= self.node_limit {
            self.quota_exhausted = true;
            return true;
        }
        false
    }
}

enum LocalOracleSlice {
    Disabled,
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    Deadline(Instant),
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    Deadline {
        started_millis: f64,
        duration_millis: f64,
    },
    #[cfg(test)]
    TestPolls(u64),
}

impl LocalOracleSlice {
    fn new(slice: Option<Duration>) -> Self {
        let Some(slice) = slice else {
            return Self::Disabled;
        };

        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        {
            Instant::now()
                .checked_add(slice)
                .map(Self::Deadline)
                // An unrepresentable host deadline cannot safely permit an
                // unbounded exact solve, so fail closed as an immediate slice.
                .unwrap_or_else(|| Self::Deadline(Instant::now()))
        }

        #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
        {
            const MAX_SAFE_MILLIS: f64 = 9_007_199_254_740_991.0;
            let started_millis = performance_now_millis();
            let duration_millis = slice.as_secs_f64() * 1_000.0;
            if started_millis.is_finite()
                && duration_millis.is_finite()
                && duration_millis <= MAX_SAFE_MILLIS
            {
                Self::Deadline {
                    started_millis,
                    duration_millis,
                }
            } else {
                // As on native, fail closed rather than allowing an
                // unrepresentable deadline to turn into unlimited oracle work.
                Self::Deadline {
                    started_millis: 0.0,
                    duration_millis: 0.0,
                }
            }
        }
    }

    fn exhausted(&mut self) -> bool {
        match self {
            Self::Disabled => false,
            #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
            Self::Deadline(deadline) => Instant::now() >= *deadline,
            #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
            Self::Deadline {
                started_millis,
                duration_millis,
            } => {
                let elapsed_millis = performance_now_millis() - *started_millis;
                !elapsed_millis.is_finite() || elapsed_millis >= *duration_millis
            }
            #[cfg(test)]
            Self::TestPolls(allowed_polls) => {
                if *allowed_polls == 0 {
                    true
                } else {
                    *allowed_polls -= 1;
                    false
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuotaOracleResult {
    Exact(OracleHit),
    QuotaBackoff,
    ParentExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostBackoffMemoResult {
    Exact(OracleHit),
    Miss,
    ParentExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OracleHit {
    pub(crate) score: i32,
    pub(crate) best_action: Option<usize>,
    wdl: Wdl,
    distance: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::budget::NodeBudget;
    use crate::{
        apply_legal_action, create_initial_state, decode_action, Config, Coord, DrawReason,
        GameState, Player, Players, JUMP_RULE, REPETITION_THRESHOLD, RULESET,
    };

    fn config(ply_cap: u64) -> Config {
        Config {
            ruleset: RULESET.into(),
            rows: 9,
            columns: 9,
            start: Players {
                a: Coord { r: 8, c: 4 },
                b: Coord { r: 0, c: 4 },
            },
            goal_rows: Players { a: 0, b: 8 },
            initial_stock: Players { a: 0, b: 0 },
            jump_rule: JUMP_RULE.into(),
            repetition_threshold: REPETITION_THRESHOLD,
            ply_cap,
            first_player: Player::A,
        }
    }

    fn play(config: &Config, codes: &[usize]) -> GameState {
        let mut state = create_initial_state(config).unwrap();
        for &code in codes {
            state =
                apply_legal_action(config, &state, &decode_action(config, code).unwrap()).unwrap();
        }
        state
    }

    fn plain_terminal_minimax(
        config: &Config,
        state: &mut PreparedGameState,
    ) -> Result<OracleValue> {
        if let Some(value) = ExactZeroWallOracle::terminal(state) {
            return Ok(value);
        }
        let actions = compact_legal_codes(config, state);
        let mut best = None;
        for code in actions.iter() {
            let undo = state.apply_generated_code(config, code)?;
            let child = plain_terminal_minimax(config, state);
            assert!(state.undo_generated_code(undo));
            let child = child?;
            let candidate = OracleValue {
                wdl: child.wdl.negate(),
                distance: child.distance + 1,
                best_action: Some(code),
            };
            if best.map_or(true, |incumbent| {
                ExactZeroWallOracle::better(candidate, incumbent)
            }) {
                best = Some(candidate);
            }
            if candidate.wdl == Wdl::Win && candidate.distance == 1 {
                break;
            }
        }
        best.ok_or(crate::NormalDuelError::InvalidState)
    }

    fn solve(
        oracle: &mut ExactZeroWallOracle,
        config: &Config,
        state: &GameState,
        limit: u64,
    ) -> Option<OracleHit> {
        let mut prepared = PreparedGameState::from_game_state(config, state).unwrap();
        let mut budget = NodeBudget::new(limit);
        let mut nodes = 0;
        oracle
            .solve(config, &mut prepared, &mut budget, &mut nodes, state.ply)
            .unwrap()
    }

    #[test]
    fn exact_solver_matches_plain_terminal_minimax_on_reduced_games() {
        let cases = [
            (config(4), Vec::new()),
            (
                config(18),
                vec![67, 3, 58, 4, 49, 3, 40, 4, 31, 3, 22, 4, 13, 3],
            ),
        ];
        for (config, sequence) in cases {
            let state = play(&config, &sequence);
            let mut plain_state = PreparedGameState::from_game_state(&config, &state).unwrap();
            let expected = plain_terminal_minimax(&config, &mut plain_state).unwrap();
            let hit = solve(&mut ExactZeroWallOracle::new(), &config, &state, 100_000)
                .expect("reduced oracle completes");
            assert_eq!(hit.wdl, expected.wdl);
            assert_eq!(hit.distance, expected.distance);
            assert_eq!(hit.best_action, expected.best_action);
        }
    }

    #[test]
    fn exact_solver_honors_ply_cap_and_threefold_outcomes() {
        let cap_config = config(2);
        let cap_state = create_initial_state(&cap_config).unwrap();
        let cap_hit = solve(
            &mut ExactZeroWallOracle::new(),
            &cap_config,
            &cap_state,
            10_000,
        )
        .unwrap();
        assert_eq!(cap_hit.wdl, Wdl::Draw);
        assert_eq!(cap_hit.distance, 2);
        assert_eq!(cap_hit.score, 0);

        let repetition_config = config(20);
        let repeated = play(&repetition_config, &[75, 3, 76, 4, 75, 3, 76, 4]);
        assert_eq!(
            repeated.outcome,
            Outcome::Draw {
                reason: DrawReason::ThreefoldRepetition
            }
        );
        let repeated_hit = solve(
            &mut ExactZeroWallOracle::new(),
            &repetition_config,
            &repeated,
            10,
        )
        .unwrap();
        assert_eq!(repeated_hit.wdl, Wdl::Draw);
        assert_eq!(repeated_hit.distance, 0);
    }

    #[test]
    fn exact_memo_distinguishes_histories_and_recovers_mirrored_actions() {
        let history_config = config(5);
        let first = play(&history_config, &[75, 3, 76, 4]);
        let second = play(&history_config, &[67, 13, 76, 4]);
        let first_prepared = PreparedGameState::from_game_state(&history_config, &first).unwrap();
        let second_prepared = PreparedGameState::from_game_state(&history_config, &second).unwrap();
        assert_eq!(
            first_prepared.search_identity(&history_config),
            second_prepared.search_identity(&history_config)
        );
        assert_ne!(
            first_prepared.oracle_identity(&history_config),
            second_prepared.oracle_identity(&history_config)
        );
        let mut oracle = ExactZeroWallOracle::new();
        assert!(solve(&mut oracle, &history_config, &first, 1_000).is_some());
        assert!(oracle
            .memo_search_keys
            .contains(&second_prepared.search_identity(&history_config).key()));
        let entries_after_first: usize = oracle.memo.values().map(Vec::len).sum();
        let mut second_probe_nodes = 0;
        let mut second_probe_budget = NodeBudget::new(2);
        assert_eq!(
            oracle.probe_memo_after_backoff(
                &history_config,
                &second_prepared,
                &mut second_probe_budget,
                &mut second_probe_nodes,
                second.ply,
            ),
            PostBackoffMemoResult::Miss,
            "same board and ply with different repetition history must miss"
        );
        assert_eq!(second_probe_nodes, 1);
        assert!(solve(&mut oracle, &history_config, &second, 1_000).is_some());
        let entries_after_second: usize = oracle.memo.values().map(Vec::len).sum();
        assert!(entries_after_second > entries_after_first);

        let mirror_config = config(18);
        let sequence = [67, 3, 58, 4, 49, 3, 40, 4, 31, 3, 22, 4, 13, 3];
        let mirrored: Vec<_> = sequence
            .iter()
            .map(|&code| mirror_code(&mirror_config, code))
            .collect();
        let left = play(&mirror_config, &sequence);
        let right = play(&mirror_config, &mirrored);
        let left_hit = solve(&mut oracle, &mirror_config, &left, 100_000).unwrap();
        let right_hit = solve(&mut oracle, &mirror_config, &right, 100_000).unwrap();
        assert_eq!(left_hit.score, right_hit.score);
        assert_eq!(left_hit.wdl, right_hit.wdl);
        assert_eq!(
            right_hit.best_action,
            left_hit
                .best_action
                .map(|code| mirror_code(&mirror_config, code))
        );
    }

    #[test]
    fn post_backoff_memo_probe_is_charged_and_recovers_mirrored_exact_action() {
        let config = config(18);
        let sequence = [67, 3, 58, 4, 49, 3, 40, 4, 31, 3, 22, 4, 13, 3];
        let mirrored: Vec<_> = sequence
            .iter()
            .map(|&code| mirror_code(&config, code))
            .collect();
        let left = play(&config, &sequence);
        let right = play(&config, &mirrored);
        let mut oracle = ExactZeroWallOracle::new();
        let left_hit = solve(&mut oracle, &config, &left, 100_000).unwrap();
        let right_prepared = PreparedGameState::from_game_state(&config, &right).unwrap();

        let mut one_remaining_nodes = 10;
        let mut one_remaining_budget = NodeBudget::new(11);
        assert_eq!(
            oracle.probe_memo_after_backoff(
                &config,
                &right_prepared,
                &mut one_remaining_budget,
                &mut one_remaining_nodes,
                right.ply,
            ),
            PostBackoffMemoResult::ParentExhausted
        );
        assert_eq!(one_remaining_nodes, 11);

        let mut two_remaining_nodes = 10;
        let mut two_remaining_budget = NodeBudget::new(12);
        let exact = oracle.probe_memo_after_backoff(
            &config,
            &right_prepared,
            &mut two_remaining_budget,
            &mut two_remaining_nodes,
            right.ply,
        );
        assert_eq!(two_remaining_nodes, 11);
        assert_eq!(
            exact,
            PostBackoffMemoResult::Exact(OracleHit {
                best_action: left_hit.best_action.map(|code| mirror_code(&config, code)),
                ..left_hit
            })
        );
    }

    #[test]
    fn interruption_is_atomic_and_restores_the_exact_state() {
        let config = config(8);
        let state = create_initial_state(&config).unwrap();
        let mut prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let before = prepared.clone();
        let mut oracle = ExactZeroWallOracle::new();
        let mut budget = NodeBudget::new(3);
        let mut nodes = 0;
        assert!(oracle
            .solve(&config, &mut prepared, &mut budget, &mut nodes, state.ply,)
            .unwrap()
            .is_none());
        assert_eq!(prepared, before);
    }

    #[test]
    fn canonical_quota_and_parent_interruptions_restore_the_exact_state() {
        let config = config(128);
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();

        let mut quota_state = prepared.clone();
        let mut quota_oracle = ExactZeroWallOracle::new();
        let mut quota_parent = NodeBudget::new(100);
        let mut quota_nodes = 0;
        assert_eq!(
            quota_oracle
                .solve_with_canonical_limits(
                    &config,
                    &mut quota_state,
                    &mut quota_parent,
                    &mut quota_nodes,
                    state.ply,
                    CanonicalOracleLimits {
                        node_quota: 3,
                        wall_clock_slice: None,
                    },
                )
                .unwrap(),
            QuotaOracleResult::QuotaBackoff
        );
        assert_eq!(quota_nodes, 3);
        assert!(quota_oracle.quota_backoff_active());
        assert_eq!(quota_state, prepared);

        let mut parent_state = prepared.clone();
        let mut parent_oracle = ExactZeroWallOracle::new();
        let mut parent_budget = NodeBudget::new(2);
        let mut parent_nodes = 0;
        assert_eq!(
            parent_oracle
                .solve_with_canonical_limits(
                    &config,
                    &mut parent_state,
                    &mut parent_budget,
                    &mut parent_nodes,
                    state.ply,
                    CanonicalOracleLimits {
                        node_quota: 3,
                        wall_clock_slice: None,
                    },
                )
                .unwrap(),
            QuotaOracleResult::ParentExhausted
        );
        assert_eq!(parent_nodes, 2);
        assert!(!parent_oracle.quota_backoff_active());
        assert_eq!(parent_state, prepared);

        let mut simultaneous_state = prepared.clone();
        let mut simultaneous_oracle = ExactZeroWallOracle::new();
        let mut simultaneous_parent = NodeBudget::new(3);
        let mut simultaneous_nodes = 0;
        assert_eq!(
            simultaneous_oracle
                .solve_with_canonical_limits(
                    &config,
                    &mut simultaneous_state,
                    &mut simultaneous_parent,
                    &mut simultaneous_nodes,
                    state.ply,
                    CanonicalOracleLimits {
                        node_quota: 3,
                        wall_clock_slice: None,
                    },
                )
                .unwrap(),
            QuotaOracleResult::ParentExhausted,
            "the parent is polled before an equal node quota"
        );
        assert_eq!(simultaneous_nodes, 3);
        assert!(!simultaneous_oracle.quota_backoff_active());
        assert_eq!(simultaneous_state, prepared);
    }

    #[test]
    fn local_slice_backoff_is_sticky_restores_state_and_yields_to_parent() {
        let config = config(128);
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let zero_slice_limits = CanonicalOracleLimits {
            node_quota: 100,
            // This test exercises the local-clock path without sleeping: a
            // zero local duration is expired on its first poll.
            wall_clock_slice: Some(Duration::ZERO),
        };

        let mut local_state = prepared.clone();
        let mut local_oracle = ExactZeroWallOracle::new();
        let mut local_parent = NodeBudget::new(100);
        let mut local_nodes = 0;
        assert_eq!(
            local_oracle
                .solve_with_canonical_limits(
                    &config,
                    &mut local_state,
                    &mut local_parent,
                    &mut local_nodes,
                    state.ply,
                    zero_slice_limits,
                )
                .unwrap(),
            QuotaOracleResult::QuotaBackoff
        );
        assert_eq!(local_nodes, 0);
        assert!(local_oracle.quota_backoff_active());
        assert_eq!(local_state, prepared);

        let mut parent_state = prepared.clone();
        let mut parent_oracle = ExactZeroWallOracle::new();
        let mut exhausted_parent = NodeBudget::new(0);
        let mut parent_nodes = 0;
        assert_eq!(
            parent_oracle
                .solve_with_canonical_limits(
                    &config,
                    &mut parent_state,
                    &mut exhausted_parent,
                    &mut parent_nodes,
                    state.ply,
                    zero_slice_limits,
                )
                .unwrap(),
            QuotaOracleResult::ParentExhausted,
            "parent exhaustion wins when the local slice is also expired"
        );
        assert_eq!(parent_nodes, 0);
        assert!(!parent_oracle.quota_backoff_active());
        assert_eq!(parent_state, prepared);

        let mut restored_state = prepared.clone();
        let mut restored_oracle = ExactZeroWallOracle::new();
        let mut restored_parent = NodeBudget::new(100);
        let mut restored_nodes = 0;
        let mut test_slice =
            OracleQuotaBudget::with_test_local_expiry(&mut restored_parent, 100, 2);
        assert!(restored_oracle
            .solve(
                &config,
                &mut restored_state,
                &mut test_slice,
                &mut restored_nodes,
                state.ply,
            )
            .unwrap()
            .is_none());
        assert_eq!(restored_nodes, 1);
        assert!(test_slice.quota_exhausted);
        assert!(!test_slice.parent_exhausted);
        assert_eq!(restored_state, prepared);
    }
}
