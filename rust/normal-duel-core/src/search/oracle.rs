use std::collections::HashMap;

use crate::{
    compact_legal_codes, Config, Outcome, PreparedGameState, PreparedOracleIdentity, Result,
};

use super::budget::SearchBudget;
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
}

impl ExactZeroWallOracle {
    pub(crate) fn new() -> Self {
        Self::default()
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

    fn store(&mut self, config: &Config, identity: PreparedOracleIdentity, mut value: OracleValue) {
        if identity.mirrored() {
            value.best_action = value.best_action.map(|code| mirror_code(config, code));
        }
        let bucket = self.memo.entry(identity.key()).or_default();
        if let Some(existing) = bucket.iter_mut().find(|entry| entry.identity == identity) {
            existing.value = value;
        } else {
            bucket.push(MemoEntry { identity, value });
        }
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
        self.store(config, identity, best);
        Ok(Some(best))
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
        Ok(Some(OracleHit {
            score,
            best_action: value.best_action,
            wdl: value.wdl,
            distance: value.distance,
        }))
    }
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
        let entries_after_first: usize = oracle.memo.values().map(Vec::len).sum();
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
}
