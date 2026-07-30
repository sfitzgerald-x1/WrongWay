use std::time::Duration;

use crate::{
    compact_legal_codes, compact_pawn_codes, validate_state, CodeList, Config, GameState,
    NormalDuelError, Outcome, PreparedGameState, Result,
};

use super::budget::{DeadlineBudget, NodeBudget, SearchBudget};
use super::eval::{evaluate, MATE_SCORE};
use super::move_picker::{uses_profiled_ordering, Heuristics, OrderingHints};
use super::oracle::ExactZeroWallOracle;
use super::tt::{mirror_code, Bound, TranspositionTable};

const NEG_INFINITY: i32 = -MATE_SCORE - 1;
const POS_INFINITY: i32 = MATE_SCORE + 1;
const MAX_SEARCH_DEPTH: u8 = 128;
const MAX_TT_CAPACITY: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchOptions {
    pub max_depth: u8,
    pub transposition_capacity: usize,
    pub aspiration_window: i32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 64,
            transposition_capacity: 1 << 18,
            aspiration_window: 256,
        }
    }
}

impl SearchOptions {
    fn validate(self) -> Result<Self> {
        if self.max_depth == 0
            || self.max_depth > MAX_SEARCH_DEPTH
            || self.transposition_capacity < 2
            || self.transposition_capacity > MAX_TT_CAPACITY
            || self.aspiration_window <= 0
            || self.aspiration_window >= MATE_SCORE
        {
            return Err(NormalDuelError::InvalidSearchOptions);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchDiagnostics {
    /// Category counters for tuning, not a closed accounting of [`SearchReport::nodes`].
    /// Terminal nodes, TT returns, and the oracle's internal traversal are not
    /// fully classified here, so these values must not be summed to infer a
    /// total node count.
    pub root_action_count: usize,
    /// Number of horizon nodes evaluated by the static evaluator. Immediate
    /// pawn wins are exact tactical leaves and intentionally do not count.
    pub static_leaf_count: u64,
    /// Number of zero-depth nodes where the side to move had a legal pawn
    /// move directly onto its goal row.
    pub immediate_goal_horizon_hits: u64,
    /// Number of positive-depth zero-wall oracle calls attempted.
    pub zero_wall_oracle_queries: u64,
    /// Number of oracle calls that completed and supplied an exact result.
    pub zero_wall_oracle_solutions: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_bound_cutoffs: u64,
    pub beta_cutoffs: u64,
    pub pvs_researches: u64,
    pub aspiration_researches: u64,
    pub repetition_hint_only_probes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchReport {
    pub action_code: Option<usize>,
    /// Root-perspective score from the last fully committed iteration. A
    /// deterministic legal fallback has no searched score.
    pub score: Option<i32>,
    pub completed_depth: u8,
    pub nodes: u64,
    pub stopped: bool,
    pub principal_variation: Vec<usize>,
    /// Cumulative node totals after each fully committed iteration. An
    /// interrupted iteration is deliberately absent, so this can differ from
    /// [`Self::nodes`] when the final attempt exhausts the budget.
    pub committed_iteration_nodes: Vec<u64>,
    /// Root-perspective scores corresponding one-for-one to
    /// [`Self::committed_iteration_nodes`].
    pub committed_iteration_scores: Vec<i32>,
    pub diagnostics: SearchDiagnostics,
}

#[derive(Debug, Clone, Copy)]
struct NodeValue {
    score: i32,
    best_action: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct SearchFrame {
    previous_action: Option<usize>,
    use_tt_bounds: bool,
}

struct SearchContext<'a, B: SearchBudget> {
    config: &'a Config,
    budget: B,
    tt: TranspositionTable,
    heuristics: Heuristics,
    oracle: ExactZeroWallOracle,
    root_ply: u64,
    nodes: u64,
    diagnostics: SearchDiagnostics,
}

impl<B: SearchBudget> SearchContext<'_, B> {
    fn stopped(&mut self) -> bool {
        self.budget.exhausted(self.nodes)
    }

    fn enter_node(&mut self) -> bool {
        if self.stopped() {
            return false;
        }
        self.nodes += 1;
        true
    }

    fn terminal_score(&self, state: &PreparedGameState) -> Option<i32> {
        match state.outcome {
            Outcome::Ongoing => None,
            Outcome::Draw { .. } => Some(0),
            Outcome::Win { winner, .. } => {
                let distance = state.ply.saturating_sub(self.root_ply);
                let distance = i32::try_from(distance).unwrap_or(i32::MAX / 2);
                let score = MATE_SCORE.saturating_sub(distance);
                Some(if winner == state.position.turn {
                    score
                } else {
                    -score
                })
            }
        }
    }

    /// Return the exact score of a winning pawn move that is just beyond the
    /// caller's nominal horizon. This is intentionally derived from the
    /// compact *legal* pawn generator rather than a distance heuristic: jumps
    /// and walls can make a goal-row square reachable or blocked in ways that
    /// a coordinate-only check would get wrong.
    ///
    /// The score is from the current side-to-move perspective. Applying the
    /// move would increment `state.ply` once and flip the turn, so encode that
    /// same root-relative distance here without mutating the state.
    fn immediate_goal_horizon_score(&mut self, state: &PreparedGameState) -> Option<i32> {
        let mover = state.position.turn;
        let goal_row = *self.config.goal_rows.get(mover);
        // A pawn can change row by at most two in one legal move (a forward
        // jump). This is only an exact impossibility guard; within two rows we
        // still ask the compact legal generator so walls and permissive exits
        // decide the result.
        if state.position.pawns.get(mover).r.abs_diff(goal_row) > 2 {
            return None;
        }
        let mut pawn_actions = CodeList::new();
        compact_pawn_codes(
            self.config,
            &state.position,
            state.position.board(),
            &mut pawn_actions,
        );
        let goal_row = usize::from(goal_row);
        let columns = usize::from(self.config.columns);
        if !pawn_actions.iter().any(|code| code / columns == goal_row) {
            return None;
        }

        self.diagnostics.immediate_goal_horizon_hits += 1;
        let distance = state.ply.saturating_sub(self.root_ply).saturating_add(1);
        let distance = i32::try_from(distance).unwrap_or(i32::MAX / 2);
        Some(MATE_SCORE.saturating_sub(distance))
    }

    fn negamax(
        &mut self,
        state: &mut PreparedGameState,
        depth: u8,
        ply: usize,
        alpha: i32,
        beta: i32,
        previous_action: Option<usize>,
    ) -> Result<Option<NodeValue>> {
        self.negamax_with_tt_bounds(
            state,
            depth,
            ply,
            alpha,
            beta,
            SearchFrame {
                previous_action,
                use_tt_bounds: true,
            },
        )
    }

    fn negamax_with_tt_bounds(
        &mut self,
        state: &mut PreparedGameState,
        depth: u8,
        ply: usize,
        mut alpha: i32,
        mut beta: i32,
        frame: SearchFrame,
    ) -> Result<Option<NodeValue>> {
        let previous_action = frame.previous_action;
        if !self.enter_node() {
            return Ok(None);
        }
        if let Some(score) = self.terminal_score(state) {
            return Ok(Some(NodeValue {
                score,
                best_action: None,
            }));
        }

        // A zero-depth node is the caller's explicit horizon. Exact solving is
        // reserved for nodes the iteration is actually extending; otherwise a
        // zero-stock leaf could consume the entire remaining budget and
        // prevent an otherwise complete shallow iteration from committing.
        if depth == 0 {
            if let Some(score) = self.immediate_goal_horizon_score(state) {
                return Ok(Some(NodeValue {
                    score,
                    best_action: None,
                }));
            }
            self.diagnostics.static_leaf_count += 1;
            return Ok(Some(NodeValue {
                score: evaluate(self.config, state),
                best_action: None,
            }));
        }

        if state.position.stock.a == 0 && state.position.stock.b == 0 {
            self.diagnostics.zero_wall_oracle_queries += 1;
            if let Some(hit) = self.oracle.solve(
                self.config,
                state,
                &mut self.budget,
                &mut self.nodes,
                self.root_ply,
            )? {
                self.diagnostics.zero_wall_oracle_solutions += 1;
                return Ok(Some(NodeValue {
                    score: hit.score,
                    best_action: hit.best_action,
                }));
            }
            return Ok(None);
        }

        let identity = state.search_identity(self.config);
        self.diagnostics.tt_probes += 1;
        let entry = self.tt.probe(identity);
        let tt_action = entry.and_then(|hit| hit.best_action(self.config, identity));
        if let Some(hit) = entry {
            self.diagnostics.tt_hits += 1;
            if !identity.bounds_reusable() {
                self.diagnostics.repetition_hint_only_probes += 1;
            } else if frame.use_tt_bounds && hit.depth >= depth {
                if let Some(bound) = hit.bound {
                    match bound {
                        Bound::Exact => {
                            self.diagnostics.tt_bound_cutoffs += 1;
                            return Ok(Some(NodeValue {
                                score: hit.score,
                                best_action: tt_action,
                            }));
                        }
                        Bound::Lower => alpha = alpha.max(hit.score),
                        Bound::Upper => beta = beta.min(hit.score),
                    }
                    if alpha >= beta {
                        self.diagnostics.tt_bound_cutoffs += 1;
                        return Ok(Some(NodeValue {
                            score: hit.score,
                            best_action: tt_action,
                        }));
                    }
                }
            }
        }

        let alpha_start = alpha;
        let beta_start = beta;
        let actions = compact_legal_codes(self.config, state);
        if actions.len == 0 {
            return Err(NormalDuelError::InvalidState);
        }
        let ordering_nodes = self.nodes;
        let budget = &mut self.budget;
        let Some(ordered) = self.heuristics.order(
            self.config,
            state,
            &actions,
            OrderingHints {
                tt_action,
                ply,
                previous_action,
                canonical_mirrored: identity.mirrored(),
            },
            || budget.exhausted(ordering_nodes),
        ) else {
            return Ok(None);
        };
        let mover = state.position.turn;
        let mut best = NEG_INFINITY;
        let mut best_action = None;
        let mut best_tie_code = None;
        for (move_number, action) in ordered.into_iter().enumerate() {
            if self.stopped() {
                return Ok(None);
            }
            debug_assert_eq!(
                actions.codes[action.index] as usize, action.code,
                "ordered action retains its legal batch index"
            );
            let undo = state.apply_generated_code(self.config, action.code)?;
            let child = if move_number == 0 {
                self.negamax(state, depth - 1, ply + 1, -beta, -alpha, Some(action.code))
            } else {
                let scout = self.negamax(
                    state,
                    depth - 1,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    Some(action.code),
                );
                match scout {
                    Ok(Some(value)) => {
                        let scout_score = -value.score;
                        // A null-window result equal to alpha can still be a
                        // fail-low bound. Re-search it too: otherwise its
                        // deterministic smaller action code can be selected
                        // as though it tied the current best move.
                        if scout_score >= alpha && scout_score < beta {
                            self.diagnostics.pvs_researches += 1;
                            // Tie-breaking needs an exact value: a bounded
                            // re-search can still return alpha as a fail-bound
                            // and make a worse smaller-code action look tied.
                            // Search the entire value range and keep the
                            // scout's action only as an ordering hint.
                            self.negamax_with_tt_bounds(
                                state,
                                depth - 1,
                                ply + 1,
                                NEG_INFINITY,
                                POS_INFINITY,
                                SearchFrame {
                                    previous_action: Some(action.code),
                                    use_tt_bounds: false,
                                },
                            )
                        } else {
                            Ok(Some(value))
                        }
                    }
                    Ok(None) => Ok(None),
                    Err(error) => Err(error),
                }
            };
            let restored = state.undo_generated_code(undo);
            if !restored {
                return Err(NormalDuelError::InvalidState);
            }
            let Some(child) = child? else {
                return Ok(None);
            };
            let score = -child.score;
            let tie_code = if uses_profiled_ordering(self.config) && identity.mirrored() {
                mirror_code(self.config, action.code)
            } else {
                action.code
            };
            if score > best || (score == best && Some(tie_code) < best_tie_code) {
                best = score;
                best_action = Some(action.code);
                best_tie_code = Some(tie_code);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.diagnostics.beta_cutoffs += 1;
                self.heuristics.record_cutoff(
                    mover,
                    action.code,
                    depth,
                    ply,
                    previous_action,
                    action.code >= self.config.cells(),
                );
                break;
            }
        }

        let bound = identity
            .bounds_reusable()
            .then_some(if best <= alpha_start {
                Bound::Upper
            } else if best >= beta_start {
                Bound::Lower
            } else {
                Bound::Exact
            });
        self.tt
            .store(self.config, identity, depth, best, bound, best_action);
        Ok(Some(NodeValue {
            score: best,
            best_action,
        }))
    }

    fn principal_variation(
        &mut self,
        root: &PreparedGameState,
        depth: u8,
    ) -> Result<Option<Vec<usize>>> {
        let mut state = root.clone();
        let mut variation = Vec::with_capacity(usize::from(depth));
        for remaining in (1..=depth).rev() {
            if self.stopped() {
                return Ok(None);
            }
            if !state.outcome.is_ongoing() {
                break;
            }
            let identity = state.search_identity(self.config);
            // Search identities deliberately omit repetition history. Their
            // actions are safe ordering hints, but only a reusable identity
            // is authoritative enough to extend a reported PV.
            if !identity.bounds_reusable() {
                break;
            }
            let Some(entry) = self.tt.probe(identity) else {
                break;
            };
            if entry.bound != Some(Bound::Exact) || entry.depth < remaining {
                break;
            }
            let Some(code) = entry.best_action(self.config, identity) else {
                break;
            };
            let actions = compact_legal_codes(self.config, &state);
            if !actions.iter().any(|legal| legal == code) {
                break;
            }
            let _undo = state.apply_generated_code(self.config, code)?;
            variation.push(code);
            // The cloned line is intentionally not undone.
        }
        if self.stopped() {
            Ok(None)
        } else {
            Ok(Some(variation))
        }
    }
}

fn run<B: SearchBudget>(
    config: &Config,
    state: &GameState,
    mut budget: B,
    options: SearchOptions,
) -> Result<SearchReport> {
    let options = options.validate()?;
    let validated = validate_state(config, state)?;
    // The deadline is created by search_for before entering this shared path.
    // We cannot return before deriving a validated legal fallback, but poll
    // each bounded setup phase and skip expensive search infrastructure once
    // expiration is observed. Node mode sees the same polls without adding
    // nodes, preserving its deterministic result.
    let expired_after_validation = budget.exhausted(0);
    let root = PreparedGameState::from_game_state(config, &validated)?;
    let expired_after_preparation = budget.exhausted(0);
    let root_actions = compact_legal_codes(config, &root);
    let fallback = root_actions.iter().next();
    if !root.outcome.is_ongoing() {
        let stopped = expired_after_validation || expired_after_preparation || budget.exhausted(0);
        let score = match root.outcome {
            Outcome::Ongoing => unreachable!("terminal root already checked"),
            Outcome::Draw { .. } => 0,
            Outcome::Win { winner, .. } => {
                if winner == root.position.turn {
                    MATE_SCORE
                } else {
                    -MATE_SCORE
                }
            }
        };
        return Ok(SearchReport {
            action_code: None,
            score: Some(score),
            completed_depth: 0,
            nodes: 0,
            stopped,
            principal_variation: Vec::new(),
            committed_iteration_nodes: Vec::new(),
            committed_iteration_scores: Vec::new(),
            diagnostics: SearchDiagnostics::default(),
        });
    }
    let expired_after_actions = budget.exhausted(0);
    if expired_after_validation || expired_after_preparation || expired_after_actions {
        return Ok(SearchReport {
            action_code: fallback,
            score: None,
            completed_depth: 0,
            nodes: 0,
            stopped: true,
            principal_variation: fallback.into_iter().collect(),
            committed_iteration_nodes: Vec::new(),
            committed_iteration_scores: Vec::new(),
            diagnostics: SearchDiagnostics {
                root_action_count: root_actions.len,
                ..SearchDiagnostics::default()
            },
        });
    }
    let Some(tt) =
        TranspositionTable::new_with_budget(options.transposition_capacity, &mut budget, 0)
    else {
        return Ok(SearchReport {
            action_code: fallback,
            score: None,
            completed_depth: 0,
            nodes: 0,
            stopped: true,
            principal_variation: fallback.into_iter().collect(),
            committed_iteration_nodes: Vec::new(),
            committed_iteration_scores: Vec::new(),
            diagnostics: SearchDiagnostics {
                root_action_count: root_actions.len,
                ..SearchDiagnostics::default()
            },
        });
    };
    if budget.exhausted(0) {
        return Ok(SearchReport {
            action_code: fallback,
            score: None,
            completed_depth: 0,
            nodes: 0,
            stopped: true,
            principal_variation: fallback.into_iter().collect(),
            committed_iteration_nodes: Vec::new(),
            committed_iteration_scores: Vec::new(),
            diagnostics: SearchDiagnostics {
                root_action_count: root_actions.len,
                ..SearchDiagnostics::default()
            },
        });
    }
    let heuristics = Heuristics::new(usize::from(options.max_depth), config.policy_size());
    if budget.exhausted(0) {
        return Ok(SearchReport {
            action_code: fallback,
            score: None,
            completed_depth: 0,
            nodes: 0,
            stopped: true,
            principal_variation: fallback.into_iter().collect(),
            committed_iteration_nodes: Vec::new(),
            committed_iteration_scores: Vec::new(),
            diagnostics: SearchDiagnostics {
                root_action_count: root_actions.len,
                ..SearchDiagnostics::default()
            },
        });
    }
    let mut context = SearchContext {
        config,
        budget,
        tt,
        heuristics,
        oracle: ExactZeroWallOracle::new(),
        root_ply: root.ply,
        nodes: 0,
        diagnostics: SearchDiagnostics {
            root_action_count: root_actions.len,
            ..SearchDiagnostics::default()
        },
    };
    let mut selected = fallback;
    let mut selected_score = None;
    let mut completed_depth = 0;
    let mut stopped = false;
    let mut previous_score = evaluate(config, &root);
    let mut selected_variation: Vec<_> = fallback.into_iter().collect();
    let mut committed_iteration_nodes = Vec::with_capacity(usize::from(options.max_depth));
    let mut committed_iteration_scores = Vec::with_capacity(usize::from(options.max_depth));

    for depth in 1..=options.max_depth {
        if context.stopped() {
            stopped = true;
            break;
        }
        context.tt.next_generation();
        let mut position = root.clone();
        let attempt = if depth == 1 {
            context.negamax(&mut position, depth, 0, NEG_INFINITY, POS_INFINITY, None)?
        } else {
            let alpha = previous_score.saturating_sub(options.aspiration_window);
            let beta = previous_score.saturating_add(options.aspiration_window);
            match context.negamax(&mut position, depth, 0, alpha, beta, None)? {
                Some(value) if value.score <= alpha || value.score >= beta => {
                    if context.stopped() {
                        None
                    } else {
                        context.diagnostics.aspiration_researches += 1;
                        let mut position = root.clone();
                        context.negamax(
                            &mut position,
                            depth,
                            0,
                            NEG_INFINITY,
                            POS_INFINITY,
                            None,
                        )?
                    }
                }
                result => result,
            }
        };
        let Some(value) = attempt else {
            stopped = true;
            break;
        };
        let Some(action) = value.best_action else {
            return Err(NormalDuelError::InvalidState);
        };
        if context.stopped() {
            stopped = true;
            break;
        }
        let Some(mut candidate_variation) = context.principal_variation(&root, depth)? else {
            stopped = true;
            break;
        };
        if candidate_variation.first().copied() != Some(action) {
            candidate_variation.clear();
            candidate_variation.push(action);
        }
        if context.stopped() {
            stopped = true;
            break;
        }
        // The iteration becomes visible only after search, PV construction,
        // and a final budget check all succeed.
        selected = Some(action);
        selected_score = Some(value.score);
        previous_score = value.score;
        completed_depth = depth;
        selected_variation = candidate_variation;
        committed_iteration_nodes.push(context.nodes);
        committed_iteration_scores.push(value.score);
    }

    Ok(SearchReport {
        action_code: selected,
        score: selected_score,
        completed_depth,
        nodes: context.nodes,
        stopped,
        principal_variation: selected_variation,
        committed_iteration_nodes,
        committed_iteration_scores,
        diagnostics: context.diagnostics,
    })
}

pub(super) fn search_nodes(
    config: &Config,
    state: &GameState,
    node_budget: u64,
    options: SearchOptions,
) -> Result<SearchReport> {
    if node_budget == 0 {
        return Err(NormalDuelError::InvalidSearchBudget);
    }
    run(config, state, NodeBudget::new(node_budget), options)
}

pub(super) fn search_for(
    config: &Config,
    state: &GameState,
    duration: Duration,
    options: SearchOptions,
) -> Result<SearchReport> {
    if duration.is_zero() {
        return Err(NormalDuelError::InvalidSearchBudget);
    }
    let budget = DeadlineBudget::new(duration).ok_or(NormalDuelError::InvalidSearchBudget)?;
    run(config, state, budget, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::budget::{CheckBudget, ExactCutoffBudget};
    use crate::{
        apply_legal_action, create_initial_state, decode_action, Coord, Player, Players, JUMP_RULE,
        REPETITION_THRESHOLD, RULESET,
    };

    fn zero_wall_config() -> Config {
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
            ply_cap: 128,
            first_player: Player::A,
        }
    }

    fn shallow_search_config() -> Config {
        Config {
            rows: 7,
            columns: 7,
            start: Players {
                a: Coord { r: 6, c: 3 },
                b: Coord { r: 0, c: 3 },
            },
            goal_rows: Players { a: 0, b: 6 },
            initial_stock: Players { a: 2, b: 2 },
            ply_cap: 64,
            ..zero_wall_config()
        }
    }

    fn horizon_block_config() -> Config {
        Config {
            rows: 7,
            columns: 7,
            start: Players {
                a: Coord { r: 6, c: 3 },
                b: Coord { r: 0, c: 3 },
            },
            goal_rows: Players { a: 0, b: 6 },
            // A has exactly one wall to stop B's next goal move. B has no
            // walls, which keeps the one-extra-ply reference exhaustive and
            // compact without weakening the tactical condition under test.
            initial_stock: Players { a: 1, b: 0 },
            ply_cap: 64,
            ..zero_wall_config()
        }
    }

    fn horizon_block_state(config: &Config) -> GameState {
        let mut state = create_initial_state(config).unwrap();
        // A shuffles on its home row while B advances to row five. It is now
        // A to move; B has a direct pawn win on the following ply unless A
        // places one of the two legal horizontal blocking walls.
        for code in [44, 10, 45, 17, 44, 24, 45, 31, 44, 38] {
            state =
                apply_legal_action(config, &state, &decode_action(config, code).unwrap()).unwrap();
        }
        state
    }

    fn horizon_jump_state(config: &Config) -> GameState {
        let mut state = create_initial_state(config).unwrap();
        // On B's turn, B is two rows from its goal and A occupies the square
        // directly ahead. The permissive jump generator therefore exposes the
        // exact two-row pawn win onto B's goal row.
        for code in [44, 10, 45, 17, 44, 24, 45, 31, 38] {
            state =
                apply_legal_action(config, &state, &decode_action(config, code).unwrap()).unwrap();
        }
        state
    }

    fn plain_minimax(
        config: &Config,
        state: &mut PreparedGameState,
        depth: u8,
        root_ply: u64,
    ) -> Result<NodeValue> {
        match state.outcome {
            Outcome::Draw { .. } => {
                return Ok(NodeValue {
                    score: 0,
                    best_action: None,
                });
            }
            Outcome::Win { winner, .. } => {
                let distance = (state.ply - root_ply) as i32;
                let score = MATE_SCORE - distance;
                return Ok(NodeValue {
                    score: if winner == state.position.turn {
                        score
                    } else {
                        -score
                    },
                    best_action: None,
                });
            }
            Outcome::Ongoing => {}
        }
        if depth == 0 {
            return Ok(NodeValue {
                score: evaluate(config, state),
                best_action: None,
            });
        }
        let actions = compact_legal_codes(config, state);
        let mut best = NEG_INFINITY;
        let mut best_action = None;
        for code in actions.iter() {
            let undo = state.apply_generated_code(config, code)?;
            let child = plain_minimax(config, state, depth - 1, root_ply)?;
            assert!(state.undo_generated_code(undo));
            let score = -child.score;
            if score > best || (score == best && Some(code) < best_action) {
                best = score;
                best_action = Some(code);
            }
        }
        Ok(NodeValue {
            score: best,
            best_action,
        })
    }

    #[test]
    fn fixed_node_runs_are_reproducible_and_return_legal_moves() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let options = SearchOptions {
            max_depth: 5,
            transposition_capacity: 1 << 10,
            aspiration_window: 32,
        };
        let first = search_nodes(&config, &state, 5_000, options).unwrap();
        let second = search_nodes(&config, &state, 5_000, options).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.nodes, 5_000);
        assert_eq!(first.completed_depth, 0);
        assert_eq!(first.score, None);
        assert!(first.stopped);
        assert!(state.clone().position.turn.eq(&Player::A));
        assert!(crate::legal_action_codes(&config, &state)
            .unwrap()
            .contains(&first.action_code.unwrap()));
    }

    #[test]
    fn default_search_options_use_the_profiled_aspiration_window() {
        assert_eq!(SearchOptions::default().aspiration_window, 256);
    }

    #[test]
    fn pvs_matches_plain_minimax_on_reachable_positions() {
        let config = shallow_search_config();
        let mut state = create_initial_state(&config).unwrap();
        let mut positions = vec![state.clone()];
        for action_code in [38_usize, 10] {
            let action = decode_action(&config, action_code).unwrap();
            state = apply_legal_action(&config, &state, &action).unwrap();
            positions.push(state.clone());
        }
        for state in positions {
            let mut prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
            let expected = plain_minimax(&config, &mut prepared, 2, state.ply).unwrap();
            let report = search_nodes(
                &config,
                &state,
                500_000,
                SearchOptions {
                    max_depth: 2,
                    transposition_capacity: 1 << 10,
                    aspiration_window: 32,
                },
            )
            .unwrap();
            assert_eq!(report.completed_depth, 2);
            assert_eq!(report.score, Some(expected.score));
            assert_eq!(report.action_code, expected.best_action);
            let legal = crate::legal_action_codes(&config, &state).unwrap();
            assert!(legal.contains(&report.action_code.unwrap()));
        }
    }

    #[test]
    fn cold_depth_one_equal_wins_resolve_to_mirrored_canonical_actions() {
        let config = Config {
            initial_stock: Players { a: 1, b: 1 },
            ..zero_wall_config()
        };
        let left_codes = [67, 81, 58, 3, 49, 4, 40, 3, 31, 4, 22, 3, 13, 4];
        let right_codes = left_codes.map(|code| mirror_code(&config, code));
        let build = |codes: &[usize]| {
            let mut state = create_initial_state(&config).unwrap();
            for &code in codes {
                state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                    .unwrap();
            }
            state
        };
        let left = build(&left_codes);
        let right = build(&right_codes);
        let left_prepared = PreparedGameState::from_game_state(&config, &left).unwrap();
        let right_prepared = PreparedGameState::from_game_state(&config, &right).unwrap();
        let left_identity = left_prepared.search_identity(&config);
        let right_identity = right_prepared.search_identity(&config);
        assert_eq!(left_identity, right_identity);
        assert_ne!(left_identity.mirrored(), right_identity.mirrored());

        let winning_codes = |state: &PreparedGameState| {
            compact_legal_codes(&config, state)
                .iter()
                .filter(|&code| {
                    let mut child = state.clone();
                    child.apply_generated_code(&config, code).unwrap();
                    matches!(
                        child.outcome,
                        Outcome::Win {
                            winner: Player::A,
                            ..
                        }
                    )
                })
                .collect::<Vec<_>>()
        };
        let left_wins = winning_codes(&left_prepared);
        let right_wins = winning_codes(&right_prepared);
        assert_eq!(left_wins.len(), 2);
        assert_eq!(right_wins.len(), 2);

        let options = SearchOptions {
            max_depth: 1,
            transposition_capacity: 64,
            aspiration_window: 32,
        };
        let left_report = search_nodes(&config, &left, 100_000, options).unwrap();
        let right_report = search_nodes(&config, &right, 100_000, options).unwrap();
        assert_eq!(left_report.completed_depth, 1);
        assert_eq!(right_report.completed_depth, 1);
        assert_eq!(left_report.score, Some(MATE_SCORE - 1));
        assert_eq!(right_report.score, Some(MATE_SCORE - 1));
        assert!(left_wins.contains(&left_report.action_code.unwrap()));
        assert!(right_wins.contains(&right_report.action_code.unwrap()));
        assert_eq!(
            mirror_code(&config, left_report.action_code.unwrap()),
            right_report.action_code.unwrap()
        );
    }

    #[test]
    fn horizon_goal_detection_matches_one_extra_ply_and_blocks_forced_win() {
        let config = horizon_block_config();
        let state = horizon_block_state(&config);
        assert_eq!(state.position.turn, Player::A);
        let root = PreparedGameState::from_game_state(&config, &state).unwrap();
        let root_actions = compact_legal_codes(&config, &root);

        // Classify the moves from exact generated legal pawn destinations;
        // this does not rely on the static evaluator or assumed wall labels.
        let mut must_block = Vec::new();
        let mut loses_immediately = Vec::new();
        for code in root_actions.iter() {
            let mut child = root.clone();
            child
                .apply_generated_code(&config, code)
                .expect("root action is generated legal");
            let opponent_goal_is_legal =
                compact_legal_codes(&config, &child)
                    .iter()
                    .any(|child_code| {
                        child_code < config.cells()
                            && child_code / usize::from(config.columns)
                                == usize::from(*config.goal_rows.get(child.position.turn))
                    });
            if opponent_goal_is_legal {
                loses_immediately.push(code);
            } else {
                must_block.push(code);
            }
        }
        assert!(!must_block.is_empty());
        assert!(!loses_immediately.is_empty());

        // At the tactical leaf, the new depth-zero rule must equal a plain
        // minimax expansion by one ply, including the root-relative mate
        // distance. This is the immediate-win half of the horizon contract.
        let non_blocking = loses_immediately[0];
        let mut immediate_child = root.clone();
        immediate_child
            .apply_generated_code(&config, non_blocking)
            .expect("classified non-blocking action is legal");
        let mut reference_child = immediate_child.clone();
        let expected_leaf = plain_minimax(&config, &mut reference_child, 1, root.ply).unwrap();
        let mut context = SearchContext {
            config: &config,
            budget: NodeBudget::new(10_000),
            tt: TranspositionTable::new(64),
            heuristics: Heuristics::new(2, config.policy_size()),
            oracle: ExactZeroWallOracle::new(),
            root_ply: root.ply,
            nodes: 0,
            diagnostics: SearchDiagnostics::default(),
        };
        let immediate = context
            .negamax(
                &mut immediate_child,
                0,
                1,
                NEG_INFINITY,
                POS_INFINITY,
                Some(non_blocking),
            )
            .unwrap()
            .unwrap();
        assert_eq!(immediate.score, expected_leaf.score);
        assert_eq!(immediate.score, MATE_SCORE - 2);
        assert_eq!(context.diagnostics.immediate_goal_horizon_hits, 1);
        assert_eq!(context.diagnostics.static_leaf_count, 0);

        // The shallow full-width root must still choose one of the moves that
        // exactly removes B's generated goal move. This deliberately checks a
        // tactical contract rather than tying a shallow static evaluation to
        // the deeper reference's incidental exact action code.
        let report = search_nodes(
            &config,
            &state,
            500_000,
            SearchOptions {
                max_depth: 1,
                transposition_capacity: 256,
                aspiration_window: 256,
            },
        )
        .unwrap();
        assert_eq!(report.completed_depth, 1);
        assert!(must_block.contains(&report.action_code.unwrap()));
        assert!(report.diagnostics.immediate_goal_horizon_hits > 0);
        assert!(report.diagnostics.static_leaf_count > 0);
        assert_eq!(report.committed_iteration_nodes, vec![report.nodes]);
        assert_eq!(
            report.committed_iteration_scores,
            vec![report.score.unwrap()]
        );
    }

    #[test]
    fn horizon_goal_guard_preserves_legal_two_row_jump_wins() {
        let config = horizon_block_config();
        let state = horizon_jump_state(&config);
        assert_eq!(state.position.turn, Player::B);
        let root = PreparedGameState::from_game_state(&config, &state).unwrap();
        let mover = root.position.turn;
        assert_eq!(
            root.position
                .pawns
                .get(mover)
                .r
                .abs_diff(*config.goal_rows.get(mover)),
            2,
            "the cheap guard must leave a two-row jump eligible for exact generation"
        );
        let goal_code = usize::from(*config.goal_rows.get(mover)) * usize::from(config.columns)
            + usize::from(root.position.pawns.get(mover).c);
        assert!(
            compact_legal_codes(&config, &root)
                .iter()
                .any(|code| code == goal_code),
            "the permissive legal pawn generator exposes the forward goal jump"
        );

        let mut reference = root.clone();
        let expected = plain_minimax(&config, &mut reference, 1, root.ply).unwrap();
        let mut searched = root.clone();
        let mut context = SearchContext {
            config: &config,
            budget: NodeBudget::new(10_000),
            tt: TranspositionTable::new(64),
            heuristics: Heuristics::new(2, config.policy_size()),
            oracle: ExactZeroWallOracle::new(),
            root_ply: root.ply,
            nodes: 0,
            diagnostics: SearchDiagnostics::default(),
        };
        let value = context
            .negamax(&mut searched, 0, 0, NEG_INFINITY, POS_INFINITY, None)
            .unwrap()
            .unwrap();
        assert_eq!(value.score, expected.score);
        assert_eq!(value.score, MATE_SCORE - 1);
        assert_eq!(context.diagnostics.immediate_goal_horizon_hits, 1);
        assert_eq!(context.diagnostics.static_leaf_count, 0);
    }

    #[test]
    fn zero_stock_horizon_leaf_does_not_starve_depth_two_iteration() {
        let config = Config {
            rows: 7,
            columns: 7,
            start: Players {
                a: Coord { r: 6, c: 3 },
                b: Coord { r: 0, c: 3 },
            },
            goal_rows: Players { a: 0, b: 6 },
            initial_stock: Players { a: 4, b: 4 },
            ply_cap: 512,
            ..zero_wall_config()
        };
        let mut state = create_initial_state(&config).unwrap();
        for code in [95, 115, 78, 83, 102, 71] {
            state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                .unwrap();
        }
        let mut prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let expected = plain_minimax(&config, &mut prepared, 2, state.ply).unwrap();
        let report = search_nodes(
            &config,
            &state,
            2_000_000,
            SearchOptions {
                max_depth: 2,
                transposition_capacity: 2,
                aspiration_window: 1,
            },
        )
        .unwrap();
        assert_eq!(expected.score, 6);
        assert_eq!(expected.best_action, Some(38));
        assert_eq!(report.completed_depth, 2);
        assert_eq!(report.score, Some(expected.score));
        assert_eq!(report.action_code, expected.best_action);
    }

    #[test]
    fn positive_depth_zero_stock_root_still_uses_exact_oracle() {
        let config = Config {
            ply_cap: 18,
            ..zero_wall_config()
        };
        let mut state = create_initial_state(&config).unwrap();
        for code in [67, 3, 58, 4, 49, 3, 40, 4, 31, 3, 22, 4, 13, 3] {
            state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                .unwrap();
        }
        let report = search_nodes(
            &config,
            &state,
            100_000,
            SearchOptions {
                max_depth: 1,
                transposition_capacity: 64,
                aspiration_window: 32,
            },
        )
        .unwrap();
        assert_eq!(report.completed_depth, 1);
        assert_eq!(report.action_code, Some(4));
        assert_eq!(report.score, Some(MATE_SCORE - 1));
        assert_eq!(report.diagnostics.zero_wall_oracle_queries, 1);
        assert_eq!(report.diagnostics.zero_wall_oracle_solutions, 1);
    }

    #[test]
    fn exhausted_budget_returns_deterministic_legal_fallback() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let report = search_nodes(
            &config,
            &state,
            1,
            SearchOptions {
                max_depth: 8,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(report.completed_depth, 0);
        assert!(report.stopped);
        assert_eq!(report.score, None);
        assert!(report.committed_iteration_nodes.is_empty());
        assert!(report.committed_iteration_scores.is_empty());
        assert_eq!(
            report.action_code,
            crate::legal_action_codes(&config, &state)
                .unwrap()
                .into_iter()
                .next()
        );
    }

    #[test]
    fn wall_clock_deadline_returns_the_last_completed_or_fallback_move() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let report = search_for(
            &config,
            &state,
            Duration::from_nanos(1),
            SearchOptions::default(),
        )
        .unwrap();
        assert!(report.stopped);
        assert!(crate::legal_action_codes(&config, &state)
            .unwrap()
            .contains(&report.action_code.unwrap()));
        assert_eq!(report.score, None);
    }

    #[test]
    fn expired_setup_returns_the_legal_fallback_without_allocating_search_state() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let report = run(
            &config,
            &state,
            CheckBudget::new(0),
            SearchOptions {
                max_depth: 8,
                transposition_capacity: 1 << 20,
                aspiration_window: 32,
            },
        )
        .unwrap();
        assert!(report.stopped);
        assert_eq!(report.completed_depth, 0);
        assert_eq!(report.score, None);
        assert_eq!(
            report.diagnostics.root_action_count,
            crate::legal_action_codes(&config, &state).unwrap().len()
        );
        assert_eq!(
            report.principal_variation,
            report.action_code.into_iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn terminal_root_reports_a_real_score_without_a_move() {
        let config = zero_wall_config();
        let mut state = create_initial_state(&config).unwrap();
        for code in [67, 3, 58, 4, 49, 3, 40, 4, 31, 3, 22, 4, 13, 3, 4] {
            state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                .unwrap();
        }
        assert!(matches!(state.outcome, Outcome::Win { .. }));
        let report = search_nodes(&config, &state, 1, SearchOptions::default()).unwrap();
        assert_eq!(report.action_code, None);
        assert_eq!(report.completed_depth, 0);
        assert!(report.score.is_some());
        assert!(report.principal_variation.is_empty());
    }

    #[test]
    fn injected_stop_restores_the_exact_prepared_root() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let mut prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let before = prepared.clone();
        let options = SearchOptions {
            max_depth: 4,
            transposition_capacity: 64,
            aspiration_window: 32,
        };
        let mut context = SearchContext {
            config: &config,
            budget: CheckBudget::new(2),
            tt: TranspositionTable::new(options.transposition_capacity),
            heuristics: Heuristics::new(usize::from(options.max_depth), config.policy_size()),
            oracle: ExactZeroWallOracle::new(),
            root_ply: prepared.ply,
            nodes: 0,
            diagnostics: SearchDiagnostics::default(),
        };
        assert!(context
            .negamax(&mut prepared, 4, 0, NEG_INFINITY, POS_INFINITY, None)
            .unwrap()
            .is_none());
        assert_eq!(prepared, before);
    }

    #[test]
    fn iteration_commit_is_atomic_across_search_and_pv_cutoffs() {
        let config = shallow_search_config();
        let state = create_initial_state(&config).unwrap();
        let options = SearchOptions {
            max_depth: 1,
            transposition_capacity: 64,
            aspiration_window: 32,
        };
        let completed = search_nodes(&config, &state, 100_000, options).unwrap();
        assert_eq!(completed.completed_depth, 1);
        let exact_nodes = completed.nodes;

        let after_search = run(&config, &state, NodeBudget::new(exact_nodes), options).unwrap();
        assert_eq!(after_search.completed_depth, 0);
        assert_eq!(after_search.score, None);
        assert!(after_search.committed_iteration_nodes.is_empty());
        assert!(after_search.committed_iteration_scores.is_empty());

        for allowed_polls in 1..=3 {
            let cutoff = run(
                &config,
                &state,
                ExactCutoffBudget::new(exact_nodes, allowed_polls),
                options,
            )
            .unwrap();
            assert_eq!(cutoff.completed_depth, 0, "poll {allowed_polls}");
            assert_eq!(cutoff.score, None);
            assert!(cutoff.committed_iteration_nodes.is_empty());
            assert!(cutoff.committed_iteration_scores.is_empty());
            assert_eq!(
                cutoff.principal_variation,
                cutoff.action_code.into_iter().collect::<Vec<_>>()
            );
        }
        let committed = run(
            &config,
            &state,
            ExactCutoffBudget::new(exact_nodes, 4),
            options,
        )
        .unwrap();
        assert_eq!(committed.completed_depth, 1);
        assert!(committed.score.is_some());
        assert_eq!(committed.committed_iteration_nodes, vec![exact_nodes]);
        assert_eq!(
            committed.committed_iteration_scores,
            vec![committed.score.unwrap()]
        );
    }

    #[test]
    fn root_relative_scores_do_not_depend_on_recursive_ply_parameter() {
        let config = shallow_search_config();
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let solve_at = |ply| {
            let mut position = prepared.clone();
            let mut context = SearchContext {
                config: &config,
                budget: NodeBudget::new(100_000),
                tt: TranspositionTable::new(256),
                heuristics: Heuristics::new(8, config.policy_size()),
                oracle: ExactZeroWallOracle::new(),
                root_ply: prepared.ply,
                nodes: 0,
                diagnostics: SearchDiagnostics::default(),
            };
            context
                .negamax(&mut position, 2, ply, NEG_INFINITY, POS_INFINITY, None)
                .unwrap()
                .unwrap()
                .score
        };
        assert_eq!(solve_at(0), solve_at(5));
    }

    #[test]
    fn principal_variation_stops_before_history_dependent_tt_hint() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let root = PreparedGameState::from_game_state(&config, &state).unwrap();
        let root_action = compact_legal_codes(&config, &root).iter().next().unwrap();
        let mut child = root.clone();
        child
            .apply_generated_code(&config, root_action)
            .expect("root action is generated legal");
        assert!(root.search_identity(&config).bounds_reusable());
        assert!(!child.search_identity(&config).bounds_reusable());
        let child_action = compact_legal_codes(&config, &child).iter().next().unwrap();

        let mut context = SearchContext {
            config: &config,
            budget: NodeBudget::new(100),
            tt: TranspositionTable::new(64),
            heuristics: Heuristics::new(4, config.policy_size()),
            oracle: ExactZeroWallOracle::new(),
            root_ply: root.ply,
            nodes: 0,
            diagnostics: SearchDiagnostics::default(),
        };
        context.tt.store(
            &config,
            root.search_identity(&config),
            2,
            0,
            Some(Bound::Exact),
            Some(root_action),
        );
        // This is legal in this concrete history, but its identity deliberately
        // omits repetition context. It must remain an ordering hint only.
        context.tt.store(
            &config,
            child.search_identity(&config),
            1,
            0,
            None,
            Some(child_action),
        );
        assert_eq!(
            context.principal_variation(&root, 2).unwrap(),
            Some(vec![root_action])
        );
        context.tt.store(
            &config,
            root.search_identity(&config),
            2,
            0,
            Some(Bound::Lower),
            Some(root_action),
        );
        assert_eq!(context.principal_variation(&root, 2).unwrap(), Some(vec![]));
    }

    #[test]
    fn root_diagnostics_cover_every_legal_action() {
        let config = zero_wall_config();
        let state = create_initial_state(&config).unwrap();
        let report = search_nodes(
            &config,
            &state,
            1_000,
            SearchOptions {
                max_depth: 2,
                ..SearchOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            report.diagnostics.root_action_count,
            crate::legal_action_codes(&config, &state).unwrap().len()
        );
    }

    #[test]
    fn searched_moves_remain_legal_across_random_reachable_states() {
        let mut config = zero_wall_config();
        config.initial_stock = Players { a: 1, b: 1 };
        let mut state = create_initial_state(&config).unwrap();
        let mut random = 0x4d59_5df4_d0f3_3173_u64;
        for _ in 0..24 {
            if !state.outcome.is_ongoing() {
                break;
            }
            let legal = crate::legal_action_codes(&config, &state).unwrap();
            let report = search_nodes(
                &config,
                &state,
                300,
                SearchOptions {
                    max_depth: 3,
                    transposition_capacity: 256,
                    aspiration_window: 32,
                },
            )
            .unwrap();
            let selected = report.action_code.expect("ongoing state has a move");
            assert!(legal.contains(&selected));
            assert_eq!(report.principal_variation.first().copied(), Some(selected));

            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let code = legal[random as usize % legal.len()];
            state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn full_research_ignores_reusable_wall_child_bound_and_keeps_minimax_action() {
        let config = Config {
            initial_stock: Players { a: 1, b: 2 },
            ..shallow_search_config()
        };
        let state = create_initial_state(&config).unwrap();
        let root = PreparedGameState::from_game_state(&config, &state).unwrap();
        let wall = compact_legal_codes(&config, &root)
            .iter()
            .find(|code| *code >= config.cells())
            .expect("root has a legal wall");
        let mut child = root.clone();
        child
            .apply_generated_code(&config, wall)
            .expect("generated wall is legal");
        let identity = child.search_identity(&config);
        assert!(
            identity.bounds_reusable(),
            "walls begin a fresh repetition epoch"
        );

        // Depth three keeps this above the scout's immediate horizon while the
        // remaining stock stays small enough to keep the plain reference cheap.
        let mut reference = child.clone();
        let expected = plain_minimax(&config, &mut reference, 3, root.ply).unwrap();
        let wrong_action = compact_legal_codes(&config, &child)
            .iter()
            .find(|&code| Some(code) != expected.best_action)
            .expect("child has a non-best legal action");

        let new_context = || SearchContext {
            config: &config,
            budget: NodeBudget::new(2_000_000),
            tt: TranspositionTable::new(64),
            heuristics: Heuristics::new(4, config.policy_size()),
            oracle: ExactZeroWallOracle::new(),
            root_ply: root.ply,
            nodes: 0,
            diagnostics: SearchDiagnostics::default(),
        };
        // This is the lower bound the equality scout can leave at the child
        // root. With beta at that boundary, normal TT probing returns it
        // immediately, including its non-exact action.
        let mut shortcut = new_context();
        shortcut.tt.store(
            &config,
            identity,
            3,
            expected.score,
            Some(Bound::Lower),
            Some(wrong_action),
        );
        let mut shortcut_child = child.clone();
        let shortcut_value = shortcut
            .negamax(
                &mut shortcut_child,
                3,
                1,
                NEG_INFINITY,
                expected.score,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(shortcut.nodes, 1);
        assert_eq!(shortcut_value.best_action, Some(wrong_action));

        let mut full_research = new_context();
        full_research.tt.store(
            &config,
            identity,
            3,
            expected.score,
            Some(Bound::Lower),
            Some(wrong_action),
        );
        let mut research_child = child.clone();
        let value = full_research
            .negamax_with_tt_bounds(
                &mut research_child,
                3,
                1,
                NEG_INFINITY,
                POS_INFINITY,
                SearchFrame {
                    previous_action: None,
                    use_tt_bounds: false,
                },
            )
            .unwrap()
            .unwrap();
        assert!(full_research.nodes > 1, "the child was traversed exactly");
        assert_eq!(value.score, expected.score);
        assert_eq!(value.best_action, expected.best_action);
        assert_eq!(research_child, child, "search remains transactional");
    }
}
