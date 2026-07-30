use crate::{
    compact_pawn_codes, Board, CodeList, Config, Coord, Orientation, Player, PreparedGameState,
    Wall,
};

use super::tt::mirror_code;

const TT_BONUS: i32 = 1_000_000_000;
const KILLER_ONE_BONUS: i32 = 800_000_000;
const KILLER_TWO_BONUS: i32 = 700_000_000;
const COUNTER_BONUS: i32 = 600_000_000;

// Verified immediate goal defense occupies its own band below every learned
// special-move hint and above the ordinary history band.
const IMMEDIATE_GOAL_DEFENSE_BASE: i32 = 550_000_000;
const IMMEDIATE_GOAL_DEFENSE_CEILING: i32 = COUNTER_BONUS - 1;
const NORMAL_WALL_CEILING: i32 = IMMEDIATE_GOAL_DEFENSE_BASE - 1;

const OPPONENT_PATH_EDGE_BONUS: i32 = 2_000_000;
const OWN_PATH_EDGE_PENALTY: i32 = 3_000_000;
const OPPONENT_FINAL_EDGE_BONUS: i32 = 8_000_000;
const OWN_FINAL_EDGE_PENALTY: i32 = 12_000_000;
const WALL_TACTICAL_FLOOR: i32 = -12_000_000;
const WALL_TACTICAL_CEILING: i32 = 48_000_000;

const MAX_CELLS: usize = 81;
const MAX_GOAL_MOVES: usize = 9;
#[cfg(test)]
const PACKED_PROFILE_FLOODS_PER_NODE: usize = 2;

pub(crate) fn uses_profiled_ordering(config: &Config) -> bool {
    config.rows == 9 && config.columns == 9
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrderedAction {
    pub(crate) index: usize,
    pub(crate) code: usize,
}

#[derive(Debug, Clone, Copy)]
struct ScoredAction {
    action: OrderedAction,
    score: i32,
    canonical_code: u16,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OrderingHints {
    pub(crate) tt_action: Option<usize>,
    pub(crate) ply: usize,
    pub(crate) previous_action: Option<usize>,
    pub(crate) canonical_mirrored: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EdgeMask {
    down: u128,
    right: u128,
}

impl EdgeMask {
    fn wall_intersection_count(self, profile: Self) -> u32 {
        debug_assert!(
            self.down == 0 || self.right == 0,
            "one wall blocks only one physical edge orientation"
        );
        if self.down == 0 {
            (self.right & profile.right).count_ones()
        } else {
            (self.down & profile.down).count_ones()
        }
    }
}

#[derive(Clone)]
struct DistanceLayers {
    levels: [u128; MAX_CELLS],
    len: usize,
}

impl DistanceLayers {
    fn new_until(config: &Config, board: Board, sources: u128, target: u128) -> Self {
        debug_assert_ne!(sources, 0);
        debug_assert_ne!(target, 0);
        let columns = usize::from(config.columns);
        let valid = (1_u128 << config.cells()) - 1;
        let mut right_column = 0_u128;
        for row in 0..config.rows {
            right_column |=
                1_u128 << (usize::from(row) * columns + usize::from(config.columns - 1));
        }

        let mut levels = [0_u128; MAX_CELLS];
        levels[0] = sources;
        let mut len = 1;
        let mut seen = sources;
        let mut frontier = sources;
        while frontier & target == 0 {
            let upward = (frontier >> columns) & !board.blocked_down;
            let downward = ((frontier & !board.blocked_down) << columns) & valid;
            let leftward = (frontier >> 1) & !right_column & !board.blocked_right;
            let rightward = ((frontier & !right_column & !board.blocked_right) << 1) & valid;
            frontier = (upward | downward | leftward | rightward) & !seen;
            if frontier == 0 {
                break;
            }
            debug_assert!(len < MAX_CELLS);
            levels[len] = frontier;
            len += 1;
            seen |= frontier;
        }
        Self { levels, len }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShortestPathProfile {
    distance: u8,
    shortest: EdgeMask,
    final_step: EdgeMask,
}

impl ShortestPathProfile {
    fn new(
        config: &Config,
        board: Board,
        from: Coord,
        goal_row: u8,
        stopped: &mut impl FnMut() -> bool,
    ) -> Option<Self> {
        let columns = usize::from(config.columns);
        let valid = (1_u128 << config.cells()) - 1;
        let mut right_column = 0_u128;
        for row in 0..config.rows {
            right_column |=
                1_u128 << (usize::from(row) * columns + usize::from(config.columns - 1));
        }
        let source = 1_u128 << (usize::from(from.r) * columns + usize::from(from.c));
        let goal =
            ((1_u128 << columns) - 1) << (usize::from(goal_row) * usize::from(config.columns));
        let goal_layers = DistanceLayers::new_until(config, board, goal, source);
        if stopped() {
            return None;
        }
        let distance = goal_layers.len - 1;
        debug_assert!(distance < MAX_CELLS);

        let mut shortest = EdgeMask::default();
        let mut final_step = EdgeMask::default();
        let mut source_level = source;
        for source_distance in 0..distance {
            let goal_distance = distance - source_distance - 1;
            let goal_level = goal_layers.levels[goal_distance];
            let upward = (source_level >> columns) & !board.blocked_down;
            let downward = ((source_level & !board.blocked_down) << columns) & valid;
            let leftward = (source_level >> 1) & !right_column & !board.blocked_right;
            let rightward = ((source_level & !right_column & !board.blocked_right) << 1) & valid;
            let next_level = (upward | downward | leftward | rightward) & goal_level;
            debug_assert_ne!(next_level, 0);

            // A physical edge belongs to the shortest-path DAG exactly when
            // it advances the source-reachable frontier into the next lower
            // goal-distance layer.
            let down = ((source_level & (next_level >> columns))
                | ((source_level >> columns) & next_level))
                & !board.blocked_down
                & valid;
            let right = ((source_level & (next_level >> 1)) | ((source_level >> 1) & next_level))
                & !right_column
                & !board.blocked_right
                & valid;
            let edges = EdgeMask { down, right };
            shortest.down |= edges.down;
            shortest.right |= edges.right;
            if goal_distance == 0 {
                final_step = edges;
            }
            source_level = next_level;
        }

        Some(Self {
            distance: u8::try_from(distance).expect("normal duel path distance fits in u8"),
            shortest,
            final_step,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct WallFeatures {
    opponent_path_hits: u32,
    own_path_hits: u32,
    opponent_final_hits: u32,
    own_final_hits: u32,
    blocks_immediate_goal: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ImmediateGoalThreats {
    paths: [EdgeMask; MAX_GOAL_MOVES],
    len: usize,
}

impl ImmediateGoalThreats {
    fn new(config: &Config, state: &PreparedGameState, board: Board, opponent: Player) -> Self {
        let from = *state.position.pawns.get(opponent);
        let goal_row = *config.goal_rows.get(opponent);
        if from.r.abs_diff(goal_row) > 2 {
            return Self::default();
        }

        let mut opponent_position = state.position;
        opponent_position.turn = opponent;
        let mut pawn_codes = CodeList::new();
        compact_pawn_codes(config, &opponent_position, board, &mut pawn_codes);
        let middle = *state.position.pawns.get(opponent.other());
        let mut threats = Self::default();
        for code in pawn_codes
            .iter()
            .filter(|code| code / usize::from(config.columns) == usize::from(goal_row))
        {
            let destination = Coord {
                r: goal_row,
                c: (code % usize::from(config.columns)) as u8,
            };
            let direct = from.r.abs_diff(destination.r) + from.c.abs_diff(destination.c) == 1;
            let path = if direct {
                edge_between(config, from, destination)
            } else {
                let first = edge_between(config, from, middle);
                let second = edge_between(config, middle, destination);
                EdgeMask {
                    down: first.down | second.down,
                    right: first.right | second.right,
                }
            };
            debug_assert!(threats.len < MAX_GOAL_MOVES);
            threats.paths[threats.len] = path;
            threats.len += 1;
        }
        threats
    }

    fn all_blocked_by(self, wall_edges: EdgeMask) -> bool {
        self.len > 0
            && self.paths[..self.len]
                .iter()
                .all(|&path| wall_edges.wall_intersection_count(path) > 0)
    }
}

// General wall-chain extension ordering is deliberately deferred: the exact
// endpoint map and per-wall intersection failed the <=5% full-width-node
// overhead gate. The complete wall set remains searchable, and a future
// policy/ablation milestone can reintroduce the signal with measured evidence.
struct WallOrderingContext {
    own: ShortestPathProfile,
    opponent: ShortestPathProfile,
    opponent_coord: Coord,
    immediate_goal_threats: ImmediateGoalThreats,
}

impl WallOrderingContext {
    fn new(
        config: &Config,
        state: &PreparedGameState,
        stopped: &mut impl FnMut() -> bool,
    ) -> Option<Self> {
        let board = state.position.board();
        let mover = state.position.turn;
        let opponent = mover.other();
        let own_coord = *state.position.pawns.get(mover);
        let opponent_coord = *state.position.pawns.get(opponent);
        let own = ShortestPathProfile::new(
            config,
            board,
            own_coord,
            *config.goal_rows.get(mover),
            stopped,
        )?;
        let opponent_profile = ShortestPathProfile::new(
            config,
            board,
            opponent_coord,
            *config.goal_rows.get(opponent),
            stopped,
        )?;
        let immediate_goal_threats = ImmediateGoalThreats::new(config, state, board, opponent);

        Some(Self {
            own,
            opponent: opponent_profile,
            opponent_coord,
            immediate_goal_threats,
        })
    }

    fn features(&self, config: &Config, wall: Wall) -> WallFeatures {
        let edges = wall_edges(config, wall);
        let own_path_hits = edges.wall_intersection_count(self.own.shortest);
        WallFeatures {
            opponent_path_hits: edges.wall_intersection_count(self.opponent.shortest),
            own_path_hits,
            opponent_final_hits: if self.opponent.distance <= 3 {
                edges.wall_intersection_count(self.opponent.final_step)
            } else {
                0
            },
            own_final_hits: if self.own.distance <= 3 {
                edges.wall_intersection_count(self.own.final_step)
            } else {
                0
            },
            blocks_immediate_goal: self.immediate_goal_threats.all_blocked_by(edges),
        }
    }

    fn static_tactical_score(&self, features: WallFeatures) -> i32 {
        let own_final_penalty = if self.own.distance <= 3 {
            OWN_FINAL_EDGE_PENALTY
        } else {
            0
        };
        (features.opponent_path_hits as i32 * OPPONENT_PATH_EDGE_BONUS)
            .saturating_sub(features.own_path_hits as i32 * OWN_PATH_EDGE_PENALTY)
            .saturating_add(features.opponent_final_hits as i32 * OPPONENT_FINAL_EDGE_BONUS)
            .saturating_sub(features.own_final_hits as i32 * own_final_penalty)
            .clamp(WALL_TACTICAL_FLOOR, WALL_TACTICAL_CEILING)
    }

    fn immediate_goal_defense(&self, features: WallFeatures) -> bool {
        features.blocks_immediate_goal
    }
}

fn wall_from_code(config: &Config, code: usize) -> Wall {
    debug_assert!(code >= config.cells() && code < config.policy_size());
    let anchors = config.anchors_per_axis();
    let offset = code - config.cells();
    let anchor = offset % anchors;
    Wall {
        orientation: if offset < anchors {
            Orientation::Horizontal
        } else {
            Orientation::Vertical
        },
        r: (anchor / usize::from(config.columns - 1)) as u8,
        c: (anchor % usize::from(config.columns - 1)) as u8,
    }
}

fn edge_between(config: &Config, from: Coord, to: Coord) -> EdgeMask {
    let columns = usize::from(config.columns);
    if from.c == to.c && from.r.abs_diff(to.r) == 1 {
        let row = usize::from(from.r.min(to.r));
        return EdgeMask {
            down: 1_u128 << (row * columns + usize::from(from.c)),
            right: 0,
        };
    }
    debug_assert!(from.r == to.r && from.c.abs_diff(to.c) == 1);
    let column = usize::from(from.c.min(to.c));
    EdgeMask {
        down: 0,
        right: 1_u128 << (usize::from(from.r) * columns + column),
    }
}

fn wall_edges(config: &Config, wall: Wall) -> EdgeMask {
    let columns = usize::from(config.columns);
    match wall.orientation {
        Orientation::Horizontal => {
            let first = usize::from(wall.r) * columns + usize::from(wall.c);
            EdgeMask {
                down: (1_u128 << first) | (1_u128 << (first + 1)),
                right: 0,
            }
        }
        Orientation::Vertical => {
            let first = usize::from(wall.r) * columns + usize::from(wall.c);
            EdgeMask {
                down: 0,
                right: (1_u128 << first) | (1_u128 << (first + columns)),
            }
        }
    }
}

pub(crate) struct Heuristics {
    killers: Vec<[Option<usize>; 2]>,
    history: [Vec<i32>; 2],
    counter: Vec<Option<usize>>,
}

impl Heuristics {
    pub(crate) fn new(max_depth: usize, policy_size: usize) -> Self {
        Self {
            killers: vec![[None, None]; max_depth.saturating_add(2)],
            history: std::array::from_fn(|_| vec![0; policy_size]),
            counter: vec![None; policy_size],
        }
    }

    fn player_index(player: Player) -> usize {
        match player {
            Player::A => 0,
            Player::B => 1,
        }
    }

    fn learned_priority(
        &self,
        code: usize,
        tt_action: Option<usize>,
        ply: usize,
        previous_action: Option<usize>,
    ) -> Option<i32> {
        if Some(code) == tt_action {
            return Some(TT_BONUS);
        }
        let killers = self.killers.get(ply).copied().unwrap_or([None, None]);
        if Some(code) == killers[0] {
            return Some(KILLER_ONE_BONUS);
        }
        if Some(code) == killers[1] {
            return Some(KILLER_TWO_BONUS);
        }
        if previous_action
            .and_then(|previous| self.counter.get(previous))
            .copied()
            .flatten()
            == Some(code)
        {
            return Some(COUNTER_BONUS);
        }
        None
    }

    fn legacy_score(
        &self,
        config: &Config,
        state: &PreparedGameState,
        code: usize,
        hints: OrderingHints,
    ) -> i32 {
        if let Some(priority) =
            self.learned_priority(code, hints.tt_action, hints.ply, hints.previous_action)
        {
            return priority;
        }
        let history = self.history[Self::player_index(state.position.turn)][code];
        if code < config.cells() {
            let row = (code / usize::from(config.columns)) as i32;
            let progress = match state.position.turn {
                Player::A => i32::from(config.rows) - row,
                Player::B => row + 1,
            };
            return history.saturating_add(progress * 256);
        }

        let anchors = config.anchors_per_axis();
        let anchor = (code - config.cells()) % anchors;
        let row = (anchor / usize::from(config.columns - 1)) as i32;
        let column = (anchor % usize::from(config.columns - 1)) as i32;
        let opponent = *state.position.pawns.get(state.position.turn.other());
        let proximity =
            32 - (row - i32::from(opponent.r)).abs() - (column - i32::from(opponent.c)).abs();
        let center = i32::from(config.columns - 2) / 2;
        history
            .saturating_add(proximity * 64)
            .saturating_sub((column - center).abs() * 8)
    }

    fn legacy_order(
        &self,
        config: &Config,
        state: &PreparedGameState,
        actions: &CodeList,
        hints: OrderingHints,
    ) -> Vec<OrderedAction> {
        let mut ordered: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(index, code)| {
                (
                    OrderedAction { index, code },
                    self.legacy_score(config, state, code, hints),
                )
            })
            .collect();
        ordered.sort_unstable_by(|(left_action, left_score), (right_action, right_score)| {
            right_score
                .cmp(left_score)
                .then_with(|| left_action.code.cmp(&right_action.code))
        });
        ordered.into_iter().map(|(action, _)| action).collect()
    }

    fn score(
        &self,
        config: &Config,
        state: &PreparedGameState,
        wall_context: Option<&WallOrderingContext>,
        code: usize,
        hints: OrderingHints,
    ) -> i32 {
        if let Some(priority) =
            self.learned_priority(code, hints.tt_action, hints.ply, hints.previous_action)
        {
            return priority;
        }
        let history = self.history[Self::player_index(state.position.turn)][code];
        if code < config.cells() {
            let row = (code / usize::from(config.columns)) as i32;
            let progress = match state.position.turn {
                Player::A => i32::from(config.rows) - row,
                Player::B => row + 1,
            };
            return history.saturating_add(progress * 256);
        }

        let context = wall_context.expect("a wall-bearing action list builds path profiles");
        let wall = wall_from_code(config, code);
        let features = context.features(config, wall);
        let tactical = context.static_tactical_score(features);

        // The old location signal remains only as a deterministic low-order
        // tie-break. Doubled cell/junction coordinates keep it mirror exact.
        let row_distance =
            (2 * i32::from(wall.r) + 1 - 2 * i32::from(context.opponent_coord.r)).abs();
        let column_distance =
            (2 * i32::from(wall.c) + 1 - 2 * i32::from(context.opponent_coord.c)).abs();
        let proximity = (64 - row_distance - column_distance) * 32;
        let center_distance = (2 * i32::from(wall.c) - i32::from(config.columns - 2)).abs();
        let location = proximity.saturating_sub(center_distance * 4);

        if context.immediate_goal_defense(features) {
            return IMMEDIATE_GOAL_DEFENSE_BASE
                .saturating_add(tactical.max(0))
                .saturating_add(history / 16)
                .saturating_add(location)
                .min(IMMEDIATE_GOAL_DEFENSE_CEILING);
        }
        history
            .saturating_add(tactical)
            .saturating_add(location)
            .min(NORMAL_WALL_CEILING)
    }

    pub(crate) fn order(
        &self,
        config: &Config,
        state: &PreparedGameState,
        actions: &CodeList,
        hints: OrderingHints,
        mut stopped: impl FnMut() -> bool,
    ) -> Option<Vec<OrderedAction>> {
        if !uses_profiled_ordering(config) {
            return Some(self.legacy_order(config, state, actions, hints));
        }
        if stopped() {
            return None;
        }
        let has_walls = actions.iter().any(|code| code >= config.cells());
        let wall_context = if has_walls {
            Some(WallOrderingContext::new(config, state, &mut stopped)?)
        } else {
            None
        };
        let mut ordered = Vec::with_capacity(actions.len);
        for (index, code) in actions.iter().enumerate() {
            if index != 0 && index % 64 == 0 && stopped() {
                return None;
            }
            ordered.push(ScoredAction {
                action: OrderedAction { index, code },
                score: self.score(config, state, wall_context.as_ref(), code, hints),
                canonical_code: u16::try_from(if hints.canonical_mirrored {
                    mirror_code(config, code)
                } else {
                    code
                })
                .expect("normal duel policy code fits in u16"),
            });
        }
        ordered.sort_unstable_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.canonical_code.cmp(&right.canonical_code))
                .then_with(|| left.action.code.cmp(&right.action.code))
        });
        if stopped() {
            None
        } else {
            Some(ordered.into_iter().map(|scored| scored.action).collect())
        }
    }

    pub(crate) fn record_cutoff(
        &mut self,
        player: Player,
        code: usize,
        depth: u8,
        ply: usize,
        previous_action: Option<usize>,
        is_wall: bool,
    ) {
        if is_wall {
            if let Some(killers) = self.killers.get_mut(ply) {
                if killers[0] != Some(code) {
                    killers[1] = killers[0];
                    killers[0] = Some(code);
                }
            }
        }
        let bonus = i32::from(depth).saturating_mul(i32::from(depth)).max(1);
        let history = &mut self.history[Self::player_index(player)][code];
        *history = history.saturating_add(bonus).min(500_000_000);
        if let Some(previous) = previous_action {
            self.counter[previous] = Some(code);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::{DeadlineBudget, SearchBudget};
    use super::*;
    use crate::{
        apply_legal_action, compact_legal_codes, create_initial_state, decode_action, Config,
        GameState, Players, JUMP_RULE, REPETITION_THRESHOLD, RULESET,
    };
    use std::collections::{BTreeSet, VecDeque};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    fn config(stock: u64) -> Config {
        const SIZE: u8 = 9;
        Config {
            ruleset: RULESET.into(),
            rows: SIZE,
            columns: SIZE,
            start: Players {
                a: Coord {
                    r: SIZE - 1,
                    c: SIZE / 2,
                },
                b: Coord { r: 0, c: SIZE / 2 },
            },
            goal_rows: Players { a: 0, b: SIZE - 1 },
            initial_stock: Players { a: stock, b: stock },
            jump_rule: JUMP_RULE.into(),
            repetition_threshold: REPETITION_THRESHOLD,
            ply_cap: 512,
            first_player: Player::A,
        }
    }

    fn apply_codes(config: &Config, codes: &[usize]) -> GameState {
        let mut state = create_initial_state(config).unwrap();
        for &code in codes {
            state =
                apply_legal_action(config, &state, &decode_action(config, code).unwrap()).unwrap();
        }
        state
    }

    fn generated_goal_pawn_codes(config: &Config, state: &PreparedGameState) -> Vec<usize> {
        let goal_row = usize::from(*config.goal_rows.get(state.position.turn));
        compact_legal_codes(config, state)
            .iter()
            .filter(|&code| code < config.cells() && code / usize::from(config.columns) == goal_row)
            .collect()
    }

    fn scalar_distances(config: &Config, board: Board, sources: &[Coord]) -> [u8; MAX_CELLS] {
        let mut distances = [u8::MAX; MAX_CELLS];
        let mut queue = VecDeque::new();
        for &source in sources {
            let index = usize::from(source.r) * usize::from(config.columns) + usize::from(source.c);
            distances[index] = 0;
            queue.push_back(source);
        }
        while let Some(from) = queue.pop_front() {
            let from_index =
                usize::from(from.r) * usize::from(config.columns) + usize::from(from.c);
            for (dr, dc) in [(-1_i16, 0_i16), (0, -1), (0, 1), (1, 0)] {
                let row = i16::from(from.r) + dr;
                let column = i16::from(from.c) + dc;
                if row < 0
                    || row >= i16::from(config.rows)
                    || column < 0
                    || column >= i16::from(config.columns)
                {
                    continue;
                }
                let to = Coord {
                    r: row as u8,
                    c: column as u8,
                };
                let to_index = usize::from(to.r) * usize::from(config.columns) + usize::from(to.c);
                if distances[to_index] != u8::MAX || board.edge_blocked(config, from, to) {
                    continue;
                }
                distances[to_index] = distances[from_index] + 1;
                queue.push_back(to);
            }
        }
        distances
    }

    fn scalar_profile(
        config: &Config,
        board: Board,
        from: Coord,
        goal_row: u8,
    ) -> ShortestPathProfile {
        let source_distances = scalar_distances(config, board, &[from]);
        let goals: Vec<_> = (0..config.columns)
            .map(|c| Coord { r: goal_row, c })
            .collect();
        let goal_distances = scalar_distances(config, board, &goals);
        let source_index = usize::from(from.r) * usize::from(config.columns) + usize::from(from.c);
        let distance = goal_distances[source_index];
        assert_ne!(distance, u8::MAX, "legal states retain a goal path");
        let mut shortest = EdgeMask::default();
        let mut final_step = EdgeMask::default();
        let on_shortest = |left: usize, right: usize| {
            (source_distances[left] != u8::MAX
                && goal_distances[right] != u8::MAX
                && source_distances[left]
                    .saturating_add(1)
                    .saturating_add(goal_distances[right])
                    == distance)
                || (source_distances[right] != u8::MAX
                    && goal_distances[left] != u8::MAX
                    && source_distances[right]
                        .saturating_add(1)
                        .saturating_add(goal_distances[left])
                        == distance)
        };
        let columns = usize::from(config.columns);
        for row in 0..usize::from(config.rows) {
            for column in 0..columns {
                let current = row * columns + column;
                let down_neighbor = current + columns;
                if row + 1 < usize::from(config.rows)
                    && !board.edge_blocked(
                        config,
                        Coord {
                            r: row as u8,
                            c: column as u8,
                        },
                        Coord {
                            r: row as u8 + 1,
                            c: column as u8,
                        },
                    )
                    && on_shortest(current, down_neighbor)
                {
                    shortest.down |= 1_u128 << current;
                    if goal_distances[current] == 0 || goal_distances[down_neighbor] == 0 {
                        final_step.down |= 1_u128 << current;
                    }
                }
                let right_neighbor = current + 1;
                if column + 1 < columns
                    && !board.edge_blocked(
                        config,
                        Coord {
                            r: row as u8,
                            c: column as u8,
                        },
                        Coord {
                            r: row as u8,
                            c: column as u8 + 1,
                        },
                    )
                    && on_shortest(current, right_neighbor)
                {
                    shortest.right |= 1_u128 << current;
                    if goal_distances[current] == 0 || goal_distances[right_neighbor] == 0 {
                        final_step.right |= 1_u128 << current;
                    }
                }
            }
        }
        ShortestPathProfile {
            distance,
            shortest,
            final_step,
        }
    }

    fn ordered_codes(
        heuristics: &Heuristics,
        config: &Config,
        state: &PreparedGameState,
        actions: &CodeList,
        tt_action: Option<usize>,
        ply: usize,
        previous_action: Option<usize>,
    ) -> Vec<usize> {
        let canonical_mirrored = state.search_identity(config).mirrored();
        heuristics
            .order(
                config,
                state,
                actions,
                OrderingHints {
                    tt_action,
                    ply,
                    previous_action,
                    canonical_mirrored,
                },
                || false,
            )
            .expect("unbounded test ordering completes")
            .into_iter()
            .map(|action| action.code)
            .collect()
    }

    fn shuffled(actions: &CodeList, mut seed: u64) -> CodeList {
        let mut codes: Vec<_> = actions.iter().collect();
        for index in (1..codes.len()).rev() {
            next_random(&mut seed);
            let swap = seed as usize % (index + 1);
            codes.swap(index, swap);
        }
        let mut result = CodeList::new();
        for code in codes {
            result.push(code);
        }
        result
    }

    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn ordered_output_is_a_deterministic_full_permutation_across_reachable_states() {
        let config = config(3);
        let mut state = create_initial_state(&config).unwrap();
        let mut random = 0xd1b5_4a32_d192_ed0a_u64;
        for sample in 0..18 {
            if !state.outcome.is_ongoing() {
                break;
            }
            let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
            let before = prepared.clone();
            let legal = compact_legal_codes(&config, &prepared);
            let permuted = shuffled(&legal, random);
            let heuristics = Heuristics::new(8, config.policy_size());
            let canonical = ordered_codes(&heuristics, &config, &prepared, &legal, None, 0, None);
            let reordered =
                ordered_codes(&heuristics, &config, &prepared, &permuted, None, 0, None);
            assert_eq!(canonical, reordered, "sample {sample}");
            assert_eq!(canonical.len(), legal.len);
            assert_eq!(
                canonical.iter().copied().collect::<BTreeSet<_>>(),
                legal.iter().collect::<BTreeSet<_>>()
            );
            assert_eq!(prepared, before, "ordering is read-only");

            let next = legal
                .iter()
                .nth(next_random(&mut random) as usize % legal.len)
                .unwrap();
            state = apply_legal_action(&config, &state, &decode_action(&config, next).unwrap())
                .unwrap();
        }
    }

    #[test]
    fn mirror_equivalent_states_have_mirror_equivalent_ordering() {
        let config = config(3);
        let mut left = apply_codes(&config, &[75, 13]);
        let mut right = apply_codes(
            &config,
            &[mirror_code(&config, 75), mirror_code(&config, 13)],
        );
        let left_wall = crate::legal_action_codes(&config, &left)
            .unwrap()
            .into_iter()
            .find(|&code| code >= config.cells())
            .unwrap();
        let right_wall = mirror_code(&config, left_wall);
        left = apply_legal_action(&config, &left, &decode_action(&config, left_wall).unwrap())
            .unwrap();
        right = apply_legal_action(
            &config,
            &right,
            &decode_action(&config, right_wall).unwrap(),
        )
        .unwrap();
        let left = PreparedGameState::from_game_state(&config, &left).unwrap();
        let right = PreparedGameState::from_game_state(&config, &right).unwrap();
        let left_identity = left.search_identity(&config);
        let right_identity = right.search_identity(&config);
        assert_eq!(left_identity, right_identity);
        assert_ne!(left_identity.mirrored(), right_identity.mirrored());

        let heuristics = Heuristics::new(8, config.policy_size());
        let left_actions = compact_legal_codes(&config, &left);
        let right_actions = compact_legal_codes(&config, &right);
        let left_order = ordered_codes(&heuristics, &config, &left, &left_actions, None, 0, None);
        let right_order =
            ordered_codes(&heuristics, &config, &right, &right_actions, None, 0, None);
        assert_eq!(
            left_order
                .into_iter()
                .map(|code| mirror_code(&config, code))
                .collect::<Vec<_>>(),
            right_order
        );
    }

    #[test]
    fn stockless_pawn_only_ties_use_canonical_mirror_ordering() {
        let config = config(0);
        let left = apply_codes(&config, &[75, 13]);
        let right = apply_codes(
            &config,
            &[mirror_code(&config, 75), mirror_code(&config, 13)],
        );
        let left = PreparedGameState::from_game_state(&config, &left).unwrap();
        let right = PreparedGameState::from_game_state(&config, &right).unwrap();
        let left_identity = left.search_identity(&config);
        let right_identity = right.search_identity(&config);
        assert_eq!(left_identity, right_identity);
        assert_ne!(left_identity.mirrored(), right_identity.mirrored());

        let left_actions = compact_legal_codes(&config, &left);
        let right_actions = compact_legal_codes(&config, &right);
        assert!(left_actions.iter().all(|code| code < config.cells()));
        assert!(right_actions.iter().all(|code| code < config.cells()));
        let left_lateral_count = left_actions
            .iter()
            .filter(|&code| code / usize::from(config.columns) == usize::from(config.rows - 1))
            .count();
        let right_lateral_count = right_actions
            .iter()
            .filter(|&code| code / usize::from(config.columns) == usize::from(config.rows - 1))
            .count();
        assert_eq!(left_lateral_count, 2, "fixture contains a tied pawn pair");
        assert_eq!(
            right_lateral_count, 2,
            "mirrored fixture contains the tied pair"
        );

        let heuristics = Heuristics::new(8, config.policy_size());
        let left_order = ordered_codes(&heuristics, &config, &left, &left_actions, None, 0, None);
        let right_order =
            ordered_codes(&heuristics, &config, &right, &right_actions, None, 0, None);
        assert_eq!(
            left_order
                .into_iter()
                .map(|code| mirror_code(&config, code))
                .collect::<Vec<_>>(),
            right_order
        );
    }

    #[test]
    fn verified_immediate_goal_defenses_lead_irrelevant_zero_gain_walls() {
        let config = Config {
            initial_stock: Players { a: 1, b: 0 },
            ..config(1)
        };
        let state = apply_codes(
            &config,
            &[75, 13, 76, 22, 75, 31, 76, 40, 75, 49, 76, 58, 75, 67],
        );
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let actions = compact_legal_codes(&config, &prepared);
        let mut never_stop = || false;
        let context = WallOrderingContext::new(&config, &prepared, &mut never_stop).unwrap();
        let non_defense = actions
            .iter()
            .filter(|&code| code < config.cells())
            .find(|&code| {
                let mut child = prepared.clone();
                child.apply_generated_code(&config, code).unwrap();
                !generated_goal_pawn_codes(&config, &child).is_empty()
            })
            .expect("an ordinary move exposes the opponent's immediate goal move");
        let blocking: Vec<_> = actions
            .iter()
            .filter(|&code| code >= config.cells())
            .filter(|&code| {
                let wall = wall_from_code(&config, code);
                let features = context.features(&config, wall);
                features.blocks_immediate_goal
            })
            .collect();
        let irrelevant = actions
            .iter()
            .filter(|&code| code >= config.cells())
            .find(|&code| {
                let features = context.features(&config, wall_from_code(&config, code));
                features.opponent_path_hits == 0 && features.own_path_hits == 0
            })
            .expect("fixture retains a legal zero-gain wall");
        assert!(!blocking.is_empty());
        let mut undefended_child = prepared.clone();
        undefended_child
            .apply_generated_code(&config, non_defense)
            .unwrap();
        assert!(
            !generated_goal_pawn_codes(&config, &undefended_child).is_empty(),
            "the threat is independently present after a non-defense"
        );
        for &code in &blocking {
            let mut defended_child = prepared.clone();
            defended_child.apply_generated_code(&config, code).unwrap();
            assert!(
                generated_goal_pawn_codes(&config, &defended_child).is_empty(),
                "classified wall {code} independently removes every immediate goal move"
            );
        }
        let heuristics = Heuristics::new(8, config.policy_size());
        let hints = OrderingHints {
            canonical_mirrored: prepared.search_identity(&config).mirrored(),
            ..OrderingHints::default()
        };
        for &code in &blocking {
            assert!(
                heuristics.score(&config, &prepared, Some(&context), code, hints,)
                    >= IMMEDIATE_GOAL_DEFENSE_BASE,
                "verified defense {code} occupies the immediate-defense band"
            );
        }

        let order = ordered_codes(&heuristics, &config, &prepared, &actions, None, 0, None);
        let first_block = blocking
            .iter()
            .map(|code| {
                order
                    .iter()
                    .position(|candidate| candidate == code)
                    .unwrap()
            })
            .min()
            .unwrap();
        assert!(first_block < order.iter().position(|&code| code == irrelevant).unwrap());
        assert!(
            order.contains(&irrelevant),
            "zero-gain walls remain full-width"
        );
    }

    #[test]
    fn distance_two_final_edge_hit_without_a_goal_move_is_not_immediate_defense() {
        let config = Config {
            initial_stock: Players { a: 2, b: 0 },
            ..config(2)
        };
        let state = apply_codes(&config, &[81, 13, 75, 22, 76, 31, 75, 40, 76, 49, 75, 58]);
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let actions = compact_legal_codes(&config, &prepared);
        for code in actions.iter().filter(|&code| code < config.cells()) {
            let mut child = prepared.clone();
            child.apply_generated_code(&config, code).unwrap();
            assert!(
                generated_goal_pawn_codes(&config, &child).is_empty(),
                "the opponent has no immediate goal move after ordinary action {code}"
            );
        }

        let candidate_code = config.cells() + 7 * usize::from(config.columns - 1) + 3;
        assert!(
            actions.iter().any(|code| code == candidate_code),
            "review fixture keeps H-7-3 legal"
        );
        let mut never_stop = || false;
        let context = WallOrderingContext::new(&config, &prepared, &mut never_stop).unwrap();
        let features = context.features(&config, wall_from_code(&config, candidate_code));
        assert!(
            features.opponent_final_hits > 0,
            "the wall still has the geometric final-edge signal"
        );
        assert!(!features.blocks_immediate_goal);
        let hints = OrderingHints {
            canonical_mirrored: prepared.search_identity(&config).mirrored(),
            ..OrderingHints::default()
        };
        let score = Heuristics::new(8, config.policy_size()).score(
            &config,
            &prepared,
            Some(&context),
            candidate_code,
            hints,
        );
        assert!(
            score < IMMEDIATE_GOAL_DEFENSE_BASE,
            "H-7-3 must remain outside the verified immediate-defense band"
        );
    }

    #[test]
    fn near_goal_own_route_preservation_leads_self_blocking_zero_gain_walls() {
        let config = config(1);
        let state = apply_codes(&config, &[67, 3, 58, 2, 49, 1, 40, 0, 31, 9, 22, 0]);
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let actions = compact_legal_codes(&config, &prepared);
        let mut never_stop = || false;
        let context = WallOrderingContext::new(&config, &prepared, &mut never_stop).unwrap();
        assert_eq!(context.own.distance, 2);
        let safe = actions
            .iter()
            .filter(|&code| code >= config.cells())
            .find(|&code| {
                let features = context.features(&config, wall_from_code(&config, code));
                features.opponent_path_hits == 0
                    && features.own_path_hits == 0
                    && features.own_final_hits == 0
            })
            .expect("fixture exposes a safe zero-gain wall");
        let self_blocking = actions
            .iter()
            .filter(|&code| code >= config.cells() && code != safe)
            .find(|&code| {
                let features = context.features(&config, wall_from_code(&config, code));
                features.opponent_path_hits == 0 && features.own_final_hits > 0
            })
            .expect("fixture exposes a wall that obstructs the urgent own route");
        let order = ordered_codes(
            &Heuristics::new(8, config.policy_size()),
            &config,
            &prepared,
            &actions,
            None,
            0,
            None,
        );
        assert!(
            order.iter().position(|&code| code == safe).unwrap()
                < order
                    .iter()
                    .position(|&code| code == self_blocking)
                    .unwrap()
        );
        assert!(order.contains(&self_blocking));
    }

    #[test]
    fn tt_killer_counter_and_history_bands_keep_their_precedence() {
        let config = Config {
            initial_stock: Players { a: 1, b: 0 },
            ..config(1)
        };
        let state = apply_codes(
            &config,
            &[75, 13, 76, 22, 75, 31, 76, 40, 75, 49, 76, 58, 75, 67],
        );
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let actions = compact_legal_codes(&config, &prepared);
        let codes: Vec<_> = actions.iter().take(6).collect();
        assert_eq!(codes.len(), 6);
        let [tt, killer_one, killer_two, counter, history, ordinary] =
            <[usize; 6]>::try_from(codes).unwrap();
        let previous = ordinary;
        let mut heuristics = Heuristics::new(8, config.policy_size());
        heuristics.killers[0] = [Some(killer_one), Some(killer_two)];
        heuristics.counter[previous] = Some(counter);
        heuristics.history[Heuristics::player_index(prepared.position.turn)][history] = 500_000_000;
        let order = ordered_codes(
            &heuristics,
            &config,
            &prepared,
            &actions,
            Some(tt),
            0,
            Some(previous),
        );
        assert_eq!(&order[..4], &[tt, killer_one, killer_two, counter]);
        assert!(
            order.iter().position(|&code| code == history).unwrap()
                < order.iter().position(|&code| code == ordinary).unwrap()
        );
    }

    #[test]
    fn wall_edge_and_shortest_path_masks_are_exact() {
        let config = config(2);
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let board = prepared.position.board();
        let profile =
            ShortestPathProfile::new(&config, board, prepared.position.pawns.a, 0, &mut || false)
                .unwrap();
        assert_eq!(profile.distance, 8);
        for row in 0..8 {
            let expected = 1_u128 << (row * 9 + 4);
            assert_ne!(profile.shortest.down & expected, 0);
        }
        assert_eq!(profile.shortest.down.count_ones(), 8);
        assert_eq!(profile.shortest.right, 0);
        assert_eq!(profile.final_step.down, 1_u128 << 4);

        let horizontal = wall_edges(
            &config,
            Wall {
                orientation: Orientation::Horizontal,
                r: 3,
                c: 2,
            },
        );
        assert_eq!(horizontal.down, (1_u128 << 29) | (1_u128 << 30));
        assert_eq!(horizontal.right, 0);
        let vertical = wall_edges(
            &config,
            Wall {
                orientation: Orientation::Vertical,
                r: 3,
                c: 2,
            },
        );
        assert_eq!(vertical.down, 0);
        assert_eq!(vertical.right, (1_u128 << 29) | (1_u128 << 38));
    }

    #[test]
    fn packed_profiles_match_scalar_oracle_across_seeded_walled_states() {
        const SEEDS: [u64; 6] = [
            0x01a2_868e_95d4_6a8f,
            0x2ec3_f6a8_496b_d61d,
            0x51f4_2d72_9cb8_8015,
            0x76c8_ade4_b364_1f29,
            0xa823_17b9_6e4c_d053,
            0xd947_5c01_82fa_3be7,
        ];
        let config = config(10);
        let mut checked_profiles = 0_usize;
        let mut saw_branched_dag = false;
        let mut saw_mixed_orientation_dag = false;
        let mut saw_edge_pawn = [false; 2];

        for (seed_index, seed) in SEEDS.into_iter().enumerate() {
            let mut random = seed;
            let mut state = create_initial_state(&config).unwrap();
            for ply in 0..64 {
                if !state.outcome.is_ongoing() {
                    break;
                }
                let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
                let board = prepared.position.board();
                if board.walls.count() > 0 {
                    let mut never_stop = || false;
                    for player in [Player::A, Player::B] {
                        let from = *prepared.position.pawns.get(player);
                        let goal = *config.goal_rows.get(player);
                        let packed =
                            ShortestPathProfile::new(&config, board, from, goal, &mut never_stop)
                                .unwrap();
                        let scalar = scalar_profile(&config, board, from, goal);
                        assert_eq!(
                            packed, scalar,
                            "seed {seed:#x}, ply {ply}, player {player:?}"
                        );

                        let shortest_edges =
                            packed.shortest.down.count_ones() + packed.shortest.right.count_ones();
                        saw_branched_dag |= shortest_edges > u32::from(packed.distance);
                        saw_mixed_orientation_dag |=
                            packed.shortest.down != 0 && packed.shortest.right != 0;
                        let player_index = match player {
                            Player::A => 0,
                            Player::B => 1,
                        };
                        saw_edge_pawn[player_index] |= from.c == 0 || from.c == config.columns - 1;
                        checked_profiles += 1;
                    }
                }

                let legal = compact_legal_codes(&config, &prepared);
                let wall_codes: Vec<_> = legal
                    .iter()
                    .filter(|&code| code >= config.cells())
                    .collect();
                let pawn_codes: Vec<_> =
                    legal.iter().filter(|&code| code < config.cells()).collect();
                let roll = next_random(&mut random);
                let code = if ply < 8 && !wall_codes.is_empty() {
                    wall_codes[roll as usize % wall_codes.len()]
                } else if seed_index == 0 {
                    *pawn_codes
                        .iter()
                        .min_by_key(|&&code| {
                            (
                                code % usize::from(config.columns),
                                code / usize::from(config.columns),
                            )
                        })
                        .expect("an ongoing state has a legal pawn move")
                } else if !wall_codes.is_empty() && roll & 3 == 0 {
                    wall_codes[roll as usize % wall_codes.len()]
                } else {
                    pawn_codes[roll as usize % pawn_codes.len()]
                };
                state = apply_legal_action(&config, &state, &decode_action(&config, code).unwrap())
                    .unwrap();
            }
        }

        assert!(
            checked_profiles >= 100,
            "seed coverage exercises many walled profiles"
        );
        assert!(
            saw_branched_dag,
            "seed coverage includes a branched shortest DAG"
        );
        assert!(
            saw_mixed_orientation_dag,
            "seed coverage includes both edge orientations"
        );
        assert!(
            saw_edge_pawn.into_iter().all(|covered| covered),
            "seed coverage reaches a board edge for both players"
        );
    }

    #[test]
    fn interrupted_ordering_is_atomic_and_never_returns_a_partial_permutation() {
        let config = config(3);
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let before = prepared.clone();
        let actions = compact_legal_codes(&config, &prepared);
        let heuristics = Heuristics::new(8, config.policy_size());
        for stop_at in 1..=6 {
            let mut polls = 0;
            let result = heuristics.order(
                &config,
                &prepared,
                &actions,
                OrderingHints {
                    canonical_mirrored: prepared.search_identity(&config).mirrored(),
                    ..OrderingHints::default()
                },
                || {
                    polls += 1;
                    polls >= stop_at
                },
            );
            assert!(result.is_none(), "poll {stop_at} interrupts ordering");
            assert_eq!(prepared, before);
        }
        let completed = heuristics
            .order(
                &config,
                &prepared,
                &actions,
                OrderingHints {
                    canonical_mirrored: prepared.search_identity(&config).mirrored(),
                    ..OrderingHints::default()
                },
                || false,
            )
            .unwrap();
        assert_eq!(completed.len(), actions.len);
    }

    #[test]
    fn packed_profile_flood_budget_stays_below_five_percent_of_open_wall_validation() {
        // On an open board every anchor reaches both path-preservation floods.
        // The ordering profile adds two packed goal-distance floods total,
        // one per player; shortest-DAG source reachability then walks those
        // cached layers. Work is independent of wall count, with no per-wall
        // distance refinement. O(1) mask intersections are deliberately not
        // represented as flood-equivalent work.
        const SIZE: usize = 9;
        let anchors = 2 * (SIZE - 1) * (SIZE - 1);
        let legality_floods = 2 * anchors;
        assert!(
            PACKED_PROFILE_FLOODS_PER_NODE * 100 <= legality_floods * 5,
            "9x9: {} profile floods vs {legality_floods} legality floods",
            PACKED_PROFILE_FLOODS_PER_NODE
        );
    }

    fn legacy_location_order(
        heuristics: &Heuristics,
        config: &Config,
        state: &PreparedGameState,
        actions: &CodeList,
        hints: OrderingHints,
    ) -> Vec<usize> {
        heuristics
            .legacy_order(config, state, actions, hints)
            .into_iter()
            .map(|action| action.code)
            .collect()
    }

    fn visit_full_width_children(
        config: &Config,
        root: &PreparedGameState,
        order: impl IntoIterator<Item = usize>,
    ) {
        let mut state = root.clone();
        for code in order {
            let undo = state.apply_generated_code(config, code).unwrap();
            black_box(super::super::eval::evaluate(config, &state));
            assert!(state.undo_generated_code(undo));
        }
    }

    fn paired_wall_ordering_samples(iterations: usize, samples: usize) -> (Vec<f64>, Duration) {
        const WARMUP: usize = 20;
        let config = config(10);
        let state = create_initial_state(&config).unwrap();
        let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();
        let heuristics = Heuristics::new(8, config.policy_size());
        let canonical_mirrored = prepared.search_identity(&config).mirrored();
        let hints = OrderingHints {
            canonical_mirrored,
            ..OrderingHints::default()
        };
        let mut deadline_budget = DeadlineBudget::new(Duration::from_secs(24 * 60 * 60)).unwrap();
        let mut run_legacy = || {
            let actions = black_box(compact_legal_codes(&config, &prepared));
            let order = black_box(legacy_location_order(
                &heuristics,
                &config,
                &prepared,
                &actions,
                hints,
            ));
            visit_full_width_children(&config, &prepared, order);
        };
        let mut run_profiled = || {
            let actions = black_box(compact_legal_codes(&config, &prepared));
            let order = black_box(
                heuristics
                    .order(&config, &prepared, &actions, hints, || {
                        deadline_budget.exhausted(0)
                    })
                    .unwrap(),
            );
            visit_full_width_children(
                &config,
                &prepared,
                order.into_iter().map(|action| action.code),
            );
        };
        for _ in 0..WARMUP {
            run_legacy();
            run_profiled();
        }
        let mut shortest_window = Duration::MAX;
        let mut measure = |run: &mut dyn FnMut()| {
            let started = Instant::now();
            for _ in 0..iterations {
                run();
            }
            let elapsed = started.elapsed();
            shortest_window = shortest_window.min(elapsed);
            elapsed
        };
        let mut paired_ratios = Vec::with_capacity(samples);
        // Balance each pair as ABBA or BAAB. Averaging the two timings per
        // variant cancels first-order thermal/scheduler drift while the
        // alternating outer order avoids favoring one implementation.
        for sample in 0..samples {
            let (legacy, profiled) = if sample % 2 == 0 {
                let legacy_first = measure(&mut run_legacy);
                let profiled_first = measure(&mut run_profiled);
                let profiled_second = measure(&mut run_profiled);
                let legacy_second = measure(&mut run_legacy);
                (
                    legacy_first + legacy_second,
                    profiled_first + profiled_second,
                )
            } else {
                let profiled_first = measure(&mut run_profiled);
                let legacy_first = measure(&mut run_legacy);
                let legacy_second = measure(&mut run_legacy);
                let profiled_second = measure(&mut run_profiled);
                (
                    legacy_first + legacy_second,
                    profiled_first + profiled_second,
                )
            };
            paired_ratios.push(profiled.as_secs_f64() / legacy.as_secs_f64());
        }
        (paired_ratios, shortest_window)
    }

    #[test]
    #[ignore = "run in release mode for the dedicated wall-ordering overhead gate"]
    fn dedicated_wall_ordering_overhead_microbenchmark() {
        // Release calibration measured shortest timing windows of 272 ms,
        // 463 ms, and 969 ms at 5k, 10k, and 20k iterations respectively.
        // Twenty thousand is the first fixed candidate near one second,
        // limiting scheduler/timer noise without affecting routine tests
        // because this gate remains explicitly ignored outside release runs.
        const ITERATIONS: usize = 20_000;
        const SAMPLES: usize = 21;
        // For 21 independent paired samples, the 15th order statistic is a
        // one-sided 96.1% distribution-free upper confidence bound for the
        // population median: P(Binomial(21, 0.5) <= 14) = 0.9608.
        const MEDIAN_UPPER_BOUND_INDEX: usize = 14;
        let (mut paired_ratios, shortest_window) =
            paired_wall_ordering_samples(ITERATIONS, SAMPLES);
        paired_ratios.sort_by(f64::total_cmp);
        let median = paired_ratios[paired_ratios.len() / 2];
        let median_upper_bound = paired_ratios[MEDIAN_UPPER_BOUND_INDEX];
        let worst = *paired_ratios.last().unwrap();
        eprintln!(
            "9x9 paired median={median:.4} median_upper_96={median_upper_bound:.4} \
             worst={worst:.4} samples={SAMPLES} shortest_window={shortest_window:?}"
        );
        assert!(
            median_upper_bound <= 1.05,
            "9x9 96% upper confidence bound for median wall ordering overhead \
             {:.2}% exceeds 5% (median {:.2}%, worst pair {:.2}%)",
            (median_upper_bound - 1.0) * 100.0,
            (median - 1.0) * 100.0,
            (worst - 1.0) * 100.0
        );
    }
}
