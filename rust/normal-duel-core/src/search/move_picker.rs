use crate::{CodeList, Config, Player, PreparedGameState};

const TT_BONUS: i32 = 1_000_000_000;
const KILLER_ONE_BONUS: i32 = 800_000_000;
const KILLER_TWO_BONUS: i32 = 700_000_000;
const COUNTER_BONUS: i32 = 600_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OrderedAction {
    pub(crate) index: usize,
    pub(crate) code: usize,
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

    fn score(
        &self,
        config: &Config,
        state: &PreparedGameState,
        code: usize,
        tt_action: Option<usize>,
        ply: usize,
        previous_action: Option<usize>,
    ) -> i32 {
        if Some(code) == tt_action {
            return TT_BONUS;
        }
        let killers = self.killers.get(ply).copied().unwrap_or([None, None]);
        if Some(code) == killers[0] {
            return KILLER_ONE_BONUS;
        }
        if Some(code) == killers[1] {
            return KILLER_TWO_BONUS;
        }
        if previous_action
            .and_then(|previous| self.counter.get(previous))
            .copied()
            .flatten()
            == Some(code)
        {
            return COUNTER_BONUS;
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

        // Walls remain full-width. This inexpensive score only changes their
        // order, preferring anchors near the opponent and central files.
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

    pub(crate) fn order(
        &self,
        config: &Config,
        state: &PreparedGameState,
        actions: &CodeList,
        tt_action: Option<usize>,
        ply: usize,
        previous_action: Option<usize>,
    ) -> Vec<OrderedAction> {
        let mut ordered: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(index, code)| {
                (
                    OrderedAction { index, code },
                    self.score(config, state, code, tt_action, ply, previous_action),
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
