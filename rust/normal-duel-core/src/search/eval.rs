use crate::{compact_pawn_codes, Board, CodeList, Config, Player, PreparedGameState};

pub(crate) const MATE_SCORE: i32 = 1_000_000;
pub(crate) const EVAL_LIMIT: i32 = MATE_SCORE / 2;

const DISTANCE_WEIGHT: i32 = 100;
const STOCK_WEIGHT: i32 = 12;
const TEMPO_WEIGHT: i32 = 6;
const ROBUSTNESS_WEIGHT: i32 = 8;

fn player_index(player: Player) -> usize {
    match player {
        Player::A => 0,
        Player::B => 1,
    }
}

pub(crate) fn distances(config: &Config, state: &PreparedGameState) -> [u8; 2] {
    let board = state.position.board();
    [
        board
            .shortest_distance(config, state.position.pawns.a, config.goal_rows.a)
            .expect("legal states preserve an A path"),
        board
            .shortest_distance(config, state.position.pawns.b, config.goal_rows.b)
            .expect("legal states preserve a B path"),
    ]
}

fn alternate_path_robustness(
    config: &Config,
    state: &PreparedGameState,
    board: Board,
    player: Player,
) -> i32 {
    let mut position = state.position;
    position.turn = player;
    let mut codes = CodeList::new();
    compact_pawn_codes(config, &position, board, &mut codes);
    let goal = *config.goal_rows.get(player);
    let mut child_distances = [u8::MAX; 6];
    let mut count = 0_usize;
    for code in codes.iter().take(child_distances.len()) {
        let destination = crate::Coord {
            r: (code / usize::from(config.columns)) as u8,
            c: (code % usize::from(config.columns)) as u8,
        };
        if let Some(distance) = board.shortest_distance(config, destination, goal) {
            child_distances[count] = distance;
            count += 1;
        }
    }
    if count == 0 {
        return 0;
    }
    let best = *child_distances[..count]
        .iter()
        .min()
        .expect("non-empty child distances");
    child_distances[..count]
        .iter()
        .filter(|&&distance| distance <= best.saturating_add(1))
        .count() as i32
}

/// Evaluation v1, always from the current side-to-move perspective.
///
/// Only validated features are included: wall-only shortest path, remaining
/// stock, tempo, and near-best alternate pawn exits. Wall mobility and chain
/// bonuses remain ordering features until match evidence justifies them.
pub(crate) fn evaluate(config: &Config, state: &PreparedGameState) -> i32 {
    let perspective = state.position.turn;
    let opponent = perspective.other();
    let board = state.position.board();
    let distance = distances(config, state);
    let own_distance = i32::from(distance[player_index(perspective)]);
    let opponent_distance = i32::from(distance[player_index(opponent)]);
    let own_stock = *state.position.stock.get(perspective) as i128;
    let opponent_stock = *state.position.stock.get(opponent) as i128;
    let stock_delta = own_stock - opponent_stock;
    let robustness = alternate_path_robustness(config, state, board, perspective)
        - alternate_path_robustness(config, state, board, opponent);
    (i128::from(opponent_distance - own_distance) * i128::from(DISTANCE_WEIGHT)
        + stock_delta * i128::from(STOCK_WEIGHT)
        + i128::from(TEMPO_WEIGHT)
        + i128::from(robustness) * i128::from(ROBUSTNESS_WEIGHT))
    .clamp(i128::from(-EVAL_LIMIT), i128::from(EVAL_LIMIT)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        create_initial_state, Coord, Players, JUMP_RULE, MAX_JS_SAFE_INTEGER, REPETITION_THRESHOLD,
        RULESET,
    };

    fn config(initial_stock: Players<u64>) -> Config {
        Config {
            ruleset: RULESET.into(),
            rows: 9,
            columns: 9,
            start: Players {
                a: Coord { r: 8, c: 4 },
                b: Coord { r: 0, c: 4 },
            },
            goal_rows: Players { a: 0, b: 8 },
            initial_stock,
            jump_rule: JUMP_RULE.into(),
            repetition_threshold: REPETITION_THRESHOLD,
            ply_cap: 200,
            first_player: Player::A,
        }
    }

    #[test]
    fn max_safe_integer_stock_asymmetry_clamps_for_both_signs() {
        for (initial_stock, expected) in [
            (
                Players {
                    a: MAX_JS_SAFE_INTEGER,
                    b: 0,
                },
                EVAL_LIMIT,
            ),
            (
                Players {
                    a: 0,
                    b: MAX_JS_SAFE_INTEGER,
                },
                -EVAL_LIMIT,
            ),
        ] {
            let config = config(initial_stock);
            assert!(config.validate().is_ok());
            let state = create_initial_state(&config).unwrap();
            let prepared = PreparedGameState::from_game_state(&config, &state).unwrap();

            assert_eq!(evaluate(&config, &prepared), expected);
        }
    }
}
