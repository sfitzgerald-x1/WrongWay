//! Deterministic pool of varied reachable states, shared by the hot-path
//! parity tests and the hot-path benchmark.
//!
//! Trajectories differ in how strongly they prefer wall placements, so the
//! pool spans pawn-only openings through wall-dense mid-games. Wall density is
//! where a wrong shortest-path prefilter shows up first: with few walls almost
//! every candidate is legal regardless.

#![allow(dead_code)]

use wrongway_normal_duel::{
    apply_legal_action, create_initial_state, legal_actions, Action, Config, Coord, GameState,
    Player, Players, JUMP_RULE, REPETITION_THRESHOLD, RULESET,
};

pub fn canonical_config() -> Config {
    Config {
        ruleset: RULESET.to_owned(),
        rows: 9,
        columns: 9,
        start: Players {
            a: Coord { r: 8, c: 4 },
            b: Coord { r: 0, c: 4 },
        },
        goal_rows: Players { a: 0, b: 8 },
        initial_stock: Players { a: 10, b: 10 },
        jump_rule: JUMP_RULE.to_owned(),
        repetition_threshold: REPETITION_THRESHOLD,
        ply_cap: 200,
        first_player: Player::A,
    }
}

fn next_lcg32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// `count` ongoing states, deterministic for a given `count`.
pub fn state_pool(config: &Config, count: usize) -> Vec<GameState> {
    let mut pool: Vec<GameState> = Vec::with_capacity(count);
    let mut trajectory = 0_u32;
    while pool.len() < count {
        trajectory += 1;
        let mut random = trajectory.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
        // 0, 25, 50, 75 and 95 percent wall preference across trajectories.
        let wall_bias = [0_u32, 25, 50, 75, 95][(trajectory % 5) as usize];
        let length = 4 + (trajectory % 41) as usize;
        let Ok(mut state) = create_initial_state(config) else {
            break;
        };
        for ply in 0..length {
            if !state.outcome.is_ongoing() {
                break;
            }
            let Ok(actions) = legal_actions(config, &state) else {
                break;
            };
            if actions.is_empty() {
                break;
            }
            let prefer_wall = (next_lcg32(&mut random) >> 8) % 100 < wall_bias;
            let matches_preference =
                |action: &Action| matches!(action, Action::Wall { .. }) == prefer_wall;
            let available = actions.iter().filter(|a| matches_preference(a)).count();
            let action = if available == 0 {
                &actions[(next_lcg32(&mut random) >> 8) as usize % actions.len()]
            } else {
                let index = (next_lcg32(&mut random) >> 8) as usize % available;
                actions
                    .iter()
                    .filter(|a| matches_preference(a))
                    .nth(index)
                    .expect("index is below the filtered count")
            };
            let Ok(next) = apply_legal_action(config, &state, action) else {
                break;
            };
            state = next;
            // Skip the first few plies: an empty board exercises no filtering.
            if ply >= 2 && state.outcome.is_ongoing() {
                pool.push(state.clone());
                if pool.len() == count {
                    break;
                }
            }
        }
    }
    pool
}
