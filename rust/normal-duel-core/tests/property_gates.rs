//! Fixed-seed randomized regression gates for the mutable search representation
//! and the public normal-duel-v1 rules API.
//!
//! These tests intentionally use a tiny in-tree LCG rather than a property-test
//! crate. That keeps the cases reproducible on native and `wasm32` toolchains,
//! while still exercising a varied set of reachable 7x7 and 9x9 positions.

use wrongway_normal_duel::{
    adjudicate, apply_action, canonical_zobrist_key, create_initial_state, fast_perft_with_options,
    has_path, legal_action_codes, legal_actions, legal_position_actions, normalize_position,
    perft_with_options, Action, Config, Coord, DrawReason, GameState, GoalReason, NormalDuelError,
    Outcome, PerftOptions, Player, Players, Position, PreparedGameState, SearchPosition, Wall,
    JUMP_RULE, MAX_PERFT_NODES_HARD_CAP, REPETITION_THRESHOLD, RULESET,
};

const SEEDS: [u32; 4] = [0, 1, 0x1234_5678, u32::MAX];
const STEPS_PER_SEED: usize = 10;

fn config(size: u8, first_player: Player) -> Config {
    Config {
        ruleset: RULESET.to_owned(),
        rows: size,
        columns: size,
        start: Players {
            a: Coord {
                r: size - 1,
                c: size / 2,
            },
            b: Coord { r: 0, c: size / 2 },
        },
        goal_rows: Players { a: 0, b: size - 1 },
        initial_stock: Players { a: 10, b: 10 },
        jump_rule: JUMP_RULE.to_owned(),
        repetition_threshold: REPETITION_THRESHOLD,
        ply_cap: 200,
        first_player,
    }
}

fn next_lcg32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn choose<'a, T>(items: &'a [T], random: &mut u32) -> &'a T {
    assert!(!items.is_empty(), "cannot select from an empty action list");
    &items[(next_lcg32(random) as usize) % items.len()]
}

fn mirror_coord(config: &Config, coord: Coord) -> Coord {
    Coord {
        r: coord.r,
        c: config.columns - 1 - coord.c,
    }
}

fn mirror_action(config: &Config, action: &Action) -> Action {
    match action {
        Action::Pawn { to } => Action::Pawn {
            to: mirror_coord(config, *to),
        },
        Action::Wall { wall } => {
            let wall = Wall::parse(config, wall).expect("legal action walls parse");
            Action::Wall {
                wall: Wall {
                    orientation: wall.orientation,
                    r: wall.r,
                    c: config.columns - 2 - wall.c,
                }
                .text(),
            }
        }
    }
}

fn mirror_position(config: &Config, position: &Position) -> Position {
    Position {
        pawns: Players {
            a: mirror_coord(config, position.pawns.a),
            b: mirror_coord(config, position.pawns.b),
        },
        walls: position
            .walls
            .iter()
            .map(
                |text| match mirror_action(config, &Action::Wall { wall: text.clone() }) {
                    Action::Wall { wall } => wall,
                    Action::Pawn { .. } => unreachable!("wall mirrors to a wall"),
                },
            )
            .collect(),
        stock: position.stock,
        turn: position.turn,
    }
}

fn assert_legal_codes_and_wall_paths(config: &Config, state: &GameState) {
    let codes = legal_action_codes(config, state).expect("reachable ongoing state has legal moves");
    assert!(
        !codes.is_empty(),
        "ongoing state must expose a legal action"
    );
    assert!(
        codes.windows(2).all(|pair| pair[0] < pair[1]),
        "legal action codes must be sorted and unique"
    );
    assert_eq!(
        PreparedGameState::from_game_state(config, state)
            .unwrap()
            .legal_action_codes(config)
            .unwrap(),
        codes,
        "validated-once compact legal codes match the immutable engine"
    );
    assert_eq!(
        fast_perft_with_options(config, state, 1, PerftOptions::default()).unwrap(),
        codes.len() as u64,
        "compact depth-one perft matches the immutable legal count"
    );
    let depth_two_options = PerftOptions {
        max_nodes: MAX_PERFT_NODES_HARD_CAP,
    };
    assert_eq!(
        fast_perft_with_options(config, state, 2, depth_two_options).unwrap(),
        perft_with_options(config, state, 2, depth_two_options).unwrap(),
        "compact depth-two traversal matches immutable transitions"
    );

    for action in legal_actions(config, state)
        .expect("reachable ongoing state has legal actions")
        .into_iter()
        .filter(|action| matches!(action, Action::Wall { .. }))
    {
        let Action::Wall { wall } = action else {
            unreachable!("filter retains only wall actions");
        };
        let mut walls = state.position.walls.clone();
        walls.push(wall);
        assert!(
            has_path(config, state.position.pawns.a, config.goal_rows.a, &walls)
                .expect("legal wall path check accepts the position"),
            "every legal wall keeps player A's goal reachable"
        );
        assert!(
            has_path(config, state.position.pawns.b, config.goal_rows.b, &walls)
                .expect("legal wall path check accepts the position"),
            "every legal wall keeps player B's goal reachable"
        );
    }
}

#[test]
fn fixed_seed_reachable_states_preserve_search_undo_legality_and_paths() {
    for size in [7, 9] {
        for first_player in [Player::A, Player::B] {
            let config = config(size, first_player);
            for initial_seed in SEEDS {
                let mut random = initial_seed;
                let mut state = create_initial_state(&config).expect("valid initial state");
                for step in 0..STEPS_PER_SEED {
                    assert!(
                        state.outcome.is_ongoing(),
                        "short fixed trace remains ongoing"
                    );
                    assert_legal_codes_and_wall_paths(&config, &state);

                    let legal = legal_actions(&config, &state).expect("reachable state is legal");
                    let walls = legal
                        .iter()
                        .filter(|action| matches!(action, Action::Wall { .. }))
                        .cloned()
                        .collect::<Vec<_>>();
                    let pawns = legal
                        .iter()
                        .filter(|action| matches!(action, Action::Pawn { .. }))
                        .cloned()
                        .collect::<Vec<_>>();
                    let action = if step % 2 == 0 && !walls.is_empty() {
                        choose(&walls, &mut random).clone()
                    } else {
                        choose(&pawns, &mut random).clone()
                    };

                    let mut search = SearchPosition::from_position(&config, &state.position)
                        .expect("reachable position initializes the search state");
                    let before = search.to_position(&config).unwrap();
                    let undo = search
                        .apply_unchecked(&config, &action)
                        .expect("publicly legal action is search-applicable");
                    search.undo(&config, undo);
                    assert_eq!(
                        search.to_position(&config).unwrap(),
                        before,
                        "apply/undo must exactly restore seed {initial_seed:#010x}, {size}x{size}, {first_player:?}, step {step}"
                    );

                    state = apply_action(&config, &state, &action)
                        .expect("selected legal action applies through the checked API");
                }
            }
        }
    }
}

#[test]
fn left_right_mirror_is_an_involution_with_equivalent_actions_and_canonical_hash() {
    for size in [7, 9] {
        let config = config(size, Player::A);
        for initial_seed in SEEDS {
            let mut random = initial_seed;
            let mut state = create_initial_state(&config).expect("valid initial state");
            for step in 0..STEPS_PER_SEED {
                let original = normalize_position(&config, &state.position).unwrap();
                let mirrored = normalize_position(&config, &mirror_position(&config, &original))
                    .expect("reflection preserves a legal position");
                assert_eq!(
                    normalize_position(&config, &mirror_position(&config, &mirrored)).unwrap(),
                    original,
                    "left/right reflection must be an involution"
                );
                assert_eq!(
                    canonical_zobrist_key(&config, &original),
                    canonical_zobrist_key(&config, &mirrored),
                    "reflection shares the canonical search key"
                );

                let mirrored_codes = legal_position_actions(&config, &mirrored)
                    .unwrap()
                    .iter()
                    .map(|action| action.policy_code(&config).unwrap())
                    .collect::<Vec<_>>();
                let mut reflected_codes = legal_position_actions(&config, &original)
                    .unwrap()
                    .iter()
                    .map(|action| mirror_action(&config, action).policy_code(&config).unwrap())
                    .collect::<Vec<_>>();
                reflected_codes.sort_unstable();
                assert_eq!(
                    reflected_codes, mirrored_codes,
                    "reflection maps the complete canonical legal action list"
                );

                let legal = legal_actions(&config, &state).unwrap();
                let action = choose(&legal, &mut random).clone();
                state = apply_action(&config, &state, &action).unwrap();
                assert!(
                    state.outcome.is_ongoing(),
                    "short fixed trace remains ongoing at seed {initial_seed:#010x}, step {step}"
                );
            }
        }
    }
}

#[test]
fn adjudication_order_terminal_rejection_and_first_player_b_are_stable() {
    let mut cap_config = config(7, Player::B);
    cap_config.ply_cap = 2;

    let initial = create_initial_state(&cap_config).expect("valid B-first initial state");
    assert_eq!(initial.position.turn, Player::B);
    assert_eq!(cap_config.expected_turn(0), Player::B);
    assert_eq!(cap_config.expected_turn(1), Player::A);
    let opening_action = legal_actions(&cap_config, &initial)
        .unwrap()
        .into_iter()
        .next()
        .expect("initial board has a legal action");
    let after_opening = apply_action(&cap_config, &initial, &opening_action).unwrap();
    assert_eq!(after_opening.position.turn, Player::A);
    assert_eq!(after_opening.ply, 1);

    assert_eq!(
        adjudicate(&cap_config, Some(Player::A), 3, cap_config.ply_cap),
        Ok(Outcome::Win {
            winner: Player::A,
            reason: GoalReason::Goal,
        }),
        "a goal outranks repetition and the ply cap"
    );
    assert_eq!(
        adjudicate(&cap_config, None, 3, cap_config.ply_cap),
        Ok(Outcome::Draw {
            reason: DrawReason::ThreefoldRepetition,
        }),
        "threefold outranks the ply cap"
    );
    assert_eq!(
        adjudicate(&cap_config, None, 2, cap_config.ply_cap),
        Ok(Outcome::Draw {
            reason: DrawReason::PlyCap,
        })
    );

    let mut terminal = initial.clone();
    terminal.outcome = Outcome::Win {
        winner: Player::B,
        reason: GoalReason::Goal,
    };
    assert_eq!(
        apply_action(&cap_config, &terminal, &opening_action),
        Err(NormalDuelError::TerminalState),
        "terminal states reject every action before transition validation"
    );
}
