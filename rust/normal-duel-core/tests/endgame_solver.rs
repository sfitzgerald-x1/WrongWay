//! The exact zero-stock solver, checked against actual play.
//!
//! A solver is the kind of component that is confidently wrong: it returns a clean
//! verdict for every position and nothing about a wrong one looks wrong. So the
//! load-bearing test does not inspect the table, it PLAYS the line the table
//! recommends and requires the game to end the way the table said, in the number of
//! plies the table said. If move generation here ever drifts from the engine's --
//! the jump rule being the obvious candidate -- the rollout diverges and this fails.
use wrongway_normal_duel::endgame::{solve_layout, Endgame};
use wrongway_normal_duel::{
    apply_legal_action, decode_action, Action, Config, Coord,
    create_initial_state, legal_actions, GameState, Player, Players,
};

#[path = "common/state_pool.rs"]
mod state_pool;

fn config() -> Config {
    state_pool::canonical_config()
}

/// Play random legal moves, preferring wall placements, until both stocks are
/// spent and the game is still running.
///
/// Hand-building a zero-stock state does not work: stock 0/0 means twenty walls
/// are on the board, and the engine rejects a state whose stock and wall list
/// disagree. Reaching one by play also means the layout is one the game can
/// actually produce, which a hand-picked twenty walls might not be.
fn played_to_zero_stock(config: &Config, seed: u32) -> Option<GameState> {
    let mut random = seed.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
    let mut next = || { random ^= random << 13; random ^= random >> 17; random ^= random << 5; random };
    let mut state = create_initial_state(config).ok()?;
    for _ in 0..400 {
        if !state.outcome.is_ongoing() { return None; }
        if state.position.stock.a == 0 && state.position.stock.b == 0 {
            return Some(state);
        }
        let actions = legal_actions(config, &state).ok()?;
        if actions.is_empty() { return None; }
        // Prefer walls: the point is to exhaust stock before someone crosses.
        let walls: Vec<&Action> = actions.iter().filter(|a| matches!(a, Action::Wall { .. })).collect();
        let action = if !walls.is_empty() && next() % 10 < 8 {
            walls[(next() as usize) % walls.len()].clone()
        } else {
            actions[(next() as usize) % actions.len()].clone()
        };
        state = apply_legal_action(config, &state, &action).ok()?;
    }
    None
}

#[test]
fn a_race_with_no_interaction_is_plain_arithmetic() {
    // Pawns on opposite corners never meet, so the result is exactly what counting
    // moves says: eight each, and whoever moves first arrives first.
    let config = config();
    let table = solve_layout(&config, &[]).expect("solves");
    assert_eq!(table.len(), config.cells() * config.cells() * 2);
    let a = Coord { r: 8, c: 0 };
    let b = Coord { r: 0, c: 8 };
    for (turn, expected) in [(Player::A, Player::A), (Player::B, Player::B)] {
        match table.lookup(&config, Players { a, b }, turn).expect("on board") {
            Endgame::Wins { player, plies } => {
                assert_eq!(player, expected, "whoever moves first should win an open race");
                assert_eq!(plies, 15, "eight moves interleaved with seven");
            }
            Endgame::Draw => panic!("an open race is not a draw"),
        }
    }
}

#[test]
fn pawn_interaction_beats_the_distance_count() {
    // THE reason this is a solver and not a distance formula. These pawns are the
    // same eight moves from their goals as the corner case above, and A moves
    // first -- but they share a file, so they collide in the middle and the hop
    // rule costs A a tempo. Counting moves says A wins in 15; the game says B wins
    // in 16. Any endgame shortcut built on shortest paths gets this backwards.
    let config = config();
    let table = solve_layout(&config, &[]).expect("solves");
    let same_file = table
        .lookup(&config, Players { a: Coord { r: 8, c: 4 }, b: Coord { r: 0, c: 4 } }, Player::A)
        .expect("on board");
    assert_eq!(
        same_file,
        Endgame::Wins { player: Player::B, plies: 16 },
        "sharing a file must change the result; if this ever reads A in 15 the \
         solver has been replaced by distance counting"
    );
}

#[test]
fn playing_the_solved_line_ends_exactly_as_the_table_says() {
    // The check that catches a solver whose move generation has drifted from the
    // engine's: replay its own recommendation and require the reported winner in
    // the reported number of plies. A table can be confidently wrong and look
    // entirely normal; a rollout cannot.
    let config = config();
    let mut checked = 0;
    for seed in 1..40_u32 {
        let Some(start) = played_to_zero_stock(&config, seed) else { continue };
        let walls = start.position.walls.clone();
        let table = solve_layout(&config, &walls).expect("solves");
        let Some(Endgame::Wins { player, plies }) =
            table.lookup(&config, start.position.pawns, start.position.turn) else { continue };

        let mut game = start;
        let mut played = 0_u32;
        let winner = loop {
            if game.position.pawns.a.r == config.goal_rows.a { break Player::A; }
            if game.position.pawns.b.r == config.goal_rows.b { break Player::B; }
            assert!(played <= plies, "seed {seed}: the line ran past the promised {plies} plies");
            let code = table
                .best_move(&config, &walls, game.position.pawns, game.position.turn)
                .expect("a solved position has a move");
            let action = decode_action(&config, usize::from(code)).expect("legal code");
            assert!(matches!(action, Action::Pawn { .. }), "zero stock leaves only pawn moves");
            game = apply_legal_action(&config, &game, &action)
                .expect("the solver's move must be legal");
            played += 1;
        };
        assert_eq!(winner, player, "seed {seed}: rollout produced a different winner");
        assert_eq!(played, plies, "seed {seed}: rollout took a different number of plies");
        checked += 1;
        if checked >= 8 { break; }
    }
    assert!(checked >= 5, "only {checked} rollouts ran; the generator is not reaching zero stock");
}
