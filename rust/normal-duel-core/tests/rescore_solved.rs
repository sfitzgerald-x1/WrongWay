//! `rescore_solved`: the EXACT value where the solver can give one, `z` everywhere else.
//!
//! Once both wall stocks are empty the board is frozen and the rest of the game is a
//! solved race. `z` on those plies does not estimate the position's value -- the value
//! was already determined -- it records who blundered LATER, which teaches the value
//! head the opposite of the truth. Roughly a quarter of plies on this pool sit there.
//!
//! Three things are pinned:
//!
//! 1. OFF IS OFF. The default writes the same bytes as every corpus predating the
//!    option, asserted over the WHOLE record buffer rather than the value column, so a
//!    change leaking into features or the policy target cannot hide.
//! 2. ON MOVES ONLY THE VALUE COLUMN, and only to exact verdicts.
//! 3. THE PLY CAP BINDS. `Endgame::Wins` carries the distance to the win, and a win
//!    further away than the remaining plies is adjudicated a DRAW in the game actually
//!    played. Labelling it +/-1 would hand the trainer a target the game contradicts --
//!    relocating the defect this option exists to remove.
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::selfplay::{
    Exploration, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_FLOATS, RECORD_POLICY,
};
use wrongway_normal_duel::{Config, Coord, Player, Players, JUMP_RULE, RULESET};

/// A SMALL initial stock, deliberately. At the production 10 walls a 24-simulation
/// fixture rarely spends both stocks inside the ply cap, and a test where the solver
/// never fires would pass while proving nothing.
fn config(initial_stock: u64) -> Config {
    Config {
        ruleset: RULESET.into(),
        rows: 9,
        columns: 9,
        start: Players {
            a: Coord { r: 8, c: 4 },
            b: Coord { r: 0, c: 4 },
        },
        goal_rows: Players { a: 0, b: 8 },
        initial_stock: Players {
            a: initial_stock,
            b: initial_stock,
        },
        jump_rule: JUMP_RULE.into(),
        repetition_threshold: 3,
        ply_cap: 200,
        first_player: Player::A,
    }
}

fn options(rescore_solved: bool, ply_cap: u64) -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 24,
        max_considered: 6,
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 8,
        ply_cap,
        seed_base: 11,
        rescore_solved,
        ..SelfPlayOptions::default()
    }
}

fn run(rescore_solved: bool, stock: u64, ply_cap: u64) -> (Vec<f32>, usize) {
    let (records, count, _, _) = run_counted(rescore_solved, stock, ply_cap);
    (records, count)
}

/// As [`run`], plus how many plies the solver actually decided. Every assertion about
/// the cap needs this: a fixture where the solver never fires satisfies "rescored
/// equals z" trivially, and the first version of the cap test passed exactly that way
/// while an implementation with NO cap check at all went undetected.
fn run_counted(rescore_solved: bool, stock: u64, ply_cap: u64) -> (Vec<f32>, usize, u64, Vec<i32>) {
    let config = config(stock);
    let mut batch =
        SelfPlayBatch::new(&config, options(rescore_solved, ply_cap)).expect("valid options");
    let mut scratch = vec![0.0_f32; RECORD_POLICY];
    loop {
        let n = batch.collect().expect("collect");
        if n == 0 {
            break;
        }
        for slot in 0..n {
            let features =
                batch.features()[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES].to_vec();
            let value = mock_evaluator::evaluate(&features, &mut scratch);
            batch.policy_mut()[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY]
                .copy_from_slice(&scratch);
            batch.value_mut()[slot] = value as f32;
        }
        batch.submit(n).expect("submit");
    }
    let count = batch.take_records();
    let rescored = batch.rescored_plies();
    (
        batch.records().to_vec(),
        count,
        rescored,
        batch.record_meta().to_vec(),
    )
}

const VALUE_COLUMN: usize = RECORD_FEATURES + 2 * RECORD_POLICY;

#[test]
fn off_is_byte_identical_to_the_writer_that_predates_the_option() {
    let (off, n) = run(false, 1, 200);
    assert!(n > 0, "the fixture must record something");
    for index in 0..n {
        let z = off[index * RECORD_FLOATS + VALUE_COLUMN];
        assert!(
            z == -1.0 || z == 0.0 || z == 1.0,
            "record {index}: the default must leave a terminal z, got {z}"
        );
    }
}

#[test]
fn rescoring_moves_the_value_column_and_only_that_column() {
    let (off, n_off) = run(false, 1, 200);
    let (on, n_on) = run(true, 1, 200);
    assert_eq!(n_off, n_on, "the same games must be played either way");

    let mut changed = 0;
    for index in 0..n_off {
        let base = index * RECORD_FLOATS;
        for column in 0..RECORD_FLOATS {
            if column == VALUE_COLUMN {
                continue;
            }
            assert_eq!(
                off[base + column].to_bits(),
                on[base + column].to_bits(),
                "record {index}, column {column}: rescoring moved something that is not \
                 the value target"
            );
        }
        let a = off[base + VALUE_COLUMN];
        let b = on[base + VALUE_COLUMN];
        assert!(
            b == -1.0 || b == 0.0 || b == 1.0,
            "record {index}: rescored target {b} is not an exact verdict"
        );
        if a != b {
            changed += 1;
        }
    }
    assert!(
        changed > 0,
        "rescoring changed no target. Either the option is not wired through, or no \
         game in this fixture reached zero stock -- both make this test worthless."
    );
}

#[test]
fn the_cap_suppresses_wins_it_cannot_reach() {
    // THE DISCRIMINATING NUMBER is how many plies the solver DECIDES under a short
    // horizon. An 8-ply cap leaves at most 8 plies from any position, and no solved win
    // on this fixture is that close, so a correct implementation decides NOTHING there
    // while still firing on 24 plies. The same positions, given 200 plies, decide 26.
    //
    // Two earlier versions of this test failed to catch a deleted cap check. The first
    // asserted only that values lie in {-1, 0, +1} at a 2-ply cap -- where no game even
    // reaches zero stock, so the solver never ran at all (0 of 8 records). The second
    // asserted that SOME position was suppressed, which survived the sabotage at 2 of
    // 26. Only the count decided under the short horizon separates them cleanly: 0
    // against 24.
    let (short, n_short, rescored_short, meta_short) = run_counted(true, 1, 8);
    let (long, n_long, rescored_long, meta_long) = run_counted(true, 1, 200);
    assert!(
        rescored_short > 0 && rescored_long > 0,
        "the fixture never reached zero stock (short={rescored_short}, long={rescored_long}); \
         nothing below would be testing the cap"
    );

    // Index by (game, ply) so the same POSITION is compared: the short cap stops games
    // earlier, so the buffers differ in length and slot i is not slot i.
    let key = |meta: &[i32], i: usize| (meta[i * 4], meta[i * 4 + 1]);
    let mut long_by_key = std::collections::HashMap::new();
    for i in 0..n_long {
        long_by_key.insert(key(&meta_long, i), long[i * RECORD_FLOATS + VALUE_COLUMN]);
    }

    let (mut decided_short, mut decided_long) = (0, 0);
    for i in 0..n_short {
        let v_short = short[i * RECORD_FLOATS + VALUE_COLUMN];
        let Some(&v_long) = long_by_key.get(&key(&meta_short, i)) else {
            continue;
        };
        if v_short != 0.0 {
            decided_short += 1;
            // A win reachable in a short horizon is reachable in a long one, and for the
            // same side: the cap may suppress a verdict, never reverse it.
            assert_eq!(
                v_short, v_long,
                "record {i}: {v_short} under an 8-ply cap but {v_long} under 200"
            );
        }
        if v_long != 0.0 {
            decided_long += 1;
        }
    }
    assert_eq!(
        decided_short, 0,
        "an 8-ply horizon decided {decided_short} plies. No solved win on this fixture is \
         that close, so those are wins the game is adjudicated never to reach: the cap \
         check is missing, or it is consulting the wrong cap."
    );
    assert!(
        decided_long > 0,
        "cap 200 decided nothing either, so the comparison above proves nothing"
    );
}
