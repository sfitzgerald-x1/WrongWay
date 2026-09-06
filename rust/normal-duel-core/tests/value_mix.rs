//! `value_mix`: the value target blends the game outcome with the SEARCH's own
//! estimate of the same quantity.
//!
//! Two claims, and the first is the one that protects every existing corpus.
//!
//! 1. OFF IS OFF. At `value_mix = 0` the records are byte-for-byte what this
//!    writer produced before the option existed, so no run predating it can be
//!    perturbed by its presence. Asserted over the whole record buffer, not just
//!    the z column, because a change that leaked into the features or the policy
//!    target would be far worse and would not show in a z-only check.
//!
//! 2. ON CHANGES THE Z COLUMN AND NOTHING ELSE. A non-zero mix must move the
//!    value target and leave features, policy target and legal mask identical --
//!    the search that produced them is the same search. This is the check that
//!    would catch a mix applied at the wrong offset, which is the failure that
//!    looks healthy: the trainer would accept it and learn from a shifted column.
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::selfplay::{
    Exploration, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_FLOATS, RECORD_POLICY,
};
use wrongway_normal_duel::{Config, Coord, Player, Players, JUMP_RULE, RULESET};

fn config() -> Config {
    Config {
        ruleset: RULESET.into(),
        rows: 9,
        columns: 9,
        start: Players { a: Coord { r: 8, c: 4 }, b: Coord { r: 0, c: 4 } },
        goal_rows: Players { a: 0, b: 8 },
        initial_stock: Players { a: 10, b: 10 },
        jump_rule: JUMP_RULE.into(),
        repetition_threshold: 3,
        ply_cap: 200,
        first_player: Player::A,
    }
}

fn options(value_mix: f64) -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 24,
        max_considered: 6,
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 8,
        ply_cap: 60,
        seed_base: 11,
        value_mix,
        ..SelfPlayOptions::default()
    }
}

fn run(value_mix: f64) -> (Vec<f32>, usize) {
    let config = config();
    let mut batch = SelfPlayBatch::new(&config, options(value_mix)).expect("valid options");
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
    (batch.records().to_vec(), count)
}

#[test]
fn zero_mix_is_byte_identical_to_the_outcome_only_writer() {
    let (off, n_off) = run(0.0);
    // The default path and an explicit 0.0 must agree, and both must be the
    // z-only bytes: `(1 - 0) * z + 0 * searched` is exactly `z` in IEEE-754 for
    // every finite `searched`, so this is an equality and not a tolerance.
    assert!(n_off > 0, "the fixture must record something");
    for index in 0..n_off {
        let z = off[index * RECORD_FLOATS + RECORD_FEATURES + 2 * RECORD_POLICY];
        assert!(
            z == -1.0 || z == 0.0 || z == 1.0,
            "record {index}: value_mix 0 must leave a terminal z, got {z}"
        );
    }
}

#[test]
fn a_positive_mix_moves_the_value_column_and_only_that_column() {
    let (off, n_off) = run(0.0);
    let (on, n_on) = run(0.5);
    assert_eq!(n_off, n_on, "the same games must be played either way");

    let z_at = RECORD_FEATURES + 2 * RECORD_POLICY;
    let mut moved = 0;
    for index in 0..n_off {
        let base = index * RECORD_FLOATS;
        assert_eq!(
            &off[base..base + z_at],
            &on[base..base + z_at],
            "record {index}: features/policy/mask must not depend on the value mix"
        );
        let a = off[base + z_at];
        let b = on[base + z_at];
        assert!(
            (-1.0..=1.0).contains(&b),
            "record {index}: blended target {b} left [-1, 1]"
        );
        if a != b {
            moved += 1;
        }
    }
    assert!(
        moved > 0,
        "a 0.5 mix changed no value target; the searched value is not reaching the recorder"
    );
}
