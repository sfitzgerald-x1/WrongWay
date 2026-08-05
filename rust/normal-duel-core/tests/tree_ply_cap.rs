//! The search tree's ply-cap arm, exercised where it can actually be observed.
//!
//! `create_child` re-derives adjudication (goal, then repetition, then ply cap)
//! rather than asking the engine, and the cap it reads is `config.ply_cap`. Two
//! earlier attempts to cover the cap arm did not:
//!
//! 1. Setting `SelfPlayOptions::ply_cap` moves the *outer loop's* stop, not the
//!    tree's. `config.ply_cap` stayed at 200 and a 64-simulation tree rooted
//!    below ply 12 cannot descend to 200.
//! 2. Moving the cap onto the config and asserting via `outcomes()` /
//!    `plies_played()` that a game draws at the cap does not discriminate
//!    either: the game loop adjudicates `config.ply_cap` independently
//!    (`selfplay.rs`), so those outcomes are produced with or without the
//!    tree's arm.
//!
//! Both passed with `create_child`'s cap branch stubbed to `false`, which is the
//! only check that means anything here.
//!
//! What the arm uniquely controls is whether a child *needs the network*. A
//! child at the cap is terminal with value 0, so it is never handed out as a
//! leaf. Rooted one ply below the cap, every child is terminal: the tree asks
//! for exactly one evaluation, the root's, and then finishes. Without the arm
//! each child becomes an ordinary leaf and the count jumps to the budget. That
//! is observable from outside, and it is what this test measures.

use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::puct::{PuctParams, PuctTreeSearch, RootContext};
use wrongway_normal_duel::selfplay::{RECORD_FEATURES, RECORD_POLICY};
use wrongway_normal_duel::{
    apply_legal_action, create_initial_state, legal_actions, Config, Coord, GameState, Player,
    Players, JUMP_RULE, RULESET,
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
        initial_stock: Players { a: 10, b: 10 },
        jump_rule: JUMP_RULE.into(),
        repetition_threshold: 3,
        ply_cap,
        first_player: Player::A,
    }
}

/// Walk `plies` wall placements so the position advances without either pawn
/// approaching a goal — no game may end for any reason but the cap.
fn advance(config: &Config, plies: u64) -> GameState {
    let mut state = create_initial_state(config).expect("initial state");
    let mut seed = 0x2545_f491_u32;
    for _ in 0..plies {
        let actions = legal_actions(config, &state).expect("legal actions");
        let walls: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, wrongway_normal_duel::Action::Wall { .. }))
            .collect();
        assert!(!walls.is_empty(), "no wall available");
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let action = walls[(seed >> 8) as usize % walls.len()];
        state = apply_legal_action(config, &state, action).expect("apply");
        assert!(state.outcome.is_ongoing(), "the walk ended the game early");
    }
    state
}

/// Count the leaf evaluations a search requests, driving it to completion.
fn leaf_evaluations(config: &Config, state: &GameState, simulations: u32) -> usize {
    let params = PuctParams {
        simulations,
        max_considered: 8,
        c_puct: 1.25,
    };
    let root = RootContext::from_state(config, state).expect("root context");
    let mut tree =
        PuctTreeSearch::new(config, root, params, Lcg32::new(99)).expect("tree constructs");

    let mut features = vec![0.0_f32; RECORD_FEATURES];
    let mut policy = vec![0.0_f32; RECORD_POLICY];
    let mut evaluations = 0;
    while !tree.is_done() {
        if !tree.next_leaf(config, &mut features).expect("next_leaf") {
            break;
        }
        // A flat non-zero value: if a capped child were evaluated instead of
        // scored as a draw, this is what would flow up the tree.
        for slot in policy.iter_mut() {
            *slot = 1.0;
        }
        tree.submit(config, &policy, 0.75).expect("submit");
        evaluations += 1;
    }
    evaluations
}

#[test]
fn a_root_one_ply_below_the_cap_needs_only_the_root_evaluated() {
    const CAP: u64 = 14;
    let capped = config(CAP);
    let state = advance(&capped, CAP - 1);
    assert_eq!(state.ply, CAP - 1);
    assert!(state.outcome.is_ongoing());

    // Every child sits at exactly `CAP`, so the cap arm makes all of them
    // terminal draws and none is ever handed out as a leaf.
    let capped_evaluations = leaf_evaluations(&capped, &state, 64);
    assert_eq!(
        capped_evaluations, 1,
        "expected only the root to be evaluated, got {capped_evaluations}"
    );

    // The same walk under a distant cap: the children are ordinary leaves, so
    // the search spends its budget. This is the control that proves the
    // assertion above is about the cap and not about the position. The state is
    // rebuilt under the roomy config rather than reused, because a `GameState`
    // is only valid against the config it was produced with.
    let roomy = config(200);
    let roomy_state = advance(&roomy, CAP - 1);
    assert_eq!(
        roomy_state.position, state.position,
        "the two walks diverged"
    );
    let roomy_evaluations = leaf_evaluations(&roomy, &roomy_state, 64);
    assert!(
        roomy_evaluations > 8,
        "control run evaluated only {roomy_evaluations} leaves; the comparison is meaningless"
    );
}
