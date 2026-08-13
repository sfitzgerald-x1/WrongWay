//! `subtree_visits_after`, the read-only diagnostic behind the tree-reuse question.
//!
//! It reports what a NEXT search would inherit if it reused this tree instead of
//! rebuilding from the new position. The whole decision about whether to build tree
//! reuse rests on the distribution this produces, so a diagnostic that silently
//! over-reports would send a week of work in the wrong direction.
//!
//! The three cases it must keep apart:
//!
//!   never expanded         -> None   (the path is not in the tree at all)
//!   expanded, unvisited    -> Some(0) (the move was scored but never descended into)
//!   visited                -> Some(n)
//!
//! Collapsing the middle case into `None` would understate coverage; collapsing it
//! into a positive count would flatter reuse. Both are tested, plus the invariant that
//! matters most for the reuse argument: the visits under the root's children cannot
//! exceed the root's own visits.

use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::puct::{PuctParams, PuctTreeSearch, RootContext};
use wrongway_normal_duel::selfplay::{RECORD_FEATURES, RECORD_POLICY};
use wrongway_normal_duel::{
    create_initial_state, legal_action_codes, Config, Coord, Player, Players, JUMP_RULE, RULESET,
};

fn standard_config() -> Config {
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
        ply_cap: 200,
        first_player: Player::A,
    }
}

/// Run a search to completion against a deterministic stand-in evaluator.
///
/// The policy is deliberately NOT uniform: a uniform prior spreads visits evenly and
/// would make the inheritance look artificially flat, which is the very quantity under
/// test. Varying it by code concentrates the tree the way a real network does.
fn run(simulations: u32, max_considered: u32) -> (PuctTreeSearch, Config) {
    let config = standard_config();
    let state = create_initial_state(&config).expect("initial state");
    let params = PuctParams {
        simulations,
        max_considered,
        c_puct: 1.25,
    };
    let root = RootContext::from_state(&config, &state).expect("root context");
    let mut tree =
        PuctTreeSearch::new(&config, root, params, Lcg32::new(7)).expect("tree constructs");

    let mut features = vec![0.0_f32; RECORD_FEATURES];
    let mut policy = vec![0.0_f32; RECORD_POLICY];
    while !tree.is_done() {
        if !tree.next_leaf(&config, &mut features).expect("next_leaf") {
            break;
        }
        tree.pending_leaf_mask(&config, &mut policy)
            .expect("pending mask");
        // The mask is 1.0 on legal codes; weight them unevenly, then normalise.
        let mut sum = 0.0_f32;
        for (i, slot) in policy.iter_mut().enumerate() {
            if *slot > 0.0 {
                *slot = 1.0 + ((i % 7) as f32);
                sum += *slot;
            }
        }
        if sum > 0.0 {
            for slot in policy.iter_mut() {
                *slot /= sum;
            }
        }
        tree.submit(&config, &policy, 0.1).expect("submit");
    }
    (tree, config)
}

#[test]
fn an_unexpanded_path_is_none_and_a_played_path_carries_visits() {
    let (tree, config) = run(512, 12);
    let result = tree.result();

    // The chosen action is the most-visited candidate, so its subtree is non-empty.
    let chosen = tree
        .subtree_visits_after(&[result.action_code])
        .expect("the chosen action must be in the tree");
    assert!(
        chosen > 0,
        "the chosen action has no visits under it; the diagnostic is reading the wrong node"
    );

    // Root visits bound every child's. If this fails, the walk is landing on the
    // wrong node and every inheritance number would be inflated.
    let root_total: u32 = result.visit_counts.iter().map(|(_, v)| *v).sum();
    assert!(
        chosen <= root_total,
        "subtree visits {chosen} exceed the root's total {root_total}"
    );

    // A code that is not legal at the root was never expanded.
    let state = create_initial_state(&config).expect("initial state");
    let legal = legal_action_codes(&config, &state).expect("legal codes");
    let illegal = (0u16..)
        .find(|c| !legal.contains(&(*c as usize)))
        .expect("some code is illegal at the opening position");
    assert_eq!(
        tree.subtree_visits_after(&[illegal]),
        None,
        "an illegal code must report None, not a count"
    );
}

#[test]
fn a_considered_but_unvisited_action_reports_zero_rather_than_none() {
    // A budget SMALLER than the considered set is what produces this: with 4
    // simulations over 16 candidates the first halving round cannot give every
    // candidate even one visit, so twelve root edges are scored and never descended
    // into. (16 sims over 16 candidates does not work -- every candidate gets exactly
    // one visit and the case never arises, which is how this test first failed.)
    // Those edges must read as 0 -- "in the tree, nothing under it" -- not as None.
    let (tree, _config) = run(4, 16);
    let result = tree.result();

    let zero_visit_codes: Vec<u16> = result
        .visit_counts
        .iter()
        .filter(|(_, v)| *v == 0)
        .map(|(code, _)| *code)
        .collect();
    assert!(
        !zero_visit_codes.is_empty(),
        "no unvisited candidate in this configuration; the case is untested"
    );
    for code in zero_visit_codes {
        assert_eq!(
            tree.subtree_visits_after(&[code]),
            Some(0),
            "candidate {code} was considered but unvisited; expected Some(0)"
        );
    }
}

#[test]
fn a_two_ply_path_is_bounded_by_its_one_ply_prefix() {
    // This is the shape the reuse question actually asks: our move, then their reply.
    // The grandchild's inheritance can never exceed its parent's, and that bound is
    // what makes a small measurement here decisive against reuse.
    let (tree, config) = run(512, 12);
    let result = tree.result();
    let ours = result.action_code;
    let parent = tree.subtree_visits_after(&[ours]).expect("our move is present");

    let state = create_initial_state(&config).expect("initial state");
    let legal = legal_action_codes(&config, &state).expect("legal codes");
    let mut checked = 0;
    for reply in legal.iter().map(|c| *c as u16) {
        if let Some(grandchild) = tree.subtree_visits_after(&[ours, reply]) {
            assert!(
                grandchild <= parent,
                "reply {reply}: grandchild {grandchild} exceeds parent {parent}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no reply was found in the tree at all; the two-ply walk never succeeded"
    );
}

