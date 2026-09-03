//! Extracting a subtree and resuming a search on it.
//!
//! The load-bearing check is the repetition window. A node's `rep_count` is
//! relative to its OWN window, and moving the root moves every descendant's
//! window -- so extraction cannot copy counts across, it has to recompute them.
//! Getting that wrong changes threefold adjudication SILENTLY: the search sees
//! draws that are not there, or misses ones that are, and nothing else in the
//! result looks different. So the resumed root is compared against a FRESH search
//! at the same position, which derives its window from the game state instead.
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{PuctParams, PuctResult, PuctTreeSearch};
use wrongway_normal_duel::{
    apply_legal_action, create_initial_state, decode_action, Config, GameState, NN_INPUT_PLANES,
};

#[path = "common/state_pool.rs"]
mod state_pool;

fn params() -> PuctParams {
    PuctParams { simulations: 256, max_considered: 12, c_puct: 1.25 }
}

fn drive(search: &mut PuctTreeSearch, config: &Config) -> PuctResult {
    let mut features = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];
    let mut policy = vec![0.0_f32; config.policy_size()];
    while search.next_leaf(config, &mut features).expect("advances") {
        let value = mock_evaluator::evaluate(&features, &mut policy);
        search.submit(config, &policy, value).expect("well formed");
    }
    search.result()
}

fn fresh(config: &Config, state: &GameState, seed: u32) -> PuctTreeSearch {
    PuctTreeSearch::from_state(config, state, params(), Lcg32::new(seed)).expect("starts")
}

#[test]
fn an_extracted_root_carries_the_window_a_fresh_search_would_have() {
    // Extraction follows the move THIS search chose, not one a differently-seeded
    // search chose. The root carries an edge for every legal move but a child only
    // for the candidates it considered, and two searches with different Gumbel
    // draws consider different twelve -- so extracting along someone else's move
    // misses for a reason that has nothing to do with extraction.
    let config = state_pool::canonical_config();
    let mut state = create_initial_state(&config).expect("initial");
    let mut checked = 0;
    for ply in 0..14_u32 {
        if !state.outcome.is_ongoing() { break; }
        let mut search = fresh(&config, &state, 4_000 + ply);
        let result = drive(&mut search, &config);
        let action = decode_action(&config, result.action_code as usize).expect("legal");
        let next = apply_legal_action(&config, &state, &action).expect("legal");

        let sub = search
            .into_subtree(&config, &[result.action_code])
            .expect("the search's own chosen move must be in its tree");
        let want = fresh(&config, &next, 1);
        let mut got: Vec<(u32, u32)> = sub.root_window().entries().to_vec();
        let mut expected: Vec<(u32, u32)> = want.root_window().entries().to_vec();
        got.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            got, expected,
            "ply {ply}: rebased window differs from the one a fresh search derives from the state"
        );
        assert_eq!(sub.root_rep_count(), want.root_rep_count(), "ply {ply}: rep count");
        checked += 1;
        state = next;
    }
    assert!(checked >= 10, "only {checked} extractions were actually checked");
}

#[test]
fn extraction_survives_being_applied_twice_in_a_row() {
    // Two successive extractions with a resumed search in between -- the sequence a
    // game performs -- so the window is rebased on a tree that is itself inherited.
    let config = state_pool::canonical_config();
    let state = create_initial_state(&config).expect("initial");
    let mut search = fresh(&config, &state, 31);
    let first_result = drive(&mut search, &config);
    let first_action = decode_action(&config, first_result.action_code as usize).expect("legal");
    let after_one = apply_legal_action(&config, &state, &first_action).expect("legal");
    let sub = search.into_subtree(&config, &[first_result.action_code]).expect("own move");

    let mut resumed = PuctTreeSearch::resume(&config, sub, params(), Lcg32::new(32)).expect("resumes");
    let second_result = drive(&mut resumed, &config);
    let second_action = decode_action(&config, second_result.action_code as usize).expect("legal");
    let after_two = apply_legal_action(&config, &after_one, &second_action).expect("legal");
    let sub2 = resumed
        .into_subtree(&config, &[second_result.action_code])
        .expect("the resumed search's own chosen move must be in its tree");

    let want = fresh(&config, &after_two, 1);
    let mut got: Vec<(u32, u32)> = sub2.root_window().entries().to_vec();
    let mut expected: Vec<(u32, u32)> = want.root_window().entries().to_vec();
    got.sort_unstable();
    expected.sort_unstable();
    assert_eq!(got, expected, "window after two successive extractions");
    assert_eq!(sub2.root_rep_count(), want.root_rep_count(), "rep count after two");
}

#[test]
fn a_resumed_search_inherits_a_tree_and_still_completes() {
    let config = state_pool::canonical_config();
    let state = create_initial_state(&config).expect("initial");
    let mut search = fresh(&config, &state, 77);
    let result = drive(&mut search, &config);
    let before = search.node_count();
    let sub = search.into_subtree(&config, &[result.action_code]).expect("own move");
    let inherited = sub.node_count();
    assert!(inherited > 1, "extraction produced a bare root, nothing was inherited");
    assert!(inherited < before, "the subtree cannot be larger than the tree it came from");

    let mut resumed = PuctTreeSearch::resume(&config, sub, params(), Lcg32::new(5)).expect("resumes");
    let out = drive(&mut resumed, &config);
    // The schedule is re-run from scratch, so the budget is spent in full on THIS
    // search regardless of what was inherited.
    assert_eq!(out.simulations_used, 256, "a resumed search must spend its own budget");
    assert!(out.visit_counts.iter().any(|(_, v)| *v > 0), "no candidate was visited");
    decode_action(&config, out.action_code as usize).expect("a legal action");
}

#[test]
fn a_path_that_was_never_expanded_yields_nothing() {
    let config = state_pool::canonical_config();
    let state = create_initial_state(&config).expect("initial");
    let mut search = fresh(&config, &state, 3);
    drive(&mut search, &config);
    // 208 is a wall code the search will not have expanded a full reply under at
    // this budget; the point is that a miss returns None rather than a bad tree.
    assert!(search.into_subtree(&config, &[9_999]).is_none(), "an unknown code must miss");
}

#[test]
fn a_restart_keeps_inherited_value_but_not_inherited_exploration() {
    // The separation reuse depends on. `visits` carries the value estimate, which is
    // the whole benefit; `visits_since` is what the halving schedule ranks on, and it
    // must start this search at zero. Ranking on the total measured -177 Elo at the
    // site's default budget, because a candidate that was good before the opponent
    // moved arrived with a visit count no fresh candidate could match.
    let config = state_pool::canonical_config();
    let state = create_initial_state(&config).expect("initial");
    let mut search = fresh(&config, &state, 4_242);
    let result = drive(&mut search, &config);

    // A search that never resumed must be unaffected: the two counters agree.
    for (visits, since) in search.root_edge_visits() {
        assert_eq!(visits, since, "a fresh search must keep the two counters identical");
    }

    let mut sub = search.into_subtree(&config, &[result.action_code]).expect("own move");
    sub.restart(params(), Lcg32::new(9)).expect("restarts");
    let after = sub.root_edge_visits();
    assert!(
        after.iter().any(|(visits, _)| *visits > 0),
        "the inherited value estimates were thrown away with the exploration counts"
    );
    assert!(
        after.iter().all(|(_, since)| *since == 0),
        "inherited exploration is still visible to the halving schedule"
    );
}
