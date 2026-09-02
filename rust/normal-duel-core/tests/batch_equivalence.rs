//! Batched collection against the sequential search it is supposed to replace.
//!
//! The whole claim of `collect_leaves` at `virtual_loss == 0.0` is that it is a
//! SCHEDULING change and not an algorithm change: sequential halving visits its
//! survivors on a fixed rota that no evaluation influences, and distinct
//! survivors own disjoint subtrees, so evaluating one leaf from each at the same
//! time cannot change what the search decides.
//!
//! That claim is not obviously true. Batching reorders `create_child` against
//! `expand`, so nodes and edges are allocated in a different interleaving, and
//! every value that lands on a node is a floating-point addition whose result
//! depends on the order it arrives in. So this compares the FULL result --
//! including the improved policy's floats -- rather than just the chosen move.
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{PuctParams, PuctResult, PuctTreeSearch};
use wrongway_normal_duel::{
    apply_legal_action, create_initial_state, decode_action, Config, GameState, NN_INPUT_PLANES,
};

#[path = "common/state_pool.rs"]
mod state_pool;

const SIMULATIONS: u32 = 256;
const MAX_CONSIDERED: u32 = 12;
const C_PUCT: f64 = 1.25;

fn params() -> PuctParams {
    PuctParams { simulations: SIMULATIONS, max_considered: MAX_CONSIDERED, c_puct: C_PUCT }
}

fn sequential(config: &Config, state: &GameState, seed: u32) -> PuctResult {
    let mut search = PuctTreeSearch::from_state(config, state, params(), Lcg32::new(seed))
        .expect("an ongoing state starts a search");
    let mut features = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];
    let mut policy = vec![0.0_f32; config.policy_size()];
    while search.next_leaf(config, &mut features).expect("advances") {
        let value = mock_evaluator::evaluate(&features, &mut policy);
        search.submit(config, &policy, value).expect("well formed");
    }
    search.result()
}

/// Returns the result and the batch sizes actually achieved.
fn batched(config: &Config, state: &GameState, seed: u32, max_n: usize, vl: f64)
    -> (PuctResult, Vec<usize>) {
    let mut search = PuctTreeSearch::from_state(config, state, params(), Lcg32::new(seed))
        .expect("an ongoing state starts a search");
    search.set_virtual_loss(vl).expect("a finite non-negative penalty");
    let stride = NN_INPUT_PLANES * config.cells();
    let width = config.policy_size();
    let mut features = vec![0.0_f32; stride * max_n];
    let mut policies = vec![0.0_f32; width * max_n];
    let mut values = vec![0.0_f64; max_n];
    let mut sizes = Vec::new();
    loop {
        let n = search.collect_leaves(config, &mut features, max_n).expect("advances");
        if n == 0 { break; }
        sizes.push(n);
        for i in 0..n {
            values[i] = mock_evaluator::evaluate(
                &features[i * stride..(i + 1) * stride],
                &mut policies[i * width..(i + 1) * width],
            );
        }
        search.submit_batch(config, &policies, &values, n).expect("well formed");
    }
    (search.result(), sizes)
}

fn assert_identical(a: &PuctResult, b: &PuctResult, what: &str) {
    assert_eq!(a.action_code, b.action_code, "{what}: chosen action");
    assert_eq!(a.visit_counts, b.visit_counts, "{what}: visit counts");
    assert_eq!(a.considered, b.considered, "{what}: considered set");
    assert_eq!(a.simulations_used, b.simulations_used, "{what}: simulations used");
    assert_eq!(a.max_depth_reached, b.max_depth_reached, "{what}: max depth");
    assert_eq!(a.root_value.to_bits(), b.root_value.to_bits(), "{what}: root value bits");
    assert_eq!(a.improved_policy.len(), b.improved_policy.len(), "{what}: policy length");
    for (i, (x, y)) in a.improved_policy.iter().zip(b.improved_policy.iter()).enumerate() {
        assert_eq!(x.0, y.0, "{what}: policy code at {i}");
        assert_eq!(x.1.to_bits(), y.1.to_bits(), "{what}: policy weight bits at {i}");
    }
}

#[test]
fn zero_virtual_loss_batching_is_bit_identical_to_the_sequential_search() {
    let config = state_pool::canonical_config();
    // A WIDE pool, not a handful. The first version of this test used 12 states and
    // passed against a search that silently truncated itself -- 241 simulations of
    // 256 -- because the bug needed a visit landing on a terminal node followed by a
    // repeat of that candidate, which only shows up in deeper, wall-heavy positions.
    // `state_pool` walks trajectories of varying length and wall bias, so breadth
    // here is what reaches the corner cases.
    let states = state_pool::state_pool(&config, 160);
    for (index, state) in states.iter().enumerate() {
        let seed = 1_000 + index as u32;
        let want = sequential(&config, state, seed);
        for max_n in [2_usize, 8, 32] {
            let (got, _) = batched(&config, state, seed, max_n, 0.0);
            assert_identical(&want, &got, &format!("state {index}, max_n {max_n}"));
        }
    }
}

#[test]
fn batching_actually_batches_and_is_bounded_by_the_surviving_candidates() {
    // If this ever reports every batch as 1, the API is a no-op dressed up as an
    // optimisation and the equivalence test above would still pass.
    let config = state_pool::canonical_config();
    let state = &state_pool::state_pool(&config, 1)[0];
    let (_, sizes) = batched(&config, state, 7, 64, 0.0);
    let biggest = *sizes.iter().max().expect("at least one batch");
    let evaluations: usize = sizes.iter().sum();
    assert!(biggest > 1, "no batch ever held more than one leaf");
    assert!(
        biggest <= MAX_CONSIDERED as usize,
        "a zero-penalty batch cannot exceed the considered set, got {biggest}"
    );
    // The drain phase is serial, so the saving is real but bounded well under the
    // batch width. Recorded as a range so a regression in either direction shows.
    let speedup = evaluations as f64 / sizes.len() as f64;
    assert!(
        speedup > 1.5 && speedup < (MAX_CONSIDERED as f64),
        "mean batch {speedup:.2} outside the expected band for sequential halving"
    );
}

#[test]
fn a_virtual_loss_changes_the_search_and_lifts_the_batch_size() {
    // The penalty is the knob that trades exactness for parallelism. If it did
    // nothing, an A/B of "batched vs not" would measure noise -- so prove it
    // moves both the batch size and the search itself before trusting any
    // strength comparison built on it.
    let config = state_pool::canonical_config();
    let state = &state_pool::state_pool(&config, 1)[0];
    let (_, safe_sizes) = batched(&config, state, 7, 32, 0.0);
    let (_, vl_sizes) = batched(&config, state, 7, 32, 1.0);
    let safe_mean = safe_sizes.iter().sum::<usize>() as f64 / safe_sizes.len() as f64;
    let vl_mean = vl_sizes.iter().sum::<usize>() as f64 / vl_sizes.len() as f64;
    assert!(vl_mean > safe_mean, "a penalty must permit larger batches ({vl_mean} vs {safe_mean})");
    assert!(
        *vl_sizes.iter().max().unwrap() > MAX_CONSIDERED as usize,
        "with a penalty a batch may exceed the considered set"
    );
}

#[test]
fn a_negative_or_non_finite_penalty_is_refused() {
    let config = state_pool::canonical_config();
    let state = &state_pool::state_pool(&config, 1)[0];
    let mut search = PuctTreeSearch::from_state(&config, state, params(), Lcg32::new(1))
        .expect("starts");
    assert!(search.set_virtual_loss(-0.1).is_err());
    assert!(search.set_virtual_loss(f64::NAN).is_err());
    assert!(search.set_virtual_loss(f64::INFINITY).is_err());
    assert!(search.set_virtual_loss(0.0).is_ok());
}

#[test]
fn a_walked_game_never_diverges_and_never_truncates() {
    // Sampled positions are not enough. The pool draws independent states from random
    // trajectories, and 160 of those passed against a search that silently truncated
    // itself to 241 simulations of 256. The failure needed a visit landing on a
    // terminal node followed by a repeat of that candidate, which is reached by
    // following the search's OWN moves into the positions it steers toward -- not by
    // sampling around them.
    //
    // Truncation is checked explicitly as well as through the result comparison: a
    // batched search that quietly stops early still returns a well-formed move, and
    // `simulations_used` is the only field that says it did less work.
    let config = state_pool::canonical_config();
    let mut state = create_initial_state(&config).expect("an initial state");
    for ply in 0..60_u32 {
        if !state.outcome.is_ongoing() {
            break;
        }
        let seed = 12_345_u32 ^ ply.wrapping_mul(2_654_435_761);
        let want = sequential(&config, &state, seed);
        for max_n in [8_usize, 32] {
            let (got, _) = batched(&config, &state, seed, max_n, 0.0);
            assert_identical(&want, &got, &format!("ply {ply}, max_n {max_n}"));
            assert_eq!(
                got.simulations_used, want.simulations_used,
                "ply {ply}, max_n {max_n}: batched search truncated itself"
            );
        }
        let action = decode_action(&config, want.action_code as usize).expect("a legal code");
        state = apply_legal_action(&config, &state, &action).expect("a legal move");
    }
}
