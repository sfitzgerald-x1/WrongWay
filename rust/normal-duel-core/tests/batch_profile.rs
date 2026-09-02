//! What batch sizes a SINGLE game's search actually achieves.
//!
//! The inference server's batch ceiling only matters if the search asks for
//! batches that large. Sequential halving spends roughly equal budget per round
//! while the survivor count halves, so the last rounds are narrow and the drain
//! phase is one leaf at a time -- which caps the average far below the widest
//! round. This prints the distribution rather than inferring it from a ratio.
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{PuctParams, PuctTreeSearch};
use wrongway_normal_duel::{Config, NN_INPUT_PLANES};

#[path = "common/state_pool.rs"]
mod state_pool;

fn profile(config: &Config, sims: u32, maxc: u32, cap: usize, vl: f64) -> (f64, usize, usize, usize) {
    let state = &state_pool::state_pool(config, 1)[0];
    let mut search = PuctTreeSearch::from_state(
        config, state,
        PuctParams { simulations: sims, max_considered: maxc, c_puct: 1.25 },
        Lcg32::new(11),
    ).expect("starts");
    search.set_virtual_loss(vl).expect("valid");
    let stride = NN_INPUT_PLANES * config.cells();
    let width = config.policy_size();
    let mut features = vec![0.0_f32; stride * cap];
    let mut policies = vec![0.0_f32; width * cap];
    let mut values = vec![0.0_f64; cap];
    let (mut batches, mut evals, mut biggest) = (0usize, 0usize, 0usize);
    loop {
        let n = search.collect_leaves(config, &mut features, cap).expect("advances");
        if n == 0 { break; }
        for i in 0..n {
            values[i] = mock_evaluator::evaluate(
                &features[i * stride..(i + 1) * stride],
                &mut policies[i * width..(i + 1) * width]);
        }
        search.submit_batch(config, &policies, &values, n).expect("well formed");
        batches += 1; evals += n; biggest = biggest.max(n);
    }
    (evals as f64 / batches as f64, biggest, batches, evals)
}

#[test]
fn a_single_game_cannot_fill_a_large_batch_without_a_penalty() {
    let config = state_pool::canonical_config();
    println!("\n sims  maxc  cap   vl   mean batch  max batch  round trips  evals");
    for (sims, maxc) in [(512_u32, 24_u32), (2048, 96), (4096, 96)] {
        for cap in [16_usize, 64, 256] {
            let (mean, biggest, batches, evals) = profile(&config, sims, maxc, cap, 0.0);
            println!("{sims:5}  {maxc:4}  {cap:3}  0.0  {mean:10.2}  {biggest:9}  {batches:11}  {evals:5}");
        }
    }
    println!("-- with a virtual loss, the drain phase can batch too --");
    // Swept, not sampled at one value: the batch a penalty can fill is exactly what
    // decides whether virtual loss is worth its strength cost, and the first
    // measurement of it was inflated by duplicate leaves in a single batch.
    for vl in [0.5_f64, 1.0, 3.0, 10.0, 30.0] {
        let (mean, biggest, batches, evals) = profile(&config, 4096, 96, 256, vl);
        println!(" 4096    96  256 {vl:5.1} {mean:10.2}  {biggest:9}  {batches:11}  {evals:5}");
    }
    // The claim under test: raising the SERVER's ceiling cannot help a lone game
    // whose search never asks for more than a handful of leaves at a time.
    let (mean_64, _, _, _) = profile(&config, 4096, 96, 64, 0.0);
    let (mean_256, _, _, _) = profile(&config, 4096, 96, 256, 0.0);
    assert!((mean_256 - mean_64).abs() < 1.0,
        "a 4x larger cap changed the mean batch from {mean_64:.2} to {mean_256:.2}");
}

#[test]
fn a_batch_never_collects_the_same_leaf_twice() {
    // A leaf stays unexpanded until its evaluation is submitted, so nothing stops a
    // second descent in the SAME batch from landing on it again -- the virtual loss
    // is the only thing steering later descents away, and a penalty too small to
    // change the argmax does not steer at all. submit_batch would then expand one
    // node twice, allocating a second edge list and orphaning the first.
    //
    // This is the mechanism behind a U-shaped strength curve in the penalty, where a
    // SMALLER penalty measures far worse than a medium one. That shape is not a
    // property of virtual loss; it is this bug.
    use std::collections::HashSet;
    let config = state_pool::canonical_config();
    let state = &state_pool::state_pool(&config, 1)[0];
    for vl in [0.05_f64, 0.25, 1.0, 3.0] {
        let mut search = PuctTreeSearch::from_state(
            &config, state,
            PuctParams { simulations: 512, max_considered: 12, c_puct: 1.25 },
            Lcg32::new(5),
        ).expect("starts");
        search.set_virtual_loss(vl).expect("valid");
        let stride = NN_INPUT_PLANES * config.cells();
        let width = config.policy_size();
        let cap = 64_usize;
        let mut features = vec![0.0_f32; stride * cap];
        let mut policies = vec![0.0_f32; width * cap];
        let mut values = vec![0.0_f64; cap];
        let mut duplicate_batches = 0_usize;
        loop {
            let n = search.collect_leaves(&config, &mut features, cap).expect("advances");
            if n == 0 { break; }
            let leaves = search.pending_leaf_ids();
            let unique: HashSet<u32> = leaves.iter().copied().collect();
            if unique.len() != leaves.len() { duplicate_batches += 1; }
            for i in 0..n {
                values[i] = mock_evaluator::evaluate(
                    &features[i * stride..(i + 1) * stride],
                    &mut policies[i * width..(i + 1) * width]);
            }
            search.submit_batch(&config, &policies, &values, n).expect("well formed");
        }
        assert_eq!(duplicate_batches, 0, "virtual loss {vl}: {duplicate_batches} batches held a repeated leaf");
    }
}
