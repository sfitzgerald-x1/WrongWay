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
    for cap in [64_usize, 256] {
        let (mean, biggest, batches, evals) = profile(&config, 4096, 96, cap, 1.0);
        println!(" 4096    96  {cap:3}  1.0  {mean:10.2}  {biggest:9}  {batches:11}  {evals:5}");
    }
    // The claim under test: raising the SERVER's ceiling cannot help a lone game
    // whose search never asks for more than a handful of leaves at a time.
    let (mean_64, _, _, _) = profile(&config, 4096, 96, 64, 0.0);
    let (mean_256, _, _, _) = profile(&config, 4096, 96, 256, 0.0);
    assert!((mean_256 - mean_64).abs() < 1.0,
        "a 4x larger cap changed the mean batch from {mean_64:.2} to {mean_256:.2}");
}
