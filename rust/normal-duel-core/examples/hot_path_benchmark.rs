//! ns/call for the allocation-free hot path against the existing generator,
//! over one shared pool of mid-game states.
//!
//! Run with `cargo run --release --example hot_path_benchmark`.

use std::hint::black_box;
use std::time::Instant;

use wrongway_normal_duel::{
    encode_state_into, legal_action_codes_fast_stats, legal_position_action_codes, Config,
    GameState, SearchPosition, MAX_POLICY_CODES, NN_INPUT_PLANES,
};

#[path = "../tests/common/state_pool.rs"]
mod state_pool;

const STATES: usize = 150;
const REPEATS: usize = 200;

fn per_call_nanos(elapsed: std::time::Duration, calls: usize) -> f64 {
    elapsed.as_secs_f64() * 1e9 / calls as f64
}

fn main() {
    let config = state_pool::canonical_config();
    let states: Vec<GameState> = state_pool::state_pool(&config, STATES);
    let search: Vec<SearchPosition> = states
        .iter()
        .map(|state| {
            SearchPosition::from_position(&config, &state.position).expect("pool is normalized")
        })
        .collect();
    let calls = STATES * REPEATS;

    let mut codes = [0_u16; MAX_POLICY_CODES];
    let mut planes = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];

    // Warm up so the first pass does not pay for cold caches.
    for position in &search {
        black_box(position.legal_action_codes_fast(&config, &mut codes));
    }

    let start = Instant::now();
    let mut checksum = 0_usize;
    for _ in 0..REPEATS {
        for position in &search {
            checksum += position.legal_action_codes_fast(&config, &mut codes);
            black_box(&codes);
        }
    }
    let fast = per_call_nanos(start.elapsed(), calls);

    let start = Instant::now();
    let mut reference_checksum = 0_usize;
    for _ in 0..REPEATS {
        for state in &states {
            let generated =
                legal_position_action_codes(&config, &state.position).expect("pool is normalized");
            reference_checksum += generated.len();
            black_box(&generated);
        }
    }
    let reference = per_call_nanos(start.elapsed(), calls);

    // `legal_position_action_codes` re-normalizes and returns a fresh Vec, so
    // also time the `Position`-taking fast entry point: it pays the same wall
    // parsing and bitboard rebuild, isolating the generation win itself.
    let start = Instant::now();
    for _ in 0..REPEATS {
        for state in &states {
            let (count, _) = legal_action_codes_fast_stats(&config, &state.position, &mut codes)
                .expect("pool is normalized");
            black_box((count, &codes));
        }
    }
    let from_position = per_call_nanos(start.elapsed(), calls);

    let start = Instant::now();
    for _ in 0..REPEATS {
        for state in &states {
            encode_state_into(&config, state, &mut planes).expect("pool is canonical 9x9");
            black_box(&planes);
        }
    }
    let encode = per_call_nanos(start.elapsed(), calls);

    assert_eq!(
        checksum, reference_checksum,
        "the two generators must agree on how many actions exist"
    );

    let (candidates, geometry_ok, touching, searches) = filter_stats(&config, &states);
    println!("pool: {STATES} mid-game states, {REPEATS} repeats ({calls} calls each)");
    println!("legal_position_action_codes  {reference:9.0} ns/call");
    println!(
        "legal_action_codes_fast      {from_position:9.0} ns/call  ({:.1}x)  from &Position",
        reference / from_position
    );
    println!(
        "legal_action_codes_fast      {fast:9.0} ns/call  ({:.1}x)  from &SearchPosition",
        reference / fast
    );
    println!("encode_state_into            {encode:9.0} ns/call");
    println!();
    println!("prefilter over the same pool:");
    println!(
        "  wall candidates per node      {:.1}",
        candidates as f64 / STATES as f64
    );
    println!(
        "  geometry-legal per node       {:.1}",
        geometry_ok as f64 / STATES as f64
    );
    println!(
        "  touching a shortest path      {:.1} ({:.1}% of candidates)",
        touching as f64 / STATES as f64,
        100.0 * touching as f64 / candidates as f64
    );
    println!(
        "  has_path calls per node       {:.1} (was {:.1})",
        searches as f64 / STATES as f64 + 2.0,
        2.0 * candidates as f64 / STATES as f64
    );
}

fn filter_stats(config: &Config, states: &[GameState]) -> (u64, u64, u64, u64) {
    let mut codes = [0_u16; MAX_POLICY_CODES];
    let mut totals = (0_u64, 0_u64, 0_u64, 0_u64);
    for state in states {
        let (_, stats) = legal_action_codes_fast_stats(config, &state.position, &mut codes)
            .expect("pool is normalized");
        totals.0 += u64::from(stats.candidates);
        totals.1 += u64::from(stats.geometry_ok);
        totals.2 += u64::from(stats.path_touching);
        totals.3 += u64::from(stats.searches);
    }
    totals
}
