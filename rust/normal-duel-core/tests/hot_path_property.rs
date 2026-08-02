//! The shortest-path prefilter must be exact, not a heuristic: over a wide
//! pool of reachable positions `legal_action_codes_fast` has to return exactly
//! what the two-searches-per-candidate generator returns.

use wrongway_normal_duel::{
    legal_action_codes_fast_stats, legal_position_action_codes, MAX_POLICY_CODES,
};

#[path = "common/state_pool.rs"]
mod state_pool;

const STATES: usize = 2_500;

#[test]
fn fast_generation_equals_the_reference_generator_over_varied_positions() {
    let config = state_pool::canonical_config();
    let states = state_pool::state_pool(&config, STATES);
    assert_eq!(states.len(), STATES);

    let mut codes = [0_u16; MAX_POLICY_CODES];
    let mut dense_positions = 0_usize;
    let mut mismatches = Vec::new();
    let mut candidates = 0_u64;
    let mut searched = 0_u64;

    for (index, state) in states.iter().enumerate() {
        let expected = legal_position_action_codes(&config, &state.position)
            .expect("pool positions are normalized");
        let (count, stats) = legal_action_codes_fast_stats(&config, &state.position, &mut codes)
            .expect("pool positions are normalized");
        let actual: Vec<usize> = codes[..count].iter().map(|code| *code as usize).collect();
        if actual != expected {
            mismatches.push(format!(
                "state {index} (ply {}, {} walls): fast {actual:?} vs reference {expected:?}",
                state.ply,
                state.position.walls.len()
            ));
        }
        if state.position.walls.len() >= 8 {
            dense_positions += 1;
        }
        candidates += u64::from(stats.candidates);
        searched += u64::from(stats.path_touching);
    }

    assert!(
        mismatches.is_empty(),
        "{} of {STATES} positions disagreed:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Guard the guard: a pool of near-empty boards would pass trivially, since
    // a wrong filter only shows up once walls actually threaten a path.
    assert!(
        dense_positions >= STATES / 10,
        "the pool must include wall-dense positions, got {dense_positions}"
    );
    assert!(
        candidates > 0 && searched < candidates,
        "the filter must skip work"
    );
    println!(
        "{STATES} positions, {candidates} wall candidates, {searched} needed a real search \
         ({:.1}%)",
        100.0 * searched as f64 / candidates as f64
    );
}

#[test]
fn ascending_order_and_buffer_contract_hold() {
    let config = state_pool::canonical_config();
    let states = state_pool::state_pool(&config, 64);
    let mut codes = [0_u16; MAX_POLICY_CODES];
    for state in &states {
        let (count, _) = legal_action_codes_fast_stats(&config, &state.position, &mut codes)
            .expect("pool positions are normalized");
        assert!(count > 0, "an ongoing position always has a legal action");
        assert!(codes[..count].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(codes[..count]
            .iter()
            .all(|code| usize::from(*code) < config.policy_size()));
    }
    // A short buffer is refused rather than silently truncated.
    let mut short = [0_u16; 4];
    assert!(legal_action_codes_fast_stats(&config, &states[0].position, &mut short).is_err());
}
