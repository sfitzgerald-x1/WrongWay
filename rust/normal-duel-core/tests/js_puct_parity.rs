//! Cross-engine parity for the batched PUCT search.
//!
//! This is the acceptance criterion for moving the tree into Rust: for the same
//! seed, the same RNG stream and the same deterministic evaluator, the Rust
//! search must reach *identical* decisions to `js/normal-duel-puct-search.mjs` —
//! the same visit counts, the same chosen action, the same root value and the
//! same number of simulations spent. Not close: identical.
//!
//! The reference answers come from shelling out to node against the real `js/`
//! modules, never from a Rust reimplementation of them. The evaluator is
//! `crate::mock_evaluator` / `js/normal-duel-mock-evaluator.mjs`, a pure
//! function of the feature vector, so no ONNX or float-library difference can
//! confound the comparison.
//!
//! The test also reports the smallest Gumbel-ranking gap it saw. That is the
//! margin protecting the one quantity that is not bit-reproducible by
//! construction — see `src/js_math.rs` — and printing it turns "no ordering
//! flipped" into a number.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{PuctParams, PuctResult, PuctTreeSearch};
use wrongway_normal_duel::{Config, GameState, NN_INPUT_PLANES};

#[path = "common/state_pool.rs"]
mod state_pool;

/// At least the 200 distinct mid-game states the acceptance criterion asks for.
const STATES: usize = 220;

#[derive(Debug, Clone, Copy)]
struct Case {
    simulations: u32,
    max_considered: u32,
    c_puct: f64,
    seed: u32,
}

/// Spans the schedule's shapes: a budget below one visit per candidate, a
/// budget that halves several times, the single-candidate drain, and a zero
/// budget where the winner is decided by halving alone.
const CASES: [Case; 6] = [
    Case {
        simulations: 0,
        max_considered: 4,
        c_puct: 1.25,
        seed: 11,
    },
    Case {
        simulations: 1,
        max_considered: 8,
        c_puct: 1.25,
        seed: 22,
    },
    Case {
        simulations: 8,
        max_considered: 2,
        c_puct: 1.25,
        seed: 33,
    },
    Case {
        simulations: 16,
        max_considered: 4,
        c_puct: 0.75,
        seed: 44,
    },
    Case {
        simulations: 32,
        max_considered: 8,
        c_puct: 1.25,
        seed: 55,
    },
    Case {
        simulations: 48,
        max_considered: 16,
        c_puct: 2.5,
        seed: 66,
    },
];

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate sits two directories below the repository root")
}

fn config_json(config: &Config) -> Value {
    serde_json::to_value(config).expect("config serializes")
}

/// Doubles cross the harness boundary as hex bit patterns.
///
/// Not paranoia: serde_json 1.0.151 parses `0.48560023307800293` one ULP high,
/// so reading these as JSON numbers would fail a search that is exact. Hex
/// bit patterns take the decimal formatter and parser out of the comparison
/// entirely, the same reasoning `js_hot_path_parity.rs` uses for raw f32 bytes.
fn double_bits(value: &Value) -> u64 {
    u64::from_str_radix(
        value.as_str().expect("bit patterns cross as hex strings"),
        16,
    )
    .expect("16 hex digits")
}

/// Run the JavaScript search over the whole (state, case) grid.
fn javascript_reference(config: &Config, states: &[GameState]) -> Vec<Value> {
    let root = repository_root();
    let directory =
        std::env::temp_dir().join(format!("normal-duel-puct-parity-{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("temp directory is creatable");
    let input = directory.join("input.json");
    let output = directory.join("results.json");

    let cases: Vec<Value> = CASES
        .iter()
        .map(|case| {
            json!({
                "simulations": case.simulations,
                "maxConsidered": case.max_considered,
                "cPuct": case.c_puct,
                "seed": case.seed,
            })
        })
        .collect();

    std::fs::write(
        &input,
        serde_json::to_vec(&json!({
            "config": config_json(config),
            "states": states,
            "cases": cases,
        }))
        .expect("input serializes"),
    )
    .expect("temp input is writable");

    let run = Command::new("node")
        .current_dir(&root)
        .arg("scripts/dump-normal-duel-puct-parity.mjs")
        .arg(&input)
        .arg(&output)
        .output()
        .expect("node must be on PATH to run the cross-engine parity harness");
    assert!(
        run.status.success(),
        "JS reference harness failed ({}):\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );

    let parsed: Value =
        serde_json::from_slice(&std::fs::read(&output).expect("harness wrote the results file"))
            .expect("results file is JSON");
    parsed["results"]
        .as_array()
        .expect("results is an array")
        .clone()
}

/// Drive one Rust search to completion against the mock evaluator.
fn rust_search(config: &Config, state: &GameState, case: Case) -> PuctResult {
    let mut search = PuctTreeSearch::from_state(
        config,
        state,
        PuctParams {
            simulations: case.simulations,
            max_considered: case.max_considered,
            c_puct: case.c_puct,
        },
        Lcg32::new(case.seed),
    )
    .expect("an ongoing 9x9 state starts a search");

    let mut features = vec![0.0_f32; NN_INPUT_PLANES * config.cells()];
    let mut policy = vec![0.0_f32; config.policy_size()];
    let mut evaluations = 0_u32;
    while search
        .next_leaf(config, &mut features)
        .expect("the search advances")
    {
        let value = mock_evaluator::evaluate(&features, &mut policy);
        search
            .submit(config, &policy, value)
            .expect("the mock evaluation is well formed");
        evaluations += 1;
        assert!(
            evaluations <= case.simulations + 1,
            "a simulation performs at most one evaluation, plus one for the root"
        );
    }
    search.result()
}

#[test]
fn rust_batched_puct_matches_the_javascript_search_exactly() {
    let config = state_pool::canonical_config();
    let mut states = state_pool::state_pool(&config, STATES + 40);
    // "Distinct" is part of the claim, so make it true rather than likely.
    states.sort_by(|left, right| left.position_key.cmp(&right.position_key));
    states.dedup_by(|left, right| left.position_key == right.position_key);
    assert!(
        states.len() >= STATES,
        "the pool yielded {} distinct states, wanted {STATES}",
        states.len()
    );
    states.truncate(STATES);

    let reference = javascript_reference(&config, &states);
    assert_eq!(reference.len(), states.len() * CASES.len());

    let mut compared = 0_usize;
    let mut tightest_gap = f64::INFINITY;
    let mut simulations_total = 0_u64;

    for (state_index, state) in states.iter().enumerate() {
        for (case_index, case) in CASES.iter().enumerate() {
            let expected = &reference[state_index * CASES.len() + case_index];
            let actual = rust_search(&config, state, *case);
            let where_ = format!(
                "state {state_index} (ply {}), case {case_index} (simulations {}, maxConsidered {})",
                state.ply, case.simulations, case.max_considered
            );

            let expected_counts: Vec<(u16, u32)> = expected["visitCounts"]
                .as_array()
                .expect("visitCounts is an array")
                .iter()
                .map(|pair| {
                    let pair = pair.as_array().expect("each entry is [code, visits]");
                    (
                        pair[0].as_u64().expect("code is unsigned") as u16,
                        pair[1].as_u64().expect("visits are unsigned") as u32,
                    )
                })
                .collect();
            let expected_considered: Vec<u16> = expected["considered"]
                .as_array()
                .expect("considered is an array")
                .iter()
                .map(|code| code.as_u64().expect("code is unsigned") as u16)
                .collect();

            assert_eq!(
                actual.visit_counts, expected_counts,
                "visitCounts at {where_}"
            );
            assert_eq!(
                actual.action_code,
                expected["actionCode"]
                    .as_u64()
                    .expect("actionCode is unsigned") as u16,
                "actionCode at {where_}"
            );
            assert_eq!(
                actual.root_value.to_bits(),
                double_bits(&expected["rootValueBits"]),
                "rootValue at {where_}"
            );
            assert_eq!(
                actual.simulations_used,
                expected["simulationsUsed"]
                    .as_u64()
                    .expect("simulationsUsed is unsigned") as u32,
                "simulationsUsed at {where_}"
            );
            assert_eq!(
                actual.max_depth_reached,
                expected["maxDepthReached"]
                    .as_u64()
                    .expect("maxDepthReached is unsigned") as u32,
                "maxDepthReached at {where_}"
            );
            assert_eq!(
                actual.considered, expected_considered,
                "considered at {where_}"
            );

            // How close the Gumbel ranking came to a tie. A `Math.log`
            // disagreement is bounded by an ULP of the logit (~1e-16 relative),
            // so a gap many orders of magnitude larger is the reason no
            // ordering can flip.
            let mut scores: Vec<f64> = expected["scoreBits"]
                .as_array()
                .expect("scoreBits is an array")
                .iter()
                .map(|score| f64::from_bits(double_bits(score)))
                .collect();
            scores.sort_by(|left, right| right.partial_cmp(left).expect("scores are finite"));
            for pair in scores.windows(2) {
                tightest_gap = tightest_gap.min(pair[0] - pair[1]);
            }

            simulations_total += u64::from(actual.simulations_used);
            compared += 1;
        }
    }

    println!(
        "PUCT parity: {compared} searches over {} distinct states x {} configurations, \
         {simulations_total} simulations, all exact. Tightest Gumbel ranking gap {tightest_gap:e}.",
        states.len(),
        CASES.len()
    );
    assert_eq!(compared, STATES * CASES.len());
    assert!(
        tightest_gap > 1e-9,
        "the Gumbel ranking came within {tightest_gap:e} of a tie, which is close enough to a \
         1-ULP Math.log disagreement to be worth investigating"
    );
}
