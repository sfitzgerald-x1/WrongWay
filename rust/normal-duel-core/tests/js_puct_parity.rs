//! Cross-engine parity for the batched PUCT search, under the `v3` version
//! split.
//!
//! This was the acceptance criterion for moving the tree into Rust: for the same
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
//! What `v3` did to that, and what is done about it
//! ------------------------------------------------
//! `puct-az-tree-v3` replaced this project's `sigma` with `mctx`'s
//! `qtransform_completed_by_mix_value` in BOTH places it was used. One of those
//! is the sequential-halving ranking, so the two engines now visit different
//! children and can finish on different actions. That is the change, not a
//! regression: the plan states it, `src/puct.rs` states it, and the correctness
//! anchor moved with it — from "identical to our old JS" to "identical to mctx"
//! (`tests/qtransform_goldens.rs`) plus the property suite
//! (`tests/qtransform_properties.rs`).
//!
//! Deleting or ignoring this file would have thrown away everything the
//! qtransform does NOT touch, which is most of the search. It follows the
//! convention `scripts/diagnose-selfplay-driver-parity.mjs` already set when the
//! `v2` targets made its line comparison meaningless: the by-design divergence
//! is compared SEPARATELY and reported, and the verdict is computed from what
//! still means what it always meant. Concretely, two tests:
//!
//! 1. `rust_batched_puct_matches_the_javascript_search_exactly_without_halving`
//!    runs the full, unweakened comparison at `max_considered = 1`. With one
//!    candidate the halving loop never executes in either engine — the JS goes
//!    straight to `while (budget > 0) visit(survivors[0])` and the Rust to
//!    `draining_single` — so the qtransform is never consulted and every
//!    quantity is still required to be identical. What that covers is the whole
//!    of the tree below the root: `select_edge`'s PUCT and FPU, the repetition
//!    window, terminality and adjudication order, the backup's negation per ply,
//!    the depth accounting and the budget. At 64 simulations down a single root
//!    move it is a deeper tree than any case in the halving grid built.
//!
//! 2. `the_halving_grid_still_agrees_on_everything_the_qtransform_cannot_reach`
//!    runs the original grid and splits it. The Gumbel considered set, the root
//!    value and the simulation accounting are asserted identical, because the
//!    qtransform provably cannot reach them: the considered set is chosen by
//!    `g + logit` before a single simulation runs, the root value is the
//!    network's own evaluation of the root, and the budget is consumed by a
//!    schedule that depends on the NUMBER of survivors, never on which. The
//!    halving-dependent quantities — visit counts, chosen action, depth reached
//!    — are counted and reported, and the test asserts the divergence is
//!    NON-ZERO: a `v3` that agreed with `v1` everywhere would be a `v3` that had
//!    been reverted.
//!
//! `PuctResult::improved_policy` is compared in neither, and never was. The
//! JavaScript records normalised visit counts, and comparing an improved policy
//! across the engines would mean comparing `Math.exp` to `f64::exp` bit for bit;
//! `exp` is not an IEEE-754 operation any more than `Math.log` is, which is the
//! entire reason `src/js_math.rs` exists. The production self-play driver is the
//! Rust/wasm `SelfPlayBatch`, not this reference.
//!
//! The tests also report the smallest Gumbel-ranking gap they saw. That is the
//! margin protecting the one quantity that is not bit-reproducible by
//! construction — see `src/js_math.rs` — and printing it turns "no ordering
//! flipped" into a number.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{
    PuctParams, PuctResult, PuctTreeSearch, JS_REFERENCE_SEARCH_VERSION, PUCT_SEARCH_VERSION,
    SUPERSEDED_SEARCH_VERSION,
};
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

/// Halving never runs at `max_considered = 1`, in either engine, so the
/// qtransform is never consulted and the comparison stays at full strength.
///
/// The budget ladder is chosen for depth: one simulation, then three that all
/// pour their whole budget down a single root move, which is the deepest tree
/// this suite builds and the only place the repetition window gets more than a
/// couple of plies to go wrong in.
const EXACT_CASES: [Case; 4] = [
    Case {
        simulations: 1,
        max_considered: 1,
        c_puct: 1.25,
        seed: 101,
    },
    Case {
        simulations: 8,
        max_considered: 1,
        c_puct: 1.25,
        seed: 202,
    },
    Case {
        simulations: 24,
        max_considered: 1,
        c_puct: 0.75,
        seed: 303,
    },
    Case {
        simulations: 64,
        max_considered: 1,
        c_puct: 2.5,
        seed: 404,
    },
];

/// Spans the schedule's shapes: a budget below one visit per candidate, a
/// budget that halves several times, and the single-candidate drain.
///
/// There is no zero-budget case. There used to be, covering the winner being
/// decided by halving alone, but `puctSearch` now rejects `simulations: 0` --
/// with no simulations the visit counts are empty and `effectiveVisitCounts`
/// returns a one-hot, which the cluster shard worker would record as a policy
/// target. Parity over an input the JS side refuses is not meaningful, so the
/// case is a budget of 1 at the same `max_considered` instead.
///
/// Unchanged from `v2`, deliberately: this is the grid the split is measured
/// over, so moving it would make the divergence count incomparable with the
/// searches these cases used to agree on exactly.
const SPLIT_CASES: [Case; 6] = [
    Case {
        simulations: 1,
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

/// Run the JavaScript search over the whole (state, case) grid, returning the
/// harness's whole output: `{ version, results }`.
///
/// `label` keeps two tests in the same process off each other's scratch files;
/// cargo runs them on separate threads and the pid alone would collide.
fn javascript_reference(
    config: &Config,
    states: &[GameState],
    cases: &[Case],
    label: &str,
) -> Value {
    let root = repository_root();
    let directory = std::env::temp_dir().join(format!(
        "normal-duel-puct-parity-{}-{label}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("temp directory is creatable");
    let input = directory.join("input.json");
    let output = directory.join("results.json");

    let cases: Vec<Value> = cases
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

    serde_json::from_slice(&std::fs::read(&output).expect("harness wrote the results file"))
        .expect("results file is JSON")
}

/// The freeze, asserted rather than asserted-in-a-comment.
///
/// The JavaScript reference is frozen at `v1` on purpose (see the module docs)
/// and the Rust search is on `v3`. All three halves are pinned here: the
/// reference must still report the version this crate believes it is frozen at,
/// the Rust side must not be back on it, and — new at `v3` — the Rust side must
/// not have fallen back to `v2` either. `v2` is the record-format bump WITHOUT
/// the qtransform, so a revert of D1/D2 that remembered to move the version
/// string would land exactly there.
///
/// Runs the harness over an empty grid, so it costs one node start and no
/// searches.
#[test]
fn the_javascript_reference_is_frozen_at_its_own_version() {
    let config = state_pool::canonical_config();
    let reported = javascript_reference(&config, &[], &SPLIT_CASES, "version");
    assert_eq!(
        reported["version"].as_str(),
        Some(JS_REFERENCE_SEARCH_VERSION),
        "js/normal-duel-puct-search.mjs reports a version this crate does not expect; if the \
         reference was deliberately moved, move JS_REFERENCE_SEARCH_VERSION with it and say what \
         the parity suite now covers"
    );
    assert_ne!(
        PUCT_SEARCH_VERSION, JS_REFERENCE_SEARCH_VERSION,
        "the Rust search is back on the reference's version; the two are only allowed to agree \
         once the qtransform and the improved policy exist in both engines"
    );
    assert_ne!(
        PUCT_SEARCH_VERSION, SUPERSEDED_SEARCH_VERSION,
        "the Rust search is back on v2, the record format WITHOUT the reference qtransform; if \
         D1/D2 were deliberately reverted, the goldens and property suites have to go with them"
    );
    assert_eq!(
        PUCT_SEARCH_VERSION, "puct-az-tree-v3",
        "the version string names the search design; a new one needs its own parity story"
    );
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

/// The 220 distinct mid-game states both tests run over.
fn distinct_states(config: &Config) -> Vec<GameState> {
    let mut states = state_pool::state_pool(config, STATES + 40);
    // "Distinct" is part of the claim, so make it true rather than likely.
    states.sort_by(|left, right| left.position_key.cmp(&right.position_key));
    states.dedup_by(|left, right| left.position_key == right.position_key);
    assert!(
        states.len() >= STATES,
        "the pool yielded {} distinct states, wanted {STATES}",
        states.len()
    );
    states.truncate(STATES);
    states
}

fn expected_visit_counts(expected: &Value) -> Vec<(u16, u32)> {
    expected["visitCounts"]
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
        .collect()
}

fn expected_considered(expected: &Value) -> Vec<u16> {
    expected["considered"]
        .as_array()
        .expect("considered is an array")
        .iter()
        .map(|code| code.as_u64().expect("code is unsigned") as u16)
        .collect()
}

/// How close the Gumbel ranking came to a tie, over EVERY legal root action.
///
/// A `Math.log` disagreement is bounded by an ULP of the logit (~1e-16
/// relative), so a gap many orders of magnitude larger is the reason no
/// ordering can flip. The gap that matters is the one at the cut — between the
/// worst kept candidate and the best discarded one — so the harness dumps all
/// the scores and this takes the tightest adjacent pair anywhere in the
/// ranking, which is a bound on it. Restricted to the considered set, as it was
/// through `v2`, it could not see the cut at all, and at `max_considered = 1`
/// it saw no pair and called that infinite slack.
///
/// Returns `(gap, ranked)`, where `ranked` is false for the twelve states in
/// the pool that have exactly one legal action — both stocks spent and the pawn
/// walled down to a single step. There is no ordering to protect there, and
/// counting those as "no tie observed" would be counting them as evidence.
fn tightest_gumbel_gap(expected: &Value, running: f64) -> (f64, bool) {
    let mut scores: Vec<f64> = expected["allScoreBits"]
        .as_array()
        .expect("allScoreBits is an array")
        .iter()
        .map(|score| f64::from_bits(double_bits(score)))
        .collect();
    if scores.len() < 2 {
        return (running, false);
    }
    scores.sort_by(|left, right| right.partial_cmp(left).expect("scores are finite"));
    let mut tightest = running;
    for pair in scores.windows(2) {
        tightest = tightest.min(pair[0] - pair[1]);
    }
    (tightest, true)
}

/// The quantities the qtransform provably cannot reach, asserted identical in
/// both tests.
///
/// `considered` is chosen by `g + logit` before a single simulation runs.
/// `rootValue` is the network's evaluation of the root, taken outside the
/// budget. `simulationsUsed` is spent by a schedule whose shape depends on the
/// NUMBER of survivors in each round, never on which ones they are.
fn assert_version_independent(actual: &PuctResult, expected: &Value, where_: &str) {
    assert_eq!(
        actual.considered,
        expected_considered(expected),
        "considered at {where_}"
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
}

#[test]
fn rust_batched_puct_matches_the_javascript_search_exactly_without_halving() {
    let config = state_pool::canonical_config();
    let states = distinct_states(&config);

    let reference = javascript_reference(&config, &states, &EXACT_CASES, "exact");
    let reference = reference["results"]
        .as_array()
        .expect("results is an array")
        .clone();
    assert_eq!(reference.len(), states.len() * EXACT_CASES.len());

    let mut compared = 0_usize;
    let mut tightest_gap = f64::INFINITY;
    let mut ranked = 0_usize;
    let mut simulations_total = 0_u64;
    let mut deepest = 0_u32;

    for (state_index, state) in states.iter().enumerate() {
        for (case_index, case) in EXACT_CASES.iter().enumerate() {
            let expected = &reference[state_index * EXACT_CASES.len() + case_index];
            let actual = rust_search(&config, state, *case);
            let where_ = format!(
                "state {state_index} (ply {}), case {case_index} (simulations {})",
                state.ply, case.simulations
            );

            assert_version_independent(&actual, expected, &where_);
            assert_eq!(
                actual.visit_counts,
                expected_visit_counts(expected),
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
                actual.max_depth_reached,
                expected["maxDepthReached"]
                    .as_u64()
                    .expect("maxDepthReached is unsigned") as u32,
                "maxDepthReached at {where_}"
            );

            let (gap, has_ranking) = tightest_gumbel_gap(expected, tightest_gap);
            tightest_gap = gap;
            ranked += usize::from(has_ranking);
            deepest = deepest.max(actual.max_depth_reached);
            simulations_total += u64::from(actual.simulations_used);
            compared += 1;
        }
    }

    println!(
        "PUCT parity (no halving): {compared} searches over {} distinct states x {} \
         configurations, {simulations_total} simulations, deepest tree {deepest} plies, all \
         exact. Tightest Gumbel ranking gap {tightest_gap:e} over {ranked} ranked roots.",
        states.len(),
        EXACT_CASES.len()
    );
    assert_eq!(compared, STATES * EXACT_CASES.len());
    assert!(
        deepest >= 3,
        "the deepest tree reached {deepest} plies, which is not deep enough for this test to be \
         covering the descent"
    );
    assert!(
        ranked * 10 > compared * 9,
        "only {ranked} of {compared} searches had a root ranking to measure"
    );
    assert!(
        tightest_gap > 1e-9,
        "the Gumbel ranking came within {tightest_gap:e} of a tie, which is close enough to a \
         1-ULP Math.log disagreement to be worth investigating"
    );
}

#[test]
fn the_halving_grid_still_agrees_on_everything_the_qtransform_cannot_reach() {
    let config = state_pool::canonical_config();
    let states = distinct_states(&config);

    let reference = javascript_reference(&config, &states, &SPLIT_CASES, "grid");
    let reference = reference["results"]
        .as_array()
        .expect("results is an array")
        .clone();
    assert_eq!(reference.len(), states.len() * SPLIT_CASES.len());

    let mut compared = 0_usize;
    let mut tightest_gap = f64::INFINITY;
    let mut ranked = 0_usize;
    let mut simulations_total = 0_u64;
    let mut differing_counts = 0_usize;
    let mut differing_actions = 0_usize;
    let mut differing_depths = 0_usize;

    for (state_index, state) in states.iter().enumerate() {
        for (case_index, case) in SPLIT_CASES.iter().enumerate() {
            let expected = &reference[state_index * SPLIT_CASES.len() + case_index];
            let actual = rust_search(&config, state, *case);
            let where_ = format!(
                "state {state_index} (ply {}), case {case_index} (simulations {}, maxConsidered {})",
                state.ply, case.simulations, case.max_considered
            );

            assert_version_independent(&actual, expected, &where_);

            // By design under v3, and counted rather than asserted.
            if actual.visit_counts != expected_visit_counts(expected) {
                differing_counts += 1;
            }
            if u64::from(actual.action_code) != expected["actionCode"].as_u64().unwrap() {
                differing_actions += 1;
            }
            if u64::from(actual.max_depth_reached) != expected["maxDepthReached"].as_u64().unwrap()
            {
                differing_depths += 1;
            }

            let (gap, has_ranking) = tightest_gumbel_gap(expected, tightest_gap);
            tightest_gap = gap;
            ranked += usize::from(has_ranking);
            simulations_total += u64::from(actual.simulations_used);
            compared += 1;
        }
    }

    let percent = |count: usize| 100.0 * count as f64 / compared as f64;
    println!(
        "PUCT version split: {compared} searches over {} distinct states x {} configurations, \
         {simulations_total} simulations. Considered set, root value and budget identical in all \
         {compared}. Halving-dependent divergence (by design, v1 vs v3): visit counts \
         {differing_counts} ({:.1}%), chosen action {differing_actions} ({:.1}%), depth reached \
         {differing_depths} ({:.1}%). Tightest Gumbel ranking gap {tightest_gap:e} over \
         {ranked} ranked roots.",
        states.len(),
        SPLIT_CASES.len(),
        percent(differing_counts),
        percent(differing_actions),
        percent(differing_depths)
    );
    assert_eq!(compared, STATES * SPLIT_CASES.len());
    assert!(
        ranked * 10 > compared * 9,
        "only {ranked} of {compared} searches had a root ranking to measure"
    );
    assert!(
        tightest_gap > 1e-9,
        "the Gumbel ranking came within {tightest_gap:e} of a tie, which is close enough to a \
         1-ULP Math.log disagreement to be worth investigating"
    );
    // The divergence is the change. If it vanished, D1 was reverted -- and this
    // grid would silently go back to being a full-strength parity test against
    // an implementation the plan deliberately moved away from.
    assert!(
        differing_counts > 0,
        "the v3 halving reached the same visit counts as the frozen v1 reference on all \
         {compared} searches; the qtransform is not reaching the halving ranking"
    );
}
