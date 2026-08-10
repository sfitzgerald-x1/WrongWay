//! Cross-implementation goldens for the Gumbel-MuZero qtransform.
//!
//! The authority for `puct.rs`'s completed-Q, its qtransform and its improved
//! policy is DeepMind's `mctx`, not this repository's reading of the paper —
//! and not the previous behaviour, which `v3` deliberately breaks with. The
//! plan doc that specifies `v3` transcribes `v_mix` by hand and says so:
//! *"the formula transcribed here is our reading of the paper; the mctx goldens
//! are the authority, and any discrepancy resolves toward mctx."* This file is
//! that authority, wired to the build.
//!
//! `scripts/gen-qtransform-goldens.py` runs
//! `mctx.qtransform_completed_by_mix_value` and `gumbel_muzero_policy`'s
//! `action_weights` over 234 synthetic roots — all 216 combinations of prior
//! shape x visit pattern x Q range, at independently drawn action counts and
//! visit budgets, plus the 18 edge cases the plan pins by name — and commits
//! what came back as `fixtures/qtransform-mctx-goldens.json`. Nothing in this
//! file recomputes `mctx`; it only compares.
//!
//! **The tolerance is 1e-6 in f64 and does not move.** A mismatch is a bug in
//! `puct.rs`, to be fixed there and the fixtures regenerated. Widening the band
//! to get green would discard the only external check this design has.
//!
//! Where the comparison is not literally like-for-like
//! ---------------------------------------------------
//! Two conversions stand between the fixture and `root_qtransform`, both
//! bounded far below the tolerance and both measured by the test itself, which
//! prints the largest deviation it saw at every stage.
//!
//! `mctx` takes prior *logits*; this engine stores masked, renormalised prior
//! *probabilities*. The generator hands `mctx` `log(max(p, 1e-9))` — the
//! engine's own `POLICY_FLOOR` — and both consumers of the prior are invariant
//! to the positive rescaling `mctx`'s softmax then applies (see
//! `puct::effective_prior`). The logarithm itself is `math.log` in the
//! generator and `js_math::js_log` here, which can differ by an ULP of a number
//! near -20.
//!
//! And an `Edge` accumulates a `value_sum`, not a mean, so a fixture Q-value
//! makes the round trip `(n * q) / n`. That is exact for a power-of-two `n` and
//! within an ULP otherwise.

use std::collections::BTreeMap;

use serde_json::Value;
use wrongway_normal_duel::puct::{root_qtransform, ActionStats};

const GOLDENS: &str = include_str!("fixtures/qtransform-mctx-goldens.json");

/// Fixed by the plan, and not to be relaxed. See the module docs.
const TOLERANCE: f64 = 1e-6;

/// The edge cases the plan pins by name. Every one of them must be present in
/// the fixture file, so a regeneration that quietly dropped one fails here
/// rather than passing with a thinner grid.
const REQUIRED_CASES: [&str; 18] = [
    "edge/all-unvisited",
    "edge/all-unvisited-negative-root",
    "edge/max-visits-zero",
    "edge/one-visited",
    "edge/one-visited-last",
    "edge/all-q-equal-visited",
    "edge/all-q-equal-mixed",
    "edge/all-q-equal-zero",
    "edge/all-negative",
    "edge/negative-root-positive-q",
    "edge/single-action-visited",
    "edge/single-action-unvisited",
    "edge/alpha-prior-tiny-visited-mass",
    "edge/alpha-prior-zero-mass-visited",
    "edge/alpha-prior-one-hot-unvisited",
    "edge/q-at-both-bounds",
    "edge/q-one-ulp-apart",
    "edge/huge-visit-count",
];

fn numbers(value: &Value, field: &str, case: &str) -> Vec<f64> {
    value[field]
        .as_array()
        .unwrap_or_else(|| panic!("{case}: `{field}` must be an array"))
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .unwrap_or_else(|| panic!("{case}: `{field}` must hold numbers"))
        })
        .collect()
}

/// The largest absolute deviation across one stage, or a panic naming the
/// action that blew the tolerance.
fn compare(actual: &[f64], expected: &[f64], stage: &str, case: &str) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{case}: {stage} has {} entries, the fixture has {}",
        actual.len(),
        expected.len()
    );
    let mut worst = 0.0_f64;
    for (index, (ours, theirs)) in actual.iter().zip(expected).enumerate() {
        assert!(
            ours.is_finite(),
            "{case}: {stage}[{index}] is {ours}, and mctx produced {theirs}"
        );
        let deviation = (ours - theirs).abs();
        assert!(
            deviation <= TOLERANCE,
            "{case}: {stage}[{index}] is {ours}, mctx says {theirs} (off by {deviation:e}). \
             mctx is the authority: fix puct.rs and regenerate the fixtures. Do not widen \
             the tolerance."
        );
        worst = worst.max(deviation);
    }
    worst
}

#[test]
fn our_qtransform_matches_mctx_over_the_whole_fixture_grid() {
    let goldens: Value = serde_json::from_str(GOLDENS).expect("the goldens file is JSON");
    let cases = goldens["cases"]
        .as_array()
        .expect("the goldens file holds a `cases` array");
    assert!(
        cases.len() >= 230,
        "only {} golden cases; the plan asks for ~200",
        cases.len()
    );
    // 6 prior shapes x 6 visit patterns x 6 Q ranges, enumerated rather than
    // sampled. A regeneration that quietly went back to sampling them from one
    // counter -- which is what the first version of the generator did, covering
    // 64 of the combinations -- fails here.
    let grid = cases
        .iter()
        .filter(|case| {
            case["name"]
                .as_str()
                .unwrap_or_default()
                .starts_with("grid/")
        })
        .count();
    assert_eq!(grid, 6 * 6 * 6, "the shape grid is no longer enumerated");

    // The fixture is only an authority if it says what produced it.
    assert_eq!(
        goldens["dtype"].as_str(),
        Some("float64"),
        "a golden generated in f32 cannot be compared at 1e-6"
    );
    assert_eq!(goldens["params"]["valueScale"].as_f64(), Some(0.1));
    assert_eq!(goldens["params"]["maxvisitInit"].as_f64(), Some(50.0));
    assert_eq!(goldens["params"]["epsilon"].as_f64(), Some(1e-8));
    assert_eq!(goldens["policyFloor"].as_f64(), Some(1e-9));

    let mut worst: BTreeMap<&str, (f64, String)> = BTreeMap::new();
    let mut seen: Vec<&str> = Vec::with_capacity(cases.len());
    let mut actions = 0_usize;

    for case in cases {
        let name = case["name"].as_str().expect("every case is named");
        seen.push(name);

        let priors = numbers(case, "priors", name);
        let qvalues = numbers(case, "qvalues", name);
        let visits: Vec<u32> = case["visits"]
            .as_array()
            .expect("`visits` is an array")
            .iter()
            .map(|entry| entry.as_u64().expect("visit counts are unsigned") as u32)
            .collect();
        assert_eq!(priors.len(), visits.len(), "{name}: ragged case");
        assert_eq!(priors.len(), qvalues.len(), "{name}: ragged case");

        let stats: Vec<ActionStats> = (0..priors.len())
            .map(|index| ActionStats {
                prior: priors[index],
                visits: visits[index],
                qvalue: qvalues[index],
            })
            .collect();
        let raw_value = case["rawValue"].as_f64().expect("`rawValue` is a number");
        let ours = root_qtransform(&stats, raw_value);

        let expected_mixed = case["mixedValue"]
            .as_f64()
            .expect("`mixedValue` is a number");
        let stages = [
            (
                "mixedValue",
                compare(&[ours.mixed_value], &[expected_mixed], "mixedValue", name),
            ),
            (
                "completed",
                compare(
                    &ours.completed,
                    &numbers(case, "completed", name),
                    "completed",
                    name,
                ),
            ),
            (
                "transformed",
                compare(
                    &ours.transformed,
                    &numbers(case, "transformed", name),
                    "transformed",
                    name,
                ),
            ),
            (
                "actionWeights",
                compare(
                    &ours.action_weights,
                    &numbers(case, "actionWeights", name),
                    "actionWeights",
                    name,
                ),
            ),
        ];
        for (stage, deviation) in stages {
            let entry = worst.entry(stage).or_insert((0.0, String::new()));
            if deviation > entry.0 {
                *entry = (deviation, name.to_owned());
            }
        }
        actions += stats.len();
    }

    for required in REQUIRED_CASES {
        assert!(
            seen.contains(&required),
            "the fixture no longer contains the pinned edge case `{required}`"
        );
    }

    let report: Vec<String> = worst
        .iter()
        .map(|(stage, (deviation, case))| format!("{stage} {deviation:e} at {case}"))
        .collect();
    println!(
        "mctx goldens: {} cases, {actions} actions, all within {TOLERANCE:e}. Worst deviation \
         per stage: {}.",
        cases.len(),
        report.join(", ")
    );
}

/// The tolerance is a constant of the design, not a dial. If a future change
/// needs it wider, that is a finding about the change.
#[test]
fn the_golden_tolerance_is_the_one_the_plan_fixed() {
    assert_eq!(TOLERANCE, 1e-6);
}
