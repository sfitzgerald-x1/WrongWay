//! `RootMode::Classic` — the AZ-classic control arm — and the guard that the
//! Gumbel root it sits beside did not move.
//!
//! Two separate claims live here and they are tested in opposite directions.
//!
//! The first is a *negative* claim: adding a second root algorithm must not
//! perturb the default one. That cannot be shown by any assertion written
//! against the new code alone, so it is pinned to digests
//! ([`GUMBEL_ARGMAX_DIGEST`], [`GUMBEL_TEMPERATURE_DIGEST`]) computed by running
//! this file's `digest` on the build *before* `RootMode` existed. Regenerating
//! them is therefore a visible, deliberate act, exactly as a fixture change is.
//!
//! The second is a *positive* one: Classic really is plain PUCT with
//! visit-count targets, and not the Gumbel path wearing a different name.

use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::puct::{PuctParams, PuctResult, PuctTreeSearch, RootMode};
use wrongway_normal_duel::selfplay::{
    Exploration, GameOutcome, SelfPlayBatch, SelfPlayOptions, RECORD_FEATURES, RECORD_FLOATS,
    RECORD_META_FIELDS, RECORD_POLICY,
};
use wrongway_normal_duel::{
    create_initial_state, legal_action_codes, Config, Coord, Player, Players, JUMP_RULE, RULESET,
};

fn config() -> Config {
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

struct Run {
    records: Vec<f32>,
    meta: Vec<i32>,
    count: usize,
    outcomes: Vec<GameOutcome>,
    plies: Vec<u64>,
}

fn run(options: SelfPlayOptions) -> Run {
    let config = config();
    let mut batch = SelfPlayBatch::new(&config, options).expect("valid options");
    let mut scratch = vec![0.0_f32; RECORD_POLICY];
    loop {
        let n = batch.collect().expect("collect");
        if n == 0 {
            break;
        }
        for slot in 0..n {
            let features =
                batch.features()[slot * RECORD_FEATURES..(slot + 1) * RECORD_FEATURES].to_vec();
            let value = mock_evaluator::evaluate(&features, &mut scratch);
            batch.policy_mut()[slot * RECORD_POLICY..(slot + 1) * RECORD_POLICY]
                .copy_from_slice(&scratch);
            batch.value_mut()[slot] = value as f32;
        }
        batch.submit(n).expect("submit");
    }
    let count = batch.take_records();
    Run {
        records: batch.records().to_vec(),
        meta: batch.record_meta().to_vec(),
        count,
        outcomes: batch.outcomes(),
        plies: batch.plies_played(),
    }
}

/// FNV-1a over every byte a self-play run produces: the record floats' exact bit
/// patterns (features, policy target, legal mask, z), the per-record metadata
/// (game, ply, turn, played action code), and each game's outcome and length.
///
/// Bit patterns, not approximate comparisons: the claim being pinned is
/// byte-for-byte identity, and a target that moved in the eighth decimal is
/// still a target that moved.
fn digest(run: &Run) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    };
    eat(&(run.count as u64).to_le_bytes());
    for value in &run.records {
        eat(&value.to_bits().to_le_bytes());
    }
    for field in &run.meta {
        eat(&field.to_le_bytes());
    }
    for outcome in &run.outcomes {
        eat(&[match outcome {
            GameOutcome::Ongoing => 0,
            GameOutcome::Win(Player::A) => 1,
            GameOutcome::Win(Player::B) => 2,
            GameOutcome::Draw => 3,
        }]);
    }
    for ply in &run.plies {
        eat(&ply.to_le_bytes());
    }
    hash
}

/// Pure argmax play: no draw is taken after the search, so this digest covers
/// the search's own decisions and the recorded improved-policy targets.
fn gumbel_argmax_options() -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 64,
        max_considered: 8,
        ply_cap: 24,
        seed_base: 20_260_809,
        ..SelfPlayOptions::default()
    }
}

/// The production recipe's shape: the temperature phase draws one word per ply
/// from the same stream the search runs on, so this digest also covers the
/// stream offset.
fn gumbel_temperature_options() -> SelfPlayOptions {
    SelfPlayOptions {
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 16,
        ..gumbel_argmax_options()
    }
}

/// Computed on `scott/completed-q-targets` at 69a7f35 — the commit this branch
/// is based on, before `RootMode` existed — by running this same file with the
/// two constants set to 0 and reading the panic message.
const GUMBEL_ARGMAX_DIGEST: u64 = 0x7977_465b_8b0d_d648;
const GUMBEL_TEMPERATURE_DIGEST: u64 = 0xcb4d_f111_6f90_19e2;

/// The negative claim: the default root is byte-for-byte what it was.
#[test]
fn the_default_gumbel_root_is_unchanged_by_the_classic_arm() {
    let argmax = digest(&run(gumbel_argmax_options()));
    let temperature = digest(&run(gumbel_temperature_options()));
    assert_eq!(
        (argmax, temperature),
        (GUMBEL_ARGMAX_DIGEST, GUMBEL_TEMPERATURE_DIGEST),
        "the default Gumbel root produced different records or decisions; \
         actual argmax {argmax:#018x}, actual temperature {temperature:#018x}"
    );
}

#[test]
fn the_two_gumbel_fixtures_are_not_accidentally_the_same_run() {
    // Guards the digests above against a copy-paste that would make one of them
    // vacuous, and confirms the temperature phase actually moves the games.
    assert_ne!(GUMBEL_ARGMAX_DIGEST, GUMBEL_TEMPERATURE_DIGEST);
    let argmax = run(gumbel_argmax_options());
    let temperature = run(gumbel_temperature_options());
    let played = |r: &Run| -> Vec<i32> {
        r.meta
            .chunks(RECORD_META_FIELDS)
            .map(|fields| fields[3])
            .collect()
    };
    assert_ne!(played(&argmax), played(&temperature));
    assert!(argmax.count > 0 && temperature.count > 0);
}

/* ------------------------------------------------------------------ *
 * The classic root, at the search level
 * ------------------------------------------------------------------ */

/// Drive one search to completion against the mock network and return what it
/// decided, plus the draw stream as it left it.
fn search(root_mode: RootMode, simulations: u32, seed: u32) -> (PuctResult, Lcg32) {
    let config = config();
    let state = create_initial_state(&config).expect("initial state");
    let params = PuctParams {
        simulations,
        max_considered: 8,
        c_puct: 1.25,
        root_mode,
    };
    let mut tree = PuctTreeSearch::from_state(&config, &state, params, Lcg32::new(seed))
        .expect("an ongoing 9x9 state starts a search");

    let mut features = vec![0.0_f32; RECORD_FEATURES];
    let mut policy = vec![0.0_f32; RECORD_POLICY];
    while tree.next_leaf(&config, &mut features).expect("next_leaf") {
        let value = mock_evaluator::evaluate(&features, &mut policy);
        tree.submit(&config, &policy, value).expect("submit");
    }
    (tree.result(), tree.rng())
}

fn legal_root_actions() -> usize {
    let config = config();
    let state = create_initial_state(&config).expect("initial state");
    legal_action_codes(&config, &state)
        .expect("legal action codes")
        .len()
}

/// The whole of the classic root's contract, at the one place every part of it
/// is visible together.
#[test]
fn the_classic_root_spends_its_budget_on_plain_puct_and_targets_the_visit_distribution() {
    const SIMULATIONS: u32 = 256;
    let (result, _) = search(RootMode::Classic, SIMULATIONS, 4242);
    let legal = legal_root_actions();
    assert!(legal > 20, "the opening position has {legal} legal actions");

    // No candidate set: every legal action is a root edge, and every root edge
    // is reported. Under Gumbel `considered` is `max_considered` long.
    assert_eq!(result.considered.len(), legal);
    assert_eq!(result.visit_counts.len(), legal);
    assert_eq!(result.improved_policy.len(), legal);

    // Every simulation backs up through exactly one root edge, so the counts
    // account for the whole budget. This is also what makes the target's
    // denominator the budget rather than a schedule.
    let total: u32 = result.visit_counts.iter().map(|(_, n)| *n).sum();
    assert_eq!(result.simulations_used, SIMULATIONS);
    assert_eq!(total, SIMULATIONS);

    // The target IS N(a) / sum N -- exactly, not approximately.
    for ((code, visits), (target_code, probability)) in result
        .visit_counts
        .iter()
        .zip(result.improved_policy.iter())
    {
        assert_eq!(code, target_code, "the two lists must share an ordering");
        assert_eq!(
            *probability,
            f64::from(*visits) / f64::from(total),
            "code {code} took {probability} of the target on {visits} of {total} visits"
        );
    }
    let mass: f64 = result.improved_policy.iter().map(|(_, p)| *p).sum();
    assert!((mass - 1.0).abs() < 1e-12, "the target sums to {mass}");

    // Concentration is bounded by 1.0 by construction, and a budget spread over
    // several actions is what makes the bound informative rather than vacuous.
    let top = result
        .improved_policy
        .iter()
        .map(|(_, p)| *p)
        .fold(0.0_f64, f64::max);
    assert!(top <= 1.0, "concentration {top} exceeds 1");
    let visited = result.visit_counts.iter().filter(|(_, n)| *n > 0).count();
    assert!(
        visited > 3,
        "only {visited} root actions were visited at {SIMULATIONS} simulations; \
         a visit-count target needs a budget that actually spreads"
    );

    // Support is the visited set and nothing else: the classic target does not
    // cover actions the search never tried, which is the property that
    // distinguishes it from the improved policy and the reason it is only
    // defensible at a large budget.
    assert!(
        visited < legal,
        "every one of {legal} legal actions was visited; the support claim is untested here"
    );
    for ((_, visits), (code, probability)) in result
        .visit_counts
        .iter()
        .zip(result.improved_policy.iter())
    {
        assert_eq!(
            *visits > 0,
            *probability > 0.0,
            "code {code} carries {probability} of the target on {visits} visits"
        );
    }

    // The played move is the most-visited action, ties to the lowest code.
    let best = result
        .visit_counts
        .iter()
        .fold(None::<(u16, u32)>, |best, (code, visits)| match best {
            Some((_, top)) if *visits <= top => best,
            _ => Some((*code, *visits)),
        })
        .expect("a non-empty root");
    assert_eq!(result.action_code, best.0);
}

/// The classic root takes no Gumbel draws — it has none to take.
///
/// Worth pinning for its own sake: it is what makes the main stream's offset a
/// pure function of the ply index under this arm too, and therefore what D3's
/// separately keyed Dirichlet stream has to preserve.
#[test]
fn the_classic_root_consumes_nothing_from_the_draw_stream() {
    let before = Lcg32::new(31_337);
    let (_, after) = search(RootMode::Classic, 64, 31_337);
    assert_eq!(
        after, before,
        "the classic root advanced the game's draw stream"
    );
    // The control: the Gumbel root takes one draw per legal root action, so the
    // assertion above is about the mode and not about searches in general.
    let (_, gumbel_after) = search(RootMode::Gumbel, 64, 31_337);
    assert_ne!(gumbel_after, before);
}

/// Same seed, same search. The classic root has no draws of its own, so this is
/// really a statement that nothing else in the tree is order-dependent.
#[test]
fn the_classic_root_is_reproducible() {
    let (first, _) = search(RootMode::Classic, 96, 11);
    let (second, _) = search(RootMode::Classic, 96, 11);
    assert_eq!(first, second);
}

/// The two roots are genuinely different searches, not one search behind two
/// names. Without this every assertion above could be satisfied by the Gumbel
/// path with a relabelled target.
#[test]
fn the_two_root_modes_disagree_from_the_same_seed() {
    let (classic, _) = search(RootMode::Classic, 128, 909);
    let (gumbel, _) = search(RootMode::Gumbel, 128, 909);
    assert_ne!(classic.considered.len(), gumbel.considered.len());
    assert_ne!(classic.improved_policy, gumbel.improved_policy);
    // The Gumbel target covers every legal action including the unvisited ones;
    // the classic target covers only what the tree touched. That difference is
    // the arm.
    let classic_support = classic
        .improved_policy
        .iter()
        .filter(|(_, p)| *p > 0.0)
        .count();
    let gumbel_support = gumbel
        .improved_policy
        .iter()
        .filter(|(_, p)| *p > 0.0)
        .count();
    assert!(
        classic_support < gumbel_support,
        "classic support {classic_support} against Gumbel support {gumbel_support}"
    );
}

/* ------------------------------------------------------------------ *
 * The classic arm, at the self-play level
 * ------------------------------------------------------------------ */

fn classic_options() -> SelfPlayOptions {
    SelfPlayOptions {
        games: 3,
        simulations: 128,
        max_considered: 8,
        root_mode: RootMode::Classic,
        ply_cap: 16,
        seed_base: 20_260_809,
        temperature: 1.0,
        temperature_moves: 8,
        ..SelfPlayOptions::default()
    }
}

/// A record's policy target and legal mask, as f64, indexed by action code.
fn target_and_mask(run: &Run, index: usize) -> (Vec<f64>, Vec<f64>) {
    let base = index * RECORD_FLOATS + RECORD_FEATURES;
    let read = |from: usize| -> Vec<f64> {
        run.records[from..from + RECORD_POLICY]
            .iter()
            .map(|value| f64::from(*value))
            .collect()
    };
    (read(base), read(base + RECORD_POLICY))
}

/// Argmax of a target, ties to the lowest code — the rule `most_visited` uses.
fn argmax(target: &[f64]) -> u16 {
    let mut chosen = 0_u16;
    let mut best = f64::NEG_INFINITY;
    for (code, probability) in target.iter().enumerate() {
        if *probability > best {
            best = *probability;
            chosen = code as u16;
        }
    }
    chosen
}

/// The recorded target keeps every contract the trainer's loader depends on —
/// a normalised distribution, exactly zero on illegal codes — while being the
/// visit distribution rather than a softmax.
#[test]
fn classic_records_are_visit_distributions_that_still_satisfy_the_loader() {
    let out = run(classic_options());
    assert!(out.count > 0, "the batch recorded nothing");

    let mut multi_support = 0_usize;
    for index in 0..out.count {
        let (target, mask) = target_and_mask(&out, index);
        let legal = mask.iter().filter(|m| **m > 0.0).count();
        let mut mass = 0.0_f64;
        let mut support = 0_usize;
        for (code, probability) in target.iter().enumerate() {
            if mask[code] == 0.0 {
                assert_eq!(
                    *probability, 0.0,
                    "record {index}: illegal code {code} carries {probability}"
                );
            }
            if *probability > 0.0 {
                mass += *probability;
                support += 1;
            }
        }
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "record {index}: target sums to {mass}, outside the loader's 1e-3 band"
        );
        assert!(
            support <= legal,
            "record {index}: {support} nonzero entries against {legal} legal codes"
        );
        if support > 1 {
            multi_support += 1;
        } else {
            assert_eq!(
                legal, 1,
                "record {index}: one-hot target at a position with {legal} legal codes"
            );
        }
        // The signature that separates this arm from the Gumbel one: at 128
        // simulations the tree cannot touch every legal action, so a target that
        // covered them all would not be a visit distribution.
        if legal > 1 {
            assert!(
                support < legal,
                "record {index}: all {legal} legal codes carry mass; that is not a visit \
                 distribution at this budget"
            );
        }
    }
    assert!(
        multi_support * 10 > out.count * 9,
        "only {multi_support} of {} targets carry a real distribution",
        out.count
    );
}

/// AlphaZero's played-move rule, both halves of it: sampled from the visit
/// counts inside the temperature phase, the argmax outside it.
///
/// The recorded target *is* the visit distribution in this mode, so "sampled
/// from visit counts" and "sampled from the recorded target" are the same
/// sentence — which is why this arm needed no second sampler. The test reads the
/// record, so it would fail if that stopped being true.
#[test]
fn classic_play_samples_visit_counts_in_the_opening_and_plays_the_argmax_after() {
    let options = classic_options();
    let phase = options.temperature_moves;
    let out = run(options);
    assert!(out.count > 0, "the batch recorded nothing");

    let mut sampled_off_argmax = 0_usize;
    let mut inside = 0_usize;
    let mut outside = 0_usize;
    for index in 0..out.count {
        let meta = &out.meta[index * RECORD_META_FIELDS..(index + 1) * RECORD_META_FIELDS];
        let (ply, played) = (meta[1] as u64, meta[3] as u16);
        let (target, _) = target_and_mask(&out, index);
        let best = argmax(&target);
        if ply < phase {
            inside += 1;
            if played != best {
                sampled_off_argmax += 1;
            }
        } else {
            outside += 1;
            assert_eq!(
                played, best,
                "record {index} at ply {ply} is past the temperature phase but played {played} \
                 rather than the most-visited {best}"
            );
        }
    }
    assert!(inside > 0 && outside > 0, "the run never left the phase");
    assert!(
        sampled_off_argmax > 0,
        "not one of {inside} in-phase plies sampled away from the argmax; the temperature phase \
         is doing nothing"
    );

    // And with the phase switched off, every ply is the argmax.
    let argmax_only = run(SelfPlayOptions {
        temperature_moves: 0,
        ..classic_options()
    });
    for index in 0..argmax_only.count {
        let meta = &argmax_only.meta[index * RECORD_META_FIELDS..(index + 1) * RECORD_META_FIELDS];
        let (target, _) = target_and_mask(&argmax_only, index);
        assert_eq!(meta[3] as u16, argmax(&target), "record {index}");
    }
}

/// Same seed, same games.
#[test]
fn classic_self_play_is_reproducible_from_the_seed() {
    let first = run(classic_options());
    let second = run(classic_options());
    assert_eq!(first.count, second.count);
    assert_eq!(first.meta, second.meta);
    assert!(first.records == second.records);
    assert_eq!(digest(&first), digest(&second));
}

/// Switching the arm on has to change the data. If it did not, every assertion
/// above could be passing on Gumbel records.
#[test]
fn the_classic_arm_produces_different_data_from_the_default() {
    let classic = digest(&run(classic_options()));
    let gumbel = digest(&run(SelfPlayOptions {
        root_mode: RootMode::Gumbel,
        ..classic_options()
    }));
    assert_ne!(classic, gumbel);
}
