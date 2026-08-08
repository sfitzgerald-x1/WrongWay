//! Guards on the exploration recipe self-play uses to pick the played move.
//!
//! The load-bearing claim is the one that cost an earlier run 114 iterations:
//! changing *which move is played* must not change *what is recorded*. The
//! recorded policy target stays the search's full visit distribution for the
//! state actually visited, whichever recipe chose the move. `meanTargetEntropy`
//! in the shard workers is the production form of this check; these tests are
//! the same assertion where it can fail the build.

use wrongway_normal_duel::mock_evaluator;
use wrongway_normal_duel::selfplay::{
    sample_visit_temperature, Exploration, GameOutcome, SelfPlayBatch, SelfPlayOptions,
    RECORD_FEATURES, RECORD_FLOATS, RECORD_POLICY,
};
use wrongway_normal_duel::{Config, Coord, Player, Players, JUMP_RULE, RULESET};

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

/// Drive a whole batch to completion against the deterministic mock network.
fn run(options: SelfPlayOptions) -> Run {
    run_with(&config(), options)
}

/// As [`run`], but over a caller-supplied config. The tree re-derives
/// adjudication from `config.ply_cap`, not `SelfPlayOptions::ply_cap`, so a test
/// that wants the tree's cap arm executed has to move the one on the config.
fn run_with(config: &Config, options: SelfPlayOptions) -> Run {
    let config = config.clone();
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
            // The mock hands back priors already; `expand` renormalises over the
            // legal codes, so masking here is unnecessary for these guards.
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

fn base_options() -> SelfPlayOptions {
    SelfPlayOptions {
        games: 4,
        simulations: 24,
        max_considered: 8,
        ply_cap: 40,
        seed_base: 7,
        ..SelfPlayOptions::default()
    }
}

fn temperature_options() -> SelfPlayOptions {
    SelfPlayOptions {
        exploration: Exploration::VisitTemperature,
        temperature: 1.0,
        temperature_moves: 20,
        ..base_options()
    }
}

/// Shannon entropy of every recorded policy target, and the number of targets
/// that carry more than one non-zero code.
fn target_stats(run: &Run) -> (f64, usize) {
    let mut entropy_sum = 0.0_f64;
    let mut multi_support = 0;
    for i in 0..run.count {
        let base = i * RECORD_FLOATS + RECORD_FEATURES;
        let target = &run.records[base..base + RECORD_POLICY];
        let mut h = 0.0_f64;
        let mut support = 0;
        let mut mass = 0.0_f64;
        for &p in target {
            if p > 0.0 {
                let p = f64::from(p);
                h -= p * p.ln();
                mass += p;
                support += 1;
            }
        }
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "record {i}: policy target sums to {mass}, not 1 -- it is not a distribution"
        );
        if support > 1 {
            multi_support += 1;
        } else {
            // A one-code target is only defensible when the position had one
            // legal move, i.e. the search had nothing to distribute over. Any
            // other single-support target is the one-hot regression.
            let mask = &run.records[base + RECORD_POLICY..base + 2 * RECORD_POLICY];
            let legal = mask.iter().filter(|&&m| m > 0.0).count();
            assert_eq!(
                legal, 1,
                "record {i}: one-hot policy target at a position with {legal} legal codes"
            );
        }
        entropy_sum += h;
    }
    (entropy_sum / run.count as f64, multi_support)
}

/// The invariant. Sampling the played move must leave the recorded target a full
/// visit distribution, exactly as argmax play does.
#[test]
fn visit_temperature_keeps_policy_targets_as_full_distributions() {
    let sampled = run(temperature_options());
    let argmax = run(base_options());
    assert!(sampled.count > 0 && argmax.count > 0);

    let (sampled_entropy, sampled_multi) = target_stats(&sampled);
    let (argmax_entropy, argmax_multi) = target_stats(&argmax);

    assert!(
        sampled_entropy > 0.0,
        "mean target entropy under temperature sampling is {sampled_entropy}; \
         0 means one-hot targets, the bug this guard exists for"
    );
    // Not `== count`: a position with one legal move records a one-code target
    // and that is correct. `target_stats` asserts each such record really had a
    // single legal code, so this only pins that forced moves stay a small tail.
    assert!(
        sampled_multi * 10 > sampled.count * 9,
        "only {sampled_multi} of {} targets carry a real distribution",
        sampled.count
    );
    // Not a claim that the two runs visit the same states -- they do not, that is
    // the point -- only that the target's shape is a property of the search and
    // not of how the move was chosen.
    assert!(argmax_entropy > 0.0);
    assert!(argmax_multi * 10 > argmax.count * 9);
}

/// The v2 record format's own acceptance criterion, where it can fail the build.
///
/// The visit-count target had two measurable defects on live shards: it put mass
/// on 8.2 codes out of 40.4 legal ones, and its top two entries were
/// *bit-identical* in 75.5% of positions because sequential halving hands both
/// finalists the same visit count whoever won. The improved policy is a softmax
/// over every legal root action, so both numbers should move decisively, and it
/// must still be exactly zero everywhere the mask says illegal.
///
/// This runs at the production search shape (128 simulations, 16 considered),
/// which is the shape the ladder degeneracy was measured at.
#[test]
fn improved_policy_targets_cover_the_legal_moves_and_separate_the_top_two() {
    let out = run(SelfPlayOptions {
        games: 3,
        simulations: 128,
        max_considered: 16,
        ply_cap: 24,
        seed_base: 20_260_808,
        temperature: 1.0,
        temperature_moves: 12,
        ..SelfPlayOptions::default()
    });
    assert!(out.count > 0, "the batch recorded nothing");

    let mut legal_total = 0_usize;
    let mut nonzero_total = 0_usize;
    let mut tied_top_two = 0_usize;
    let mut entropies: Vec<f64> = Vec::with_capacity(out.count);

    for i in 0..out.count {
        let base = i * RECORD_FLOATS + RECORD_FEATURES;
        let target = &out.records[base..base + RECORD_POLICY];
        let mask = &out.records[base + RECORD_POLICY..base + 2 * RECORD_POLICY];

        let mut mass = 0.0_f64;
        let mut entropy = 0.0_f64;
        let mut top = [0.0_f32; 2];
        let mut legal = 0_usize;
        let mut nonzero = 0_usize;
        for (code, (&p, &m)) in target.iter().zip(mask.iter()).enumerate() {
            if m == 1.0 {
                legal += 1;
            } else {
                // The trainer's contract: not "small", exactly zero.
                assert_eq!(
                    p, 0.0,
                    "record {i}: illegal code {code} carries {p} of the target"
                );
            }
            if p > 0.0 {
                nonzero += 1;
                mass += f64::from(p);
                entropy -= f64::from(p) * f64::from(p).ln();
                if p > top[0] {
                    top[1] = top[0];
                    top[0] = p;
                } else if p > top[1] {
                    top[1] = p;
                }
            }
        }
        assert!(
            (mass - 1.0).abs() < 1e-3,
            "record {i}: target sums to {mass}, outside the loader's 1e-3 band"
        );
        assert!(
            nonzero <= legal,
            "record {i}: {nonzero} nonzero entries against {legal} legal codes"
        );
        if legal > 1 && top[0] == top[1] {
            tied_top_two += 1;
        }
        legal_total += legal;
        nonzero_total += nonzero;
        entropies.push(entropy);
    }

    let mean_legal = legal_total as f64 / out.count as f64;
    let mean_nonzero = nonzero_total as f64 / out.count as f64;
    let tied = tied_top_two as f64 / out.count as f64;
    let mean_entropy = entropies.iter().sum::<f64>() / out.count as f64;
    let distinct_entropies = {
        let mut bits: Vec<u64> = entropies.iter().map(|h| h.to_bits()).collect();
        bits.sort_unstable();
        bits.dedup();
        bits.len()
    };
    println!(
        "v2 targets over {} records: mean legal {mean_legal:.1}, mean nonzero {mean_nonzero:.1}, \
         top1==top2 {:.1}%, mean entropy {mean_entropy:.3} nats over {distinct_entropies} \
         distinct values",
        out.count,
        tied * 100.0
    );

    // The visit target put mass on ~20% of the legal moves. A softmax puts mass
    // on all of them, less the tail that underflows f32 at a large sigma gap.
    assert!(
        mean_nonzero > 0.9 * mean_legal,
        "mean nonzero {mean_nonzero:.1} against mean legal {mean_legal:.1}: the target is still \
         concentrated on a subset of the legal moves"
    );
    // 75.5% under the ladder. Exact f32 ties are essentially impossible under a
    // softmax over distinct scores.
    assert!(
        tied < 0.05,
        "top1 == top2 in {:.1}% of records, the ladder signature",
        tied * 100.0
    );
    // The ladder made entropy a function of the candidate count alone, so it
    // clustered on a handful of constants. Position-dependent entropy is the
    // signal the policy loss was missing.
    assert!(
        distinct_entropies * 2 > out.count,
        "only {distinct_entropies} distinct target entropies across {} records",
        out.count
    );
}

/// A sampled move is not the argmax move, at least sometimes. Without this the
/// test above would also pass on a temperature phase that silently did nothing.
#[test]
fn visit_temperature_actually_diverges_from_argmax() {
    let sampled = run(temperature_options());
    let argmax = run(base_options());
    // Action code is meta field 3 of 4.
    let played = |r: &Run| -> Vec<i32> { r.meta.chunks(4).map(|m| m[3]).collect() };
    assert_ne!(
        played(&sampled),
        played(&argmax),
        "temperature sampling produced the identical move sequence as argmax"
    );
}

/// Same seed, same games -- the reproducibility the whole run design rests on.
#[test]
fn visit_temperature_is_reproducible_from_the_seed() {
    let a = run(temperature_options());
    let b = run(temperature_options());
    assert_eq!(a.count, b.count);
    assert_eq!(a.meta, b.meta);
    assert!(
        a.records == b.records,
        "two runs from the same seed produced different records"
    );
}

/// The stream contract: exactly one word per ply inside the phase, zero after.
/// A batch whose temperature phase is longer than its ply cap consumes a draw at
/// every ply; one with `temperature_moves = 0` consumes none, which is why the
/// latter must reproduce the pre-change argmax stream bit for bit.
#[test]
fn zero_temperature_moves_consumes_no_draws() {
    let disabled = run(SelfPlayOptions {
        temperature_moves: 0,
        ..temperature_options()
    });
    let epsilon_off = run(SelfPlayOptions {
        exploration: Exploration::UniformEpsilon,
        epsilon: 0.0,
        ..base_options()
    });
    assert_eq!(disabled.meta, epsilon_off.meta);
    assert!(disabled.records == epsilon_off.records);
}

/// The legacy path stays reachable and stays different.
#[test]
fn uniform_epsilon_remains_selectable() {
    let epsilon = run(SelfPlayOptions {
        exploration: Exploration::UniformEpsilon,
        epsilon: 0.5,
        ..base_options()
    });
    let (entropy, multi) = target_stats(&epsilon);
    assert!(entropy > 0.0);
    assert!(multi * 10 > epsilon.count * 9);
    let argmax = run(base_options());
    let played = |r: &Run| -> Vec<i32> { r.meta.chunks(4).map(|m| m[3]).collect() };
    assert_ne!(played(&epsilon), played(&argmax));
}

#[test]
fn sampler_respects_the_distribution() {
    let mut target = vec![0.0_f32; RECORD_POLICY];
    target[3] = 0.25;
    target[7] = 0.75;

    // T = 1: the cumulative boundary sits at 0.25.
    assert_eq!(sample_visit_temperature(&target, 1.0, 0.0), Some(3));
    assert_eq!(sample_visit_temperature(&target, 1.0, 0.2), Some(3));
    assert_eq!(sample_visit_temperature(&target, 1.0, 0.3), Some(7));
    assert_eq!(sample_visit_temperature(&target, 1.0, 0.999), Some(7));

    // A cold temperature concentrates on the mode: at T = 0.1 the 0.25 code
    // holds 0.25^10 / (0.25^10 + 0.75^10) ~ 1.7e-3 of the mass.
    assert_eq!(sample_visit_temperature(&target, 0.1, 0.01), Some(7));
    // A hot one flattens toward uniform: at T = 100 the split is ~0.494/0.506.
    assert_eq!(sample_visit_temperature(&target, 100.0, 0.4), Some(3));

    // No mass anywhere leaves the caller on the argmax.
    assert_eq!(
        sample_visit_temperature(&vec![0.0; RECORD_POLICY], 1.0, 0.5),
        None
    );
}

/// A temperature that would make `1/T` non-finite is refused at construction
/// rather than producing NaN weights and an arbitrary move.
#[test]
fn invalid_temperature_is_rejected() {
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let options = SelfPlayOptions {
            temperature,
            temperature_moves: 10,
            ..base_options()
        };
        assert!(
            SelfPlayBatch::new(&config(), options).is_err(),
            "temperature {temperature} should be refused"
        );
    }
    // Not read when the phase is empty, so not checked then.
    assert!(SelfPlayBatch::new(
        &config(),
        SelfPlayOptions {
            temperature: 0.0,
            temperature_moves: 0,
            ..base_options()
        }
    )
    .is_ok());
}

/// `simulations == 0` must be refused. Under the v1 record format it produced
/// the degenerate target directly: the search returned empty visit counts,
/// `effective_visit_counts` fell back to a one-hot, and every recorded policy
/// target became one-hot at a position with ~130 legal codes -- silently, at
/// full record count, which cost an earlier run 114 flat iterations. The v2
/// improved-policy target cannot degenerate that way, but a search that runs no
/// simulations visits nothing, so every completed Q is the root value and the
/// target is just the network's own prior read back. Still worth nothing, still
/// refused.
#[test]
fn zero_simulations_is_rejected_rather_than_emitting_one_hot_targets() {
    let cfg = config();
    let mut options = base_options();
    options.simulations = 0;
    let err = SelfPlayBatch::new(&cfg, options).expect_err("zero simulations must be refused");
    assert_eq!(err.to_string(), "invalid_simulations");
}

/// An options object in the PRE-CHANGE wire format carries `epsilon` and no
/// `exploration` key. Because `exploration` defaults to VisitTemperature and
/// `temperature_moves` to 0, that combination used to parse cleanly and run pure
/// argmax with NO exploration, while `epsilon` was range-checked and never read.
/// An old driver must fail loudly instead of quietly stopping exploring.
#[test]
fn contradictory_exploration_options_are_rejected() {
    let cfg = config();
    let mut epsilon_under_temperature = base_options();
    epsilon_under_temperature.exploration = Exploration::VisitTemperature;
    epsilon_under_temperature.epsilon = 0.15;
    assert_eq!(
        SelfPlayBatch::new(&cfg, epsilon_under_temperature)
            .expect_err("epsilon under visit-temperature must be refused")
            .to_string(),
        "contradictory_exploration"
    );

    let mut moves_under_epsilon = base_options();
    moves_under_epsilon.exploration = Exploration::UniformEpsilon;
    moves_under_epsilon.epsilon = 0.15;
    moves_under_epsilon.temperature_moves = 24;
    assert_eq!(
        SelfPlayBatch::new(&cfg, moves_under_epsilon)
            .expect_err("temperature_moves under uniform-epsilon must be refused")
            .to_string(),
        "contradictory_exploration"
    );
}

/// `games` and `ply_cap` feed an up-front `Vec::with_capacity` whose product is
/// unbounded without these checks. On wasm32 that is an abort across the
/// boundary rather than a readable error, and even in-range values can reserve
/// over a gigabyte, so both must come back as `invalid_*` strings.
#[test]
fn oversized_games_and_ply_cap_are_rejected_rather_than_reserving_unboundedly() {
    let config = config();

    let mut options = SelfPlayOptions {
        games: 100_000,
        ..SelfPlayOptions::default()
    };
    let err = SelfPlayBatch::new(&config, options.clone()).unwrap_err();
    assert_eq!(err.reason(), "invalid_games");

    options = SelfPlayOptions {
        ply_cap: u64::from(u32::MAX) + 1,
        ..SelfPlayOptions::default()
    };
    let err = SelfPlayBatch::new(&config, options.clone()).unwrap_err();
    assert_eq!(err.reason(), "invalid_ply_cap");

    options = SelfPlayOptions {
        ply_cap: 0,
        ..SelfPlayOptions::default()
    };
    let err = SelfPlayBatch::new(&config, options).unwrap_err();
    assert_eq!(err.reason(), "invalid_ply_cap");

    // In-range factors whose PRODUCT is out of range. This is the case the
    // per-factor caps miss: 512 x 2048 x 1229 x 4 B is ~5.15 GB, past wasm32's
    // address space entirely, so without a byte bound the top of the allowed
    // range aborts instead of erroring.
    let roomy = Config {
        ply_cap: 2048,
        ..config.clone()
    };
    let err = SelfPlayBatch::new(
        &roomy,
        SelfPlayOptions {
            games: 512,
            ply_cap: 2048,
            ..SelfPlayOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(err.reason(), "invalid_buffer_length");

    // The canonical shard shape stays accepted.
    let ok = SelfPlayOptions {
        games: 32,
        ply_cap: 200,
        ..SelfPlayOptions::default()
    };
    assert!(SelfPlayBatch::new(&config, ok).is_ok());
}

/// The outer loop's stop and the game-level cap adjudication.
///
/// This deliberately does NOT claim to cover the search tree's cap arm. It once
/// did, and the claim was false twice over: setting `SelfPlayOptions::ply_cap`
/// moves only this loop's stop, and even moving the cap onto the config does not
/// discriminate, because the game loop adjudicates `config.ply_cap` itself -- so
/// the outcomes below are produced with or without `create_child`'s cap branch.
/// Both earlier versions passed with that branch stubbed to `false`.
///
/// The tree's arm is covered in `tests/tree_ply_cap.rs`, which counts leaf
/// evaluations and does fail when the branch is removed (1 vs 65). What is worth
/// pinning here is the pairing this file owns: games stop at the config cap and
/// are scored `Draw` there, and the option can stop a game earlier.
#[test]
fn games_stop_at_the_config_cap_and_are_scored_as_draws() {
    const CAP: u64 = 12;
    let capped = Config {
        ply_cap: CAP,
        ..config()
    };

    let out = run_with(
        &capped,
        SelfPlayOptions {
            games: 6,
            simulations: 64,
            max_considered: 8,
            ply_cap: 64, // above CAP: the config is what binds here
            seed_base: 4321,
            ..SelfPlayOptions::default()
        },
    );
    assert!(out.count > 0, "the batch recorded nothing");

    let mut drew_at_cap = 0;
    for (game, (outcome, ply)) in out.outcomes.iter().zip(out.plies.iter()).enumerate() {
        assert!(
            *ply <= CAP,
            "game {game} reached ply {ply}, past the config cap of {CAP}"
        );
        if *ply == CAP {
            assert_eq!(
                *outcome,
                GameOutcome::Draw,
                "game {game} stopped at the cap but was not scored as a draw"
            );
            drew_at_cap += 1;
        }
    }
    assert!(drew_at_cap > 0, "no game reached the cap");

    // The option can only stop a game *earlier* than the config cap.
    let short = run_with(
        &capped,
        SelfPlayOptions {
            games: 6,
            simulations: 64,
            max_considered: 8,
            ply_cap: 5,
            seed_base: 4321,
            ..SelfPlayOptions::default()
        },
    );
    for (game, ply) in short.plies.iter().enumerate() {
        assert!(
            *ply <= 5,
            "game {game} ran to ply {ply} against an option cap of 5"
        );
    }

    for record in out.records.chunks_exact(RECORD_FLOATS) {
        let target = &record[RECORD_FEATURES..RECORD_FEATURES + RECORD_POLICY];
        let sum: f32 = target.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "policy target sums to {sum}, not 1"
        );
    }
}
