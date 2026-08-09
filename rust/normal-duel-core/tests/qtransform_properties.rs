//! Property suite for the `v3` qtransform (V1 of the design-fix plan).
//!
//! The goldens next door (`qtransform_goldens.rs`) prove we compute the same
//! numbers as `mctx` on ~200 recorded roots. That is a strong check and a
//! narrow one: it says nothing about inputs nobody thought to record. These are
//! the claims `v3` is being adopted FOR, asserted over a randomised sweep and
//! over the degenerate cases by hand.
//!
//! Two of them are written so they cannot pass on a reverted implementation:
//! `bounded_sharpening_...` keeps the `v2` expression in the test and asserts it
//! VIOLATES the bound on a case `v3` satisfies, and
//! `a_search_that_learned_nothing_sharpens_nothing` pins the exact-zero boost
//! that the min-max rescale's epsilon guard produces.
//!
//! Randomised, not exhaustive, but reproducible: the sweep uses the engine's own
//! `Lcg32` at fixed seeds, the same choice `tests/property_gates.rs` makes and
//! for the same reason — it behaves identically on native and `wasm32`.

use wrongway_normal_duel::js_math::Lcg32;
use wrongway_normal_duel::puct::{root_qtransform, ActionStats};

/// `mctx`'s `maxvisit_init` and `value_scale`, restated here on purpose: a test
/// that imported the constants it is checking would follow them anywhere.
const MAXVISIT_INIT: f64 = 50.0;
const VALUE_SCALE: f64 = 0.1;

/// The `v2` expression this change replaced: `(50 + maxN) * 1.0` applied to raw
/// completed-Q in `[-1, 1]`. Kept so the bound below can be shown to be a
/// property of `v3` specifically and not of arithmetic in general.
fn v2_sigma(q: f64, max_visits: u32) -> f64 {
    (50.0 + f64::from(max_visits)) * 1.0 * q
}

fn uniform(rng: &mut Lcg32) -> f64 {
    (f64::from(rng.next_u32()) + 0.5) / 4_294_967_296.0
}

fn range(rng: &mut Lcg32, low: f64, high: f64) -> f64 {
    low + (high - low) * uniform(rng)
}

/// A random root: a normalised prior, a visit pattern, Q-values, a raw value.
///
/// The shapes are drawn from the same families the goldens grid uses, so the
/// sweep is the goldens' domain with 500x the samples and no oracle.
fn random_root(rng: &mut Lcg32) -> (Vec<ActionStats>, f64) {
    let counts = [1_usize, 2, 3, 5, 9, 17, 40, 64, 130];
    let actions = counts[(rng.next_u32() as usize) % counts.len()];
    let top_visits = [0_u32, 1, 2, 7, 30, 128, 4096][(rng.next_u32() as usize) % 7];

    let mut priors: Vec<f64> = (0..actions)
        .map(|_| match rng.next_u32() % 4 {
            0 => 1.0,
            1 => uniform(rng),
            2 => uniform(rng).powi(8),
            // Exactly zero: a legal action the network gave no mass at all,
            // which is what POLICY_FLOOR exists for.
            _ => 0.0,
        })
        .collect();
    let mass: f64 = priors.iter().sum();
    if mass > 0.0 {
        for prior in &mut priors {
            *prior /= mass;
        }
    } else {
        priors.fill(1.0 / actions as f64);
    }

    let stats = priors
        .iter()
        .map(|prior| {
            let visits = if top_visits == 0 || rng.next_u32() % 3 == 0 {
                0
            } else {
                1 + rng.next_u32() % top_visits
            };
            ActionStats {
                prior: *prior,
                visits,
                // Q lives in [-1, 1]: it is a backed-up game value.
                qvalue: if visits > 0 {
                    range(rng, -1.0, 1.0)
                } else {
                    0.0
                },
            }
        })
        .collect();
    (stats, range(rng, -1.0, 1.0))
}

fn max_visits(stats: &[ActionStats]) -> u32 {
    stats.iter().map(|stat| stat.visits).max().unwrap_or(0)
}

/// The renormalised prior the improved policy is a reweighting OF: the engine's
/// `POLICY_FLOOR` applied, then normalised, which is what a flat boost leaves
/// behind.
fn floored_prior(stats: &[ActionStats]) -> Vec<f64> {
    let floored: Vec<f64> = stats.iter().map(|stat| stat.prior.max(1e-9)).collect();
    let mass: f64 = floored.iter().sum();
    floored.into_iter().map(|prior| prior / mass).collect()
}

/// **Bounded sharpening.** One iteration of policy improvement multiplies a
/// prior by at most `exp((50 + maxN) * 0.1)`.
///
/// This is the whole point of D1. Fitting the network to `pi'` and searching
/// again with the result is a feedback loop, so the per-iteration ratio is the
/// base of an exponential in iterations: `v2`'s span of `2 * (50 + maxN)` is not
/// twenty times sharper than `v3`'s, it is `e^152` times sharper at 128
/// simulations, which is the measured warm-start decay this plan is fixing.
#[test]
fn bounded_sharpening_holds_over_the_whole_sweep() {
    let mut rng = Lcg32::new(20_260_809);
    let mut worst_ratio_over_bound = 0.0_f64;
    let mut roots = 0_usize;

    for _ in 0..4_000 {
        let (stats, raw_value) = random_root(&mut rng);
        let priors = floored_prior(&stats);
        let weights = root_qtransform(&stats, raw_value).action_weights;
        let bound = ((MAXVISIT_INIT + f64::from(max_visits(&stats))) * VALUE_SCALE).exp();

        for (index, (weight, prior)) in weights.iter().zip(&priors).enumerate() {
            let ratio = weight / prior;
            assert!(
                ratio <= bound * (1.0 + 1e-12),
                "action {index} of a {}-action root was sharpened by {ratio}, past the \
                 exp((50 + maxN) * 0.1) = {bound} bound",
                stats.len()
            );
            worst_ratio_over_bound = worst_ratio_over_bound.max(ratio / bound);
        }
        roots += 1;
    }
    println!(
        "bounded sharpening: {roots} roots, worst ratio reached {:.4} of its bound",
        worst_ratio_over_bound
    );
    // The bound has to be reachable, or it is not measuring anything.
    assert!(
        worst_ratio_over_bound > 0.5,
        "no root came within half the bound; the sweep is not exercising it"
    );
}

/// The counterexample that makes the test above fail if D1 is reverted.
///
/// The bound is only reachable at all by an action whose prior is small enough
/// to have room to grow: `pi'(a)/p(a)` cannot exceed `1/p(a)` however hot the
/// softmax runs. So the case is the one warm-start decay is actually made of —
/// a move the network has nearly written off, which the tree then finds. At
/// 128 visits with a Q gap of 1.6, `v3` lifts it by at most `exp(17.8)`; `v2`
/// lifts it by `exp(284.8)`, which saturates at the ceiling of `1/p = 1e9` and
/// clears the bound by a factor of twenty.
///
/// Kept as arithmetic on the `v2` expression rather than as a recorded number,
/// so it cannot rot.
#[test]
fn the_v2_expression_violates_the_bound_this_suite_asserts() {
    let mut stats = vec![ActionStats {
        prior: 1e-9,
        visits: 128,
        qvalue: 1.0,
    }];
    for _ in 0..9 {
        stats.push(ActionStats {
            prior: (1.0 - 1e-9) / 9.0,
            visits: 128,
            qvalue: -0.6,
        });
    }
    let bound = ((MAXVISIT_INIT + 128.0) * VALUE_SCALE).exp();

    let weights = root_qtransform(&stats, 0.0).action_weights;
    let v3_ratio = weights[0] / stats[0].prior;
    assert!(
        v3_ratio <= bound,
        "v3 sharpened by {v3_ratio}, past its own bound {bound}"
    );

    // The same root under `v2`: pi' proportional to p * exp(sigma(q)).
    let v2_scores: Vec<f64> = stats
        .iter()
        .map(|stat| stat.prior.max(1e-9).ln() + v2_sigma(stat.qvalue, 128))
        .collect();
    let highest = v2_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials: Vec<f64> = v2_scores.iter().map(|s| (s - highest).exp()).collect();
    let total: f64 = exponentials.iter().sum();
    let v2_ratio = exponentials[0] / total / stats[0].prior;

    assert!(
        v2_ratio > bound,
        "the v2 expression sharpened by {v2_ratio}, which does NOT violate the {bound} bound -- \
         this test has stopped protecting D1 from being reverted"
    );
    println!(
        "bound {bound:e}: v3 reached {v3_ratio:e}, v2 reached {v2_ratio:e} \
         ({:.0}x past the bound)",
        v2_ratio / bound
    );
}

/// **No signal, no sharpening.** When every completed Q-value is equal the
/// search has separated nothing, `min == max`, and the epsilon guard is the only
/// reason the rescale is defined at all. It must produce a boost of exactly
/// `0.0` for every action, leaving `pi'` the renormalised prior.
///
/// The boost is asserted at exact `f64` equality, because that part IS exact:
/// `(c - min)` is `0.0` and `0.0 / 1e-8` is `0.0`. `pi'` is asserted to 1e-12
/// relative instead, because the softmax reaches it through `exp(log(p))`, and
/// a logarithm followed by an exponential is a round trip of one or two ULP —
/// mathematically the identity, not bitwise.
#[test]
fn a_search_that_learned_nothing_sharpens_nothing() {
    // Every shape that can produce a flat completion: nothing visited, so v_mix
    // is the raw value everywhere; everything visited at one Q; and a mixture
    // where v_mix lands exactly on the visited Q.
    let cases: Vec<(&str, Vec<ActionStats>, f64)> = vec![
        (
            "all unvisited",
            vec![
                ActionStats {
                    prior: 0.55,
                    visits: 0,
                    qvalue: 0.0,
                },
                ActionStats {
                    prior: 0.3,
                    visits: 0,
                    qvalue: 0.0,
                },
                ActionStats {
                    prior: 0.1,
                    visits: 0,
                    qvalue: 0.0,
                },
                ActionStats {
                    prior: 0.05,
                    visits: 0,
                    qvalue: 0.0,
                },
            ],
            -0.42,
        ),
        (
            "all visited at one Q",
            vec![
                // Powers of two, so `(n * q) / n` is exact and the completions
                // are equal bit for bit rather than nearly.
                ActionStats {
                    prior: 0.5,
                    visits: 8,
                    qvalue: 0.375,
                },
                ActionStats {
                    prior: 0.25,
                    visits: 4,
                    qvalue: 0.375,
                },
                ActionStats {
                    prior: 0.25,
                    visits: 16,
                    qvalue: 0.375,
                },
            ],
            0.9,
        ),
        (
            "v_mix lands on the visited Q",
            vec![
                ActionStats {
                    prior: 0.7,
                    visits: 4,
                    qvalue: -0.5,
                },
                ActionStats {
                    prior: 0.3,
                    visits: 0,
                    qvalue: 0.0,
                },
            ],
            -0.5,
        ),
        (
            "one action, whatever it measured",
            vec![ActionStats {
                prior: 1.0,
                visits: 32,
                qvalue: 0.123,
            }],
            0.7,
        ),
        (
            "a prior of exactly zero survives the floor",
            vec![
                ActionStats {
                    prior: 0.0,
                    visits: 0,
                    qvalue: 0.0,
                },
                ActionStats {
                    prior: 1.0,
                    visits: 0,
                    qvalue: 0.0,
                },
            ],
            0.25,
        ),
    ];

    for (name, stats, raw_value) in cases {
        let outcome = root_qtransform(&stats, raw_value);
        for (index, boost) in outcome.transformed.iter().enumerate() {
            assert_eq!(
                *boost, 0.0,
                "{name}: action {index} took a boost of {boost} from a search that separated \
                 nothing; the min-max epsilon guard is not holding"
            );
        }
        let priors = floored_prior(&stats);
        for (index, (weight, prior)) in outcome.action_weights.iter().zip(&priors).enumerate() {
            let relative = (weight - prior).abs() / prior;
            assert!(
                relative < 1e-12,
                "{name}: action {index} got {weight} where the renormalised prior is {prior} \
                 (relative {relative:e}); a flat boost must leave the prior alone"
            );
        }
    }
}

/// The same claim over the sweep: wherever the completed values come out equal,
/// the boost is exactly flat.
///
/// The sweep forces the two shapes where the completions are equal *bit for
/// bit* — every action unvisited, so every completion is the raw value itself;
/// or every action visited on one Q at a power-of-two count, so `(n * q) / n`
/// reproduces `q` exactly. The mixed shape, where a `v_mix` computed by a
/// weighted mean has to land on a Q it is mathematically but not bitwise equal
/// to, is a separate test with a separate claim: see
/// `a_v_mix_rounding_crumb_stays_a_crumb`.
#[test]
fn equal_completions_are_flat_across_the_sweep() {
    let mut rng = Lcg32::new(4_242_424);
    let mut unvisited_roots = 0_usize;
    let mut visited_roots = 0_usize;

    for _ in 0..4_000 {
        let (mut stats, _) = random_root(&mut rng);
        let q = range(&mut rng, -1.0, 1.0);
        let visited = rng.next_u32() % 2 == 0;
        for stat in &mut stats {
            stat.visits = if visited {
                // A power-of-two visit count keeps `(n * q) / n` exact.
                1 << (rng.next_u32() % 8)
            } else {
                0
            };
            stat.qvalue = q;
        }
        if visited {
            visited_roots += 1;
        } else {
            unvisited_roots += 1;
        }

        let outcome = root_qtransform(&stats, q);
        for (index, boost) in outcome.transformed.iter().enumerate() {
            assert_eq!(
                *boost, 0.0,
                "action {index}: boost {boost} on a root where every completion is {q}"
            );
        }
    }
    println!(
        "no-signal property: {visited_roots} all-visited and {unvisited_roots} all-unvisited \
         degenerate roots, all exactly flat"
    );
}

/// The one place the degenerate case is *not* bit-exact, stated rather than
/// hidden.
///
/// When a root mixes visited and unvisited actions on a single Q value, the
/// unvisited ones are completed with `v_mix`, whose weighted mean reproduces
/// that Q to within an ULP but not exactly. The min-max denominator is then a
/// crumb of order `1e-16`, the epsilon guard replaces it with `1e-8`, and the
/// rescale therefore multiplies that crumb by `1e8` before the visit scale
/// multiplies it again — an amplification of up to `1e8 * (50 + maxN) * 0.1`.
///
/// That sounds alarming and is not. Measured over this sweep — up to 130
/// actions at up to 4096 visits, the widest shapes the engine can reach — the
/// largest amplified boost is 8.1e-5 in logit space, which moves `pi'` by parts
/// per hundred thousand against a target the trainer's loader accepts at 1e-3.
/// It is asserted here so that if the epsilon guard is ever changed, the size
/// of what it was guarding is on record.
#[test]
fn a_v_mix_rounding_crumb_stays_a_crumb() {
    let mut rng = Lcg32::new(6_060_842);
    let mut worst = 0.0_f64;

    for _ in 0..4_000 {
        let (mut stats, _) = random_root(&mut rng);
        if stats.len() < 2 {
            continue;
        }
        let q = range(&mut rng, -1.0, 1.0);
        for (index, stat) in stats.iter_mut().enumerate() {
            stat.visits = if index % 2 == 0 {
                1 + rng.next_u32() % 4096
            } else {
                0
            };
            stat.qvalue = q;
        }
        let outcome = root_qtransform(&stats, q);
        for boost in &outcome.transformed {
            worst = worst.max(boost.abs());
        }
        for (index, weight) in outcome.action_weights.iter().enumerate() {
            let prior = floored_prior(&stats)[index];
            assert!(
                (weight - prior).abs() / prior < 1e-4,
                "action {index}: {weight} against a renormalised prior of {prior}"
            );
        }
    }
    // 8.1e-5 measured; the band is deliberately close so that an order-of-
    // magnitude change in the amplification cannot pass unnoticed.
    assert!(
        worst < 2e-4,
        "a v_mix rounding crumb was amplified to a boost of {worst}, which is no longer a crumb"
    );
    println!("v_mix rounding crumb: worst amplified boost {worst:e} in logit space");
}

/// **v_mix interpolates.** It lies between the raw value and the prior-weighted
/// mean of the visited Q-values, equals the raw value at `N = 0`, and converges
/// on the weighted mean as `N` grows.
#[test]
fn v_mix_lies_between_the_raw_value_and_the_visited_mean() {
    let mut rng = Lcg32::new(777_001);
    let mut unvisited_roots = 0_usize;
    let mut interpolated = 0_usize;

    for _ in 0..4_000 {
        let (stats, raw_value) = random_root(&mut rng);
        let mixed = root_qtransform(&stats, raw_value).mixed_value;

        let visited: Vec<&ActionStats> = stats.iter().filter(|stat| stat.visits > 0).collect();
        if visited.is_empty() {
            assert_eq!(
                mixed, raw_value,
                "with no visits at all, v_mix must be the raw value untouched"
            );
            unvisited_roots += 1;
            continue;
        }

        let mass: f64 = visited.iter().map(|stat| stat.prior.max(1e-9)).sum();
        let weighted: f64 = visited
            .iter()
            .map(|stat| stat.prior.max(1e-9) * stat.qvalue / mass)
            .sum();
        let (low, high) = if raw_value <= weighted {
            (raw_value, weighted)
        } else {
            (weighted, raw_value)
        };
        // 1e-9 of slack for the accumulation order, not for the claim.
        assert!(
            mixed >= low - 1e-9 && mixed <= high + 1e-9,
            "v_mix {mixed} is outside [{low}, {high}] (raw {raw_value}, weighted {weighted})"
        );
        interpolated += 1;
    }
    assert!(unvisited_roots > 100 && interpolated > 1_000);
    println!("v_mix: {interpolated} interpolating roots, {unvisited_roots} with no visits at all");
}

/// **D2 bites.** The unvisited set is completed with `v_mix`, so it sits where
/// the tree's evidence puts it — not at the bottom of the range because the
/// value head happened to run cold.
///
/// This is the discriminating shape, and it needs three actions to exist: with
/// two, the min-max rescale pins one end to 0 and the other to 1 whatever the
/// completion was, so `v2` and `v3` are indistinguishable. With a third action
/// the completion decides WHICH action is the minimum. Under `v3` the weakest
/// visited action is, and the unvisited one lands strictly inside. Under `v2`'s
/// raw root value the unvisited one is pinned at 0 and the weakest visited
/// action is lifted off the floor — an inversion of exactly the "the
/// unconsidered set is crushed wholesale" mechanism D2 exists to remove.
#[test]
fn the_unvisited_set_is_not_pinned_to_the_bottom_by_a_cold_root_value() {
    let stats = [
        ActionStats {
            prior: 0.3,
            visits: 8,
            qvalue: 0.9,
        },
        ActionStats {
            prior: 0.3,
            visits: 8,
            qvalue: 0.1,
        },
        ActionStats {
            prior: 0.4,
            visits: 0,
            qvalue: 0.0,
        },
    ];
    // v_mix = (-1.0 + 16 * 0.5) / 17.
    let expected_mix = 7.0 / 17.0;
    let raw_value = -1.0;
    let outcome = root_qtransform(&stats, raw_value);

    assert!(
        (outcome.mixed_value - expected_mix).abs() < 1e-12,
        "v_mix is {}, expected {expected_mix}",
        outcome.mixed_value
    );
    assert!(
        (outcome.completed[2] - expected_mix).abs() < 1e-12,
        "the unvisited action was completed with {}, not v_mix; D2 has been reverted",
        outcome.completed[2]
    );

    let ceiling = (MAXVISIT_INIT + 8.0) * VALUE_SCALE;
    assert_eq!(
        outcome.transformed[0], ceiling,
        "the best visited action must take the top of the range"
    );
    assert_eq!(
        outcome.transformed[1], 0.0,
        "the WEAKEST VISITED action must be the minimum; if the unvisited one is, the \
         completion is the raw root value again"
    );
    assert!(
        outcome.transformed[2] > 1.0 && outcome.transformed[2] < ceiling,
        "the unvisited action took {}, which is not strictly inside [0, {ceiling}]",
        outcome.transformed[2]
    );
}

/// N -> large drives v_mix onto the weighted mean, and the approach is
/// monotone. The raw value is deliberately at the far end of the range so a
/// failure to converge is visible rather than lost in rounding.
#[test]
fn v_mix_converges_on_the_weighted_mean_as_visits_grow() {
    let weighted = (0.25 * 0.8 + 0.75 * -0.4) / 1.0;
    let mut previous = f64::INFINITY;
    for visits in [1_u32, 2, 8, 64, 1_024, 65_536] {
        let stats = [
            ActionStats {
                prior: 0.25,
                visits,
                qvalue: 0.8,
            },
            ActionStats {
                prior: 0.75,
                visits,
                qvalue: -0.4,
            },
        ];
        let mixed = root_qtransform(&stats, 1.0).mixed_value;
        assert!(
            mixed > weighted,
            "v_mix {mixed} undershot the mean {weighted}"
        );
        assert!(
            mixed < previous,
            "v_mix must fall monotonically toward the mean"
        );
        previous = mixed;
    }
    assert!(
        (previous - weighted).abs() < 1e-4,
        "v_mix stalled at {previous} instead of reaching {weighted}"
    );
}

/// **The distribution contract, unchanged.** `pi'` is a distribution over
/// exactly the actions it was given: finite, non-negative, summing to 1, and
/// exactly `1.0` when there is only one action.
///
/// The illegal-code half of this contract lives where illegal codes exist —
/// `tests/selfplay_exploration.rs`, over the recorder's output — because an
/// edge list has none by construction.
#[test]
fn the_improved_policy_is_always_a_distribution() {
    let mut rng = Lcg32::new(31_337);
    let mut worst_mass_error = 0.0_f64;
    let mut single_action_roots = 0_usize;

    for _ in 0..4_000 {
        let (stats, raw_value) = random_root(&mut rng);
        let weights = root_qtransform(&stats, raw_value).action_weights;
        assert_eq!(weights.len(), stats.len());

        let mut mass = 0.0_f64;
        for (index, weight) in weights.iter().enumerate() {
            assert!(
                weight.is_finite() && *weight >= 0.0,
                "action {index} carries {weight}"
            );
            mass += *weight;
        }
        worst_mass_error = worst_mass_error.max((mass - 1.0).abs());
        assert!(
            (mass - 1.0).abs() < 1e-12,
            "pi' sums to {mass} over {} actions",
            stats.len()
        );
        if stats.len() == 1 {
            assert_eq!(weights[0], 1.0, "a single action must take exactly 1.0");
            assert_eq!(weights[0] as f32, 1.0_f32, "and exactly 1.0 in f32 too");
            single_action_roots += 1;
        }
    }
    assert!(single_action_roots > 100);
    println!(
        "distribution contract: worst |sum - 1| = {worst_mass_error:e} over 4000 roots, \
         {single_action_roots} of them forced"
    );
}

/// **Monotonicity.** The transform never reorders two actions by value, which is
/// the property the halving ranking depends on: a better completed Q must never
/// buy a smaller boost.
#[test]
fn the_transform_never_reorders_two_actions_by_value() {
    let mut rng = Lcg32::new(90_210);
    for _ in 0..2_000 {
        let (stats, raw_value) = random_root(&mut rng);
        let outcome = root_qtransform(&stats, raw_value);
        for i in 0..stats.len() {
            for j in 0..stats.len() {
                if outcome.completed[i] < outcome.completed[j] {
                    assert!(
                        outcome.transformed[i] <= outcome.transformed[j],
                        "completed {} < {} but the boosts are {} and {}",
                        outcome.completed[i],
                        outcome.completed[j],
                        outcome.transformed[i],
                        outcome.transformed[j]
                    );
                }
            }
        }
    }
}

/// **The transform's range.** Every boost lands in `[0, (50 + maxN) * 0.1]`,
/// with both ends attained whenever two completed values differ. `v2` could
/// return anything in `[-(50 + maxN), (50 + maxN)]`, so this fails on a revert.
#[test]
fn every_boost_lands_in_the_bounded_range() {
    let mut rng = Lcg32::new(5_150);
    for _ in 0..4_000 {
        let (stats, raw_value) = random_root(&mut rng);
        let outcome = root_qtransform(&stats, raw_value);
        let ceiling = (MAXVISIT_INIT + f64::from(max_visits(&stats))) * VALUE_SCALE;

        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for boost in &outcome.transformed {
            assert!(
                *boost >= 0.0 && *boost <= ceiling,
                "a boost of {boost} left [0, {ceiling}]"
            );
            lowest = lowest.min(*boost);
            highest = highest.max(*boost);
        }

        let spread = outcome
            .completed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            - outcome
                .completed
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
        if spread > 1e-8 {
            assert_eq!(lowest, 0.0, "the worst action must sit at exactly 0");
            assert!(
                (highest - ceiling).abs() < 1e-9,
                "the best action took {highest}, not the full {ceiling}"
            );
        }
    }
}
