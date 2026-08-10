#!/usr/bin/env python3
"""Generate cross-implementation goldens for the Gumbel-MuZero qtransform.

The authority for `puct.rs`'s completed-Q, its qtransform and its improved
policy is DeepMind's `mctx`, not this repository's reading of the paper. This
script runs `mctx.qtransform_completed_by_mix_value` -- the exact function
`mctx.gumbel_muzero_policy` uses for both the sequential-halving ranking and
`action_weights` -- over 234 synthetic roots and writes what it produced to
`rust/normal-duel-core/tests/fixtures/qtransform-mctx-goldens.json`.

`rust/normal-duel-core/tests/qtransform_goldens.rs` then asserts our numbers
match, in f64, within 1e-6. A mismatch is resolved by fixing the Rust and
regenerating, never by loosening the tolerance.

Running it
----------

    uv venv --python 3.12 /tmp/mctx-venv
    uv pip install --python /tmp/mctx-venv/bin/python "jax[cpu]" mctx
    /tmp/mctx-venv/bin/python scripts/gen-qtransform-goldens.py

The venv is scratch and deliberately outside the repository: nothing here is
part of the build, and the fixtures are the committed artifact.

Input conventions, and why they are what they are
-------------------------------------------------

*Priors cross as probabilities.* Our search stores edge priors as the network's
policy masked to the legal actions and renormalised, so they are probabilities
summing to 1, not logits. `mctx` takes logits, so each case's priors are handed
over as `log(max(p, POLICY_FLOOR))` with the engine's own `POLICY_FLOOR = 1e-9`
-- the same floor `puct.rs` applies before taking a logarithm. Both consumers of
the prior are invariant to a positive rescaling of it (the improved policy is a
softmax, so a constant shift of the logits cancels; `v_mix`'s weighted mean
divides by the sum of the visited priors, so a constant factor cancels), which
is why `mctx` renormalising the floored priors through its softmax does not
change any golden.

*Unvisited actions carry a poison Q.* `mctx` documents the Q-value of an
unvisited action as undefined and replaces it with the completion; the fixtures
put `POISON_Q` there so that any implementation which reads it instead of
completing it fails loudly rather than subtly. Our `Edge` has `value_sum = 0`
at zero visits, so the Rust side cannot read it by construction -- the poison
is there to keep that true.

*Every action is legal.* Our improved policy is built over the root's edge list,
which is exactly the legal action codes; illegal codes never reach it and are
written as exact `0.0` by the recorder. `mctx`'s `invalid_actions` masking has
no counterpart in the edge list, so it is out of scope here and is covered by
`tests/selfplay_exploration.rs` instead.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import chex  # noqa: E402  (must follow the x64 flag)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import mctx  # noqa: E402
from mctx._src import qtransforms  # noqa: E402
from mctx._src import tree as tree_lib  # noqa: E402

# `puct.rs`'s POLICY_FLOOR: a legal action the network gave exactly zero
# probability is very unlikely, not impossible.
POLICY_FLOOR = 1e-9

# What the fixtures put in `qvalues` wherever `visits == 0`. Large enough that
# reading it instead of completing it cannot hide inside a 1e-6 tolerance.
POISON_Q = 9.0

# `qtransform_completed_by_mix_value` defaults, which are also D1's constants.
VALUE_SCALE = 0.1
MAXVISIT_INIT = 50.0
EPSILON = 1e-8


def synthetic_tree(priors, visits, qvalues, raw_value):
    """A one-node unbatched `mctx` tree with exactly the statistics given.

    `tree.qvalues(node)` is `children_rewards + children_discounts *
    children_values`, so zero rewards and unit discounts make the Q-values read
    back exactly as supplied.
    """
    num_actions = len(priors)
    logits = np.array([math.log(max(p, POLICY_FLOOR)) for p in priors], dtype=np.float64)
    return tree_lib.Tree(
        node_visits=jnp.asarray([int(sum(visits))], dtype=jnp.int32),
        raw_values=jnp.asarray([raw_value], dtype=jnp.float64),
        # Unused by this qtransform, but a Tree must be whole. Deliberately NOT
        # equal to `raw_value`: `qtransform_completed_by_mix_value` reads
        # `raw_values`, and a fixture where the two agreed could not tell us so.
        node_values=jnp.asarray([raw_value + 0.5], dtype=jnp.float64),
        parents=jnp.asarray([tree_lib.Tree.NO_PARENT], dtype=jnp.int32),
        action_from_parent=jnp.asarray([tree_lib.Tree.NO_PARENT], dtype=jnp.int32),
        children_index=jnp.full((1, num_actions), tree_lib.Tree.UNVISITED, dtype=jnp.int32),
        children_prior_logits=jnp.asarray([logits], dtype=jnp.float64),
        children_visits=jnp.asarray([visits], dtype=jnp.int32),
        children_rewards=jnp.zeros((1, num_actions), dtype=jnp.float64),
        children_discounts=jnp.ones((1, num_actions), dtype=jnp.float64),
        children_values=jnp.asarray([qvalues], dtype=jnp.float64),
        embeddings=jnp.zeros((1,), dtype=jnp.float64),
        root_invalid_actions=jnp.zeros((num_actions,), dtype=jnp.float64),
        extra_data=None,
    )


def golden(name, priors, visits, qvalues, raw_value):
    """Run `mctx` over one synthetic root and record everything it computed."""
    priors = [float(p) for p in priors]
    visits = [int(v) for v in visits]
    qvalues = [POISON_Q if v == 0 else float(q) for q, v in zip(qvalues, visits)]
    raw_value = float(raw_value)

    tree = synthetic_tree(priors, visits, qvalues, raw_value)
    node = jnp.asarray(tree_lib.Tree.ROOT_INDEX)

    # The three stages, taken from `mctx` itself rather than reimplemented: the
    # mixed value, the completion, and the full transform.
    prior_probs = jax.nn.softmax(tree.children_prior_logits[0])
    mixed_value = qtransforms._compute_mixed_value(  # noqa: SLF001
        tree.raw_values[0],
        qvalues=tree.qvalues(node),
        visit_counts=tree.children_visits[0],
        prior_probs=prior_probs,
    )
    completed = qtransforms._complete_qvalues(  # noqa: SLF001
        tree.qvalues(node),
        visit_counts=tree.children_visits[0],
        value=mixed_value,
    )
    transformed = qtransforms.qtransform_completed_by_mix_value(
        tree,
        node,
        value_scale=VALUE_SCALE,
        maxvisit_init=MAXVISIT_INIT,
        epsilon=EPSILON,
    )
    # Exactly `gumbel_muzero_policy`'s `action_weights`, with no invalid
    # actions to mask.
    action_weights = jax.nn.softmax(tree.children_prior_logits[0] + transformed)

    chex.assert_trees_all_equal_shapes(transformed, completed, action_weights)
    return {
        "name": name,
        "priors": priors,
        "visits": visits,
        "qvalues": qvalues,
        "rawValue": raw_value,
        "mixedValue": float(mixed_value),
        "completed": [float(x) for x in completed],
        "transformed": [float(x) for x in transformed],
        "actionWeights": [float(x) for x in action_weights],
    }


def normalised(weights):
    total = float(np.sum(weights))
    return [float(w) / total for w in weights]


def edge_cases():
    """The cases the plan pins by name. Each one is a guard, not a sample."""
    cases = []

    # All-unvisited root: no tree evidence at all, so v_mix must collapse to the
    # raw value, every completion must be equal, and the transform must be flat.
    cases.append(("edge/all-unvisited", [0.5, 0.3, 0.15, 0.05], [0, 0, 0, 0], None, 0.42))
    cases.append(("edge/all-unvisited-negative-root", [0.25] * 4, [0, 0, 0, 0], None, -0.9))

    # maxN = 0 is the same arithmetic reached from the other direction: the
    # visit scale is `maxvisit_init` alone.
    cases.append(("edge/max-visits-zero", normalised([7, 2, 1]), [0, 0, 0], None, 0.0))

    # Exactly one visited action: v_mix's weighted mean has a single term, so
    # the visited prior cancels entirely and only that action's Q survives.
    cases.append(("edge/one-visited", [0.4, 0.35, 0.25], [3, 0, 0], [0.6, None, None], -0.2))
    cases.append(("edge/one-visited-last", [0.1, 0.2, 0.7], [0, 0, 11], [None, None, -0.4], 0.8))

    # All Qs equal, every action visited: min == max, so the epsilon guard is
    # the only thing keeping the rescale defined. Must give a flat transform and
    # therefore pi' == the renormalised prior.
    cases.append(("edge/all-q-equal-visited", [0.5, 0.2, 0.2, 0.1], [4, 4, 4, 4], [0.3] * 4, 0.3))
    # Same, but reached with a mixture of visited and unvisited actions whose
    # v_mix happens to land on the visited Q.
    cases.append(("edge/all-q-equal-mixed", [0.6, 0.4], [5, 0], [-0.25, None], -0.25))
    cases.append(("edge/all-q-equal-zero", [0.25] * 4, [2, 3, 4, 5], [0.0] * 4, 0.0))

    # Negative Q throughout, including a root value outside the visited range,
    # which is the configuration where completing with the raw value instead of
    # v_mix crushes the unconsidered set.
    cases.append((
        "edge/all-negative",
        [0.4, 0.3, 0.2, 0.1],
        [6, 2, 0, 0],
        [-0.9, -0.3, None, None],
        -0.05,
    ))
    cases.append((
        "edge/negative-root-positive-q",
        [0.3, 0.3, 0.2, 0.2],
        [8, 4, 0, 0],
        [0.7, 0.55, None, None],
        -1.0,
    ))

    # Single legal action: the softmax over one element is exactly 1.0.
    cases.append(("edge/single-action-visited", [1.0], [7], [0.33], 0.1))
    cases.append(("edge/single-action-unvisited", [1.0], [0], None, -0.6))

    # Priors with (almost) all their mass off the visited actions: the v_mix
    # denominator is the sum of the VISITED priors, so this is where it is
    # smallest. `zero-mass` uses an exact 0.0 prior on a visited action, which
    # is what the engine's POLICY_FLOOR exists for.
    cases.append((
        "edge/alpha-prior-tiny-visited-mass",
        [1e-9, 2e-9, 1.0 - 3e-9],
        [5, 3, 0],
        [0.2, -0.4, None],
        0.15,
    ))
    cases.append((
        "edge/alpha-prior-zero-mass-visited",
        [0.0, 0.0, 0.5, 0.5],
        [4, 6, 0, 0],
        [0.9, -0.9, None, None],
        0.05,
    ))
    cases.append((
        "edge/alpha-prior-one-hot-unvisited",
        [1.0 - 3e-9, 1e-9, 1e-9, 1e-9],
        [0, 2, 2, 2],
        [None, 0.1, 0.2, 0.3],
        -0.35,
    ))

    # Extremes of the Q range, and a spread of exactly 1 ULP, where the rescale
    # divides by something far below the epsilon floor.
    cases.append(("edge/q-at-both-bounds", [0.5, 0.5], [10, 10], [1.0, -1.0], 0.0))
    tight = 0.25
    cases.append((
        "edge/q-one-ulp-apart",
        [0.5, 0.5],
        [9, 9],
        [tight, math.nextafter(tight, 1.0)],
        tight,
    ))
    # A visit count large enough that the visit scale, not the Q spread, sets
    # the sharpening.
    cases.append((
        "edge/huge-visit-count",
        [0.5, 0.3, 0.2],
        [100_000, 40_000, 0],
        [0.8, -0.8, None],
        0.0,
    ))
    return cases


def grid_cases(rng):
    """A grid over priors x visit counts x Q-values x maxN.

    The three shape axes that decide the *arithmetic* -- the prior's shape, the
    visit pattern and the Q range -- are ENUMERATED, all 6 x 6 x 6 of them, so
    every combination really is represented. The two axes that only decide the
    problem's size -- the action count and the top visit count -- are drawn
    independently from `rng` for each combination.

    An earlier version derived all five axes from one counter with different
    strides, which phase-locked them: 64 of 1512 combinations, and only 12 of
    the 36 prior x visit pairs. That is the kind of hole a grid is supposed to
    close, so it is enumerated now rather than sampled.
    """
    action_counts = [2, 3, 5, 8, 16, 40, 64]
    prior_shapes = ["uniform", "dirichlet-0.1", "dirichlet-1", "dirichlet-10", "peaked", "floored"]
    visit_shapes = ["none", "one", "sparse", "ladder", "all", "heavy-tail"]
    q_shapes = ["positive", "negative", "mixed", "tight", "wide", "extreme"]
    max_visits = [0, 1, 2, 7, 30, 128, 1000]

    combinations = [
        (prior_shape, visit_shape, q_shape)
        for prior_shape in prior_shapes
        for visit_shape in visit_shapes
        for q_shape in q_shapes
    ]

    cases = []
    for index, (prior_shape, visit_shape, q_shape) in enumerate(combinations):
        num_actions = int(rng.choice(action_counts))
        top_visits = int(rng.choice(max_visits))

        if prior_shape == "uniform":
            priors = np.full(num_actions, 1.0 / num_actions)
        elif prior_shape == "peaked":
            priors = np.full(num_actions, 1e-6)
            priors[rng.integers(num_actions)] = 1.0
        elif prior_shape == "floored":
            priors = rng.dirichlet(np.full(num_actions, 1.0))
            priors[rng.integers(num_actions)] = 0.0
        else:
            alpha = float(prior_shape.split("-")[1])
            priors = rng.dirichlet(np.full(num_actions, alpha))
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(num_actions, 1.0 / num_actions)

        visits = np.zeros(num_actions, dtype=np.int64)
        if visit_shape == "none" or top_visits == 0:
            pass
        elif visit_shape == "one":
            visits[rng.integers(num_actions)] = top_visits
        elif visit_shape == "sparse":
            chosen = rng.choice(num_actions, size=max(1, num_actions // 8), replace=False)
            visits[chosen] = rng.integers(1, top_visits + 1, size=len(chosen))
            visits[chosen[0]] = top_visits
        elif visit_shape == "ladder":
            # What sequential halving actually produces: a few survivors on the
            # same count, the eliminated field on smaller ones.
            survivors = max(1, num_actions // 4)
            visits[:survivors] = top_visits
            visits[survivors : survivors * 2] = max(1, top_visits // 2)
            visits[survivors * 2 : survivors * 3] = max(1, top_visits // 4)
        elif visit_shape == "all":
            visits[:] = rng.integers(1, top_visits + 1, size=num_actions)
            visits[rng.integers(num_actions)] = top_visits
        else:  # heavy-tail
            visits[0] = top_visits
            visits[1:] = 1

        if q_shape == "positive":
            qvalues = rng.uniform(0.0, 1.0, num_actions)
        elif q_shape == "negative":
            qvalues = rng.uniform(-1.0, 0.0, num_actions)
        elif q_shape == "mixed":
            qvalues = rng.uniform(-1.0, 1.0, num_actions)
        elif q_shape == "tight":
            centre = rng.uniform(-1.0, 1.0)
            qvalues = centre + rng.uniform(-1e-7, 1e-7, num_actions)
        elif q_shape == "wide":
            qvalues = rng.choice([-1.0, 1.0], num_actions) * rng.uniform(0.5, 1.0, num_actions)
        else:  # extreme
            qvalues = rng.choice([-1.0, 1.0], num_actions)

        raw_value = float(rng.uniform(-1.0, 1.0))
        cases.append((
            f"grid/{index:03d}-{num_actions}a-{prior_shape}-{visit_shape}-{q_shape}-n{top_visits}",
            [float(p) for p in priors],
            [int(v) for v in visits],
            [float(q) for q in qvalues],
            raw_value,
        ))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rust/normal-duel-core/tests/fixtures/qtransform-mctx-goldens.json",
    )
    parser.add_argument("--seed", type=int, default=20_260_809)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    specs = edge_cases() + grid_cases(rng)

    cases = []
    for name, priors, visits, qvalues, raw_value in specs:
        if qvalues is None:
            qvalues = [POISON_Q] * len(priors)
        qvalues = [POISON_Q if q is None else q for q in qvalues]
        cases.append(golden(name, priors, visits, qvalues, raw_value))

    document = {
        "generator": "scripts/gen-qtransform-goldens.py",
        "source": "mctx.qtransform_completed_by_mix_value + gumbel_muzero_policy action_weights",
        "mctxVersion": mctx.__version__,
        "jaxVersion": jax.__version__,
        "dtype": "float64",
        "policyFloor": POLICY_FLOOR,
        "poisonQ": POISON_Q,
        "params": {
            "valueScale": VALUE_SCALE,
            "maxvisitInit": MAXVISIT_INIT,
            "epsilon": EPSILON,
        },
        "priorLogitConvention": "mctx received log(max(prior, policyFloor)) as children_prior_logits",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
