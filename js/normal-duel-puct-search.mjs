/**
 * Stage 4 PUCT tree search and deterministic self-play generator for the
 * canonical 9x9 normal duel.
 *
 * Why this exists
 * ---------------
 * `js/normal-duel-gumbel-search.mjs` expands the root children and evaluates
 * each exactly once: a depth-1 horizon with no tree behind it. A network
 * distilled from the classical engine still lost 0-24 against that engine,
 * which runs alpha-beta to depth 3+. This module keeps the Gumbel root
 * machinery (one Gumbel per legal action, sequential halving over the top `m`)
 * but makes every root "visit" run a *full tree simulation*: select by PUCT to
 * a leaf, expand it once, back the value up to the root negating at each ply.
 *
 * No strength claim is made here. Strength is measured separately against the
 * classical engine.
 *
 * Rules authority
 * ---------------
 * Every legality question, transition and adjudication goes through
 * `js/normal-duel-engine.mjs`; terminal states are scored from the *engine's*
 * outcome and are never handed to the network. Tensors go through
 * `js/normal-duel-nn-encoding.mjs`. Randomness goes through `js/lcg32.mjs`.
 * Nothing here reimplements a rule.
 *
 * Absolute action frame
 * ---------------------
 * Action codes and input planes are engine-absolute and coherent. No mirroring
 * or coordinate transform is applied anywhere in this file.
 *
 * Determinism
 * -----------
 * All randomness comes from `createLcg32(seed)`. There is no `Math.random`, no
 * `Date`, no `performance.now`. Every ordering decision — Gumbel draw order,
 * PUCT argmax, halving survivors — breaks ties by ascending action code, so the
 * same seed and evaluator reproduce a byte-identical game.
 *
 * Sign convention (the classic silent bug)
 * ----------------------------------------
 * A node's value, its edges' `Q`, and `evaluate`'s value are ALL from that
 * node's side-to-move perspective. `simulate(node)` returns a value in the
 * node's frame; the caller negates once when crossing the ply boundary. A sign
 * error here makes search actively worse than no search, so it is asserted in
 * the tests rather than trusted.
 */

import {
  applyLegalAction, createInitialState, decodeAction, legalActionCodes, policySize,
  validateConfig, validateState
} from './normal-duel-engine.mjs';
import { encodePolicyTarget, encodeState, encodeLegalPolicyTarget } from './normal-duel-nn-encoding.mjs';
import { createLcg32 } from './lcg32.mjs';

/** Frozen identifier for this search + self-play record format. */
export const PUCT_SEARCH_VERSION = 'puct-az-tree-v1';

export class PuctSearchError extends Error {
  constructor(reason) { super(reason); this.name = 'PuctSearchError'; this.reason = reason; }
}

function fail(reason) { throw new PuctSearchError(reason); }

/**
 * Floor applied inside the logit so a legal action the policy assigns exactly
 * zero probability is merely very unlikely, not `-Infinity`.
 */
const POLICY_FLOOR = 1e-9;

/** Gumbel-MuZero sigma constants, matching the landed root search. */
const C_VISIT = 50;
const C_SCALE = 1;

/** Default exploration constant if the caller does not supply one. */
export const DEFAULT_C_PUCT = 1.25;

/**
 * First-play-urgency reduction.
 *
 * FPU rule: an edge with `N === 0` takes
 *
 *     Q = clamp(parentValue - FPU_REDUCTION * sqrt(sumVisitedPrior), -1, +1)
 *
 * where `parentValue` is the node's own value estimate in its own frame (the
 * network value on expansion, refined to `W/N` once the node has been visited)
 * and `sumVisitedPrior` is the total prior mass of that node's already-visited
 * children. So the first child is optimistic-at-parent-value, and each visited
 * sibling makes the remaining unvisited ones progressively less attractive.
 * This is what lets the tree go deep on a narrow line instead of fanning out
 * over all ~130 legal actions at every ply.
 */
export const FPU_REDUCTION = 0.25;

function canonical9x9(config) {
  const checked = validateConfig(config);
  if (checked.rows !== 9 || checked.columns !== 9) fail('unsupported_board');
  return checked;
}

function positiveInteger(value, reason) {
  if (!Number.isSafeInteger(value) || value < 1) fail(reason);
  return value;
}

function nonNegativeInteger(value, reason) {
  if (!Number.isSafeInteger(value) || value < 0) fail(reason);
  return value;
}

function positiveFinite(value, reason) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) fail(reason);
  return value;
}

function clampValue(value) {
  if (value > 1) return 1;
  if (value < -1) return -1;
  return value;
}

/**
 * `u` is drawn strictly inside (0, 1) so `-log(-log(u))` is always finite.
 */
function gumbel(random) {
  const raw = random();
  if (!Number.isSafeInteger(raw) || raw < 0 || raw > 0xffff_ffff) fail('invalid_random_output');
  let u = (raw + 0.5) / 4294967296;
  if (!(u > 0)) u = Number.MIN_VALUE;
  if (!(u < 1)) u = 1 - Number.EPSILON / 2;
  return -Math.log(-Math.log(u));
}

/** Strictly increasing in `q`, so it never reorders two candidates by value. */
function sigma(q, maxVisits) {
  return (C_VISIT + maxVisits) * C_SCALE * q;
}

function readEvaluation(size, result) {
  if (result === null || typeof result !== 'object') fail('invalid_evaluation');
  const { policy, value } = result;
  if (!(policy instanceof Float32Array) && !(policy instanceof Float64Array) && !Array.isArray(policy)) fail('invalid_evaluation');
  if (policy.length !== size) fail('invalid_evaluation');
  if (typeof value !== 'number' || !Number.isFinite(value) || value < -1 || value > 1) fail('invalid_evaluation');
  // Copy: the evaluator's array is the caller's, and must not be retained or written to.
  const copied = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    const probability = policy[index];
    if (typeof probability !== 'number' || !Number.isFinite(probability) || probability < 0) fail('invalid_evaluation');
    copied[index] = probability;
  }
  return { policy: copied, value };
}

/**
 * Value of a terminal state from its own side-to-move perspective, taken from
 * the engine's adjudication. Covers goal, ply cap and threefold repetition —
 * the engine owns all three. The network is never consulted here.
 */
function terminalValue(state) {
  if (state.outcome.kind === 'win') return state.outcome.winner === state.position.turn ? 1 : -1;
  return 0;
}

/* ------------------------------------------------------------------ *
 * Tree
 * ------------------------------------------------------------------ */

/**
 * A node owns its state and one edge per legal action. Edges are stored in
 * ascending action-code order (the engine returns them that way), so argmax
 * ties resolve to the lowest code without an extra sort.
 */
function createNode(state) {
  return {
    state,
    terminal: state.outcome.kind !== 'ongoing',
    expanded: false,
    edges: null,
    visits: 0,
    valueSum: 0,
    /** Node value in its OWN frame: network value, refined to W/N once visited. */
    value: 0,
    visitedPrior: 0
  };
}

/** Node value estimate in the node's own frame, used as the FPU baseline. */
function nodeValue(node) {
  return node.visits > 0 ? node.valueSum / node.visits : node.value;
}

function edgeQ(node, edge) {
  if (edge.visits > 0) return edge.valueSum / edge.visits;
  return clampValue(nodeValue(node) - FPU_REDUCTION * Math.sqrt(node.visitedPrior));
}

/**
 * `argmax(Q + cPuct * P * sqrt(sumN) / (1 + N))` over the node's edges, with
 * `sumN` the node's total edge visits. Ties break to the lowest action code
 * because edges are in ascending code order and the comparison is strict.
 */
function selectEdge(node, cPuct) {
  const sqrtTotal = Math.sqrt(node.visits);
  let best = null;
  let bestScore = -Infinity;
  for (const edge of node.edges) {
    const score = edgeQ(node, edge) + cPuct * edge.prior * sqrtTotal / (1 + edge.visits);
    if (score > bestScore) { bestScore = score; best = edge; }
  }
  return best;
}

/**
 * Evaluate a leaf once and install its edges. Priors are the network's policy
 * masked to the legal actions and renormalised; a policy with no mass on any
 * legal action falls back to uniform, so the search never divides by zero.
 * Returns the leaf value in the leaf's own frame.
 */
function expand(context, node) {
  const { config, size, evaluate } = context;
  const legal = legalActionCodes(config, node.state);
  if (legal.length === 0) fail('no_legal_actions');

  const { policy, value } = readEvaluation(size, evaluate(config, node.state));
  let mass = 0;
  for (const code of legal) mass += policy[code];

  node.edges = legal.map((code) => ({
    code,
    prior: mass > 0 ? policy[code] / mass : 1 / legal.length,
    visits: 0,
    valueSum: 0,
    child: null
  }));
  node.expanded = true;
  node.value = value;
  node.visitedPrior = 0;
  context.evaluations += 1;
  return value;
}

/**
 * One simulation from `node`, returning the backed-up value in `node`'s frame.
 *
 * Reuses the tree's stored child states instead of cloning the tree: only the
 * one new state produced by the selected action is built per ply, and only the
 * first time that edge is taken.
 */
function simulate(context, node, depth) {
  if (depth > context.maxDepth) context.maxDepth = depth;

  // Terminal: scored by the engine, never expanded, and costs no evaluation.
  if (node.terminal) {
    const value = terminalValue(node.state);
    node.visits += 1;
    node.valueSum += value;
    return value;
  }

  if (!node.expanded) {
    const value = expand(context, node);
    node.visits += 1;
    node.valueSum += value;
    return value;
  }

  const edge = selectEdge(node, context.cPuct);
  if (edge.child === null) {
    edge.child = createNode(applyLegalAction(context.config, node.state, decodeAction(context.config, edge.code)));
  }
  if (edge.visits === 0) node.visitedPrior += edge.prior;

  // Negate at every ply: the child's value is in the child's frame.
  const value = -simulate(context, edge.child, depth + 1);

  edge.visits += 1;
  edge.valueSum += value;
  node.visits += 1;
  node.valueSum += value;
  return value;
}

/**
 * Visit counts for policy-target purposes. A search with nothing to decide (a
 * single candidate) or with a zero budget spends no simulations, so the raw
 * counts sum to zero and cannot be normalised; the played action then carries
 * the whole target.
 */
export function effectiveVisitCounts(result) {
  let total = 0;
  for (const count of result.visitCounts.values()) total += count;
  if (total > 0) return result.visitCounts;
  return new Map([[result.actionCode, 1]]);
}

/* ------------------------------------------------------------------ *
 * PUCT search with a Gumbel root
 * ------------------------------------------------------------------ */

/**
 * Gumbel root action selection over a genuine PUCT tree.
 *
 * Root: one Gumbel per legal action, drawn in ascending code order; the top
 * `m = min(maxConsidered, legalCount)` by `g + logit` are considered, and
 * sequential halving runs over them. With `m` candidates there are
 * `R = ceil(log2(m))` rounds; a round over `k` survivors allocates
 * `max(1, floor(simulations / (R * k)))` visits per candidate, handed out one
 * at a time in ascending code order, then keeps the top `ceil(k / 2)` by
 * `g + logit + sigma(qhat)`.
 *
 * The difference from the landed depth-1 search: each visit runs a full tree
 * simulation through `simulate`, so repeat visits deepen the tree instead of
 * re-reading a cached child value.
 *
 * Budget: `simulations` bounds the number of simulations, and a simulation
 * performs at most one `evaluate` call, so leaf evaluations never exceed
 * `simulations`. The single root evaluation that produces the priors and
 * `rootValue` sits outside that budget, exactly as in the landed search.
 */
export function puctSearch({ config, state, evaluate, simulations, cPuct, random, maxConsidered }) {
  const checked = canonical9x9(config);
  const validated = validateState(checked, state);
  if (typeof evaluate !== 'function') fail('invalid_evaluator');
  if (typeof random !== 'function') fail('invalid_random');
  // Positive, not merely non-negative. With no simulations every search returns
  // empty visit counts, `effectiveVisitCounts` falls back to a one-hot of the
  // played action, and the recorded policy target becomes one-hot at a position
  // with ~130 legal codes -- silently, at full record count. That degenerate
  // target removes AlphaZero's improvement ratchet and is what cost an earlier
  // run 114 flat iterations. `maxConsidered` was already required to be
  // positive, so accepting 0 here was an asymmetry rather than a decision, and
  // this is the driver that produced the bad records: the Rust port guards it,
  // but the JS shard worker is the incumbent.
  positiveInteger(simulations, 'invalid_simulations');
  positiveInteger(maxConsidered, 'invalid_max_considered');
  const exploration = cPuct === undefined ? DEFAULT_C_PUCT : positiveFinite(cPuct, 'invalid_c_puct');
  if (validated.outcome.kind !== 'ongoing') fail('terminal_state');

  const size = policySize(checked);
  const context = {
    config: checked, size, evaluate, cPuct: exploration, maxDepth: 0, evaluations: 0
  };

  const root = createNode(validated);
  // Root expansion: priors and `rootValue`, outside the simulation budget.
  const rootValue = expand(context, root);
  context.evaluations = 0;

  const candidates = root.edges.map((edge) => {
    const logit = Math.log(Math.max(edge.prior, POLICY_FLOOR));
    return { edge, code: edge.code, score: gumbel(random) + logit };
  });

  const considered = candidates
    .slice()
    .sort((left, right) => (right.score - left.score) || (left.code - right.code))
    .slice(0, Math.min(maxConsidered, candidates.length))
    .sort((left, right) => left.code - right.code);

  let budget = simulations;
  let used = 0;

  const visit = (candidate) => {
    const { edge } = candidate;
    if (edge.child === null) {
      edge.child = createNode(applyLegalAction(checked, validated, decodeAction(checked, edge.code)));
    }
    if (edge.visits === 0) root.visitedPrior += edge.prior;
    const value = -simulate(context, edge.child, 1);
    edge.visits += 1;
    edge.valueSum += value;
    root.visits += 1;
    root.valueSum += value;
    budget -= 1;
    used += 1;
  };

  const qhat = (candidate) => (
    candidate.edge.visits > 0 ? candidate.edge.valueSum / candidate.edge.visits : rootValue
  );

  let survivors = considered;
  const rounds = Math.max(1, Math.ceil(Math.log2(Math.max(survivors.length, 2))));
  while (survivors.length > 1) {
    const perCandidate = Math.max(1, Math.floor(simulations / (rounds * survivors.length)));
    for (let pass = 0; pass < perCandidate && budget > 0; pass += 1) {
      for (const candidate of survivors) {
        if (budget <= 0) break;
        visit(candidate);
      }
    }
    let maxVisits = 0;
    for (const candidate of survivors) maxVisits = Math.max(maxVisits, candidate.edge.visits);
    const ranked = survivors
      .slice()
      .sort((left, right) => {
        const delta = (right.score + sigma(qhat(right), maxVisits)) - (left.score + sigma(qhat(left), maxVisits));
        return delta || (left.code - right.code);
      });
    survivors = ranked.slice(0, Math.ceil(survivors.length / 2)).sort((left, right) => left.code - right.code);
  }

  // A single candidate still deserves the budget: it deepens the tree and gives
  // a real `rootValue` refinement, and the played action is forced anyway.
  if (survivors.length === 1) {
    while (budget > 0) visit(survivors[0]);
  }

  const winner = survivors[0];
  const visitCounts = new Map(considered.map((candidate) => [candidate.code, candidate.edge.visits]));

  let total = 0;
  for (const candidate of considered) total += candidate.edge.visits;
  const improvedPolicy = total > 0
    ? encodePolicyTarget(checked, visitCounts)
    : encodePolicyTarget(checked, new Map([[winner.code, 1]]));

  return Object.freeze({
    action: Object.freeze(decodeAction(checked, winner.code)),
    actionCode: winner.code,
    visitCounts,
    improvedPolicy,
    rootValue,
    simulationsUsed: used,
    maxDepthReached: context.maxDepth,
    considered: Object.freeze(considered.map((candidate) => candidate.code))
  });
}

/* ------------------------------------------------------------------ *
 * Self-play
 * ------------------------------------------------------------------ */

/**
 * Play one full game with `puctSearch` at every ply. Record shape is identical
 * to `selfPlayGame` in `js/normal-duel-gumbel-search.mjs` (`features`,
 * `policyTarget`, `visitCounts`, `z` per ply), so `train.py` needs no change.
 *
 * A single `createLcg32(seed)` stream drives the whole game. Every played
 * action is asserted to be a member of `legalActionCodes` for the exact state
 * it is played from — 100% legality is checked, not assumed.
 */
export function selfPlayGamePuct({ config, evaluate, simulations, cPuct, maxConsidered, seed, plyCap }) {
  const checked = canonical9x9(config);
  if (typeof evaluate !== 'function') fail('invalid_evaluator');
  // Positive, not merely non-negative. With no simulations every search returns
  // empty visit counts, `effectiveVisitCounts` falls back to a one-hot of the
  // played action, and the recorded policy target becomes one-hot at a position
  // with ~130 legal codes -- silently, at full record count. That degenerate
  // target removes AlphaZero's improvement ratchet and is what cost an earlier
  // run 114 flat iterations. `maxConsidered` was already required to be
  // positive, so accepting 0 here was an asymmetry rather than a decision, and
  // this is the driver that produced the bad records: the Rust port guards it,
  // but the JS shard worker is the incumbent.
  positiveInteger(simulations, 'invalid_simulations');
  positiveInteger(maxConsidered, 'invalid_max_considered');
  const exploration = cPuct === undefined ? DEFAULT_C_PUCT : positiveFinite(cPuct, 'invalid_c_puct');
  const cap = plyCap === undefined ? checked.plyCap : positiveInteger(plyCap, 'invalid_ply_cap');
  const random = createLcg32(seed); // throws on a non-uint32 seed

  const plies = [];
  let state = createInitialState(checked);

  while (state.outcome.kind === 'ongoing' && plies.length < cap) {
    const turn = state.position.turn;
    const result = puctSearch({
      config: checked, state, evaluate, simulations, cPuct: exploration, maxConsidered, random
    });

    const legal = legalActionCodes(checked, state);
    if (!legal.includes(result.actionCode)) fail('illegal_search_action');

    const features = encodeState(checked, state);
    const policyTarget = encodeLegalPolicyTarget(checked, state, effectiveVisitCounts(result));

    plies.push({
      ply: state.ply,
      turn,
      actionCode: result.actionCode,
      visitCounts: new Map([...result.visitCounts.entries()].sort((left, right) => left[0] - right[0])),
      features,
      policyTarget,
      z: 0
    });

    state = applyLegalAction(checked, state, decodeAction(checked, result.actionCode));
  }

  const outcome = Object.freeze({ ...state.outcome });
  for (const record of plies) {
    record.z = outcome.kind === 'win' ? (record.turn === outcome.winner ? 1 : -1) : 0;
    Object.freeze(record);
  }

  return Object.freeze({
    seed,
    version: PUCT_SEARCH_VERSION,
    plies: Object.freeze(plies),
    outcome,
    finalPly: plies.length
  });
}
