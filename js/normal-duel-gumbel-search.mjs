/**
 * Stage 4 Gumbel-AlphaZero root search and deterministic self-play generator
 * for the canonical 9x9 normal duel (`docs/ai-engine-plan.md:424-455`).
 *
 * This is the *validation layer* the plan asks for before any training runs:
 * the complete self-play loop — search, replay, evaluation — exercised on CPU
 * with a stub evaluator, so that determinism and 100% legality are proven
 * before a network exists. Nothing here has learned knowledge, and no strength
 * claim can be made from it.
 *
 * Rules authority
 * ---------------
 * Every legality question, transition and adjudication goes through
 * `js/normal-duel-engine.mjs`. Tensors go through `js/normal-duel-nn-encoding.mjs`.
 * Randomness goes through `js/lcg32.mjs`. Nothing here reimplements a rule.
 *
 * Absolute action frame
 * ---------------------
 * Action codes and input planes are both engine-absolute and deliberately
 * coherent (see the header of `normal-duel-nn-encoding.mjs`). No mirroring or
 * canonicalisation is applied anywhere in this file.
 *
 * Determinism
 * -----------
 * All randomness comes from `createLcg32(seed)`. There is no `Math.random`, no
 * `Date`, no `performance.now`. Where a decision could otherwise depend on
 * insertion or iteration order, candidates are sorted by action code first, so
 * the same seed and the same evaluator reproduce a byte-identical game.
 * (`Math.log` precision is implementation-defined by ECMA-262, so "identical"
 * means identical on a given engine build — which is what replay requires.)
 */

import {
  applyLegalAction, createInitialState, decodeAction, edgeBlocked, legalActionCodes, policySize,
  validateConfig, validateState
} from './normal-duel-engine.mjs';
import { encodePolicyTarget, encodeState, encodeLegalPolicyTarget } from './normal-duel-nn-encoding.mjs';
import { createLcg32 } from './lcg32.mjs';

/** Frozen identifier for this search + self-play record format. */
export const GUMBEL_SEARCH_VERSION = 'gumbel-az-root-v1';

export class GumbelSearchError extends Error {
  constructor(reason) { super(reason); this.name = 'GumbelSearchError'; this.reason = reason; }
}

function fail(reason) { throw new GumbelSearchError(reason); }

/**
 * Floor applied inside the logit so a legal action the policy assigns exactly
 * zero probability is merely very unlikely, not `-Infinity` (which would make
 * `g + logit` NaN-adjacent and destroy the ordering).
 */
const POLICY_FLOOR = 1e-9;

/** Gumbel-MuZero sigma constants; see `sigma` below. */
const C_VISIT = 50;
const C_SCALE = 1;

const DIRECTIONS = Object.freeze([[-1, 0], [1, 0], [0, -1], [0, 1]]);

function canonical9x9(config) {
  const checked = validateConfig(config);
  // Stage 4 is specified on the canonical 9x9 board only.
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

/* ------------------------------------------------------------------ *
 * Stub evaluator
 * ------------------------------------------------------------------ */

/**
 * Blocked-edge lookup for one position, built by asking the engine about every
 * orthogonal edge. Wall text is never parsed here; the H/V anchor convention
 * stays owned by the engine.
 *
 * Indexed `r * columns + c`: `h[i]` is the edge between (r,c) and (r+1,c),
 * `v[i]` the edge between (r,c) and (r,c+1) — the same convention the encoder's
 * wall planes use.
 */
function blockedEdges(config, walls) {
  const { rows, columns } = config;
  const h = new Uint8Array(rows * columns);
  const v = new Uint8Array(rows * columns);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      if (r + 1 < rows && edgeBlocked(config, { r, c }, { r: r + 1, c }, walls)) h[r * columns + c] = 1;
      if (c + 1 < columns && edgeBlocked(config, { r, c }, { r, c: c + 1 }, walls)) v[r * columns + c] = 1;
    }
  }
  return { h, v };
}

function edgeIsBlocked(config, blocked, fromR, fromC, toR, toC) {
  const { columns } = config;
  if (fromR === toR) {
    const c = Math.min(fromC, toC);
    return blocked.v[fromR * columns + c] === 1;
  }
  const r = Math.min(fromR, toR);
  return blocked.h[r * columns + fromC] === 1;
}

/**
 * Breadth-first shortest path length, in steps, from a pawn to its goal row,
 * respecting walls and ignoring the opposing pawn. Returns `Infinity` if no
 * path exists (the engine forbids that, so it is a defensive value only).
 */
function goalDistance(config, blocked, from, goalRow) {
  const { rows, columns } = config;
  const distance = new Int32Array(rows * columns).fill(-1);
  const queue = [from.r * columns + from.c];
  distance[queue[0]] = 0;
  for (let head = 0; head < queue.length; head += 1) {
    const index = queue[head];
    const r = Math.floor(index / columns); const c = index % columns;
    if (r === goalRow) return distance[index];
    for (const [dr, dc] of DIRECTIONS) {
      const nr = r + dr; const nc = c + dc;
      if (nr < 0 || nr >= rows || nc < 0 || nc >= columns) continue;
      const next = nr * columns + nc;
      if (distance[next] !== -1 || edgeIsBlocked(config, blocked, r, c, nr, nc)) continue;
      distance[next] = distance[index] + 1;
      queue.push(next);
    }
  }
  return Infinity;
}

/**
 * PLACEHOLDER evaluator. This is *not* a network and carries no learned
 * knowledge whatsoever — it exists so the Stage 4 self-play loop can be
 * validated on CPU before training. Any strength measured against it is
 * meaningless.
 *
 * Policy: uniform over the legal actions of `state`, zero everywhere else.
 * Value: a bounded, monotone function of the shortest-path difference from the
 * side-to-move's perspective, `d / (1 + |d|)` with `d = (distOpponent -
 * distMover) / 2`. Only exact IEEE arithmetic is used, so the value is
 * bit-reproducible. Terminal states return the adjudicated result directly.
 *
 * Pure: neither `config` nor `state` is mutated, and a fresh Float32Array is
 * returned every call.
 */
export function uniformStubEvaluator(config, state) {
  const checked = canonical9x9(config);
  const validated = validateState(checked, state);
  const size = policySize(checked);
  const policy = new Float32Array(size);

  const legal = legalActionCodes(checked, validated);
  if (legal.length > 0) {
    const share = 1 / legal.length;
    for (const code of legal) policy[code] = share;
  }

  const mover = validated.position.turn;
  if (validated.outcome.kind === 'win') {
    return { policy, value: validated.outcome.winner === mover ? 1 : -1 };
  }
  if (validated.outcome.kind !== 'ongoing') return { policy, value: 0 };

  const opponent = mover === 'A' ? 'B' : 'A';
  const blocked = blockedEdges(checked, validated.position.walls);
  const moverDistance = goalDistance(checked, blocked, validated.position.pawns[mover], checked.goalRows[mover]);
  const opponentDistance = goalDistance(checked, blocked, validated.position.pawns[opponent], checked.goalRows[opponent]);
  if (!Number.isFinite(moverDistance) || !Number.isFinite(opponentDistance)) return { policy, value: 0 };

  const difference = (opponentDistance - moverDistance) / 2;
  return { policy, value: difference / (1 + Math.abs(difference)) };
}

/* ------------------------------------------------------------------ *
 * Gumbel root search
 * ------------------------------------------------------------------ */

/**
 * `u` is drawn strictly inside (0, 1) — `(x + 0.5) / 2^32` for a uint32 `x` —
 * so `-log(-log(u))` is always finite; no clamping branch can ever be hit, but
 * the guard is kept so a substituted `random` cannot produce an infinity.
 */
function gumbel(random) {
  const raw = random();
  if (!Number.isSafeInteger(raw) || raw < 0 || raw > 0xffff_ffff) fail('invalid_random_output');
  let u = (raw + 0.5) / 4294967296;
  if (!(u > 0)) u = Number.MIN_VALUE;
  if (!(u < 1)) u = 1 - Number.EPSILON / 2;
  return -Math.log(-Math.log(u));
}

/**
 * Monotone transform of the estimated value used by sequential halving, in the
 * Gumbel-MuZero form:
 *
 *     sigma(q) = (C_VISIT + maxVisits) * C_SCALE * q
 *
 * with `C_VISIT = 50`, `C_SCALE = 1` and `maxVisits` the largest visit count
 * among the considered candidates at that moment. It is strictly increasing in
 * `q` (the scale is positive), so it never reorders two candidates by value; it
 * only grows the weight of the value evidence relative to `g + logit` as visits
 * accumulate.
 */
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
 * Value of a child state from the ROOT mover's perspective. Terminal children
 * are adjudicated by the engine, so `evaluate` is never asked about a finished
 * game; ongoing children are evaluated once and the value negated, since
 * `evaluate` reports from the child's side-to-move perspective.
 */
function childValue(config, size, evaluate, child) {
  if (child.outcome.kind === 'win') return child.outcome.winner === child.position.turn ? -1 : 1;
  if (child.outcome.kind !== 'ongoing') return 0;
  return -readEvaluation(size, evaluate(config, child)).value;
}

/**
 * Visit counts for policy-target purposes. A search with nothing to decide (a
 * single candidate) or with a zero budget spends no simulations, so the raw
 * counts sum to zero and cannot be normalised; the played action then carries
 * the whole target. Exposed so `selfPlayGame` and callers agree on the rule.
 */
export function effectiveVisitCounts(result) {
  let total = 0;
  for (const count of result.visitCounts.values()) total += count;
  if (total > 0) return result.visitCounts;
  return new Map([[result.actionCode, 1]]);
}

/**
 * Gumbel-AlphaZero root action selection.
 *
 * Depth 1: a simulation expands one root child, adjudicates or evaluates it,
 * and backs the value up to the root. There is no tree beyond the root children
 * at this milestone — that is deliberate and sufficient for the Stage 4 loop
 * validation, and stated plainly so nobody mistakes it for full MCTS.
 *
 * Because `evaluate` is required to be pure and depth is 1, a child's value is
 * a function of the child state alone; it is computed on the first visit and
 * reused for later visits of the same child. Repeat visits still consume the
 * budget and still count as simulations — they simply do not re-call a function
 * whose output cannot change.
 *
 * Schedule: with `m` candidates the search runs `R = ceil(log2(m))` halving
 * rounds. A round over `k` survivors allocates `max(1, floor(simulations /
 * (R * k)))` visits per candidate, handed out one visit at a time in ascending
 * action-code order so a budget that runs out mid-round is spread evenly and
 * deterministically. The round then keeps the top `ceil(k / 2)` by
 * `g + logit + sigma(qhat)`. Rounds stop when one candidate remains; if the
 * budget is exhausted first, the remaining halvings run on the statistics
 * already gathered.
 */
export function gumbelRootSearch({ config, state, evaluate, simulations, maxConsidered, random }) {
  const checked = canonical9x9(config);
  const validated = validateState(checked, state);
  if (typeof evaluate !== 'function') fail('invalid_evaluator');
  if (typeof random !== 'function') fail('invalid_random');
  positiveInteger(simulations, 'invalid_simulations');
  positiveInteger(maxConsidered, 'invalid_max_considered');
  if (validated.outcome.kind !== 'ongoing') fail('terminal_state');

  const size = policySize(checked);
  const legal = legalActionCodes(checked, validated); // engine returns ascending codes
  if (legal.length === 0) fail('no_legal_actions');

  const root = readEvaluation(size, evaluate(checked, validated));
  const rootValue = root.value;

  // One Gumbel per legal action, drawn in ascending code order so the draw
  // order — and therefore the game — never depends on iteration order.
  const candidates = legal.map((code) => {
    const logit = Math.log(Math.max(root.policy[code], POLICY_FLOOR));
    return { code, score: gumbel(random) + logit, visits: 0, valueSum: 0, cachedValue: null };
  });

  const considered = candidates
    .slice()
    .sort((left, right) => (right.score - left.score) || (left.code - right.code))
    .slice(0, Math.min(maxConsidered, candidates.length))
    .sort((left, right) => left.code - right.code);

  let budget = simulations;
  let used = 0;

  const simulate = (candidate) => {
    if (candidate.cachedValue === null) {
      const child = applyLegalAction(checked, validated, decodeAction(checked, candidate.code));
      candidate.cachedValue = childValue(checked, size, evaluate, child);
    }
    candidate.visits += 1;
    candidate.valueSum += candidate.cachedValue;
    budget -= 1;
    used += 1;
  };

  const qhat = (candidate) => (candidate.visits > 0 ? candidate.valueSum / candidate.visits : rootValue);

  let survivors = considered;
  const rounds = Math.max(1, Math.ceil(Math.log2(Math.max(survivors.length, 2))));
  while (survivors.length > 1) {
    const perCandidate = Math.max(1, Math.floor(simulations / (rounds * survivors.length)));
    for (let pass = 0; pass < perCandidate && budget > 0; pass += 1) {
      for (const candidate of survivors) {
        if (budget <= 0) break;
        simulate(candidate);
      }
    }
    let maxVisits = 0;
    for (const candidate of survivors) maxVisits = Math.max(maxVisits, candidate.visits);
    const ranked = survivors
      .slice()
      .sort((left, right) => {
        const delta = (right.score + sigma(qhat(right), maxVisits)) - (left.score + sigma(qhat(left), maxVisits));
        return delta || (left.code - right.code);
      });
    survivors = ranked.slice(0, Math.ceil(survivors.length / 2)).sort((left, right) => left.code - right.code);
  }

  const winner = survivors[0];
  const visitCounts = new Map(considered.map((candidate) => [candidate.code, candidate.visits]));

  let total = 0;
  for (const candidate of considered) total += candidate.visits;
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
    considered: Object.freeze(considered.map((candidate) => candidate.code))
  });
}

/* ------------------------------------------------------------------ *
 * Self-play
 * ------------------------------------------------------------------ */

/**
 * Play one full game with `gumbelRootSearch` at every ply, from the engine's
 * initial state until the engine reports a terminal outcome or `plyCap` plies
 * have been played.
 *
 * A single `createLcg32(seed)` stream drives the whole game, so the seed alone
 * reproduces it exactly. Every played action is asserted to be a member of
 * `legalActionCodes` for the exact state it is played from — the plan's exit
 * gate demands 100% legality, so it is checked rather than assumed.
 *
 * `z` is the game result from each ply's side-to-move perspective (+1 won, -1
 * lost, 0 draw or unresolved at `plyCap`), assigned after the game ends.
 */
export function selfPlayGame({ config, evaluate, simulations, maxConsidered, seed, plyCap }) {
  const checked = canonical9x9(config);
  const evaluator = evaluate ?? uniformStubEvaluator;
  if (typeof evaluator !== 'function') fail('invalid_evaluator');
  positiveInteger(simulations, 'invalid_simulations');
  positiveInteger(maxConsidered, 'invalid_max_considered');
  const cap = plyCap === undefined ? checked.plyCap : positiveInteger(plyCap, 'invalid_ply_cap');
  const random = createLcg32(seed); // throws on a non-uint32 seed

  const plies = [];
  let state = createInitialState(checked);

  while (state.outcome.kind === 'ongoing' && plies.length < cap) {
    const turn = state.position.turn;
    const result = gumbelRootSearch({
      config: checked, state, evaluate: evaluator, simulations, maxConsidered, random
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
    version: GUMBEL_SEARCH_VERSION,
    plies: Object.freeze(plies),
    outcome,
    finalPly: plies.length
  });
}
