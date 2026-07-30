/**
 * Deterministic perft helpers for normal-duel-v1.
 *
 * A perft count is the number of exact-depth leaf occurrences in the legal-
 * action tree, never a count of unique positions:
 * `P(state, 0) = 1`; `P(terminal, depth > 0) = 0`; and an ongoing state is
 * the sum of its children's `P(child, depth - 1)`.  In particular, a branch
 * which ends early contributes zero to every deeper exact-depth total.
 *
 * Traversal deliberately retains no search tree. `countExact` only keeps one
 * active call frame per remaining ply. `perftReport` additionally retains its
 * returned root divide: one action and one compact `childLeavesByDepth` vector
 * per root action. This is O(depth) temporary traversal memory (plus required
 * output and the rules engine's one-position legal-action list at each frame).
 */
import { applyAction, applyLegalAction, createInitialState, encodeAction, legalActions, validateState } from './normal-duel-engine.mjs';
import { createLcg32 } from './lcg32.mjs';

export const MAX_PERFT_DEPTH = 4;
/**
 * Conservative default maximum number of counted state evaluations in one
 * perft call or report. Callers can explicitly select a larger budget up to
 * `MAX_PERFT_NODES_HARD_CAP` for a deliberately bounded wall-rich probe.
 */
export const MAX_PERFT_NODES = 400;
/**
 * Absolute ceiling for an explicitly requested perft budget. The depth cap
 * and scalar traversal remain in force, preventing accidental unbounded
 * resource use while allowing reviewed larger fixture probes.
 */
export const MAX_PERFT_NODES_HARD_CAP = 5_000;
/**
 * `nodeVisits` is an exact scalar-evaluation counter, not a unique-position
 * or full game-tree-node metric. It charges the root report evaluation once
 * and every `countExact(state, remaining)` invocation. A depth-one report
 * obtains each root child's depth-zero leaf algebraically without materializing
 * a child. For report depths two and above, each root child is materialized
 * once to seed its deeper scalar counts, but that child's depth-zero leaf is
 * still algebraic and does not add a separate charge. Each requested report
 * depth is counted independently, so the same game state can be charged
 * repeatedly.
 *
 * The counter is incremented before a scalar evaluation can inspect terminal
 * state or expand legal actions. Traversal throws before the next evaluation
 * would exceed its deterministic budget.
 */

function fail(message) { throw new TypeError(`normal-duel-perft: ${message}`); }
function nonNegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) fail(`${name} must be a non-negative safe integer`);
  return value;
}
function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${name} must be a positive safe integer`);
  return value;
}
function perftDepth(value, name) {
  const depth = nonNegativeInteger(value, name);
  if (depth > MAX_PERFT_DEPTH) fail(`${name} exceeds MAX_PERFT_DEPTH (${MAX_PERFT_DEPTH})`);
  return depth;
}
function nodeBudget(options) {
  if (options === undefined) return MAX_PERFT_NODES;
  if (options === null || typeof options !== 'object' || Array.isArray(options)) fail('options must be an object');
  const keys = Object.keys(options);
  if (keys.length !== 1 || keys[0] !== 'maxNodes') fail('options must contain only maxNodes');
  const budget = positiveInteger(options.maxNodes, 'maxNodes');
  if (budget > MAX_PERFT_NODES_HARD_CAP) {
    fail(`maxNodes exceeds MAX_PERFT_NODES_HARD_CAP (${MAX_PERFT_NODES_HARD_CAP})`);
  }
  return budget;
}
function frozenAction(action) {
  return Object.freeze(action.kind === 'pawn'
    ? { kind: 'pawn', to: Object.freeze({ r: action.to.r, c: action.to.c }) }
    : { kind: 'wall', wall: action.wall });
}
function frozenReport(report) {
  return Object.freeze({
    depth: report.depth,
    leavesByDepth: Object.freeze([...report.leavesByDepth]),
    nodeVisits: report.nodeVisits,
    divide: Object.freeze(report.divide.map((entry) => Object.freeze({
      action: frozenAction(entry.action),
      actionCode: entry.actionCode,
      childLeavesByDepth: Object.freeze([...entry.childLeavesByDepth])
    })))
  });
}

function counterFor(maxNodes) {
  let visits = 0;
  return Object.freeze({
    charge() {
      if (visits >= maxNodes) fail(`node budget exceeded (${maxNodes} counted state visits)`);
      visits += 1;
    },
    get visits() { return visits; }
  });
}

/**
 * Count only P(current, remaining). It returns one scalar and retains only a
 * recursion stack; siblings are applied, counted, and discarded one at a time.
 */
function countExact(config, current, remaining, counter) {
  counter.charge();
  if (remaining === 0) return 1;
  if (current.outcome.kind !== 'ongoing') return 0;
  const actions = legalActions(config, current);
  if (remaining === 1) return actions.length;
  let total = 0;
  for (const action of actions) {
    total += countExact(config, applyLegalAction(config, current, action), remaining - 1, counter);
    if (!Number.isSafeInteger(total)) fail('count exceeds Number.MAX_SAFE_INTEGER');
  }
  return total;
}

/**
 * Count leaf occurrences from `state` to `depth`, without transposition-table
 * deduplication. Unlike `perftReport`, this is a single scalar traversal: it
 * validates once, calls `countExact` once, and retains no root divide or
 * per-depth vectors. `options.maxNodes` defaults to the conservative budget;
 * an explicit value may raise it only as far as `MAX_PERFT_NODES_HARD_CAP`.
 */
export function perft(config, state, depth, options) {
  const checkedDepth = perftDepth(depth, 'depth');
  const maxNodes = nodeBudget(options);
  const initial = validateState(config, state);
  return countExact(config, initial, checkedDepth, counterFor(maxNodes));
}

/**
 * Return exact (not cumulative) per-depth leaf totals and a root divide table.
 * `leavesByDepth[d] === P(root, d)`. The divide is canonical policy-action
 * order. For divide entry i, `childLeavesByDepth[d] ===
 * P(apply(rootAction[i]), d)`: index zero is always one, including a terminal
 * child. Consequently, for d >= 1, `leavesByDepth[d]` is the sum of every
 * root entry's `childLeavesByDepth[d - 1]`. Divide is empty at depth zero and
 * for terminal roots.
 */
export function perftReport(config, state, maxDepth, options) {
  const depth = perftDepth(maxDepth, 'maxDepth');
  const maxNodes = nodeBudget(options);
  const initial = validateState(config, state);
  const counter = counterFor(maxNodes);
  counter.charge();

  if (depth === 0 || initial.outcome.kind !== 'ongoing') {
    const leavesByDepth = [1];
    while (leavesByDepth.length <= depth) leavesByDepth.push(0);
    return frozenReport({ depth, leavesByDepth, nodeVisits: counter.visits, divide: [] });
  }

  const actions = legalActions(config, initial);
  const divide = [];
  const leavesByDepth = [1];
  for (const action of actions) {
    // P(child, 0) is known from the root action occurrence; do not materialize
    // the child merely to count a depth-zero leaf.
    const childLeavesByDepth = [1];
    if (depth > 1) {
      const child = applyLegalAction(config, initial, action);
      for (let childDepth = 1; childDepth < depth; childDepth += 1) {
        childLeavesByDepth.push(countExact(config, child, childDepth, counter));
      }
    }
    divide.push({ action, actionCode: encodeAction(config, action), childLeavesByDepth });
  }
  for (let currentDepth = 1; currentDepth <= depth; currentDepth += 1) {
    let total = 0;
    for (const entry of divide) {
      total += entry.childLeavesByDepth[currentDepth - 1];
      if (!Number.isSafeInteger(total)) fail('count exceeds Number.MAX_SAFE_INTEGER');
    }
    leavesByDepth.push(total);
  }
  return frozenReport({ depth, leavesByDepth, nodeVisits: counter.visits, divide });
}

/**
 * Replay a canonical action-code sequence from the configured initial state.
 * This is intentionally a fixture helper, not a search shortcut: every move
 * still flows through the rules engine's `applyAction` validation.
 */
export function stateFromActionCodes(config, actionCodes) {
  if (!Array.isArray(actionCodes)) fail('actionCodes must be an array');
  let state = createInitialState(config);
  for (const code of actionCodes) {
    nonNegativeInteger(code, 'action code');
    const action = legalActions(config, state).find((candidate) => encodeAction(config, candidate) === code);
    if (!action) fail(`action code ${code} is not legal at ply ${state.ply}`);
    state = applyAction(config, state, action);
  }
  return state;
}

/**
 * Generate a reproducible reachable state by selecting only canonical wall
 * actions. PRNG semantics are lcg32-v1: initialise a uint32 from `seed`,
 * advance it before each selection with
 * `(state * 1664525 + 1013904223) mod 2^32`, then choose the canonical wall
 * action at `state % walls.length`. Modulo bias is intentional.
 */
export function seededWallState(config, { seed, plies }) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) fail('seed must be a uint32');
  nonNegativeInteger(plies, 'plies');
  const initial = createInitialState(config);
  const totalStock = initial.position.stock.A + initial.position.stock.B;
  if (plies > totalStock) fail(`plies (${plies}) exceeds total initial wall stock (${totalStock})`);
  const next = createLcg32(seed);
  let state = initial;
  for (let index = 0; index < plies; index += 1) {
    if (state.outcome.kind !== 'ongoing') fail(`state became terminal at ply ${state.ply}`);
    const walls = legalActions(config, state).filter((action) => action.kind === 'wall');
    if (walls.length === 0) {
      const remaining = state.position.stock.A + state.position.stock.B;
      fail(remaining === 0 ? `wall stock exhausted at ply ${state.ply}` : `no legal wall action at ply ${state.ply}`);
    }
    state = applyAction(config, state, walls[next() % walls.length]);
  }
  return state;
}
