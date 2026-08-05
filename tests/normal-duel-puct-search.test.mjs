import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyAction, createInitialState, decodeAction, legalActionCodes, policySize
} from '../js/normal-duel-engine.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import { uniformStubEvaluator } from '../js/normal-duel-gumbel-search.mjs';
import {
  DEFAULT_C_PUCT, FPU_REDUCTION, PUCT_SEARCH_VERSION, effectiveVisitCounts, puctSearch, selfPlayGamePuct
} from '../js/normal-duel-puct-search.mjs';

const CONFIG_9X9 = Object.freeze({
  ruleset: 'normal-duel-v1', rows: 9, columns: 9,
  start: { A: { r: 8, c: 4 }, B: { r: 0, c: 4 } },
  goalRows: { A: 0, B: 8 }, initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap: 200, firstPlayer: 'A'
});

const CONFIG_7X7 = Object.freeze({
  ruleset: 'normal-duel-v1', rows: 7, columns: 7,
  start: { A: { r: 6, c: 3 }, B: { r: 0, c: 3 } },
  goalRows: { A: 0, B: 6 }, initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap: 200, firstPlayer: 'A'
});

const SIZE = policySize(CONFIG_9X9);

const pawn = (r, c) => ({ kind: 'pawn', to: { r, c } });
const wall = (text) => ({ kind: 'wall', wall: text });

function play(config, actions) {
  return actions.reduce((state, action) => applyAction(config, state, action), createInitialState(config));
}

/** A mid-game position: both pawns advanced, two walls down, 124 legal actions. */
const MID_GAME = play(CONFIG_9X9, [
  pawn(7, 4), pawn(1, 4), wall('H-3-3'), wall('V-5-2'), pawn(6, 4), pawn(2, 4)
]);

/**
 * A to move at (1, 4) with an empty goal square in front of it: exactly one
 * action ends the game as a win for A this ply.
 */
const A_WINS_IN_ONE = (() => {
  const actions = [];
  for (let step = 0; step < 7; step += 1) {
    actions.push(pawn(7 - step, 4));
    actions.push(step % 2 === 0 ? pawn(0, 5) : pawn(0, 4));
  }
  return play(CONFIG_9X9, actions);
})();

function winningCodes(config, state) {
  return legalActionCodes(config, state).filter((code) => (
    applyAction(config, state, decodeAction(config, code)).outcome.kind === 'win'
  ));
}

const search = (overrides = {}) => puctSearch({
  config: CONFIG_9X9,
  state: MID_GAME,
  evaluate: uniformStubEvaluator,
  simulations: 32,
  cPuct: DEFAULT_C_PUCT,
  maxConsidered: 8,
  random: createLcg32(7),
  ...overrides
});

const defaultGame = (overrides = {}) => selfPlayGamePuct({
  config: CONFIG_9X9, evaluate: uniformStubEvaluator, simulations: 16, cPuct: DEFAULT_C_PUCT,
  maxConsidered: 8, seed: 1234, plyCap: 40, ...overrides
});

function counted(evaluate) {
  const wrapped = (config, state) => { wrapped.calls += 1; return evaluate(config, state); };
  wrapped.calls = 0;
  return wrapped;
}

function totalVisits(visitCounts) {
  let total = 0;
  for (const count of visitCounts.values()) total += count;
  return total;
}

/**
 * Uniform priors and a value of exactly 0 everywhere, so no value signal can
 * account for the shape of the tree the search builds.
 */
function flatZeroEvaluator(config, state) {
  const policy = new Float32Array(SIZE);
  for (const code of legalActionCodes(config, state)) policy[code] = 1;
  return { policy, value: 0 };
}

/**
 * Run a search behind an evaluator that records, per distinct position, how far
 * below the root that position sits.
 *
 * This is the structural witness for "there is a tree here". It reads nothing
 * the module reports about itself — not `maxDepthReached`, not
 * `simulationsUsed` — only the positions the module actually handed to the
 * evaluator. A leaf two plies below the root can only be reached by descending
 * through a child that a *previous* simulation already expanded, so
 * `maxRelativePly >= 2` is observable proof that the tree survives between
 * simulations. Clear `edge.child` after each visit and this collapses to 1.
 */
function probeTree({
  state = MID_GAME, simulations = 64, cPuct = DEFAULT_C_PUCT,
  maxConsidered = 8, seed = 7, evaluate = uniformStubEvaluator
} = {}) {
  const relativePly = new Map();
  const probe = (config, probed) => {
    relativePly.set(probed.positionKey, probed.ply - state.ply);
    return evaluate(config, probed);
  };
  const result = puctSearch({
    config: CONFIG_9X9, state, evaluate: probe, simulations, cPuct, maxConsidered,
    random: createLcg32(seed)
  });
  const depths = [...relativePly.values()];
  return {
    result,
    distinctPositions: relativePly.size,
    maxRelativePly: Math.max(...depths),
    below: (ply) => depths.filter((depth) => depth >= ply).length,
    atPly: (ply) => depths.filter((depth) => depth === ply).length
  };
}

/* ------------------------------------------------------------------ */

test('version id and FPU constants are frozen', () => {
  assert.equal(PUCT_SEARCH_VERSION, 'puct-az-tree-v1');
  assert.equal(DEFAULT_C_PUCT, 1.25);
  assert.equal(FPU_REDUCTION, 0.25);
  assert.equal(defaultGame().version, PUCT_SEARCH_VERSION);
});

test('the fixtures are what the tests assume they are', () => {
  assert.equal(MID_GAME.outcome.kind, 'ongoing');
  assert.equal(legalActionCodes(CONFIG_9X9, MID_GAME).length, 124);
  assert.equal(A_WINS_IN_ONE.outcome.kind, 'ongoing');
  assert.equal(A_WINS_IN_ONE.position.turn, 'A');
  assert.equal(winningCodes(CONFIG_9X9, A_WINS_IN_ONE).length, 1);
});

/* --- the core regression: this is a tree, not a depth-1 root scan ---- */

test('maxDepthReached >= 3 at simulations = 32 on a mid-game position', () => {
  const result = search({ simulations: 32 });
  assert.ok(
    result.maxDepthReached >= 3,
    `expected a real tree, got maxDepthReached=${result.maxDepthReached}`
  );
});

/**
 * `maxDepthReached` is self-reported: the module increments its own counter. If
 * the tree were silently cleared between simulations the counter could still be
 * anything at all, so the two tests around this one prove nothing on their own.
 * This one only believes the evaluator.
 */
test('the tree persists between simulations: leaves below the root children are expanded', () => {
  const probed = probeTree({ simulations: 64, maxConsidered: 8 });

  // A grandchild of the root was evaluated. Reaching one requires descending
  // through an already-expanded child, i.e. a tree that outlived a simulation.
  assert.ok(
    probed.maxRelativePly >= 2,
    `no position below the root children was ever evaluated (maxRelativePly=${probed.maxRelativePly}); `
    + 'the tree is being cleared between simulations'
  );
  assert.ok(probed.below(2) >= 8, `only ${probed.below(2)} positions below the root children`);

  // A depth-1 scan can only ever evaluate the root plus its considered
  // children. Exceeding that count is the same fact counted a second way.
  assert.ok(
    probed.distinctPositions > 1 + probed.result.considered.length,
    `evaluated ${probed.distinctPositions} positions for ${probed.result.considered.length} root children: depth-1 scan`
  );

  // Expansion is cached, so a simulation never re-evaluates a node it already
  // expanded: distinct positions is exactly the root plus one new leaf per
  // simulation. Fewer would mean an expanded node was evaluated twice.
  assert.equal(probed.distinctPositions, 1 + probed.result.simulationsUsed);
});

/**
 * The FPU term is what makes the search commit to a line instead of fanning out
 * over the ~130 siblings at every ply. Under `flatZeroEvaluator` every value is
 * exactly 0 and every prior is uniform, so a node's visited and unvisited edges
 * would tie exactly — *only* the FPU reduction breaks that tie, by making each
 * unvisited sibling less attractive as its visited siblings' prior accumulates.
 *
 * `maxConsidered: 1` puts the whole budget into one subtree, so the shape below
 * that root child is a direct read on the term. With the reduction the search
 * drives 8+ plies deep off 10-ish expanded siblings; drop it and 64 simulations
 * fan out over 60+ siblings and the tree never gets past two plies. Only the
 * frozen-constants test notices otherwise, and that one cannot see the term
 * being computed and then ignored.
 */
test('FPU drives depth: the search commits to a line instead of fanning out', () => {
  const probed = probeTree({ simulations: 64, maxConsidered: 1, evaluate: flatZeroEvaluator });

  assert.ok(
    probed.maxRelativePly >= 5,
    `deepest evaluated leaf is ${probed.maxRelativePly} plies below the root; `
    + 'without the FPU reduction the search fans out instead of going deep'
  );
  assert.ok(
    probed.atPly(2) <= 20,
    `${probed.atPly(2)} siblings expanded two plies below the root out of ${probed.result.simulationsUsed} `
    + 'simulations; the search is fanning out, so the FPU reduction is not reaching selection'
  );
});

/**
 * Nothing else notices whether the prior/exploration term reaches `selectEdge`:
 * an implementation that dropped `cPuct * P * sqrt(N) / (1 + N)` entirely would
 * pass every other test in this file. A large exploration constant must produce
 * a visibly broader, shallower tree than a near-zero one over the same seed,
 * budget and evaluator; if the term is ignored the two searches are identical.
 */
test('cPuct reaches selection: exploration trades depth for breadth', () => {
  const deep = probeTree({ simulations: 96, cPuct: 0.01 });
  const broad = probeTree({ simulations: 96, cPuct: 5 });

  assert.ok(
    deep.maxRelativePly !== broad.maxRelativePly || deep.distinctPositions !== broad.distinctPositions,
    'cPuct = 0.01 and cPuct = 5 searched identically; the exploration term is not reaching selection'
  );
  assert.ok(
    deep.maxRelativePly > broad.maxRelativePly,
    `low exploration reached ${deep.maxRelativePly} plies, high exploration ${broad.maxRelativePly}`
  );
  // Breadth is the other half of the trade: the same budget goes across the
  // root children instead of down one line, so far fewer leaves sit deep.
  assert.ok(
    deep.below(4) > 2 * broad.below(4),
    `deep leaves: ${deep.below(4)} at cPuct=0.01 vs ${broad.below(4)} at cPuct=5`
  );
});

test('depth grows with the simulation budget', () => {
  const depths = [16, 32, 64].map((simulations) => search({ simulations }).maxDepthReached);
  for (const depth of depths) assert.ok(depth > 1, `depth-1 search detected: ${depths.join(',')}`);
  assert.ok(depths[2] >= depths[0], `depth should not shrink with budget: ${depths.join(',')}`);
});

/* --- backup sign ---------------------------------------------------- */

test('backup sign: an immediately winning move is played', () => {
  const [winner] = winningCodes(CONFIG_9X9, A_WINS_IN_ONE);
  // Every legal action considered, so the winning move cannot be missed by chance.
  const result = puctSearch({
    config: CONFIG_9X9,
    state: A_WINS_IN_ONE,
    evaluate: uniformStubEvaluator,
    simulations: 64,
    cPuct: DEFAULT_C_PUCT,
    maxConsidered: legalActionCodes(CONFIG_9X9, A_WINS_IN_ONE).length,
    random: createLcg32(3)
  });
  assert.equal(result.actionCode, winner);
  assert.ok(result.visitCounts.get(winner) > 0);
});

test('backup sign: a move handing the opponent a won position is avoided', () => {
  // `evaluate` reports from the CHILD's side-to-move perspective, i.e. the
  // opponent's. A child scored +1 is a disaster for us; a child scored -1 is
  // excellent. If the backup negation were inverted these two would swap and
  // the search would pick `poison` over `prize`.
  const legal = legalActionCodes(CONFIG_9X9, MID_GAME);
  const poison = legal[0];
  const prize = legal[legal.length - 1];
  const childKey = (code) => JSON.stringify(
    applyAction(CONFIG_9X9, MID_GAME, decodeAction(CONFIG_9X9, code)).position
  );
  const poisonKey = childKey(poison);
  const prizeKey = childKey(prize);

  // Root priors are concentrated on the two marked actions so the Gumbel root
  // considers both; everything else is a rounding error.
  const evaluate = (config, state) => {
    const policy = new Float32Array(SIZE);
    for (const code of legalActionCodes(config, state)) policy[code] = 1e-6;
    if (state.ply === MID_GAME.ply) { policy[poison] = 0.5; policy[prize] = 0.5; }
    const key = JSON.stringify(state.position);
    if (key === poisonKey) return { policy, value: 1 };
    if (key === prizeKey) return { policy, value: -1 };
    return { policy, value: 0 };
  };

  const result = puctSearch({
    config: CONFIG_9X9,
    state: MID_GAME,
    evaluate,
    simulations: 64,
    cPuct: DEFAULT_C_PUCT,
    maxConsidered: 4,
    random: createLcg32(11)
  });
  assert.notEqual(result.actionCode, poison);
  assert.equal(result.actionCode, prize);
});

/* --- terminal handling ---------------------------------------------- */

test('terminal children are scored by the engine, not expanded, and cost no evaluation', () => {
  const [winner] = winningCodes(CONFIG_9X9, A_WINS_IN_ONE);
  const evaluate = counted((config, state) => {
    assert.equal(state.outcome.kind, 'ongoing', 'the network was asked about a terminal state');
    return uniformStubEvaluator(config, state);
  });
  const legalCount = legalActionCodes(CONFIG_9X9, A_WINS_IN_ONE).length;
  const result = puctSearch({
    config: CONFIG_9X9,
    state: A_WINS_IN_ONE,
    evaluate,
    simulations: 40,
    cPuct: DEFAULT_C_PUCT,
    maxConsidered: legalCount,
    random: createLcg32(5)
  });
  assert.equal(result.actionCode, winner);
  // One root evaluation plus at most one leaf evaluation per simulation, and
  // strictly fewer than that because visits to the terminal winner call none.
  assert.ok(evaluate.calls <= 1 + result.simulationsUsed);
  assert.ok(evaluate.calls < 1 + result.simulationsUsed, 'terminal visits consumed an evaluation');
});

test('a terminal root state is rejected rather than searched', () => {
  const [winner] = winningCodes(CONFIG_9X9, A_WINS_IN_ONE);
  const terminal = applyAction(CONFIG_9X9, A_WINS_IN_ONE, decodeAction(CONFIG_9X9, winner));
  assert.equal(terminal.outcome.kind, 'win');
  assert.throws(() => search({ state: terminal }), /terminal_state/);
});

/* --- budget ---------------------------------------------------------- */

test('simulationsUsed never exceeds the budget and visit totals reconcile', () => {
  for (const simulations of [1, 2, 3, 5, 8, 32, 64]) {
    const result = search({ simulations });
    assert.ok(result.simulationsUsed <= simulations, `used ${result.simulationsUsed} of ${simulations}`);
    assert.equal(totalVisits(result.visitCounts), result.simulationsUsed);
  }
});

test('leaf evaluations never exceed the simulation budget', () => {
  for (const simulations of [1, 4, 32]) {
    const evaluate = counted(uniformStubEvaluator);
    const result = search({ simulations, evaluate });
    // The single root evaluation produces the priors and is outside the budget.
    assert.ok(evaluate.calls - 1 <= simulations, `${evaluate.calls - 1} leaf evals for ${simulations}`);
    assert.ok(result.simulationsUsed <= simulations);
  }
});

/**
 * A zero budget used to be accepted here and returned a legal action with a
 * normalised policy, which reads as graceful degradation and is not: with no
 * simulations the visit counts are empty, `effectiveVisitCounts` falls back to a
 * one-hot of the played action, and a self-play driver records that one-hot as
 * the policy target at a position with ~130 legal codes. Full record count, no
 * warning, and the improvement ratchet gone -- an earlier run spent 114 flat
 * iterations that way.
 *
 * `createGumbelAgent` already rejected `simulations: 0` and `maxConsidered` was
 * already required to be positive, so accepting it here was an inconsistency
 * rather than a considered choice.
 */
test('a zero simulation budget is rejected rather than yielding a one-hot target', () => {
  assert.throws(() => search({ simulations: 0 }), /invalid_simulations/);
  assert.throws(() => selfPlayGamePuct({
    config: CONFIG_9X9, evaluate: uniformStubEvaluator, simulations: 0, maxConsidered: 4, seed: 1
  }), /invalid_simulations/);

  // A budget of one still works: the guard is on zero, not on "small".
  const result = search({ simulations: 1 });
  assert.ok(legalActionCodes(CONFIG_9X9, MID_GAME).includes(result.actionCode));
  let sum = 0;
  for (const p of result.improvedPolicy) sum += p;
  assert.ok(Math.abs(sum - 1) < 1e-6);
});

/* --- determinism ----------------------------------------------------- */

test('determinism: same seed is byte-identical, different seeds may differ', () => {
  const first = defaultGame({ seed: 99 });
  const second = defaultGame({ seed: 99 });
  assert.deepEqual(first.plies.map((r) => r.actionCode), second.plies.map((r) => r.actionCode));
  assert.deepEqual(first.plies.map((r) => [...r.visitCounts]), second.plies.map((r) => [...r.visitCounts]));
  assert.deepEqual(first.outcome, second.outcome);
  for (let index = 0; index < first.plies.length; index += 1) {
    assert.deepEqual([...first.plies[index].features], [...second.plies[index].features]);
    assert.deepEqual([...first.plies[index].policyTarget], [...second.plies[index].policyTarget]);
  }
  const other = defaultGame({ seed: 4242 });
  assert.ok(other.plies.length > 0);
});

test('determinism: repeated searches from one seed agree', () => {
  const a = search();
  const b = search();
  assert.equal(a.actionCode, b.actionCode);
  assert.equal(a.maxDepthReached, b.maxDepthReached);
  assert.deepEqual([...a.visitCounts], [...b.visitCounts]);
});

/* --- legality -------------------------------------------------------- */

test('legality: a full self-play game replays through the engine with zero illegal actions', () => {
  const game = defaultGame({ simulations: 12, plyCap: 40 });
  let state = createInitialState(CONFIG_9X9);
  for (const record of game.plies) {
    assert.equal(state.outcome.kind, 'ongoing');
    assert.equal(state.ply, record.ply);
    assert.equal(state.position.turn, record.turn);
    const legal = legalActionCodes(CONFIG_9X9, state);
    assert.ok(legal.includes(record.actionCode), `illegal action ${record.actionCode} at ply ${record.ply}`);
    for (const code of record.visitCounts.keys()) assert.ok(legal.includes(code));
    state = applyAction(CONFIG_9X9, state, decodeAction(CONFIG_9X9, record.actionCode));
  }
  assert.deepEqual(state.outcome, game.outcome);
  assert.equal(game.finalPly, game.plies.length);
});

test('self-play records match the landed shape and carry z by side to move', () => {
  const game = defaultGame({ simulations: 8, plyCap: 40 });
  for (const record of game.plies) {
    assert.ok(record.features instanceof Float32Array);
    assert.ok(record.policyTarget instanceof Float32Array);
    assert.equal(record.policyTarget.length, SIZE);
    assert.ok(record.visitCounts instanceof Map);
    assert.equal(typeof record.z, 'number');
    assert.ok(Object.isFrozen(record));
    let sum = 0;
    for (const p of record.policyTarget) sum += p;
    assert.ok(Math.abs(sum - 1) < 1e-6);
  }
  if (game.outcome.kind === 'win') {
    for (const record of game.plies) {
      assert.equal(record.z, record.turn === game.outcome.winner ? 1 : -1);
    }
  } else {
    for (const record of game.plies) assert.equal(record.z, 0);
  }
});

/* --- improvedPolicy -------------------------------------------------- */

test('improvedPolicy sums to 1 and is zero on illegal codes', () => {
  const result = search({ simulations: 32 });
  assert.ok(result.improvedPolicy instanceof Float32Array);
  assert.equal(result.improvedPolicy.length, SIZE);
  assert.equal(SIZE, 209);
  const legal = new Set(legalActionCodes(CONFIG_9X9, MID_GAME));
  let sum = 0;
  for (let code = 0; code < SIZE; code += 1) {
    sum += result.improvedPolicy[code];
    if (!legal.has(code)) assert.equal(result.improvedPolicy[code], 0, `mass on illegal code ${code}`);
  }
  assert.ok(Math.abs(sum - 1) < 1e-6);
  assert.equal(effectiveVisitCounts(result), result.visitCounts);
});

test('the result is frozen and the action is legal', () => {
  const result = search();
  assert.ok(Object.isFrozen(result));
  assert.ok(Object.isFrozen(result.action));
  assert.ok(legalActionCodes(CONFIG_9X9, MID_GAME).includes(result.actionCode));
  assert.deepEqual(result.action, decodeAction(CONFIG_9X9, result.actionCode));
  assert.equal(typeof result.rootValue, 'number');
});

/* --- purity and board guard ------------------------------------------ */

test('purity: caller config, state and evaluator arrays are untouched', () => {
  const configSnapshot = JSON.stringify(CONFIG_9X9);
  const stateSnapshot = JSON.stringify(MID_GAME);

  const shared = new Float32Array(SIZE);
  const evaluate = (config, state) => {
    shared.fill(0);
    for (const code of legalActionCodes(config, state)) shared[code] = 1;
    return { policy: shared, value: 0.25 };
  };
  const result = search({ evaluate });

  assert.equal(JSON.stringify(CONFIG_9X9), configSnapshot);
  assert.equal(JSON.stringify(MID_GAME), stateSnapshot);
  // The search must not have retained or written through the shared buffer.
  assert.notEqual(result.improvedPolicy, shared);
  assert.ok(legalActionCodes(CONFIG_9X9, MID_GAME).includes(result.actionCode));
});

test('7x7 is rejected on both entry points', () => {
  assert.throws(() => puctSearch({
    config: CONFIG_7X7, state: createInitialState(CONFIG_7X7), evaluate: uniformStubEvaluator,
    simulations: 4, cPuct: DEFAULT_C_PUCT, maxConsidered: 4, random: createLcg32(1)
  }), /unsupported_board/);
  assert.throws(() => selfPlayGamePuct({
    config: CONFIG_7X7, evaluate: uniformStubEvaluator, simulations: 4, cPuct: DEFAULT_C_PUCT,
    maxConsidered: 4, seed: 1, plyCap: 4
  }), /unsupported_board/);
});

test('argument validation rejects bad budgets, cPuct and evaluators', () => {
  assert.throws(() => search({ simulations: -1 }), /invalid_simulations/);
  assert.throws(() => search({ maxConsidered: 0 }), /invalid_max_considered/);
  assert.throws(() => search({ cPuct: 0 }), /invalid_c_puct/);
  assert.throws(() => search({ cPuct: Number.NaN }), /invalid_c_puct/);
  assert.throws(() => search({ evaluate: null }), /invalid_evaluator/);
  assert.throws(() => search({ random: null }), /invalid_random/);
  assert.throws(() => search({ evaluate: () => ({ policy: new Float32Array(3), value: 0 }) }), /invalid_evaluation/);
  assert.throws(() => search({ evaluate: () => ({ policy: new Float32Array(SIZE), value: 2 }) }), /invalid_evaluation/);
});
