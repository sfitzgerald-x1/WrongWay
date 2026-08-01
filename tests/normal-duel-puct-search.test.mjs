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
  for (const simulations of [0, 1, 2, 3, 5, 8, 32, 64]) {
    const result = search({ simulations });
    assert.ok(result.simulationsUsed <= simulations, `used ${result.simulationsUsed} of ${simulations}`);
    assert.equal(totalVisits(result.visitCounts), result.simulationsUsed);
  }
});

test('leaf evaluations never exceed the simulation budget', () => {
  for (const simulations of [0, 1, 4, 32]) {
    const evaluate = counted(uniformStubEvaluator);
    const result = search({ simulations, evaluate });
    // The single root evaluation produces the priors and is outside the budget.
    assert.ok(evaluate.calls - 1 <= simulations, `${evaluate.calls - 1} leaf evals for ${simulations}`);
    assert.ok(result.simulationsUsed <= simulations);
  }
});

test('a zero budget still returns a legal action and a normalised policy', () => {
  const result = search({ simulations: 0 });
  assert.equal(result.simulationsUsed, 0);
  assert.equal(result.maxDepthReached, 0);
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
