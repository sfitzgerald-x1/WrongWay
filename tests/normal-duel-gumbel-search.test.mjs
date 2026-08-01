import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyAction, createInitialState, decodeAction, legalActionCodes, policySize
} from '../js/normal-duel-engine.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import {
  GUMBEL_SEARCH_VERSION, gumbelRootSearch, selfPlayGame, uniformStubEvaluator
} from '../js/normal-duel-gumbel-search.mjs';

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

/** A to move with no wall stock, so the search can be cornered to one action. */
const CONFIG_NO_A_STOCK = Object.freeze({ ...CONFIG_9X9, initialStock: { A: 0, B: 10 } });

const SIZE = policySize(CONFIG_9X9);

const pawn = (r, c) => ({ kind: 'pawn', to: { r, c } });
const wall = (text) => ({ kind: 'wall', wall: text });

function play(config, actions) {
  return actions.reduce((state, action) => applyAction(config, state, action), createInitialState(config));
}

function codesOf(game) { return game.plies.map((record) => record.actionCode); }

function visitsOf(game) {
  return game.plies.map((record) => [...record.visitCounts.entries()]);
}

function totalVisits(visitCounts) {
  let total = 0;
  for (const count of visitCounts.values()) total += count;
  return total;
}

const defaultGame = (overrides = {}) => selfPlayGame({
  config: CONFIG_9X9, evaluate: uniformStubEvaluator, simulations: 16, maxConsidered: 8,
  seed: 1234, plyCap: 60, ...overrides
});

test('version id is frozen', () => {
  assert.equal(GUMBEL_SEARCH_VERSION, 'gumbel-az-root-v1');
  assert.equal(defaultGame().version, GUMBEL_SEARCH_VERSION);
});

test('uniformStubEvaluator: uniform over legal, zero elsewhere, bounded value', () => {
  const states = [
    createInitialState(CONFIG_9X9),
    play(CONFIG_9X9, [pawn(7, 4), wall('H-3-3'), pawn(6, 4), wall('V-5-2')])
  ];
  for (const state of states) {
    const { policy, value } = uniformStubEvaluator(CONFIG_9X9, state);
    assert.ok(policy instanceof Float32Array);
    assert.equal(policy.length, SIZE);

    const legal = new Set(legalActionCodes(CONFIG_9X9, state));
    let sum = 0;
    for (let code = 0; code < SIZE; code += 1) {
      if (legal.has(code)) { assert.ok(policy[code] > 0); sum += policy[code]; }
      else assert.equal(policy[code], 0, `illegal code ${code} carries probability`);
    }
    assert.ok(Math.abs(sum - 1) < 1e-5, `policy sums to ${sum}`);
    // Every legal action shares the mass equally.
    const share = 1 / legal.size;
    for (const code of legal) assert.ok(Math.abs(policy[code] - share) < 1e-7);

    assert.ok(Number.isFinite(value));
    assert.ok(value >= -1 && value <= 1, `value ${value} outside [-1, 1]`);
  }
  // Symmetric start: neither side is ahead.
  assert.equal(uniformStubEvaluator(CONFIG_9X9, createInitialState(CONFIG_9X9)).value, 0);
});

test('uniformStubEvaluator rewards being closer to goal', () => {
  // A has advanced two rows and B has not moved, so A to move is ahead.
  const ahead = play(CONFIG_9X9, [pawn(7, 4), wall('H-2-2'), pawn(6, 4), wall('H-2-5')]);
  assert.equal(ahead.position.turn, 'A');
  assert.ok(uniformStubEvaluator(CONFIG_9X9, ahead).value > 0);
});

test('determinism: same seed reproduces the game exactly', () => {
  const first = defaultGame();
  const second = defaultGame();
  assert.deepEqual(codesOf(first), codesOf(second));
  assert.deepEqual(visitsOf(first), visitsOf(second));
  assert.deepEqual(first.outcome, second.outcome);
  assert.equal(first.finalPly, second.finalPly);
  for (let index = 0; index < first.plies.length; index += 1) {
    assert.deepEqual(first.plies[index].features, second.plies[index].features);
    assert.deepEqual(first.plies[index].policyTarget, second.plies[index].policyTarget);
    assert.equal(first.plies[index].z, second.plies[index].z);
  }
});

test('determinism: a different seed produces a different game', () => {
  const base = codesOf(defaultGame({ seed: 1234 }));
  const other = codesOf(defaultGame({ seed: 999_777 }));
  assert.notDeepEqual(base, other);
});

test('legality: an independent engine replay accepts every played action', () => {
  for (const seed of [1, 2, 3, 40_000]) {
    const game = defaultGame({ seed });
    let state = createInitialState(CONFIG_9X9);
    for (const record of game.plies) {
      assert.equal(state.outcome.kind, 'ongoing');
      assert.equal(record.ply, state.ply);
      assert.equal(record.turn, state.position.turn);
      const legal = legalActionCodes(CONFIG_9X9, state);
      assert.ok(legal.includes(record.actionCode),
        `seed ${seed} ply ${record.ply}: code ${record.actionCode} not legal`);
      // applyAction re-checks legality itself; it throws if the record lied.
      state = applyAction(CONFIG_9X9, state, decodeAction(CONFIG_9X9, record.actionCode));
    }
    assert.deepEqual(state.outcome, game.outcome);
  }
});

test('termination: the game ends terminally or exactly at plyCap', () => {
  for (const seed of [5, 6, 7]) {
    const game = defaultGame({ seed, plyCap: 60 });
    assert.equal(game.finalPly, game.plies.length);
    if (game.outcome.kind === 'ongoing') assert.equal(game.finalPly, 60);
    else assert.ok(game.finalPly <= 60);
  }
  // A tight cap must stop exactly there and leave the game unresolved.
  const capped = defaultGame({ seed: 5, plyCap: 4 });
  assert.equal(capped.finalPly, 4);
  assert.equal(capped.plies.length, 4);
  assert.equal(capped.outcome.kind, 'ongoing');
});

test('budget: simulationsUsed never exceeds simulations', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4)]);
  const combinations = [
    [0, 1], [0, 8], [1, 4], [3, 8], [4, 4], [16, 8], [7, 16], [2, 32],
    [64, 400], [5, 209], [32, 1]
  ];
  for (const [simulations, maxConsidered] of combinations) {
    const result = gumbelRootSearch({
      config: CONFIG_9X9, state, evaluate: uniformStubEvaluator,
      simulations, maxConsidered, random: createLcg32(11)
    });
    assert.ok(result.simulationsUsed <= simulations,
      `used ${result.simulationsUsed} of ${simulations} (m=${maxConsidered})`);
    assert.equal(totalVisits(result.visitCounts), result.simulationsUsed);
    assert.ok(result.considered.length <= maxConsidered);
    assert.ok(legalActionCodes(CONFIG_9X9, state).includes(result.actionCode));
  }
});

test('budget: whole-game simulationsUsed stays within budget every ply', () => {
  const simulations = 12;
  const game = defaultGame({ simulations, maxConsidered: 5, seed: 21, plyCap: 30 });
  for (const record of game.plies) {
    assert.ok(totalVisits(record.visitCounts) <= simulations);
  }
});

test('single legal action: chosen without spending the budget', () => {
  // A has no wall stock, so only pawn moves are available. Walk A into the
  // bottom-left corner, then let B seal the sideways exit with V-7-0.
  const state = play(CONFIG_NO_A_STOCK, [
    pawn(8, 3), wall('V-0-7'),
    pawn(8, 2), wall('V-2-7'),
    pawn(8, 1), wall('V-4-7'),
    pawn(8, 0), wall('V-7-0')
  ]);
  const legal = legalActionCodes(CONFIG_NO_A_STOCK, state);
  assert.deepEqual(legal, [7 * 9 + 0]); // exactly one: step up to (7, 0)

  const result = gumbelRootSearch({
    config: CONFIG_NO_A_STOCK, state, evaluate: uniformStubEvaluator,
    simulations: 64, maxConsidered: 16, random: createLcg32(3)
  });
  assert.equal(result.actionCode, legal[0]);
  assert.equal(result.simulationsUsed, 0);
  assert.deepEqual(result.considered, legal);
  assert.equal(result.improvedPolicy[legal[0]], 1);
  assert.deepEqual(result.action, { kind: 'pawn', to: { r: 7, c: 0 } });
});

test('improvedPolicy: sums to 1, zero off-legal, argmax legal', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), wall('H-3-3')]);
  const legal = new Set(legalActionCodes(CONFIG_9X9, state));
  for (const [simulations, maxConsidered] of [[0, 4], [1, 3], [16, 8], [48, 12]]) {
    const result = gumbelRootSearch({
      config: CONFIG_9X9, state, evaluate: uniformStubEvaluator,
      simulations, maxConsidered, random: createLcg32(77)
    });
    assert.equal(result.improvedPolicy.length, SIZE);
    let sum = 0; let best = -1; let bestValue = -Infinity;
    for (let code = 0; code < SIZE; code += 1) {
      const value = result.improvedPolicy[code];
      sum += value;
      if (!legal.has(code)) assert.equal(value, 0, `illegal code ${code} has mass`);
      if (value > bestValue) { bestValue = value; best = code; }
    }
    assert.ok(Math.abs(sum - 1) < 1e-5, `improvedPolicy sums to ${sum}`);
    assert.ok(legal.has(best), `argmax ${best} is not legal`);
  }
});

test('visitCounts total equals simulationsUsed and only covers considered codes', () => {
  const state = createInitialState(CONFIG_9X9);
  const result = gumbelRootSearch({
    config: CONFIG_9X9, state, evaluate: uniformStubEvaluator,
    simulations: 24, maxConsidered: 6, random: createLcg32(9)
  });
  assert.equal(totalVisits(result.visitCounts), result.simulationsUsed);
  assert.ok(result.simulationsUsed > 0);
  assert.deepEqual([...result.visitCounts.keys()].slice().sort((a, b) => a - b), [...result.considered]);
  const legal = new Set(legalActionCodes(CONFIG_9X9, state));
  for (const code of result.visitCounts.keys()) assert.ok(legal.has(code));
});

test('z: winner plies carry +1 and loser plies -1, by side to move', () => {
  let decisive = null;
  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const game = defaultGame({ seed, plyCap: 120 });
    if (game.outcome.kind === 'win') { decisive = game; break; }
  }
  assert.ok(decisive, 'expected at least one decisive game among the probed seeds');
  const { winner } = decisive.outcome;
  for (const record of decisive.plies) {
    assert.equal(record.z, record.turn === winner ? 1 : -1);
  }
  // Unresolved games score 0 everywhere.
  const capped = defaultGame({ seed: 1234, plyCap: 3 });
  assert.equal(capped.outcome.kind, 'ongoing');
  for (const record of capped.plies) assert.equal(record.z, 0);
});

test('purity: config and state are untouched by a search', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('H-3-3')]);
  const configBefore = structuredClone(CONFIG_9X9);
  const stateBefore = structuredClone(state);
  // A mutating evaluator must not be able to corrupt the search either.
  const evaluate = (config, position) => {
    const evaluation = uniformStubEvaluator(config, position);
    const policy = evaluation.policy;
    queueMicrotask(() => policy.fill(0));
    return evaluation;
  };
  const result = gumbelRootSearch({
    config: CONFIG_9X9, state, evaluate, simulations: 20, maxConsidered: 6, random: createLcg32(5)
  });
  assert.deepEqual(structuredClone(CONFIG_9X9), configBefore);
  assert.deepEqual(structuredClone(state), stateBefore);
  assert.ok(legalActionCodes(CONFIG_9X9, state).includes(result.actionCode));
});

test('purity: the returned record is frozen', () => {
  const result = gumbelRootSearch({
    config: CONFIG_9X9, state: createInitialState(CONFIG_9X9), evaluate: uniformStubEvaluator,
    simulations: 8, maxConsidered: 4, random: createLcg32(2)
  });
  assert.ok(Object.isFrozen(result));
  assert.throws(() => { result.actionCode = 0; }, TypeError);
  const game = defaultGame({ plyCap: 3 });
  assert.ok(Object.isFrozen(game));
  assert.ok(Object.isFrozen(game.plies));
  assert.ok(Object.isFrozen(game.plies[0]));
});

test('self-play records carry encoder-shaped features and policy targets', () => {
  const game = defaultGame({ plyCap: 5 });
  for (const record of game.plies) {
    assert.ok(record.features instanceof Float32Array);
    assert.equal(record.features.length, 8 * 81);
    assert.ok(record.policyTarget instanceof Float32Array);
    assert.equal(record.policyTarget.length, SIZE);
    let sum = 0;
    for (const value of record.policyTarget) sum += value;
    assert.ok(Math.abs(sum - 1) < 1e-5);
  }
});

test('non-canonical boards and bad arguments are rejected', () => {
  assert.throws(() => uniformStubEvaluator(CONFIG_7X7, createInitialState(CONFIG_7X7)), /unsupported_board/);
  assert.throws(() => gumbelRootSearch({
    config: CONFIG_7X7, state: createInitialState(CONFIG_7X7), evaluate: uniformStubEvaluator,
    simulations: 4, maxConsidered: 4, random: createLcg32(1)
  }), /unsupported_board/);
  assert.throws(() => selfPlayGame({
    config: CONFIG_7X7, evaluate: uniformStubEvaluator, simulations: 4, maxConsidered: 4, seed: 1, plyCap: 4
  }), /unsupported_board/);

  const state = createInitialState(CONFIG_9X9);
  const base = { config: CONFIG_9X9, state, evaluate: uniformStubEvaluator, random: createLcg32(1) };
  assert.throws(() => gumbelRootSearch({ ...base, simulations: -1, maxConsidered: 4 }), /invalid_simulations/);
  assert.throws(() => gumbelRootSearch({ ...base, simulations: 1.5, maxConsidered: 4 }), /invalid_simulations/);
  assert.throws(() => gumbelRootSearch({ ...base, simulations: 4, maxConsidered: 0 }), /invalid_max_considered/);
  assert.throws(() => gumbelRootSearch({ ...base, simulations: 4, maxConsidered: 4, evaluate: null }), /invalid_evaluator/);
  assert.throws(() => gumbelRootSearch({ ...base, simulations: 4, maxConsidered: 4, random: null }), /invalid_random/);
  // A seed outside uint32 is the RNG's own contract violation.
  assert.throws(() => selfPlayGame({
    config: CONFIG_9X9, evaluate: uniformStubEvaluator, simulations: 4, maxConsidered: 4, seed: -1, plyCap: 4
  }), TypeError);
});

test('a terminal state is refused rather than searched', () => {
  // Race A straight up the board while B shuffles sideways; A reaches row 0.
  const actions = [];
  for (let step = 0; step < 8; step += 1) {
    actions.push(pawn(7 - step, 4));
    if (step < 7) actions.push(step % 2 === 0 ? pawn(0, 5) : pawn(0, 4));
  }
  const state = play(CONFIG_9X9, actions);
  assert.deepEqual(state.outcome, { kind: 'win', winner: 'A', reason: 'goal' });
  assert.throws(() => gumbelRootSearch({
    config: CONFIG_9X9, state, evaluate: uniformStubEvaluator,
    simulations: 8, maxConsidered: 4, random: createLcg32(1)
  }), /terminal_state/);
});

test('search never returns a code outside the legal set, across many positions', () => {
  const game = defaultGame({ seed: 4242, plyCap: 40 });
  let state = createInitialState(CONFIG_9X9);
  for (const record of game.plies) {
    for (const code of record.visitCounts.keys()) {
      assert.ok(legalActionCodes(CONFIG_9X9, state).includes(code));
    }
    state = applyAction(CONFIG_9X9, state, decodeAction(CONFIG_9X9, record.actionCode));
  }
});
