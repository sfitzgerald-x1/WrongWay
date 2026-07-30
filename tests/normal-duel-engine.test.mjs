import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import * as engine from '../js/normal-duel-engine.mjs';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/normal-duel-v1-cases.json', import.meta.url), 'utf8'));
const configFor = (entry) => fixture.configs[entry.configId];
const copy = (value) => structuredClone(value);

function assertCode(fn, code, message) {
  assert.throws(fn, (error) => error instanceof engine.NormalDuelError && error.code === code, message);
}

function legacyContext(config) {
  const context = { CELL: 40, COLS: config.columns, CUR_MAP: 'duel', GOAL_2V2: {}, ORDER_2V2: [], ROWS: config.rows, WW: 8, WW2: 8 };
  vm.createContext(context);
  vm.runInContext(readFileSync(new URL('../js/game-logic.js', import.meta.url), 'utf8'), context);
  vm.runInContext(readFileSync(new URL('../js/ai.js', import.meta.url), 'utf8'), context);
  return context;
}

test('normal-duel-v1 fixtures: queries, transitions, adjudication, and rejections', () => {
  for (const entry of fixture.cases) {
    if (entry.kind === 'query') {
      const config = configFor(entry); const legal = engine.legalPositionActions(config, entry.position);
      if (entry.expect.legalActionCount !== undefined) assert.equal(legal.length, entry.expect.legalActionCount, entry.id);
      if (entry.expect.legalPawnDestinations) assert.deepEqual(engine.legalPawnDestinations(config, entry.position), entry.expect.legalPawnDestinations, entry.id);
      if (entry.action) {
        if (entry.expect.actionCode !== undefined) {
          assert.equal(engine.encodeAction(config, entry.action), entry.expect.actionCode, entry.id);
          assert.equal(legal.some((action) => engine.encodeAction(config, action) === engine.encodeAction(config, entry.action)), entry.expect.legal, entry.id);
        } else assert.equal(engine.isLegalWall(config, entry.position, entry.action.wall), entry.expect.legal, entry.id);
        if (entry.expect.pathsRemain !== undefined) {
          const walls = [...entry.position.walls, entry.action.wall];
          assert.equal(engine.hasPath(config, entry.position.pawns.A, config.goalRows.A, walls) && engine.hasPath(config, entry.position.pawns.B, config.goalRows.B, walls), entry.expect.pathsRemain, entry.id);
        }
      }
    } else if (entry.kind === 'transition') {
      assert.deepEqual(engine.applyAction(configFor(entry), entry.state, entry.action), entry.nextState, entry.id);
    } else if (entry.kind === 'adjudication') {
      assert.deepEqual(engine.adjudicate(configFor(entry), entry.facts), entry.expect.outcome, entry.id);
    } else if (entry.kind === 'rejection') {
      assert.throws(() => {
        if (entry.input.features) engine.validateConfig(entry.input);
        else engine.applyAction(fixture.configs['standard-a'], { ...engine.createInitialState(fixture.configs['standard-a']), outcome: entry.input.outcome }, entry.input.action);
      }, (error) => error.code === entry.expect.error, entry.id);
    }
  }
});

test('seeded mixed trajectories agree with legacy normal 1v1 moves and walls', () => {
  let seed = 0x1a2b3c4d;
  const nextRandom = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed; };
  for (const config of Object.values(fixture.configs)) {
    const legacy = legacyContext(config); let state = engine.createInitialState(config);
    for (let ply = 0; ply < 16; ply += 1) {
      const position = state.position; const mover = position.pawns[position.turn]; const opponent = position.pawns[position.turn === 'A' ? 'B' : 'A'];
      const oldMoves = Array.from(legacy.getMovesFrom(mover, opponent, new Set(position.walls)), ({ r, c }) => ({ r, c }))
        .sort((a, b) => engine.encodeAction(config, { kind: 'pawn', to: a }) - engine.encodeAction(config, { kind: 'pawn', to: b }));
      assert.deepEqual(engine.legalPawnDestinations(config, position), oldMoves, `moves ${config.rows}/${ply}`);
      for (let index = 0; index < 8; index += 1) {
        const r = nextRandom() % (config.rows - 1); const c = nextRandom() % (config.columns - 1); const orientation = nextRandom() & 1 ? 'H' : 'V';
        const wall = `${orientation}-${r}-${c}`;
        const expected = position.stock[position.turn] > 0 && Boolean(legacy.tryWall(wall, new Set(position.walls), position.pawns.A, position.pawns.B, config.goalRows.A, config.goalRows.B));
        assert.equal(engine.isLegalWall(config, position, wall), expected, `wall ${config.rows}/${ply}/${wall}`);
      }
      const actions = engine.legalPositionActions(config, position);
      const preferred = ply % 4 === 1 ? actions.filter(({ kind }) => kind === 'wall') : actions.filter(({ kind }) => kind === 'pawn');
      state = engine.applyAction(config, state, preferred[nextRandom() % preferred.length]);
      if (state.outcome.kind !== 'ongoing') break;
    }
  }
});

test('permissive adjacent exits directly match the legacy 1v1 generator', () => {
  for (const entry of fixture.cases.filter(({ id }) => id.startsWith('permissive-jump-'))) {
    const config = configFor(entry);
    const position = entry.position;
    const legacy = legacyContext(config);
    const mover = position.pawns[position.turn];
    const opponent = position.pawns[position.turn === 'A' ? 'B' : 'A'];
    const expected = Array.from(legacy.getMovesFrom(mover, opponent, new Set(position.walls)), ({ r, c }) => ({ r, c }))
      .sort((left, right) => engine.encodeAction(config, { kind: 'pawn', to: left })
        - engine.encodeAction(config, { kind: 'pawn', to: right }));
    assert.deepEqual(engine.legalPawnDestinations(config, position), expected, entry.id);
  }
});

test('policy actions round trip and legal results are sorted, complete, and immutable', () => {
  for (const config of Object.values(fixture.configs)) {
    assert.equal(engine.policySize(config), config.rows === 9 ? 209 : 121);
    for (let code = 0; code < engine.policySize(config); code += 1) assert.equal(engine.encodeAction(config, engine.decodeAction(config, code)), code);
    const state = engine.createInitialState(config); const before = JSON.stringify(state);
    const codes = engine.legalActionCodes(config, state);
    assert.equal(engine.positionKey(config, state.position), state.positionKey);
    assert.deepEqual(engine.legalPositionActionCodes(config, state.position), codes);
    assert.deepEqual(codes, [...codes].sort((a, b) => a - b));
    const mask = engine.legalActionMask(config, state);
    assert.deepEqual(engine.legalPositionActionMask(config, state.position), mask);
    assert.equal(mask.length, engine.policySize(config));
    assert.deepEqual([...mask].flatMap((legal, code) => legal ? [code] : []), codes);
    const next = engine.applyAction(config, state, engine.decodeAction(config, codes[0]));
    assert.equal(JSON.stringify(state), before, 'applyAction mutates input');
    assert.notEqual(next.position, state.position);
  }
  const config = fixture.configs['standard-a'];
  assertCode(() => engine.decodeAction(config, -1), 'invalid_action_code');
  assert.equal(engine.edgeBlocked(config, { r: 0, c: 0 }, { r: 1, c: 0 }, ['H-0-0']), true);
  assert.equal(engine.edgeBlocked(config, { r: 0, c: 2 }, { r: 1, c: 2 }, ['H-0-0']), false);
  const unordered = {
    pawns: { A: { r: 8, c: 4 }, B: { r: 0, c: 4 } },
    walls: ['V-4-4', 'H-0-0'],
    stock: { A: 9, B: 9 },
    turn: 'A'
  };
  assert.deepEqual(engine.normalizePosition(config, unordered).walls, ['H-0-0', 'V-4-4']);
});

test('validation is fail-closed while remaining independent of JSON object property order', () => {
  const config = fixture.configs['standard-a'];
  const initial = engine.createInitialState(config);
  const reordered = {
    outcome: { kind: 'ongoing' },
    repetitionCounts: initial.repetitionCounts.map(({ positionKey, count }) => ({ count, positionKey })),
    historyStartPly: initial.historyStartPly,
    ply: initial.ply,
    positionKey: initial.positionKey,
    position: {
      turn: initial.position.turn,
      stock: { B: initial.position.stock.B, A: initial.position.stock.A },
      walls: [],
      pawns: {
        B: { c: initial.position.pawns.B.c, r: initial.position.pawns.B.r },
        A: { c: initial.position.pawns.A.c, r: initial.position.pawns.A.r }
      }
    }
  };
  assert.deepEqual(engine.validateState(config, reordered), initial);

  const extraConfig = copy(config); extraConfig.start.A.extra = true;
  assertCode(() => engine.validateConfig(extraConfig), 'invalid_config');
  const inheritedConfig = Object.assign(Object.create({ ruleset: config.ruleset }), copy(config));
  delete inheritedConfig.ruleset;
  assertCode(() => engine.validateConfig(inheritedConfig), 'invalid_config');
  const hiddenFeature = copy(config);
  Object.defineProperty(hiddenFeature, 'hammerMode', { value: true });
  assertCode(() => engine.validateConfig(hiddenFeature), 'invalid_config');
  const unsafeStock = copy(config); unsafeStock.initialStock.A = 2 ** 54;
  assertCode(() => engine.validateConfig(unsafeStock), 'invalid_config');
  const extraPosition = copy(initial.position); extraPosition.pawns.A.extra = true;
  assertCode(() => engine.normalizePosition(config, extraPosition), 'invalid_position');
  assertCode(() => engine.encodeAction(config, { kind: 'pawn', to: { r: 7, c: 4, extra: true } }), 'invalid_action');
  const inheritedAction = Object.assign(Object.create({ kind: 'pawn' }), { to: { r: 7, c: 4 }, extra: false });
  assertCode(() => engine.encodeAction(config, inheritedAction), 'invalid_action');
  assertCode(() => engine.hasPath(config, initial.position.pawns.A, config.goalRows.A, ['not-a-wall']), 'invalid_wall');
  assertCode(() => engine.edgeBlocked(config, { r: 8, c: 4, extra: true }, { r: 7, c: 4 }, []), 'invalid_edge');
  assertCode(() => engine.adjudicate(config, {
    goalWinner: null, resultingPositionCount: 1, resultingPly: 1, externalResult: 'timeout'
  }), 'invalid_adjudication');
  assertCode(() => engine.adjudicate(config, {
    goalWinner: null, resultingPositionCount: 2 ** 54, resultingPly: 1
  }), 'invalid_adjudication');
  assertCode(() => engine.comparePositionKeys(initial.positionKey, 42), 'invalid_position_key');

  const extraState = copy(initial); extraState.externalClock = 10;
  assertCode(() => engine.validateState(config, extraState), 'invalid_state');
  const extraCount = copy(initial); extraCount.repetitionCounts[0].extra = true;
  assertCode(() => engine.validateState(config, extraCount), 'invalid_state');
});

test('state validation rejects forged repetition resets and incompatible history', () => {
  const config = fixture.configs['standard-a'];
  const initial = engine.createInitialState(config);
  const impossibleOpeningPosition = {
    pawns: { A: { r: 6, c: 4 }, B: { r: 0, c: 4 } },
    walls: [],
    stock: { A: 10, B: 10 },
    turn: 'B'
  };
  const impossibleOpeningKey = engine.positionKey(config, impossibleOpeningPosition);
  assertCode(() => engine.validateState(config, {
    position: impossibleOpeningPosition,
    positionKey: impossibleOpeningKey,
    ply: 1,
    historyStartPly: 0,
    repetitionCounts: [
      { positionKey: initial.positionKey, count: 1 },
      { positionKey: impossibleOpeningKey, count: 1 }
    ].sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey)),
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');

  const impossibleDraw = {
    ...copy(initial),
    ply: 2,
    repetitionCounts: [{ positionKey: initial.positionKey, count: 3 }],
    outcome: { kind: 'draw', reason: 'threefold_repetition' }
  };
  assertCode(() => engine.validateState(config, impossibleDraw), 'invalid_state');

  const afterA = engine.applyAction(config, initial, { kind: 'pawn', to: { r: 7, c: 4 } });
  const afterB = engine.applyAction(config, afterA, { kind: 'pawn', to: { r: 1, c: 4 } });

  const reset = copy(afterB);
  reset.historyStartPly = reset.ply;
  reset.repetitionCounts = [{ positionKey: reset.positionKey, count: 1 }];
  assertCode(() => engine.validateState(config, reset), 'invalid_state');

  const wrongTurnTotals = copy(afterB);
  wrongTurnTotals.repetitionCounts = [
    { positionKey: initial.positionKey, count: 2 },
    { positionKey: afterB.positionKey, count: 1 }
  ].sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey));
  assertCode(() => engine.validateState(config, wrongTurnTotals), 'invalid_state');

  const forgedKey = copy(afterB);
  forgedKey.repetitionCounts = [
    { positionKey: 'forged', count: 1 },
    { positionKey: afterB.positionKey, count: 2 }
  ].sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey));
  assertCode(() => engine.validateState(config, forgedKey), 'invalid_state');

  const teleportedPosition = {
    pawns: { A: { r: 3, c: 3 }, B: { r: 5, c: 5 } },
    walls: [],
    stock: { A: 10, B: 10 },
    turn: 'B'
  };
  const teleportedKey = engine.positionKey(config, teleportedPosition);
  const teleportedHistory = copy(afterB);
  teleportedHistory.repetitionCounts = [
    { positionKey: initial.positionKey, count: 1 },
    { positionKey: teleportedKey, count: 1 },
    { positionKey: afterB.positionKey, count: 1 }
  ].sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey));
  assertCode(() => engine.validateState(config, teleportedHistory), 'invalid_state');

  const forgedWinnerPosition = {
    pawns: { A: { r: 0, c: 4 }, B: { r: 0, c: 3 } },
    walls: ['H-4-0'],
    stock: { A: 10, B: 9 },
    turn: 'A'
  };
  const forgedWinnerKey = engine.positionKey(config, forgedWinnerPosition);
  assertCode(() => engine.validateState(config, {
    position: forgedWinnerPosition,
    positionKey: forgedWinnerKey,
    ply: 2,
    historyStartPly: 2,
    repetitionCounts: [{ positionKey: forgedWinnerKey, count: 1 }],
    outcome: { kind: 'win', winner: 'A', reason: 'goal' }
  }), 'invalid_state');

  const tooManyWallsPosition = {
    pawns: copy(initial.position.pawns),
    walls: ['H-0-0', 'H-2-0'],
    stock: { A: 8, B: 10 },
    turn: 'B'
  };
  const tooManyWallsKey = engine.positionKey(config, tooManyWallsPosition);
  assertCode(() => engine.validateState(config, {
    position: tooManyWallsPosition,
    positionKey: tooManyWallsKey,
    ply: 1,
    historyStartPly: 1,
    repetitionCounts: [{ positionKey: tooManyWallsKey, count: 1 }],
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');

  const wrongPayerPosition = {
    pawns: copy(initial.position.pawns),
    walls: ['H-0-0'],
    stock: { A: 10, B: 9 },
    turn: 'B'
  };
  const wrongPayerKey = engine.positionKey(config, wrongPayerPosition);
  assertCode(() => engine.validateState(config, {
    position: wrongPayerPosition,
    positionKey: wrongPayerKey,
    ply: 1,
    historyStartPly: 1,
    repetitionCounts: [{ positionKey: wrongPayerKey, count: 1 }],
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');

  const lateWrongPayerPosition = {
    pawns: copy(initial.position.pawns),
    walls: ['H-0-0'],
    stock: { A: 9, B: 10 },
    turn: 'A'
  };
  const lateWrongPayerKey = engine.positionKey(config, lateWrongPayerPosition);
  assertCode(() => engine.validateState(config, {
    position: lateWrongPayerPosition,
    positionKey: lateWrongPayerKey,
    ply: 2,
    historyStartPly: 2,
    repetitionCounts: [{ positionKey: lateWrongPayerKey, count: 1 }],
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');

  const afterMispaidWall = {
    pawns: copy(initial.position.pawns),
    walls: ['H-0-0'],
    stock: { A: 10, B: 9 },
    turn: 'B'
  };
  const afterMispaidWallKey = engine.positionKey(config, afterMispaidWall);
  const afterMispaidPawn = {
    pawns: { A: copy(initial.position.pawns.A), B: { r: 1, c: 4 } },
    walls: ['H-0-0'],
    stock: { A: 10, B: 9 },
    turn: 'A'
  };
  const afterMispaidPawnKey = engine.positionKey(config, afterMispaidPawn);
  assertCode(() => engine.validateState(config, {
    position: afterMispaidPawn,
    positionKey: afterMispaidPawnKey,
    ply: 2,
    historyStartPly: 1,
    repetitionCounts: [
      { positionKey: afterMispaidWallKey, count: 1 },
      { positionKey: afterMispaidPawnKey, count: 1 }
    ].sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey)),
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');

  const displacedHistoryPositions = [
    {
      pawns: { A: { r: 2, c: 4 }, B: { r: 0, c: 4 } },
      walls: ['H-0-0'], stock: { A: 9, B: 10 }, turn: 'B'
    },
    {
      pawns: { A: { r: 2, c: 4 }, B: { r: 1, c: 4 } },
      walls: ['H-0-0'], stock: { A: 9, B: 10 }, turn: 'A'
    },
    {
      pawns: { A: { r: 2, c: 3 }, B: { r: 1, c: 4 } },
      walls: ['H-0-0'], stock: { A: 9, B: 10 }, turn: 'B'
    }
  ];
  const displacedKeys = displacedHistoryPositions.map((position) => engine.positionKey(config, position));
  assertCode(() => engine.validateState(config, {
    position: displacedHistoryPositions[2],
    positionKey: displacedKeys[2],
    ply: 3,
    historyStartPly: 1,
    repetitionCounts: displacedKeys.map((positionKey) => ({ positionKey, count: 1 }))
      .sort((left, right) => engine.comparePositionKeys(left.positionKey, right.positionKey)),
    outcome: { kind: 'ongoing' }
  }), 'invalid_state');
});

test('immutable transitions enforce goal, repetition, cap, wall reset, and terminal behavior', () => {
  const config = fixture.configs['standard-a'];

  let goalState = engine.createInitialState(config);
  while (goalState.outcome.kind === 'ongoing') {
    const player = goalState.position.turn;
    const destinations = engine.legalPawnDestinations(config, goalState.position);
    const target = destinations.reduce((best, candidate) => {
      if (!best) return candidate;
      return player === 'A'
        ? (candidate.r < best.r ? candidate : best)
        : (candidate.r > best.r ? candidate : best);
    }, null);
    goalState = engine.applyAction(config, goalState, { kind: 'pawn', to: target });
  }
  assert.equal(goalState.outcome.kind, 'win');
  assert.equal(goalState.outcome.reason, 'goal');
  const goalWinner = goalState.outcome.winner;
  assert.deepEqual(engine.legalActions(config, goalState), []);
  assert.equal(engine.legalActionCodes(config, goalState).length, 0);
  assert.equal(engine.legalActionMask(config, goalState).reduce((sum, value) => sum + value, 0), 0);
  assert.deepEqual(engine.validateState(config, {
    outcome: { reason: 'goal', winner: goalWinner, kind: 'win' },
    repetitionCounts: goalState.repetitionCounts.map(({ positionKey, count }) => ({ count, positionKey })),
    historyStartPly: goalState.historyStartPly,
    ply: goalState.ply,
    positionKey: goalState.positionKey,
    position: goalState.position
  }).outcome, goalState.outcome);
  assertCode(() => engine.applyAction(config, goalState, { kind: 'pawn', to: { r: 0, c: 0 } }), 'terminal_state');

  let repetitionState = engine.createInitialState(config);
  const cycle = [
    { r: 8, c: 3 }, { r: 0, c: 3 }, { r: 8, c: 4 }, { r: 0, c: 4 },
    { r: 8, c: 3 }, { r: 0, c: 3 }, { r: 8, c: 4 }, { r: 0, c: 4 }
  ];
  for (const to of cycle) repetitionState = engine.applyAction(config, repetitionState, { kind: 'pawn', to });
  assert.deepEqual(repetitionState.outcome, { kind: 'draw', reason: 'threefold_repetition' });

  const capConfig = { ...copy(config), plyCap: 2 };
  let capState = engine.createInitialState(capConfig);
  capState = engine.applyAction(capConfig, capState, { kind: 'pawn', to: { r: 7, c: 4 } });
  capState = engine.applyAction(capConfig, capState, { kind: 'pawn', to: { r: 1, c: 4 } });
  assert.deepEqual(capState.outcome, { kind: 'draw', reason: 'ply_cap' });

  const wallState = engine.applyAction(config, engine.createInitialState(config), { kind: 'wall', wall: 'H-0-0' });
  assert.equal(wallState.historyStartPly, 1);
  assert.deepEqual(wallState.repetitionCounts, [{ positionKey: wallState.positionKey, count: 1 }]);
  assertCode(() => engine.applyAction(config, engine.createInitialState(config), { kind: 'pawn', to: { r: 0, c: 0 } }), 'illegal_action');
});
