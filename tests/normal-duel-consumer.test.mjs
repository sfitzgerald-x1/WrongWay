import assert from 'node:assert/strict';
import test from 'node:test';
import {
  classifyNormalDuelConsumer,
  createConsumerRouter,
  normalDuelLegalMoves,
  normalDuelMoveTowardGoal,
  normalDuelTryWall
} from '../js/normal-duel-consumer.mjs';

const normalScope = Object.freeze({
  gameMode: 'pvc', map: 'duel', duelSize: 'standard', gameTimeMode: 'none',
  chaosMode: false, hammerMode: false, dropMode: false, isRanked: false,
  initialStock: 10, is2v2: false
});
const snapshot9A = Object.freeze({
  pA: Object.freeze({ r: 8, c: 4 }), pB: Object.freeze({ r: 0, c: 4 }),
  walls: Object.freeze([]), turn: 'A'
});
const snapshot9B = Object.freeze({ ...snapshot9A, turn: 'B' });
const snapshot7 = Object.freeze({
  pA: Object.freeze({ r: 6, c: 3 }), pB: Object.freeze({ r: 0, c: 3 }),
  walls: Object.freeze([]), turn: 'A'
});

test('consumer legal-move queries preserve actual A/B roles on both board sizes', () => {
  assert.deepEqual(normalDuelLegalMoves('standard', snapshot9A), [{ r: 7, c: 4 }, { r: 8, c: 3 }, { r: 8, c: 5 }]);
  assert.deepEqual(normalDuelLegalMoves('standard', snapshot9B), [{ r: 0, c: 3 }, { r: 0, c: 5 }, { r: 1, c: 4 }]);
  assert.deepEqual(normalDuelLegalMoves('blitz', snapshot7), [{ r: 5, c: 3 }, { r: 6, c: 2 }, { r: 6, c: 4 }]);
});

test('consumer carries permissive adjacent exits through the engine', () => {
  const snapshot = {
    pA: { r: 4, c: 4 }, pB: { r: 3, c: 4 }, walls: ['H-2-4'], turn: 'A'
  };
  assert.deepEqual(normalDuelLegalMoves('standard', snapshot), [
    { r: 3, c: 3 }, { r: 3, c: 5 }, { r: 4, c: 3 }, { r: 4, c: 5 }, { r: 5, c: 4 }
  ]);
});

test('consumer wall query preserves Set|null compatibility and does not mutate snapshots', () => {
  const snapshot = { pA: { r: 8, c: 4 }, pB: { r: 0, c: 4 }, walls: new Set(['H-0-0']), turn: 'A' };
  const before = [...snapshot.walls];
  const accepted = normalDuelTryWall('standard', snapshot, 'V-1-0');
  assert.ok(accepted instanceof Set);
  assert.deepEqual([...accepted].sort(), ['H-0-0', 'V-1-0']);
  assert.equal(normalDuelTryWall('standard', snapshot, 'H-0-1'), null);
  assert.equal(normalDuelTryWall('standard', snapshot, 'not-a-wall'), null);
  assert.deepEqual([...snapshot.walls], before);
});

test('moveTowardGoal retains path preference but lets the engine establish legality', () => {
  const open = { pA: { r: 4, c: 4 }, pB: { r: 3, c: 4 }, walls: [], turn: 'A' };
  assert.deepEqual(normalDuelMoveTowardGoal('standard', open, [{ r: 4, c: 4 }, { r: 3, c: 4 }]), { r: 2, c: 4 });
  const blocked = { ...open, walls: ['H-2-4'] };
  assert.deepEqual(normalDuelMoveTowardGoal('standard', blocked, [{ r: 4, c: 4 }, { r: 3, c: 4 }]), { r: 3, c: 3 });
  assert.equal(normalDuelMoveTowardGoal('standard', blocked, [{ r: 4, c: 4 }, { r: 2, c: 4 }]), null);
});

test('scope classification is explicit and fail closed', () => {
  assert.deepEqual(classifyNormalDuelConsumer(normalScope), { eligible: true, reason: null });
  const cases = [
    ['gameMode', 'online', 'legacy_game_mode'],
    ['map', 'classic', 'legacy_map'],
    ['duelSize', 'giant', 'legacy_duel_size'],
    ['gameTimeMode', 'blitz-30', 'legacy_clock'],
    ['chaosMode', true, 'legacy_chaos_mode'],
    ['hammerMode', true, 'legacy_hammer_mode'],
    ['dropMode', 'often', 'legacy_drop_mode'],
    ['isRanked', true, 'legacy_ranked'],
    ['initialStock', 12, 'legacy_stock_config'],
    ['is2v2', true, 'legacy_two_v_two']
  ];
  for (const [key, value, reason] of cases) assert.deepEqual(classifyNormalDuelConsumer({ ...normalScope, [key]: value }), { eligible: false, reason });
  assert.deepEqual(classifyNormalDuelConsumer({ ...normalScope, chaosMode: undefined }), { eligible: false, reason: 'legacy_chaos_mode' });
  assert.deepEqual(classifyNormalDuelConsumer({ ...normalScope, futureRule: false }), { eligible: false, reason: 'legacy_invalid_scope' });
  assert.deepEqual(classifyNormalDuelConsumer(null), { eligible: false, reason: 'legacy_invalid_scope' });
});

test('router calls reference only for eligible modes and preserves caller inputs', () => {
  const calls = { reference: 0, legacy: 0 };
  const reference = {
    legalMoves: (...args) => { calls.reference += 1; return ['reference-moves', args]; },
    tryWall: (...args) => { calls.reference += 1; return ['reference-wall', args]; },
    moveTowardGoal: (...args) => { calls.reference += 1; return ['reference-path', args]; }
  };
  const legacy = {
    legalMoves: (...args) => { calls.legacy += 1; return ['legacy-moves', args]; },
    tryWall: (...args) => { calls.legacy += 1; return ['legacy-wall', args]; },
    moveTowardGoal: (...args) => { calls.legacy += 1; return ['legacy-path', args]; }
  };
  const router = createConsumerRouter({ reference, legacy });
  const snapshot = { pA: { r: 8, c: 4 }, pB: { r: 0, c: 4 }, walls: [], turn: 'A' };
  const path = [{ r: 8, c: 4 }, { r: 7, c: 4 }];
  const before = structuredClone({ scope: normalScope, snapshot, path });
  assert.equal(router.legalMoves(normalScope, 'standard', snapshot)[0], 'reference-moves');
  assert.equal(router.tryWall({ ...normalScope, gameMode: 'online' }, 'standard', snapshot, 'H-0-0')[0], 'legacy-wall');
  assert.equal(router.moveTowardGoal({ ...normalScope, hammerMode: true }, 'standard', snapshot, path)[0], 'legacy-path');
  assert.deepEqual(calls, { reference: 1, legacy: 2 });
  assert.deepEqual({ scope: normalScope, snapshot, path }, before);
});

test('browser adapter publishes a frozen consumer surface', async () => {
  const originalWindow = globalThis.window;
  try {
    globalThis.window = {};
    const adapter = await import(`../js/normal-duel-browser-adapter.mjs?test=${Date.now()}`);
    assert.equal(globalThis.window.NormalDuelConsumer, adapter.NormalDuelConsumer);
    assert.equal(typeof adapter.NormalDuelConsumer.classify, 'function');
    assert.deepEqual(adapter.NormalDuelConsumer.legalMoves(normalScope, snapshot9A), [
      { r: 7, c: 4 }, { r: 8, c: 3 }, { r: 8, c: 5 }
    ]);
    assert.throws(() => adapter.NormalDuelConsumer.legalMoves({ ...normalScope, gameMode: 'online' }, snapshot9A));
    assert.equal(Object.isFrozen(adapter.NormalDuelConsumer), true);
    assert.equal(Object.isFrozen(globalThis.window.NormalDuelConsumer), true);
  } finally {
    if (originalWindow === undefined) delete globalThis.window;
    else globalThis.window = originalWindow;
  }
});
