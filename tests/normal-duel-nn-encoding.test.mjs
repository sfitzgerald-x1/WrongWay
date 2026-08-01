import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyAction, createInitialState, encodeAction, legalActionCodes, legalActions, policySize
} from '../js/normal-duel-engine.mjs';
import {
  NN_INPUT_PLANES, NN_PLANE_LAYOUT, encodeLegalPolicyTarget, encodePolicyTarget, encodeState,
  legalMaskFloat, planeIndex
} from '../js/normal-duel-nn-encoding.mjs';

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

const CELLS = 81;
const PAWN_CODES = CELLS; // codes [0, 81) are pawn moves; wall codes follow.

const pawn = (r, c) => ({ kind: 'pawn', to: { r, c } });
const wall = (text) => ({ kind: 'wall', wall: text });

function play(config, actions) {
  return actions.reduce((state, action) => applyAction(config, state, action), createInitialState(config));
}

function plane(tensor, name) {
  const base = planeIndex(name) * CELLS;
  return tensor.slice(base, base + CELLS);
}

function hotCells(planeValues) {
  const cells = [];
  for (let index = 0; index < planeValues.length; index += 1) {
    if (planeValues[index] !== 0) cells.push({ r: Math.floor(index / 9), c: index % 9, value: planeValues[index] });
  }
  return cells;
}

/** Play legal wall actions until both sides are out of stock. */
function drainStock(config) {
  let state = createInitialState(config);
  for (let step = 0; step < 64; step += 1) {
    if (state.position.stock.A === 0 && state.position.stock.B === 0) return state;
    const wallAction = legalActions(config, state).find((action) => action.kind === 'wall');
    assert.ok(wallAction, 'a legal wall placement should exist while stock remains');
    state = applyAction(config, state, wallAction);
  }
  throw new Error('drainStock did not converge');
}

/** Both pawns marched to within a step or two of their goal rows, no walls. */
const LATE_GAME_LINE = [
  pawn(7, 4), pawn(0, 3), pawn(6, 4), pawn(1, 3), pawn(5, 4), pawn(2, 3), pawn(4, 4),
  pawn(3, 3), pawn(3, 4), pawn(4, 3), pawn(2, 4), pawn(5, 3), pawn(1, 4), pawn(6, 3)
];

/**
 * Two lines that reach mover-relative-identical positions with opposite sides
 * to move. Both are ordinary legal games from `createInitialState`.
 *
 * MIRRORED_LINE ends with B to move, A on (3,4) and B on (5,4); B crosses A
 * with a jump, which is what makes the ply parity work out. Each side spends
 * one wall so the wall planes and the stock planes are exercised too; the two
 * lines use row-mirrored walls (H-0-0 <-> H-7-0, V-6-7 <-> V-1-7) played by
 * opposite sides.
 */
const MIRRORED_LINE = [
  wall('H-0-0'), wall('V-6-7'),
  pawn(7, 4), pawn(1, 4), pawn(6, 4), pawn(2, 4), pawn(5, 4),
  pawn(3, 4), pawn(4, 4), pawn(5, 4), pawn(3, 4)
];

/** DIRECT_LINE ends with A to move, A on (3,4) and B on (5,4) — the same
 * position mirrored, with the roles swapped. Both sides detour around one
 * another. */
const DIRECT_LINE = [
  wall('V-1-7'), wall('H-7-0'),
  pawn(8, 3), pawn(0, 5), pawn(7, 3), pawn(1, 5), pawn(6, 3), pawn(2, 5), pawn(5, 3),
  pawn(3, 5), pawn(4, 3), pawn(4, 5), pawn(3, 3), pawn(5, 5), pawn(3, 4), pawn(5, 4)
];

test('layout is frozen, self-consistent and carries side to move', () => {
  assert.equal(NN_PLANE_LAYOUT.length, NN_INPUT_PLANES);
  assert.equal(NN_INPUT_PLANES, 8);
  assert.ok(Object.isFrozen(NN_PLANE_LAYOUT));
  assert.ok(NN_PLANE_LAYOUT.every((entry) => Object.isFrozen(entry) && typeof entry.name === 'string'));
  assert.equal(new Set(NN_PLANE_LAYOUT.map((entry) => entry.name)).size, NN_INPUT_PLANES);
  assert.deepEqual(NN_PLANE_LAYOUT.map((entry) => entry.name), [
    'mover_pawn', 'opponent_pawn', 'wall_horizontal', 'wall_vertical',
    'mover_stock', 'opponent_stock', 'goal_proximity', 'side_to_move'
  ]);
});

test('shape, dtype and value range', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), wall('H-4-4'), pawn(6, 4)]);
  const tensor = encodeState(CONFIG_9X9, state);
  assert.ok(tensor instanceof Float32Array);
  assert.equal(tensor.length, NN_INPUT_PLANES * CELLS);
  for (const value of tensor) {
    assert.ok(Number.isFinite(value), 'finite');
    assert.ok(value >= 0 && value <= 1, `bounded: ${value}`);
  }
});

/**
 * The coherence property this encoder exists to hold: the input frame and the
 * action frame are the same frame.
 *
 * Byte-identical encodings across mover-relative mirrored states are NO LONGER
 * expected, and their absence is correct rather than a regression. The 209-way
 * policy head speaks the engine's absolute action codes; mirroring rows in the
 * input while leaving codes absolute made two states with identical features
 * carry mirrored labels ("advance toward my goal" was code 22 in one and 58 in
 * the other), which corrupts roughly half of a self-play corpus. Absolute
 * planes plus a `side_to_move` plane keep features and labels in one frame, at
 * the cost of these two states — which really are different positions with
 * different legal moves — no longer colliding.
 */
test('coherence: the input frame and the action frame agree', () => {
  const bToMove = play(CONFIG_9X9, MIRRORED_LINE);
  const aToMove = play(CONFIG_9X9, DIRECT_LINE);

  assert.equal(bToMove.position.turn, 'B');
  assert.equal(aToMove.position.turn, 'A');
  assert.deepEqual(bToMove.position.pawns, { A: { r: 3, c: 4 }, B: { r: 5, c: 4 } });
  assert.deepEqual(aToMove.position.pawns, { A: { r: 3, c: 4 }, B: { r: 5, c: 4 } });
  assert.deepEqual(bToMove.position.walls, ['H-0-0', 'V-6-7']);
  assert.deepEqual(aToMove.position.walls, ['H-7-0', 'V-1-7']);

  // Deliberately different now: same features, different side to move.
  assert.notDeepEqual(encodeState(CONFIG_9X9, bToMove), encodeState(CONFIG_9X9, aToMove));

  for (const state of [bToMove, aToMove]) {
    const mover = state.position.turn;
    const from = state.position.pawns[mover];
    const goalRow = CONFIG_9X9.goalRows[mover];
    const step = { r: from.r + Math.sign(goalRow - from.r), c: from.c };
    const advanceCode = encodeAction(CONFIG_9X9, pawn(step.r, step.c));

    const tensor = encodeState(CONFIG_9X9, state);
    const mask = legalMaskFloat(CONFIG_9X9, state);
    const proximity = plane(tensor, 'goal_proximity');

    // The plane cell the action code names is the plane cell the mover pawn
    // will occupy: same index arithmetic, same frame.
    assert.equal(advanceCode, step.r * 9 + step.c);
    assert.deepEqual(hotCells(plane(tensor, 'mover_pawn')), [{ r: from.r, c: from.c, value: 1 }]);
    assert.equal(mask[advanceCode], 1, `${mover} advance code ${advanceCode} is legal`);
    // The input planes agree that this code moves toward the mover's goal.
    assert.ok(proximity[advanceCode] > proximity[from.r * 9 + from.c],
      'goal_proximity increases at the square the advance code names');
    assert.equal(proximity[goalRow * 9 + from.c], 1);
    assert.equal(plane(tensor, 'side_to_move')[0], mover === 'A' ? 1 : 0);
  }

  // The two states' advance codes are not interchangeable — which is exactly
  // why the input frame must not be mirrored relative to the action frame.
  const bAdvance = encodeAction(CONFIG_9X9, pawn(6, 4));
  const aAdvance = encodeAction(CONFIG_9X9, pawn(2, 4));
  assert.equal(bAdvance, 58);
  assert.equal(aAdvance, 22);
  assert.equal(legalMaskFloat(CONFIG_9X9, bToMove)[aAdvance], 0);
  assert.equal(legalMaskFloat(CONFIG_9X9, aToMove)[bAdvance], 0);
});

test('determinism: repeated encoding is byte-identical', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('V-3-3')]);
  const first = encodeState(CONFIG_9X9, state);
  const second = encodeState(CONFIG_9X9, state);
  assert.deepEqual(new Uint8Array(first.buffer.slice(0)), new Uint8Array(second.buffer.slice(0)));
});

test('pawn planes select by side to move without moving any coordinate', () => {
  const initial = createInitialState(CONFIG_9X9);
  const before = encodeState(CONFIG_9X9, initial);
  // A to move: A sits on absolute row 8, B on absolute row 0.
  assert.deepEqual(hotCells(plane(before, 'mover_pawn')), [{ r: 8, c: 4, value: 1 }]);
  assert.deepEqual(hotCells(plane(before, 'opponent_pawn')), [{ r: 0, c: 4, value: 1 }]);

  const after = encodeState(CONFIG_9X9, applyAction(CONFIG_9X9, initial, pawn(7, 4)));
  // B to move: the two planes swap roles, but every coordinate is unchanged.
  assert.deepEqual(hotCells(plane(after, 'mover_pawn')), [{ r: 0, c: 4, value: 1 }]);
  assert.deepEqual(hotCells(plane(after, 'opponent_pawn')), [{ r: 7, c: 4, value: 1 }]);
});

test('side_to_move plane is constant 1 for A and 0 for B', () => {
  const initial = createInitialState(CONFIG_9X9);
  assert.ok(plane(encodeState(CONFIG_9X9, initial), 'side_to_move').every((value) => value === 1));
  const bTurn = applyAction(CONFIG_9X9, initial, pawn(7, 4));
  assert.ok(plane(encodeState(CONFIG_9X9, bTurn), 'side_to_move').every((value) => value === 0));
});

test('stock planes are mover-relative and scaled', () => {
  const state = play(CONFIG_9X9, [wall('H-4-4'), pawn(1, 4)]);
  const tensor = encodeState(CONFIG_9X9, state);
  assert.equal(state.position.turn, 'A');
  assert.ok(plane(tensor, 'mover_stock').every((value) => value === Math.fround(0.9)));
  assert.ok(plane(tensor, 'opponent_stock').every((value) => value === 1));
});

test('wall planes use absolute rows', () => {
  const empty = encodeState(CONFIG_9X9, createInitialState(CONFIG_9X9));
  assert.ok(plane(empty, 'wall_horizontal').every((value) => value === 0));
  assert.ok(plane(empty, 'wall_vertical').every((value) => value === 0));

  const horizontal = encodeState(CONFIG_9X9, play(CONFIG_9X9, [wall('H-4-4')]));
  // H-4-4 blocks the edges below (4,4) and (4,5); B is to move, and no mirror
  // is applied, so the plane cells are the absolute upper rows.
  assert.deepEqual(hotCells(plane(horizontal, 'wall_horizontal')),
    [{ r: 4, c: 4, value: 1 }, { r: 4, c: 5, value: 1 }]);
  assert.ok(plane(horizontal, 'wall_vertical').every((value) => value === 0));

  const vertical = encodeState(CONFIG_9X9, play(CONFIG_9X9, [wall('V-4-4')]));
  // V-4-4 blocks the edges right of (4,4) and (5,4), absolute rows 4 and 5.
  assert.deepEqual(hotCells(plane(vertical, 'wall_vertical')),
    [{ r: 4, c: 4, value: 1 }, { r: 5, c: 4, value: 1 }]);
  assert.ok(plane(vertical, 'wall_horizontal').every((value) => value === 0));
});

test('goal proximity peaks on the mover goal row and flips with the mover', () => {
  const initial = createInitialState(CONFIG_9X9);
  const aValues = plane(encodeState(CONFIG_9X9, initial), 'goal_proximity');
  const bValues = plane(encodeState(CONFIG_9X9, applyAction(CONFIG_9X9, initial, pawn(7, 4))), 'goal_proximity');
  for (let r = 0; r < 9; r += 1) {
    for (let c = 0; c < 9; c += 1) {
      assert.equal(aValues[r * 9 + c], (8 - r) / 8, `A row ${r}`);
      assert.equal(bValues[r * 9 + c], r / 8, `B row ${r}`);
    }
  }
});

test('legalMaskFloat agrees with legalActionCodes mid-game, at zero stock and late', () => {
  const midGame = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('H-4-4')]);
  const zeroStock = drainStock(CONFIG_9X9);
  const lateGame = play(CONFIG_9X9, LATE_GAME_LINE);

  for (const state of [midGame, zeroStock, lateGame]) {
    const mask = legalMaskFloat(CONFIG_9X9, state);
    const legal = new Set(legalActionCodes(CONFIG_9X9, state));
    assert.equal(mask.length, policySize(CONFIG_9X9));
    assert.ok(legal.size > 0);
    for (let code = 0; code < mask.length; code += 1) {
      assert.equal(mask[code], legal.has(code) ? 1 : 0, `code ${code}`);
    }
  }

  // Zero stock: no wall code may be legal, and the stock planes read zero.
  assert.deepEqual(zeroStock.position.stock, { A: 0, B: 0 });
  const zeroMask = legalMaskFloat(CONFIG_9X9, zeroStock);
  for (let code = PAWN_CODES; code < zeroMask.length; code += 1) assert.equal(zeroMask[code], 0);
  const zeroTensor = encodeState(CONFIG_9X9, zeroStock);
  assert.ok(plane(zeroTensor, 'mover_stock').every((value) => value === 0));
  assert.ok(plane(zeroTensor, 'opponent_stock').every((value) => value === 0));

  // Late game: A stands one step from its goal row, and that step is legal.
  assert.deepEqual(lateGame.position.pawns.A, { r: 1, c: 4 });
  assert.equal(lateGame.position.turn, 'A');
  assert.equal(legalMaskFloat(CONFIG_9X9, lateGame)[encodeAction(CONFIG_9X9, pawn(0, 4))], 1);
});

test('encodePolicyTarget normalises, zeroes off-support and rejects bad input', () => {
  const size = policySize(CONFIG_9X9);
  const target = encodePolicyTarget(CONFIG_9X9, new Map([[63, 3], [72, 1], [100, 4]]));
  assert.equal(target.length, size);
  assert.ok(Math.abs([...target].reduce((sum, value) => sum + value, 0) - 1) < 1e-6);
  assert.ok(Math.abs(target[63] - 0.375) < 1e-6);
  assert.ok(Math.abs(target[72] - 0.125) < 1e-6);
  assert.ok(Math.abs(target[100] - 0.5) < 1e-6);
  for (let code = 0; code < size; code += 1) {
    if (![63, 72, 100].includes(code)) assert.equal(target[code], 0);
  }

  assert.deepEqual(encodePolicyTarget(CONFIG_9X9, { 63: 3, 72: 1, 100: 4 }), target);

  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[-1, 1]])), /invalid_action_code/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[1.5, 1]])), /invalid_action_code/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[size, 1]])), /invalid_action_code/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[0, 0], [1, 0]])), /empty_visit_counts/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map()), /empty_visit_counts/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[0, -1]])), /invalid_visit_count/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[0, Number.NaN]])), /invalid_visit_count/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, [[0, 1]]), /invalid_visit_counts/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, null), /invalid_visit_counts/);
});

test('object keys are validated, not coerced', () => {
  // Canonical decimal integer strings work, in Maps and in plain objects.
  assert.equal(encodePolicyTarget(CONFIG_9X9, { 0: 1 })[0], 1);
  assert.equal(encodePolicyTarget(CONFIG_9X9, new Map([['63', 1]]))[63], 1);

  for (const key of ['', ' ', ' 3 ', '3 ', '0x10', '03', '+3', '-1', '3.0', '1e2', 'NaN', 'Infinity']) {
    assert.throws(() => encodePolicyTarget(CONFIG_9X9, { [key]: 1 }), /invalid_action_code/, `key ${JSON.stringify(key)}`);
    assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[key, 1]])), /invalid_action_code/, `map key ${JSON.stringify(key)}`);
  }
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[null, 1]])), /invalid_action_code/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[{ code: 3 }, 1]])), /invalid_action_code/);
});

test('duplicate codes across number and string keys are rejected', () => {
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([[63, 1], ['63', 2]])), /duplicate_action_code/);
  assert.throws(() => encodePolicyTarget(CONFIG_9X9, new Map([['63', 2], [63, 1]])), /duplicate_action_code/);
  // Distinct codes are unaffected.
  const target = encodePolicyTarget(CONFIG_9X9, new Map([[63, 1], ['64', 1]]));
  assert.equal(target[63], 0.5);
  assert.equal(target[64], 0.5);
});

test('encodeLegalPolicyTarget accepts legal support and rejects everything else', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('H-4-4')]);
  const legal = legalActionCodes(CONFIG_9X9, state);
  assert.ok(legal.length >= 3);
  const [first, second, third] = legal;

  const target = encodeLegalPolicyTarget(CONFIG_9X9, state, new Map([[first, 1], [second, 3]]));
  assert.equal(target.length, policySize(CONFIG_9X9));
  assert.ok(Math.abs(target[first] - 0.25) < 1e-6);
  assert.ok(Math.abs(target[second] - 0.75) < 1e-6);
  assert.equal(target[third], 0);
  // The mask and the target live in the same absolute frame.
  const mask = legalMaskFloat(CONFIG_9X9, state);
  for (let code = 0; code < target.length; code += 1) {
    if (target[code] !== 0) assert.equal(mask[code], 1, `code ${code}`);
  }
  assert.deepEqual(encodeLegalPolicyTarget(CONFIG_9X9, state, { [first]: 1, [second]: 3 }), target);

  const illegal = [...Array(policySize(CONFIG_9X9)).keys()].find((code) => !legal.includes(code));
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_9X9, state, new Map([[illegal, 1]])), /illegal_action_code/);
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_9X9, state, new Map([[first, 1], [illegal, 1]])), /illegal_action_code/);
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_9X9, state, new Map([[policySize(CONFIG_9X9), 1]])), /invalid_action_code/);
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_9X9, state, new Map([[first, 0]])), /empty_visit_counts/);
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_9X9, state, new Map()), /empty_visit_counts/);
  assert.throws(() => encodeLegalPolicyTarget(CONFIG_7X7, createInitialState(CONFIG_7X7), new Map([[0, 1]])),
    /unsupported_board/);
});

test('encodeState rejects non-9x9 configs', () => {
  const state = createInitialState(CONFIG_7X7);
  assert.throws(() => encodeState(CONFIG_7X7, state), /unsupported_board/);
  assert.throws(() => legalMaskFloat(CONFIG_7X7, state), /unsupported_board/);
  assert.throws(() => encodePolicyTarget(CONFIG_7X7, new Map([[0, 1]])), /unsupported_board/);
});

test('encodeState rejects an untrusted state', () => {
  const state = createInitialState(CONFIG_9X9);
  const tampered = { ...state, position: { ...state.position, turn: 'B' } };
  assert.throws(() => encodeState(CONFIG_9X9, tampered), /invalid_state/);
});

test('purity: config and state are unchanged by encoding', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), wall('H-4-4'), pawn(6, 4)]);
  const config = structuredClone(CONFIG_9X9);
  const configBefore = structuredClone(config);
  const stateBefore = structuredClone(state);
  encodeState(config, state);
  legalMaskFloat(config, state);
  encodeLegalPolicyTarget(config, state, new Map([[legalActionCodes(config, state)[0], 1]]));
  assert.deepEqual(config, configBefore);
  assert.deepEqual(state, stateBefore);
});
