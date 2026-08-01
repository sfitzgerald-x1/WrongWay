import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyAction, createInitialState, legalActionCodes, policySize
} from '../js/normal-duel-engine.mjs';
import {
  NN_INPUT_PLANES, NN_PLANE_LAYOUT, encodePolicyTarget, encodeState, legalMaskFloat, planeIndex
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
 * position with the roles swapped once the board is mirrored, so the two
 * encodings must be identical. Both sides detour around one another. */
const DIRECT_LINE = [
  wall('V-1-7'), wall('H-7-0'),
  pawn(8, 3), pawn(0, 5), pawn(7, 3), pawn(1, 5), pawn(6, 3), pawn(2, 5), pawn(5, 3),
  pawn(3, 5), pawn(4, 3), pawn(4, 5), pawn(3, 3), pawn(5, 5), pawn(3, 4), pawn(5, 4)
];

test('layout is frozen, self-consistent and small', () => {
  assert.equal(NN_PLANE_LAYOUT.length, NN_INPUT_PLANES);
  assert.ok(Object.isFrozen(NN_PLANE_LAYOUT));
  assert.ok(NN_PLANE_LAYOUT.every((entry) => Object.isFrozen(entry) && typeof entry.name === 'string'));
  assert.equal(new Set(NN_PLANE_LAYOUT.map((entry) => entry.name)).size, NN_INPUT_PLANES);
  assert.ok(!NN_PLANE_LAYOUT.some((entry) => entry.name === 'side_to_move'), 'perspective transform replaces a side-to-move plane');
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

test('perspective invariance: mover-relative mirror positions encode identically', () => {
  const bToMove = play(CONFIG_9X9, MIRRORED_LINE);
  const aToMove = play(CONFIG_9X9, DIRECT_LINE);

  assert.equal(bToMove.position.turn, 'B');
  assert.equal(aToMove.position.turn, 'A');
  assert.deepEqual(bToMove.position.pawns, { A: { r: 3, c: 4 }, B: { r: 5, c: 4 } });
  assert.deepEqual(aToMove.position.pawns, { A: { r: 3, c: 4 }, B: { r: 5, c: 4 } });
  assert.notEqual(bToMove.positionKey, aToMove.positionKey);
  // Non-vacuous: both sides hold walls on the board, and the wall sets differ.
  assert.deepEqual(bToMove.position.walls, ['H-0-0', 'V-6-7']);
  assert.deepEqual(aToMove.position.walls, ['H-7-0', 'V-1-7']);
  assert.deepEqual(bToMove.position.stock, { A: 9, B: 9 });

  assert.deepEqual(encodeState(CONFIG_9X9, bToMove), encodeState(CONFIG_9X9, aToMove));
});

test('determinism: repeated encoding is byte-identical', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('V-3-3')]);
  const first = encodeState(CONFIG_9X9, state);
  const second = encodeState(CONFIG_9X9, state);
  assert.deepEqual(new Uint8Array(first.buffer.slice(0)), new Uint8Array(second.buffer.slice(0)));
});

test('mover and opponent planes follow the side to move', () => {
  const initial = createInitialState(CONFIG_9X9);
  const before = encodeState(CONFIG_9X9, initial);
  // A to move: identity orientation, A on its own back rank (row 8).
  assert.deepEqual(hotCells(plane(before, 'mover_pawn')), [{ r: 8, c: 4, value: 1 }]);
  assert.deepEqual(hotCells(plane(before, 'opponent_pawn')), [{ r: 0, c: 4, value: 1 }]);

  const after = encodeState(CONFIG_9X9, applyAction(CONFIG_9X9, initial, pawn(7, 4)));
  // B to move: rows mirror, so B (engine row 0) is now the mover on row 8 and
  // A (engine row 7) shows up in the opponent plane on row 1.
  assert.deepEqual(hotCells(plane(after, 'mover_pawn')), [{ r: 8, c: 4, value: 1 }]);
  assert.deepEqual(hotCells(plane(after, 'opponent_pawn')), [{ r: 1, c: 4, value: 1 }]);
  assert.notDeepEqual(plane(before, 'opponent_pawn'), plane(after, 'opponent_pawn'));
});

test('stock planes are mover-relative and scaled', () => {
  const state = play(CONFIG_9X9, [wall('H-4-4'), pawn(1, 4)]);
  const tensor = encodeState(CONFIG_9X9, state);
  assert.equal(state.position.turn, 'A');
  assert.ok(plane(tensor, 'mover_stock').every((value) => value === Math.fround(0.9)));
  assert.ok(plane(tensor, 'opponent_stock').every((value) => value === 1));
});

test('wall planes: empty board is zero, one wall places two segment cells', () => {
  const empty = encodeState(CONFIG_9X9, createInitialState(CONFIG_9X9));
  assert.ok(plane(empty, 'wall_horizontal').every((value) => value === 0));
  assert.ok(plane(empty, 'wall_vertical').every((value) => value === 0));

  const horizontal = encodeState(CONFIG_9X9, play(CONFIG_9X9, [wall('H-4-4')]));
  // H-4-4 blocks the edges below (4,4) and (4,5); B is to move so rows mirror.
  assert.deepEqual(hotCells(plane(horizontal, 'wall_horizontal')),
    [{ r: 3, c: 4, value: 1 }, { r: 3, c: 5, value: 1 }]);
  assert.ok(plane(horizontal, 'wall_vertical').every((value) => value === 0));

  const vertical = encodeState(CONFIG_9X9, play(CONFIG_9X9, [wall('V-4-4')]));
  // V-4-4 blocks the edges right of (4,4) and (5,4); mirrored rows 4 and 3.
  assert.deepEqual(hotCells(plane(vertical, 'wall_vertical')),
    [{ r: 3, c: 4, value: 1 }, { r: 4, c: 4, value: 1 }]);
  assert.ok(plane(vertical, 'wall_horizontal').every((value) => value === 0));
});

test('goal proximity plane peaks on the mover goal row', () => {
  const tensor = encodeState(CONFIG_9X9, createInitialState(CONFIG_9X9));
  const values = plane(tensor, 'goal_proximity');
  for (let r = 0; r < 9; r += 1) {
    for (let c = 0; c < 9; c += 1) assert.equal(values[r * 9 + c], (8 - r) / 8);
  }
});

test('legalMaskFloat agrees with legalActionCodes', () => {
  const state = play(CONFIG_9X9, [pawn(7, 4), pawn(1, 4), wall('H-4-4')]);
  const mask = legalMaskFloat(CONFIG_9X9, state);
  const legal = new Set(legalActionCodes(CONFIG_9X9, state));
  assert.equal(mask.length, policySize(CONFIG_9X9));
  assert.ok(legal.size > 0);
  for (let code = 0; code < mask.length; code += 1) {
    assert.equal(mask[code], legal.has(code) ? 1 : 0, `code ${code}`);
  }
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
  assert.deepEqual(config, configBefore);
  assert.deepEqual(state, stateBefore);
});
