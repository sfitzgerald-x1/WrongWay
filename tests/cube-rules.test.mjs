import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BARRICADE_AREA,
  BARRICADES_PER_PLAYER,
  SIZE,
  blockedEdgesForWalls,
  createInitialState,
  legalMoves,
  shortestPath,
  validateWallCandidate,
  wallDimensions,
  wallEdgeKeys
} from '../js/cube-rules.mjs';

test('the initial state begins at the centers of opposite faces', () => {
  const state = createInitialState();
  assert.deepEqual(state.players.A, { x: 4, y: 4, z: 8 });
  assert.deepEqual(state.players.B, { x: 4, y: 4, z: 0 });
  assert.equal(state.remaining.A, BARRICADES_PER_PLAYER);
  assert.equal(state.remaining.B, BARRICADES_PER_PLAYER);
});

test('a barricade controls exactly two ninths of a 9 by 9 plane', () => {
  const wall = { plane: 'XY', layer: 4, u: 0, v: 0, rotated: false };
  assert.deepEqual(wallDimensions(wall), { width: 3, height: 6 });
  assert.equal(wallEdgeKeys(wall).length, BARRICADE_AREA);
  assert.equal(BARRICADE_AREA / (SIZE * SIZE), 2 / 9);

  wall.rotated = true;
  assert.deepEqual(wallDimensions(wall), { width: 6, height: 3 });
  assert.equal(wallEdgeKeys(wall).length, BARRICADE_AREA);
});

test('initial routes cross eight steps through the cube', () => {
  const state = createInitialState();
  const blocked = blockedEdgesForWalls(state.walls);
  assert.equal(shortestPath(state.players.A, 0, blocked).length - 1, 8);
  assert.equal(shortestPath(state.players.B, 8, blocked).length - 1, 8);
});

test('a face pawn has five legal orthogonal moves', () => {
  const state = createInitialState();
  assert.equal(legalMoves(state.players.A, state.players.B, new Set()).length, 5);
});

test('an adjacent opponent opens every non-return exit as a jump', () => {
  const moves = legalMoves(
    { x: 4, y: 4, z: 5 },
    { x: 4, y: 4, z: 4 },
    new Set()
  );
  assert.equal(moves.length, 10);
  assert.ok(moves.some((move) => move.x === 4 && move.y === 4 && move.z === 3));
});

test('wall validation rejects overlapping controlled crossings', () => {
  const state = createInitialState();
  const first = { plane: 'XY', layer: 4, u: 0, v: 0, rotated: false };
  assert.equal(validateWallCandidate(first, [], state.players.A, state.players.B).valid, true);

  const overlap = { plane: 'XY', layer: 4, u: 1, v: 0, rotated: false };
  assert.equal(validateWallCandidate(overlap, [first], state.players.A, state.players.B).valid, false);
});

test('all 20 barricades can be placed while preserving both routes', () => {
  const state = createInitialState();

  for (let turn = 0; turn < BARRICADES_PER_PLAYER * 2; turn += 1) {
    let selected = null;

    search:
    for (const plane of ['XY', 'XZ', 'YZ']) {
      for (let layer = 0; layer < SIZE - 1; layer += 1) {
        for (const rotated of [false, true]) {
          const { width, height } = wallDimensions({ rotated });
          for (let u = 0; u <= SIZE - width; u += 1) {
            for (let v = 0; v <= SIZE - height; v += 1) {
              const candidate = { plane, layer, rotated, u, v };
              if (validateWallCandidate(
                candidate,
                state.walls,
                state.players.A,
                state.players.B
              ).valid) {
                selected = candidate;
                break search;
              }
            }
          }
        }
      }
    }

    assert.ok(selected, `expected a legal placement for barricade ${turn + 1}`);
    state.walls.push(selected);
  }

  const blocked = blockedEdgesForWalls(state.walls);
  assert.equal(blocked.size, BARRICADE_AREA * BARRICADES_PER_PLAYER * 2);
  assert.ok(shortestPath(state.players.A, 0, blocked));
  assert.ok(shortestPath(state.players.B, SIZE - 1, blocked));
});
