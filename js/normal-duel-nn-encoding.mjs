/**
 * Stage 4 neural-network input encoding for the canonical 9x9 normal duel.
 *
 * Pure, deterministic state -> tensor encoding. Rules, geometry and legality
 * always come from the authoritative engine (`js/normal-duel-engine.mjs`);
 * nothing here reimplements a rule.
 *
 * Canonical orientation
 * ---------------------
 * Every plane is written from the SIDE-TO-MOVE's point of view, in a frame
 * where the mover always advances toward row 0 (the top of the encoded plane)
 * and the mover's own pawn starts on the bottom row. Player A's goal row is 0,
 * so an A-to-move state is encoded with the identity row map; player B's goal
 * row is rows-1, so a B-to-move state is encoded with the row mirror
 * `R = rows - 1 - r`. Columns are never mirrored. One set of weights therefore
 * serves both sides, and two mover-relative-identical positions encode to
 * byte-identical tensors regardless of whose turn it is.
 *
 * There is deliberately NO side-to-move plane: the perspective transform above
 * makes it redundant, and the Stage 4 architecture (6 blocks x 64 channels,
 * Gumbel n~16, ~20 evaluations inside a 900 ms turn budget) pays first-layer
 * FLOPs for every input plane, so the layout is kept minimal.
 */

import {
  edgeBlocked, legalActionMask, policySize, validateConfig, validateState
} from './normal-duel-engine.mjs';

export class NormalDuelEncodingError extends Error {
  constructor(reason) { super(reason); this.name = 'NormalDuelEncodingError'; this.reason = reason; }
}

function fail(reason) { throw new NormalDuelEncodingError(reason); }

/**
 * Plane layout — the single source of truth shared by training code, the
 * inference wrapper and debugging tools. Index order is tensor plane order.
 */
export const NN_PLANE_LAYOUT = Object.freeze([
  Object.freeze({ name: 'mover_pawn', description: 'one-hot mover pawn, canonical orientation' }),
  Object.freeze({ name: 'opponent_pawn', description: 'one-hot opponent pawn, canonical orientation' }),
  Object.freeze({ name: 'wall_horizontal', description: '1 where the edge between (R,C) and (R+1,C) is wall-blocked' }),
  Object.freeze({ name: 'wall_vertical', description: '1 where the edge between (R,C) and (R,C+1) is wall-blocked' }),
  Object.freeze({ name: 'mover_stock', description: 'constant plane: mover walls remaining / initialStock[mover]' }),
  Object.freeze({ name: 'opponent_stock', description: 'constant plane: opponent walls remaining / initialStock[opponent]' }),
  Object.freeze({ name: 'goal_proximity', description: 'static row gradient (rows-1-R)/(rows-1): 1 on the mover goal row' })
]);

export const NN_INPUT_PLANES = Object.freeze(NN_PLANE_LAYOUT.length);

const PLANE_INDEX = Object.freeze(Object.fromEntries(
  NN_PLANE_LAYOUT.map((plane, index) => [plane.name, index])
));

export function planeIndex(name) {
  if (!Object.hasOwn(PLANE_INDEX, name)) fail('unknown_plane');
  return PLANE_INDEX[name];
}

function canonical9x9(config) {
  const checked = validateConfig(config);
  // Stage 4's architecture was measured on the canonical 9x9 board only.
  if (checked.rows !== 9 || checked.columns !== 9) fail('unsupported_board');
  return checked;
}

function other(player) { return player === 'A' ? 'B' : 'A'; }

/** Canonical row for an engine row: identity when the mover's goal is row 0. */
function canonicalRow(config, mover, r) {
  return config.goalRows[mover] === 0 ? r : config.rows - 1 - r;
}

/**
 * Wall geometry is read back out of the engine: for every orthogonal edge we
 * ask `edgeBlocked` whether a wall segment sits on it. No wall text is parsed
 * here, so the H/V anchor convention stays owned by the engine and the mirror
 * below is a pure coordinate transform on edges rather than on wall strings.
 */
function writeWallPlanes(config, mover, walls, out, offsetH, offsetV) {
  const { rows, columns } = config;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      if (r + 1 < rows && edgeBlocked(config, { r, c }, { r: r + 1, c }, walls)) {
        // Engine edge (r,c)-(r+1,c) maps to the canonical edge between the two
        // canonical rows; the plane cell is the upper of the two.
        const canonicalUpper = Math.min(canonicalRow(config, mover, r), canonicalRow(config, mover, r + 1));
        out[offsetH + canonicalUpper * columns + c] = 1;
      }
      if (c + 1 < columns && edgeBlocked(config, { r, c }, { r, c: c + 1 }, walls)) {
        out[offsetV + canonicalRow(config, mover, r) * columns + c] = 1;
      }
    }
  }
}

/**
 * Encode a validated game state as `NN_INPUT_PLANES * rows * columns` floats,
 * planes in `NN_PLANE_LAYOUT` order, each plane row-major, all values in [0, 1].
 * Neither `config` nor `state` is mutated.
 */
export function encodeState(config, state) {
  const checked = canonical9x9(config);
  const validated = validateState(checked, state); // never trust the caller's state
  const { rows, columns } = checked;
  const cells = rows * columns;
  const out = new Float32Array(NN_INPUT_PLANES * cells);

  const mover = validated.position.turn;
  const opponent = other(mover);
  const cell = (coord) => canonicalRow(checked, mover, coord.r) * columns + coord.c;

  out[PLANE_INDEX.mover_pawn * cells + cell(validated.position.pawns[mover])] = 1;
  out[PLANE_INDEX.opponent_pawn * cells + cell(validated.position.pawns[opponent])] = 1;

  writeWallPlanes(checked, mover, validated.position.walls, out,
    PLANE_INDEX.wall_horizontal * cells, PLANE_INDEX.wall_vertical * cells);

  const stockScale = (player) => (checked.initialStock[player] > 0
    ? validated.position.stock[player] / checked.initialStock[player]
    : 0);
  out.fill(stockScale(mover), PLANE_INDEX.mover_stock * cells, (PLANE_INDEX.mover_stock + 1) * cells);
  out.fill(stockScale(opponent), PLANE_INDEX.opponent_stock * cells, (PLANE_INDEX.opponent_stock + 1) * cells);

  // Static orientation cue: 1 on the mover's goal row (canonical row 0),
  // falling linearly to 0 on the mover's own back rank.
  const proximityBase = PLANE_INDEX.goal_proximity * cells;
  for (let r = 0; r < rows; r += 1) {
    const value = (rows - 1 - r) / (rows - 1);
    out.fill(value, proximityBase + r * columns, proximityBase + (r + 1) * columns);
  }
  return out;
}

/** Legal-action mask as floats, derived from the engine's `legalActionMask`. */
export function legalMaskFloat(config, state) {
  const checked = canonical9x9(config);
  const mask = legalActionMask(checked, state);
  const out = new Float32Array(policySize(checked));
  for (let code = 0; code < mask.length; code += 1) out[code] = mask[code] ? 1 : 0;
  return out;
}

/**
 * Normalise a `Map`/plain object of `actionCode -> visit count` into a policy
 * target that sums to 1 over the supplied codes and is zero elsewhere. Throws
 * on out-of-range or malformed codes, negative/non-finite counts, and a zero
 * total. Codes are not checked against a state here — legality filtering is the
 * caller's job via `legalMaskFloat`; only the policy index range is enforced.
 */
export function encodePolicyTarget(config, visitCounts) {
  const checked = canonical9x9(config);
  const size = policySize(checked);
  const out = new Float32Array(size);
  const entries = visitCounts instanceof Map
    ? [...visitCounts.entries()]
    : (visitCounts !== null && typeof visitCounts === 'object' && !Array.isArray(visitCounts)
      ? Object.entries(visitCounts)
      : fail('invalid_visit_counts'));

  let total = 0;
  const seen = new Set();
  const parsed = [];
  for (const [rawCode, rawCount] of entries) {
    const code = typeof rawCode === 'string' ? Number(rawCode) : rawCode;
    if (!Number.isSafeInteger(code) || code < 0 || code >= size) fail('invalid_action_code');
    if (typeof rawCount !== 'number' || !Number.isFinite(rawCount) || rawCount < 0) fail('invalid_visit_count');
    if (seen.has(code)) fail('duplicate_action_code');
    seen.add(code);
    parsed.push([code, rawCount]);
    total += rawCount;
  }
  if (!(total > 0)) fail('empty_visit_counts');
  // Sort by code so the accumulation order — and therefore the float rounding —
  // does not depend on the caller's iteration order.
  parsed.sort((left, right) => left[0] - right[0]);
  for (const [code, count] of parsed) out[code] = count / total;
  return out;
}

/**
 * Legality-checked variant used by the self-play writer: every supplied code
 * must be legal in `state`.
 */
export function encodeLegalPolicyTarget(config, state, visitCounts) {
  const checked = canonical9x9(config);
  const mask = legalMaskFloat(checked, state);
  const target = encodePolicyTarget(checked, visitCounts);
  for (let code = 0; code < target.length; code += 1) {
    if (target[code] !== 0 && mask[code] !== 1) fail('illegal_action_code');
  }
  return target;
}
