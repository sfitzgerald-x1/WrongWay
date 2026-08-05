/**
 * Stage 4 neural-network input encoding for the canonical 9x9 normal duel.
 *
 * Pure, deterministic state -> tensor encoding. Rules, geometry and legality
 * always come from the authoritative engine (`js/normal-duel-engine.mjs`);
 * nothing here reimplements a rule.
 *
 * Absolute orientation
 * --------------------
 * Every plane is indexed by the engine's own row `r`. There is no mirroring of
 * any kind. The reason is coherence with the action space: the 209-way policy
 * head speaks the engine's action codes, which are absolute (a pawn move is
 * `to.r * columns + to.c`, a wall is an offset off the absolute anchor index).
 * Nothing maps those codes into a mover-canonical frame, so an input frame that
 * mirrored rows would present features in one frame and labels in another —
 * "advance one step toward my goal" would be code 22 from an A-to-move state
 * and code 58 from the row-mirrored B-to-move state while the two encodings
 * were byte-identical. What matters is that the input frame and the policy
 * frame agree; the policy frame is absolute, so the input frame is absolute.
 *
 * Planes 0/1 (pawns) and 4/5 (stock) remain semantically mover-relative: plane
 * 0 always holds the side-to-move's pawn and plane 1 the opponent's. That is a
 * channel SELECTION, not a coordinate transform — it permutes which plane a
 * value is written into and never moves a value to a different (r, c). Cell
 * (r, c) of every plane still denotes engine square (r, c), so the mapping
 * between board squares and action codes is untouched and the action space
 * stays synchronised with the input.
 *
 * Because the frame is absolute, the network cannot infer whose turn it is from
 * geometry alone, so plane 7 `side_to_move` carries it explicitly (1.0 for A,
 * 0.0 for B), as specified in `docs/ai-engine-plan.md`. The eighth plane costs
 * `2 * 81 * 8 * 64 * 9 = 746,496` stem FLOPs versus `653,184` at seven planes,
 * i.e. +93,312 FLOPs — about 0.13% of a ~71.7 MFLOP forward pass at 6 blocks x
 * 64 channels. Coherence is worth an eighth of a percent per evaluation.
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
 * All planes are indexed by absolute engine (r, c).
 */
export const NN_PLANE_LAYOUT = Object.freeze([
  Object.freeze({ name: 'mover_pawn', description: 'one-hot side-to-move pawn at absolute (r,c)' }),
  Object.freeze({ name: 'opponent_pawn', description: 'one-hot opponent pawn at absolute (r,c)' }),
  Object.freeze({ name: 'wall_horizontal', description: '1 where the edge between (r,c) and (r+1,c) is wall-blocked' }),
  Object.freeze({ name: 'wall_vertical', description: '1 where the edge between (r,c) and (r,c+1) is wall-blocked' }),
  Object.freeze({ name: 'mover_stock', description: 'constant plane: mover walls remaining / initialStock[mover]' }),
  Object.freeze({ name: 'opponent_stock', description: 'constant plane: opponent walls remaining / initialStock[opponent]' }),
  Object.freeze({ name: 'goal_proximity', description: 'row gradient toward config.goalRows[mover]: 1 on the mover goal row, 0 on the far row' }),
  Object.freeze({ name: 'side_to_move', description: 'constant plane: 1 when A is to move, 0 when B is to move' }),
  Object.freeze({ name: 'mover_path_distance', description: 'wall-aware shortest-path distance from (r,c) to config.goalRows[mover], divided by cells-1; 1 where unreachable' }),
  Object.freeze({ name: 'opponent_path_distance', description: 'wall-aware shortest-path distance from (r,c) to config.goalRows[opponent], divided by cells-1; 1 where unreachable' })
]);

/**
 * Value written into the path-distance planes for a cell with no wall-legal
 * route to the goal row. See `NN_UNREACHABLE_DISTANCE` in the Rust port: the
 * planes have one channel and live in [0, 1], so "unreachable" has to be
 * spelled as a distance, and the top of the range is the choice that leaves
 * the ordering monotone.
 */
export const NN_UNREACHABLE_DISTANCE = 1;

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

/**
 * Wall geometry is read back out of the engine: for every orthogonal edge we
 * ask `edgeBlocked` whether a wall segment sits on it. No wall text is parsed
 * here, so the H/V anchor convention stays owned by the engine. Plane cells use
 * absolute engine coordinates, with the horizontal plane anchored on the upper
 * of the two rows an edge separates.
 */
function writeWallPlanes(config, walls, out, offsetH, offsetV) {
  const { rows, columns } = config;
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      if (r + 1 < rows && edgeBlocked(config, { r, c }, { r: r + 1, c }, walls)) {
        out[offsetH + r * columns + c] = 1;
      }
      if (c + 1 < columns && edgeBlocked(config, { r, c }, { r, c: c + 1 }, walls)) {
        out[offsetV + r * columns + c] = 1;
      }
    }
  }
}

/**
 * Wall-aware distance from every cell to `goalRow`, as a flat row-major array
 * of step counts with `-1` where walls seal a cell off from the goal row.
 *
 * One multi-source BFS seeded with the whole goal row, not 81 single-source
 * ones: the move graph is undirected, since a wall segment cuts an edge for
 * both cells it joins, so the distance from a cell to the nearest goal square
 * equals the depth at which a search started from the goal row reaches it.
 * Blockage comes from the engine's `edgeBlocked` one edge at a time, the same
 * way `writeWallPlanes` reads wall geometry, so no rule is reimplemented here.
 */
function goalDistanceField(config, walls, goalRow) {
  const { rows, columns } = config;
  const distances = new Int32Array(rows * columns).fill(-1);
  let frontier = [];
  for (let c = 0; c < columns; c += 1) {
    distances[goalRow * columns + c] = 0;
    frontier.push({ r: goalRow, c });
  }
  for (let distance = 1; frontier.length > 0; distance += 1) {
    const next = [];
    for (const from of frontier) {
      const neighbours = [
        { r: from.r - 1, c: from.c }, { r: from.r + 1, c: from.c },
        { r: from.r, c: from.c - 1 }, { r: from.r, c: from.c + 1 }
      ];
      for (const to of neighbours) {
        if (to.r < 0 || to.r >= rows || to.c < 0 || to.c >= columns) continue;
        const index = to.r * columns + to.c;
        if (distances[index] !== -1) continue;
        if (edgeBlocked(config, from, to, walls)) continue;
        distances[index] = distance;
        next.push(to);
      }
    }
    frontier = next;
  }
  return distances;
}

/**
 * Encode a validated game state as `NN_INPUT_PLANES * rows * columns` floats,
 * planes in `NN_PLANE_LAYOUT` order, each plane row-major over absolute engine
 * coordinates, all values in [0, 1]. Neither `config` nor `state` is mutated.
 */
export function encodeState(config, state) {
  const checked = canonical9x9(config);
  const validated = validateState(checked, state); // never trust the caller's state
  const { rows, columns } = checked;
  const cells = rows * columns;
  const out = new Float32Array(NN_INPUT_PLANES * cells);

  const mover = validated.position.turn;
  const opponent = other(mover);
  const cell = (coord) => coord.r * columns + coord.c;

  out[PLANE_INDEX.mover_pawn * cells + cell(validated.position.pawns[mover])] = 1;
  out[PLANE_INDEX.opponent_pawn * cells + cell(validated.position.pawns[opponent])] = 1;

  writeWallPlanes(checked, validated.position.walls, out,
    PLANE_INDEX.wall_horizontal * cells, PLANE_INDEX.wall_vertical * cells);

  const stockScale = (player) => (checked.initialStock[player] > 0
    ? validated.position.stock[player] / checked.initialStock[player]
    : 0);
  out.fill(stockScale(mover), PLANE_INDEX.mover_stock * cells, (PLANE_INDEX.mover_stock + 1) * cells);
  out.fill(stockScale(opponent), PLANE_INDEX.opponent_stock * cells, (PLANE_INDEX.opponent_stock + 1) * cells);

  // Mover-relative gradient in absolute coordinates: 1 on the mover's own goal
  // row, falling linearly to 0 on the opposite edge. Derived from
  // `config.goalRows[mover]`, so the same board yields a different plane for
  // A-to-move and B-to-move. No coordinate is mirrored to achieve this.
  const goalRow = checked.goalRows[mover];
  const proximityBase = PLANE_INDEX.goal_proximity * cells;
  for (let r = 0; r < rows; r += 1) {
    const value = (rows - 1 - Math.abs(r - goalRow)) / (rows - 1);
    out.fill(value, proximityBase + r * columns, proximityBase + (r + 1) * columns);
  }

  // Whose turn it is is not recoverable from an absolute frame, so state it.
  out.fill(mover === 'A' ? 1 : 0, PLANE_INDEX.side_to_move * cells, (PLANE_INDEX.side_to_move + 1) * cells);

  // Wall-aware shortest-path distance to each side's goal row, normalised by
  // the longest path the board can hold (`cells - 1` steps) into [0, 1].
  //
  // The classical eval is dominated by exactly this quantity — weight 100
  // against 12/6/8 for stock, tempo and robustness — so the network is handed
  // it rather than left to rediscover BFS through a stack of 3x3 kernels.
  // Mover-relative by plane SELECTION only, like planes 0/1 and 4/5: no
  // coordinate is transformed, so the frame stays absolute and synchronised
  // with the action codes.
  const longestPath = cells - 1;
  for (const [player, plane] of [[mover, 'mover_path_distance'], [opponent, 'opponent_path_distance']]) {
    const distances = goalDistanceField(checked, validated.position.walls, checked.goalRows[player]);
    const base = PLANE_INDEX[plane] * cells;
    for (let index = 0; index < cells; index += 1) {
      out[base + index] = distances[index] < 0
        ? NN_UNREACHABLE_DISTANCE
        : distances[index] / longestPath;
    }
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

/** Canonical decimal integer strings only — no whitespace, sign, hex or ''. */
const DECIMAL_CODE = /^(?:0|[1-9][0-9]*)$/;

/**
 * Object keys arrive as strings. Validate rather than coerce: `Number('')` is
 * 0, `Number(' 3 ')` is 3 and `Number('0x10')` is 16, all of which would
 * silently write a visit count to the wrong policy index.
 */
function parseActionCode(rawCode) {
  if (typeof rawCode === 'number') return rawCode;
  if (typeof rawCode === 'string' && DECIMAL_CODE.test(rawCode)) return Number(rawCode);
  return fail('invalid_action_code');
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
    const code = parseActionCode(rawCode);
    if (!Number.isSafeInteger(code) || code < 0 || code >= size) fail('invalid_action_code');
    if (typeof rawCount !== 'number' || !Number.isFinite(rawCount) || rawCount < 0) fail('invalid_visit_count');
    // A Map may hold both `63` and `'63'`; they name the same policy index.
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
 * must be legal in `state`. Both the mask and the target live in the engine's
 * absolute action frame, which is the same frame `encodeState` is indexed in.
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
