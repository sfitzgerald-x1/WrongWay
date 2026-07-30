/**
 * Stateless consumer bridge for the normal-duel-v1 reference rules.
 *
 * This module deliberately accepts only a narrow, fully explicit product scope.
 * It does not construct a GameState or apply actions: its queries are for the
 * existing client call sites that currently operate on a board snapshot.
 */
import {
  isLegalWall,
  legalPawnDestinations,
  NormalDuelError,
  validateConfig
} from './normal-duel-engine.mjs';

const NORMAL_DUEL_RULESET = 'normal-duel-v1';
const LEGACY_REASONS = Object.freeze({
  invalidScope: 'legacy_invalid_scope',
  gameMode: 'legacy_game_mode',
  map: 'legacy_map',
  duelSize: 'legacy_duel_size',
  clock: 'legacy_clock',
  chaos: 'legacy_chaos_mode',
  hammer: 'legacy_hammer_mode',
  drop: 'legacy_drop_mode',
  ranked: 'legacy_ranked',
  stock: 'legacy_stock_config',
  twoVTwo: 'legacy_two_v_two'
});

const SCOPE_KEYS = new Set([
  'gameMode', 'map', 'duelSize', 'gameTimeMode', 'chaosMode', 'hammerMode',
  'dropMode', 'isRanked', 'initialStock', 'is2v2'
]);
const SNAPSHOT_KEYS = new Set(['pA', 'pB', 'walls', 'turn']);
const COORD_KEYS = new Set(['r', 'c']);

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasExactOwnKeys(value, keys) {
  return isPlainObject(value)
    && Reflect.ownKeys(value).length === keys.size
    && [...keys].every((key) => Object.hasOwn(value, key));
}

function cloneCoord(coord) { return { r: coord.r, c: coord.c }; }
function equalCoord(left, right) { return left.r === right.r && left.c === right.c; }

function queryConfig(duelSize) {
  const dimensions = duelSize === 'standard' ? 9 : duelSize === 'blitz' ? 7 : null;
  if (dimensions === null) throw new NormalDuelError('invalid_consumer_query');
  const center = Math.floor(dimensions / 2);
  return validateConfig({
    ruleset: NORMAL_DUEL_RULESET,
    rows: dimensions,
    columns: dimensions,
    start: { A: { r: dimensions - 1, c: center }, B: { r: 0, c: center } },
    goalRows: { A: 0, B: dimensions - 1 },
    // Consumer queries have no wall ownership or remaining-stock information.
    // Give either side the normal initial allotment; this is a valid Position
    // query and never gets promoted to a stateful transition.
    initialStock: { A: 10, B: 10 },
    jumpRule: 'permissive-adjacent-exit-v1',
    repetitionThreshold: 3,
    plyCap: 200,
    firstPlayer: 'A'
  });
}

function positionFromSnapshot(duelSize, snapshot) {
  if (!hasExactOwnKeys(snapshot, SNAPSHOT_KEYS)
    || !hasExactOwnKeys(snapshot.pA, COORD_KEYS)
    || !hasExactOwnKeys(snapshot.pB, COORD_KEYS)
    || !(Array.isArray(snapshot.walls) || snapshot.walls instanceof Set)
    || (snapshot.turn !== 'A' && snapshot.turn !== 'B')) {
    throw new NormalDuelError('invalid_consumer_query');
  }
  const config = queryConfig(duelSize);
  return {
    config,
    position: {
      pawns: { A: cloneCoord(snapshot.pA), B: cloneCoord(snapshot.pB) },
      walls: [...snapshot.walls],
      stock: { A: 10, B: 10 },
      turn: snapshot.turn
    }
  };
}

/** Return why a product scope remains on the legacy implementation. */
export function classifyNormalDuelConsumer(scope) {
  if (!hasExactOwnKeys(scope, SCOPE_KEYS)) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.invalidScope });
  if (scope.gameMode !== 'pvp' && scope.gameMode !== 'pvc') return Object.freeze({ eligible: false, reason: LEGACY_REASONS.gameMode });
  if (scope.map !== 'duel') return Object.freeze({ eligible: false, reason: LEGACY_REASONS.map });
  if (scope.duelSize !== 'standard' && scope.duelSize !== 'blitz') return Object.freeze({ eligible: false, reason: LEGACY_REASONS.duelSize });
  if (scope.gameTimeMode !== 'none') return Object.freeze({ eligible: false, reason: LEGACY_REASONS.clock });
  if (scope.chaosMode !== false) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.chaos });
  if (scope.hammerMode !== false) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.hammer });
  if (scope.dropMode !== false) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.drop });
  if (scope.isRanked !== false) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.ranked });
  if (scope.initialStock !== 10) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.stock });
  if (scope.is2v2 !== false) return Object.freeze({ eligible: false, reason: LEGACY_REASONS.twoVTwo });
  return Object.freeze({ eligible: true, reason: null });
}

/**
 * Reference equivalent of the legacy getMovesFrom consumer.
 * The moving pawn is chosen by snapshot.turn, so B is never reinterpreted as A.
 */
export function normalDuelLegalMoves(duelSize, snapshot) {
  const { config, position } = positionFromSnapshot(duelSize, snapshot);
  return legalPawnDestinations(config, position).map(cloneCoord);
}

/**
 * Reference equivalent of legacy tryWall. Invalid snapshots or walls fail closed
 * as null, matching the existing call site's Set|null protocol.
 */
export function normalDuelTryWall(duelSize, snapshot, wall) {
  try {
    const { config, position } = positionFromSnapshot(duelSize, snapshot);
    return isLegalWall(config, position, wall) ? new Set([...position.walls, wall]) : null;
  } catch (error) {
    if (error instanceof NormalDuelError) return null;
    throw error;
  }
}

/**
 * Preserve the legacy path preference while asking the reference engine which
 * destinations are legal. In particular, the old hand-written jump legality is
 * not repeated here.
 */
export function normalDuelMoveTowardGoal(duelSize, snapshot, path) {
  if (!Array.isArray(path) || path.length < 2 || !hasExactOwnKeys(path[1], COORD_KEYS)) return null;
  const { config, position } = positionFromSnapshot(duelSize, snapshot);
  const mover = position.pawns[position.turn];
  const opponent = position.pawns[position.turn === 'A' ? 'B' : 'A'];
  const legal = legalPawnDestinations(config, position);
  const contains = (candidate) => legal.some((destination) => equalCoord(destination, candidate));
  const nextOnPath = path[1];

  if (!equalCoord(nextOnPath, opponent)) return contains(nextOnPath) ? cloneCoord(nextOnPath) : null;

  const dr = opponent.r - mover.r;
  const dc = opponent.c - mover.c;
  for (const [exitR, exitC] of [[dr, dc], [-dc, dr], [dc, -dr], [-dr, -dc]]) {
    const candidate = { r: opponent.r + exitR, c: opponent.c + exitC };
    if (contains(candidate)) return candidate;
  }
  return null;
}

function assertConsumerImplementation(implementation, name) {
  if (!isPlainObject(implementation)
    || typeof implementation.legalMoves !== 'function'
    || typeof implementation.tryWall !== 'function'
    || typeof implementation.moveTowardGoal !== 'function') {
    throw new TypeError(`${name} must provide legalMoves, tryWall, and moveTowardGoal functions`);
  }
}

const REFERENCE = Object.freeze({
  legalMoves: normalDuelLegalMoves,
  tryWall: normalDuelTryWall,
  moveTowardGoal: normalDuelMoveTowardGoal
});

/**
 * Route a complete consumer operation as a unit. The injected legacy seam keeps
 * unsupported modes behaviorally isolated while allowing focused migration tests.
 */
export function createConsumerRouter({ reference = REFERENCE, legacy } = {}) {
  assertConsumerImplementation(reference, 'reference');
  assertConsumerImplementation(legacy, 'legacy');
  const implementationFor = (scope) => classifyNormalDuelConsumer(scope).eligible ? reference : legacy;
  return Object.freeze({
    classify: classifyNormalDuelConsumer,
    legalMoves(scope, duelSize, snapshot) {
      return implementationFor(scope).legalMoves(duelSize, snapshot);
    },
    tryWall(scope, duelSize, snapshot, wall) {
      return implementationFor(scope).tryWall(duelSize, snapshot, wall);
    },
    moveTowardGoal(scope, duelSize, snapshot, path) {
      return implementationFor(scope).moveTowardGoal(duelSize, snapshot, path);
    }
  });
}

export const normalDuelConsumer = Object.freeze({
  classifyNormalDuelConsumer,
  createConsumerRouter,
  normalDuelLegalMoves,
  normalDuelTryWall,
  normalDuelMoveTowardGoal
});
