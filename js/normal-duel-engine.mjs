/** Pure reference rules for the normal-duel-v1 contract. */

const PLAYERS = new Set(['A', 'B']);
const DIRECTIONS = [[-1, 0], [0, -1], [0, 1], [1, 0]];
const CONFIG_KEYS = new Set([
  'ruleset', 'rows', 'columns', 'start', 'goalRows', 'initialStock',
  'jumpRule', 'repetitionThreshold', 'plyCap', 'firstPlayer'
]);
const PLAYER_KEYS = new Set(['A', 'B']);
const COORD_KEYS = new Set(['r', 'c']);
const POSITION_KEYS = new Set(['pawns', 'walls', 'stock', 'turn']);
const STATE_KEYS = new Set(['position', 'positionKey', 'ply', 'historyStartPly', 'repetitionCounts', 'outcome']);
const REPETITION_COUNT_KEYS = new Set(['positionKey', 'count']);
const PAWN_ACTION_KEYS = new Set(['kind', 'to']);
const WALL_ACTION_KEYS = new Set(['kind', 'wall']);
const ADJUDICATION_KEYS = new Set(['goalWinner', 'resultingPositionCount', 'resultingPly']);

export class NormalDuelError extends Error {
  constructor(code, message = code) {
    super(message);
    this.name = 'NormalDuelError';
    this.code = code;
  }
}

function fail(code, message) { throw new NormalDuelError(code, message); }
function other(player) { return player === 'A' ? 'B' : 'A'; }
function isObject(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function integer(value) { return Number.isSafeInteger(value); }
function ownKeysOnly(value, allowed, code = 'invalid_config') {
  if (!isObject(value)) fail(code);
  for (const key of Object.keys(value)) if (!allowed.has(key)) fail(code);
}
function exactKeys(value, expected, code) {
  ownKeysOnly(value, expected, code);
  if (Reflect.ownKeys(value).length !== expected.size) fail(code);
  for (const key of expected) if (!Object.hasOwn(value, key)) fail(code);
}
function cloneCoord({ r, c }) { return { r, c }; }
function sameCoord(left, right) { return left.r === right.r && left.c === right.c; }
function clonePosition(position) {
  return {
    pawns: { A: cloneCoord(position.pawns.A), B: cloneCoord(position.pawns.B) },
    walls: [...position.walls],
    stock: { A: position.stock.A, B: position.stock.B },
    turn: position.turn
  };
}
function compareNumbers(left, right) { return left - right; }

/** Compare canonical keys in their required UTF-8 byte ordering. */
export function comparePositionKeys(left, right) {
  if (typeof left !== 'string' || typeof right !== 'string') fail('invalid_position_key');
  const a = new TextEncoder().encode(left);
  const b = new TextEncoder().encode(right);
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

export function validateConfig(input) {
  if (isObject(input) && Object.hasOwn(input, 'features')) fail('unsupported_feature');
  exactKeys(input, CONFIG_KEYS, 'invalid_config');
  const { rows, columns } = input;
  if (!((rows === 9 && columns === 9) || (rows === 7 && columns === 7))) fail('invalid_config');
  if (input.ruleset !== 'normal-duel-v1'
    || input.jumpRule !== 'permissive-adjacent-exit-v1'
    || input.repetitionThreshold !== 3
    || !PLAYERS.has(input.firstPlayer)
    || !integer(input.plyCap) || input.plyCap < 1) fail('invalid_config');
  const expected = {
    A: { r: rows - 1, c: Math.floor(columns / 2) },
    B: { r: 0, c: Math.floor(columns / 2) }
  };
  exactKeys(input.start, PLAYER_KEYS, 'invalid_config');
  exactKeys(input.goalRows, PLAYER_KEYS, 'invalid_config');
  exactKeys(input.initialStock, PLAYER_KEYS, 'invalid_config');
  for (const player of PLAYERS) {
    const start = input.start[player];
    exactKeys(start, COORD_KEYS, 'invalid_config');
    if (start.r !== expected[player].r || start.c !== expected[player].c
      || input.goalRows?.[player] !== (player === 'A' ? 0 : rows - 1)
      || !integer(input.initialStock?.[player]) || input.initialStock[player] < 0) fail('invalid_config');
  }
  return Object.freeze({
    ruleset: input.ruleset, rows, columns,
    start: Object.freeze({ A: Object.freeze(cloneCoord(expected.A)), B: Object.freeze(cloneCoord(expected.B)) }),
    goalRows: Object.freeze({ A: 0, B: rows - 1 }),
    initialStock: Object.freeze({ A: input.initialStock.A, B: input.initialStock.B }),
    jumpRule: input.jumpRule, repetitionThreshold: input.repetitionThreshold,
    plyCap: input.plyCap, firstPlayer: input.firstPlayer
  });
}

function configOf(config) { return validateConfig(config); }
function inBounds(config, coord) {
  return isObject(coord) && Object.keys(coord).length === COORD_KEYS.size
    && Object.hasOwn(coord, 'r') && Object.hasOwn(coord, 'c')
    && integer(coord.r) && integer(coord.c)
    && coord.r >= 0 && coord.r < config.rows && coord.c >= 0 && coord.c < config.columns;
}
function anchorIndex(config, wall) { return wall.r * (config.columns - 1) + wall.c; }
function parseWall(config, text, code = 'invalid_wall') {
  const match = typeof text === 'string' && /^([HV])-(0|[1-9]\d*)-(0|[1-9]\d*)$/.exec(text);
  if (!match) fail(code);
  const wall = { orientation: match[1], r: Number(match[2]), c: Number(match[3]) };
  if (wall.r >= config.rows - 1 || wall.c >= config.columns - 1) fail(code);
  return wall;
}
function formatWall(wall) { return `${wall.orientation}-${wall.r}-${wall.c}`; }
function wallCode(config, wall) {
  const anchors = (config.rows - 1) * (config.columns - 1);
  return config.rows * config.columns + (wall.orientation === 'V' ? anchors : 0) + anchorIndex(config, wall);
}
function actionCode(config, action) {
  return action.kind === 'pawn'
    ? action.to.r * config.columns + action.to.c
    : wallCode(config, parseWall(config, action.wall, 'invalid_action'));
}
function sortedWalls(config, walls) {
  return walls.map((text) => ({ text, wall: parseWall(config, text) }))
    .sort((left, right) => wallCode(config, left.wall) - wallCode(config, right.wall));
}

function checkedWallSet(config, walls, code = 'invalid_wall') {
  if (!(Array.isArray(walls) || walls instanceof Set)) fail(code);
  const values = [...walls];
  const parsed = sortedWalls(config, values);
  if (new Set(parsed.map(({ text }) => text)).size !== parsed.length) fail(code);
  return new Set(parsed.map(({ text }) => text));
}

function edgeBlockedSet(from, to, walls) {
  if (from.r !== to.r) {
    const row = Math.min(from.r, to.r);
    return walls.has(`H-${row}-${from.c}`) || walls.has(`H-${row}-${from.c - 1}`);
  }
  const column = Math.min(from.c, to.c);
  return walls.has(`V-${from.r}-${column}`) || walls.has(`V-${from.r - 1}-${column}`);
}

function hasPathSet(config, from, goalRow, walls) {
  const queue = [from]; const seen = new Set([`${from.r},${from.c}`]);
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    if (current.r === goalRow) return true;
    for (const [dr, dc] of DIRECTIONS) {
      const next = { r: current.r + dr, c: current.c + dc };
      const key = `${next.r},${next.c}`;
      if (inBounds(config, next) && !seen.has(key) && !edgeBlockedSet(current, next, walls)) {
        seen.add(key); queue.push(next);
      }
    }
  }
  return false;
}

export function policySize(config) {
  const checked = configOf(config);
  return checked.rows * checked.columns + 2 * (checked.rows - 1) * (checked.columns - 1);
}

export function encodeAction(config, action) {
  const checked = configOf(config);
  if (!isObject(action)) fail('invalid_action');
  if (action.kind === 'pawn') {
    exactKeys(action, PAWN_ACTION_KEYS, 'invalid_action');
    if (!inBounds(checked, action.to)) fail('invalid_action');
    return actionCode(checked, action);
  }
  if (action.kind === 'wall') {
    exactKeys(action, WALL_ACTION_KEYS, 'invalid_action');
    return actionCode(checked, action);
  }
  fail('invalid_action');
}

export function decodeAction(config, code) {
  const checked = configOf(config);
  if (!integer(code) || code < 0 || code >= policySize(checked)) fail('invalid_action_code');
  const cells = checked.rows * checked.columns;
  if (code < cells) return { kind: 'pawn', to: { r: Math.floor(code / checked.columns), c: code % checked.columns } };
  const anchorCount = (checked.rows - 1) * (checked.columns - 1);
  const offset = code - cells;
  const orientation = offset < anchorCount ? 'H' : 'V';
  const anchor = offset % anchorCount;
  return { kind: 'wall', wall: `${orientation}-${Math.floor(anchor / (checked.columns - 1))}-${anchor % (checked.columns - 1)}` };
}

/** True when the orthogonal board edge is blocked by one of walls. */
export function edgeBlocked(config, from, to, walls) {
  const checked = configOf(config);
  if (!inBounds(checked, from) || !inBounds(checked, to)
    || Math.abs(from.r - to.r) + Math.abs(from.c - to.c) !== 1) fail('invalid_edge');
  return edgeBlockedSet(from, to, checkedWallSet(checked, walls));
}

export function hasPath(config, from, goalRow, walls) {
  const checked = configOf(config);
  if (!inBounds(checked, from) || !integer(goalRow) || goalRow < 0 || goalRow >= checked.rows) fail('invalid_position');
  return hasPathSet(checked, from, goalRow, checkedWallSet(checked, walls));
}

export function normalizePosition(config, input) {
  const checked = configOf(config);
  exactKeys(input, POSITION_KEYS, 'invalid_position');
  exactKeys(input.pawns, PLAYER_KEYS, 'invalid_position');
  exactKeys(input.stock, PLAYER_KEYS, 'invalid_position');
  if (!inBounds(checked, input.pawns.A) || !inBounds(checked, input.pawns.B)
    || (input.pawns.A.r === input.pawns.B.r && input.pawns.A.c === input.pawns.B.c)
    || !PLAYERS.has(input.turn) || !Array.isArray(input.walls)) fail('invalid_position');
  for (const player of PLAYERS) {
    if (!integer(input.stock?.[player]) || input.stock[player] < 0 || input.stock[player] > checked.initialStock[player]) fail('invalid_position');
  }
  const parsed = sortedWalls(checked, input.walls);
  if (new Set(parsed.map(({ text }) => text)).size !== parsed.length) fail('invalid_wall');
  const wallSet = new Set(parsed.map(({ text }) => text));
  for (const { wall, text } of parsed) {
    const same = wall.orientation === 'H' ? `H-${wall.r}-${wall.c - 1}` : `V-${wall.r - 1}-${wall.c}`;
    const next = wall.orientation === 'H' ? `H-${wall.r}-${wall.c + 1}` : `V-${wall.r + 1}-${wall.c}`;
    const cross = `${wall.orientation === 'H' ? 'V' : 'H'}-${wall.r}-${wall.c}`;
    if (wallSet.has(same) || wallSet.has(next) || wallSet.has(cross)) fail('invalid_wall_geometry', text);
  }
  if (!hasPathSet(checked, input.pawns.A, checked.goalRows.A, wallSet)
    || !hasPathSet(checked, input.pawns.B, checked.goalRows.B, wallSet)) fail('path_blocked');
  return {
    pawns: { A: cloneCoord(input.pawns.A), B: cloneCoord(input.pawns.B) },
    walls: parsed.map(({ text }) => text), stock: { A: input.stock.A, B: input.stock.B }, turn: input.turn
  };
}

export function positionKey(config, position) {
  const checked = configOf(config); const normalized = normalizePosition(checked, position);
  return positionKeyNormalized(checked, normalized);
}

function positionKeyNormalized(config, normalized) {
  const horizontal = []; const vertical = [];
  for (const text of normalized.walls) {
    const wall = parseWall(config, text);
    (wall.orientation === 'H' ? horizontal : vertical).push(anchorIndex(config, wall));
  }
  horizontal.sort(compareNumbers); vertical.sort(compareNumbers);
  const square = ({ r, c }) => r * config.columns + c;
  return JSON.stringify([config.ruleset, config.rows, config.columns, square(config.start.A), square(config.start.B), config.goalRows.A, config.goalRows.B, config.initialStock.A, config.initialStock.B, config.jumpRule, config.repetitionThreshold, config.plyCap, config.firstPlayer, square(normalized.pawns.A), square(normalized.pawns.B), horizontal, vertical, normalized.stock.A, normalized.stock.B, normalized.turn]);
}

export function legalPawnDestinations(config, position) {
  const checked = configOf(config); const normalized = normalizePosition(checked, position);
  return legalPawnDestinationsNormalized(checked, normalized);
}

function legalPawnDestinationsNormalized(config, normalized) {
  const mover = normalized.pawns[normalized.turn]; const opponent = normalized.pawns[other(normalized.turn)];
  const walls = new Set(normalized.walls); const result = new Map();
  for (const [dr, dc] of DIRECTIONS) {
    const adjacent = { r: mover.r + dr, c: mover.c + dc };
    if (!inBounds(config, adjacent) || edgeBlockedSet(mover, adjacent, walls)) continue;
    if (adjacent.r !== opponent.r || adjacent.c !== opponent.c) result.set(`${adjacent.r},${adjacent.c}`, adjacent);
    else for (const [exitR, exitC] of DIRECTIONS) {
      const destination = { r: opponent.r + exitR, c: opponent.c + exitC };
      if ((destination.r === mover.r && destination.c === mover.c) || !inBounds(config, destination)
        || edgeBlockedSet(opponent, destination, walls)) continue;
      result.set(`${destination.r},${destination.c}`, destination);
    }
  }
  return [...result.values()].sort((left, right) => (left.r * config.columns + left.c) - (right.r * config.columns + right.c)).map(cloneCoord);
}

export function isLegalWall(config, position, wallText) {
  const checked = configOf(config); const normalized = normalizePosition(checked, position);
  return isLegalWallNormalized(checked, normalized, new Set(normalized.walls), wallText);
}

function isLegalWallNormalized(config, normalized, existingWalls, wallText) {
  if (normalized.stock[normalized.turn] === 0) return false;
  let candidate;
  try { candidate = parseWall(config, wallText); } catch { return false; }
  const candidateText = formatWall(candidate);
  const walls = new Set(existingWalls);
  if (walls.has(candidateText)) return false;
  const adjacentBefore = candidate.orientation === 'H' ? `H-${candidate.r}-${candidate.c - 1}` : `V-${candidate.r - 1}-${candidate.c}`;
  const adjacentAfter = candidate.orientation === 'H' ? `H-${candidate.r}-${candidate.c + 1}` : `V-${candidate.r + 1}-${candidate.c}`;
  if (walls.has(adjacentBefore) || walls.has(adjacentAfter) || walls.has(`${candidate.orientation === 'H' ? 'V' : 'H'}-${candidate.r}-${candidate.c}`)) return false;
  walls.add(candidateText);
  return hasPathSet(config, normalized.pawns.A, config.goalRows.A, walls)
    && hasPathSet(config, normalized.pawns.B, config.goalRows.B, walls);
}

/**
 * Geometry-only action query for fixture and shadow-parity probes.
 * Stateful callers must use legalActions so terminal outcomes are honored.
 */
export function legalPositionActions(config, position) {
  const checked = configOf(config); const normalized = normalizePosition(checked, position);
  const actions = legalPawnDestinationsNormalized(checked, normalized).map((to) => ({ kind: 'pawn', to }));
  const walls = new Set(normalized.walls);
  if (normalized.stock[normalized.turn] > 0) {
    for (const orientation of ['H', 'V']) for (let r = 0; r < checked.rows - 1; r += 1) for (let c = 0; c < checked.columns - 1; c += 1) {
      const wall = `${orientation}-${r}-${c}`;
      if (isLegalWallNormalized(checked, normalized, walls, wall)) actions.push({ kind: 'wall', wall });
    }
  }
  return actions.sort((left, right) => actionCode(checked, left) - actionCode(checked, right));
}

export function legalPositionActionCodes(config, position) {
  const checked = configOf(config);
  return legalPositionActions(checked, position).map((action) => actionCode(checked, action));
}
export function legalPositionActionMask(config, position) {
  const mask = new Uint8Array(policySize(config));
  for (const code of legalPositionActionCodes(config, position)) mask[code] = 1;
  return mask;
}

export function adjudicate(config, facts) {
  const checked = configOf(config);
  exactKeys(facts, ADJUDICATION_KEYS, 'invalid_adjudication');
  const { goalWinner, resultingPositionCount, resultingPly } = facts;
  if (goalWinner !== null && !PLAYERS.has(goalWinner)) fail('invalid_adjudication');
  if (!integer(resultingPositionCount) || resultingPositionCount < 1 || !integer(resultingPly) || resultingPly < 1) fail('invalid_adjudication');
  if (goalWinner) return { kind: 'win', winner: goalWinner, reason: 'goal' };
  if (resultingPositionCount >= checked.repetitionThreshold) return { kind: 'draw', reason: 'threefold_repetition' };
  if (resultingPly >= checked.plyCap) return { kind: 'draw', reason: 'ply_cap' };
  return { kind: 'ongoing' };
}

function expectedTurn(config, ply) { return ply % 2 === 0 ? config.firstPlayer : other(config.firstPlayer); }
function completedActionCounts(config, ply) {
  return {
    [config.firstPlayer]: Math.ceil(ply / 2),
    [other(config.firstPlayer)]: Math.floor(ply / 2)
  };
}
function canonicalCounts(entries) {
  return [...entries.entries()].sort(([left], [right]) => comparePositionKeys(left, right)).map(([key, count]) => ({ positionKey: key, count }));
}

function positionFromKey(config, key) {
  if (typeof key !== 'string') fail('invalid_state');
  let fields;
  try { fields = JSON.parse(key); } catch { fail('invalid_state'); }
  const cells = config.rows * config.columns;
  const anchors = (config.rows - 1) * (config.columns - 1);
  if (!Array.isArray(fields) || fields.length !== 20
    || fields[0] !== config.ruleset || fields[1] !== config.rows || fields[2] !== config.columns
    || fields[3] !== config.start.A.r * config.columns + config.start.A.c
    || fields[4] !== config.start.B.r * config.columns + config.start.B.c
    || fields[5] !== config.goalRows.A || fields[6] !== config.goalRows.B
    || fields[7] !== config.initialStock.A || fields[8] !== config.initialStock.B
    || fields[9] !== config.jumpRule || fields[10] !== config.repetitionThreshold
    || fields[11] !== config.plyCap || fields[12] !== config.firstPlayer
    || !integer(fields[13]) || fields[13] < 0 || fields[13] >= cells
    || !integer(fields[14]) || fields[14] < 0 || fields[14] >= cells
    || !Array.isArray(fields[15]) || !Array.isArray(fields[16])
    || !integer(fields[17]) || !integer(fields[18]) || !PLAYERS.has(fields[19])) fail('invalid_state');
  const anchorList = (values) => {
    let previous = -1;
    for (const value of values) {
      if (!integer(value) || value < 0 || value >= anchors || value <= previous) fail('invalid_state');
      previous = value;
    }
    return values;
  };
  const wallText = (orientation, index) => `${orientation}-${Math.floor(index / (config.columns - 1))}-${index % (config.columns - 1)}`;
  const position = {
    pawns: {
      A: { r: Math.floor(fields[13] / config.columns), c: fields[13] % config.columns },
      B: { r: Math.floor(fields[14] / config.columns), c: fields[14] % config.columns }
    },
    walls: [...anchorList(fields[15]).map((index) => wallText('H', index)), ...anchorList(fields[16]).map((index) => wallText('V', index))],
    stock: { A: fields[17], B: fields[18] },
    turn: fields[19]
  };
  const normalized = normalizePosition(config, position);
  if (positionKeyNormalized(config, normalized) !== key) fail('invalid_state');
  return normalized;
}

function sameOutcome(actual, expected) {
  if (!isObject(actual) || actual.kind !== expected.kind) return false;
  if (expected.kind === 'ongoing') return Object.keys(actual).length === 1;
  if (expected.kind === 'win') {
    return Object.keys(actual).length === 3 && actual.winner === expected.winner && actual.reason === expected.reason;
  }
  return Object.keys(actual).length === 2 && actual.reason === expected.reason;
}

export function createInitialState(config) {
  const checked = configOf(config);
  const position = normalizePosition(checked, { pawns: checked.start, walls: [], stock: checked.initialStock, turn: checked.firstPlayer });
  const key = positionKeyNormalized(checked, position);
  return { position, positionKey: key, ply: 0, historyStartPly: 0, repetitionCounts: [{ positionKey: key, count: 1 }], outcome: { kind: 'ongoing' } };
}

export function validateState(config, input) {
  const checked = configOf(config);
  exactKeys(input, STATE_KEYS, 'invalid_state');
  if (!isObject(input.position) || !integer(input.ply) || input.ply < 0
    || !integer(input.historyStartPly) || input.historyStartPly < 0 || input.historyStartPly > input.ply
    || !Array.isArray(input.repetitionCounts) || input.repetitionCounts.length === 0 || !isObject(input.outcome)) fail('invalid_state');
  const position = normalizePosition(checked, input.position); const key = positionKeyNormalized(checked, position);
  const initialState = input.ply <= 1 || input.historyStartPly === 0 ? createInitialState(checked) : null;
  if (input.positionKey !== key || position.turn !== expectedTurn(checked, input.ply)) fail('invalid_state');
  const spentBy = {
    A: checked.initialStock.A - position.stock.A,
    B: checked.initialStock.B - position.stock.B
  };
  const spent = spentBy.A + spentBy.B;
  const completedActions = completedActionCounts(checked, input.ply);
  const completedAtHistoryStart = completedActionCounts(checked, input.historyStartPly);
  if (spent !== position.walls.length) fail('invalid_state');
  if (position.walls.length > input.ply || input.historyStartPly < position.walls.length
    || spentBy.A > completedAtHistoryStart.A || spentBy.B > completedAtHistoryStart.B) fail('invalid_state');
  if ((position.walls.length === 0) !== (input.historyStartPly === 0)) fail('invalid_state');
  if (input.historyStartPly > 0 && spentBy[expectedTurn(checked, input.historyStartPly - 1)] === 0) fail('invalid_state');
  if (input.ply === 1) {
    const mover = checked.firstPlayer;
    const opponent = other(mover);
    const firstPawnMove = position.walls.length === 0
      && sameCoord(position.pawns[opponent], initialState.position.pawns[opponent])
      && legalPawnDestinationsNormalized(checked, initialState.position)
        .some((destination) => sameCoord(destination, position.pawns[mover]));
    const firstWallMove = position.walls.length === 1
      && sameCoord(position.pawns.A, initialState.position.pawns.A)
      && sameCoord(position.pawns.B, initialState.position.pawns.B)
      && position.stock[mover] === checked.initialStock[mover] - 1
      && position.stock[opponent] === checked.initialStock[opponent]
      && isLegalWallNormalized(checked, initialState.position, new Set(), position.walls[0]);
    if (!firstPawnMove && !firstWallMove) fail('invalid_state');
  }
  for (const player of PLAYERS) {
    const distanceFromStart = Math.abs(checked.start[player].r - position.pawns[player].r)
      + Math.abs(checked.start[player].c - position.pawns[player].c);
    if (distanceFromStart > 2 * completedActions[player]) fail('invalid_state');
  }
  const counts = new Map();
  const historicalPositions = new Map();
  for (const entry of input.repetitionCounts) {
    exactKeys(entry, REPETITION_COUNT_KEYS, 'invalid_state');
    if (typeof entry.positionKey !== 'string' || !integer(entry.count) || entry.count < 1 || counts.has(entry.positionKey)) fail('invalid_state');
    const historical = positionFromKey(checked, entry.positionKey);
    if (historical.walls.join() !== position.walls.join()
      || historical.stock.A !== position.stock.A || historical.stock.B !== position.stock.B) fail('invalid_state');
    counts.set(entry.positionKey, entry.count);
    historicalPositions.set(entry.positionKey, historical);
  }
  const canonical = canonicalCounts(counts);
  const canonicalInput = input.repetitionCounts.every((entry, index) =>
    entry.positionKey === canonical[index]?.positionKey && entry.count === canonical[index]?.count);
  if (!canonicalInput || !counts.has(key)
    || [...counts.values()].reduce((sum, count) => sum + count, 0) !== input.ply - input.historyStartPly + 1) fail('invalid_state');
  if (input.historyStartPly === 0 && !counts.has(initialState.positionKey)) fail('invalid_state');
  const historyLength = input.ply - input.historyStartPly + 1;
  const maximumExactPositionOccurrences = 1 + Math.floor((historyLength - 1) / 4);
  if ([...counts.values()].some((count) => count > maximumExactPositionOccurrences)) fail('invalid_state');
  const historyFirstPlayer = expectedTurn(checked, input.historyStartPly);
  const expectedTurnCounts = {
    [historyFirstPlayer]: Math.ceil(historyLength / 2),
    [other(historyFirstPlayer)]: Math.floor(historyLength / 2)
  };
  const actualTurnCounts = { A: 0, B: 0 };
  for (const [countKey, count] of counts) actualTurnCounts[historicalPositions.get(countKey).turn] += count;
  if (actualTurnCounts.A !== expectedTurnCounts.A || actualTurnCounts.B !== expectedTurnCounts.B) fail('invalid_state');
  const movesInHistoryWindow = {
    A: completedActions.A - completedAtHistoryStart.A,
    B: completedActions.B - completedAtHistoryStart.B
  };
  for (const historical of historicalPositions.values()) {
    for (const player of PLAYERS) {
      const distance = Math.abs(historical.pawns[player].r - position.pawns[player].r)
        + Math.abs(historical.pawns[player].c - position.pawns[player].c);
      if (distance > 2 * movesInHistoryWindow[player]) fail('invalid_state');
    }
  }
  const outcome = input.outcome;
  const atGoal = {
    A: position.pawns.A.r === checked.goalRows.A,
    B: position.pawns.B.r === checked.goalRows.B
  };
  if ((atGoal.A && atGoal.B) || atGoal[position.turn]) fail('invalid_state');
  for (const [countKey, historical] of historicalPositions) {
    const historicalGoal = historical.pawns.A.r === checked.goalRows.A || historical.pawns.B.r === checked.goalRows.B;
    if (historicalGoal && (countKey !== key || counts.get(countKey) !== 1)) fail('invalid_state');
  }
  const goalWinner = atGoal.A ? 'A' : atGoal.B ? 'B' : null;
  const expected = input.ply === 0
    ? { kind: 'ongoing' }
    : adjudicate(checked, { goalWinner, resultingPositionCount: counts.get(key), resultingPly: input.ply });
  if (input.ply === 0 && key !== initialState.positionKey) fail('invalid_state');
  if (input.ply > checked.plyCap || counts.get(key) > checked.repetitionThreshold
    || [...counts.entries()].some(([countKey, count]) => countKey !== key && count >= checked.repetitionThreshold)
    || !sameOutcome(outcome, expected)) fail('invalid_state');
  return { position, positionKey: key, ply: input.ply, historyStartPly: input.historyStartPly, repetitionCounts: canonical, outcome: { ...expected } };
}

export function legalActions(config, stateInput) {
  const state = validateState(config, stateInput);
  return state.outcome.kind === 'ongoing' ? legalPositionActions(config, state.position) : [];
}

export function legalActionCodes(config, stateInput) {
  return legalActions(config, stateInput).map((action) => encodeAction(config, action));
}

export function legalActionMask(config, stateInput) {
  const mask = new Uint8Array(policySize(config));
  for (const code of legalActionCodes(config, stateInput)) mask[code] = 1;
  return mask;
}

export function applyAction(config, stateInput, action) {
  const checked = configOf(config);
  if (isObject(stateInput?.outcome) && stateInput.outcome.kind !== 'ongoing') fail('terminal_state');
  const state = validateState(checked, stateInput);
  const code = encodeAction(checked, action); const legal = legalPositionActionCodes(checked, state.position);
  if (!legal.includes(code)) fail('illegal_action');
  const mover = state.position.turn; const position = clonePosition(state.position);
  if (action.kind === 'pawn') position.pawns[mover] = cloneCoord(action.to);
  else { position.walls.push(action.wall); position.stock[mover] -= 1; }
  position.turn = other(mover);
  const normalized = normalizePosition(checked, position); const ply = state.ply + 1; const key = positionKeyNormalized(checked, normalized);
  const counts = action.kind === 'wall' ? new Map([[key, 1]]) : new Map(state.repetitionCounts.map(({ positionKey: countKey, count }) => [countKey, count]));
  if (action.kind === 'pawn') counts.set(key, (counts.get(key) ?? 0) + 1);
  const outcome = adjudicate(checked, { goalWinner: normalized.pawns[mover].r === checked.goalRows[mover] ? mover : null, resultingPositionCount: counts.get(key), resultingPly: ply });
  return { position: normalized, positionKey: key, ply, historyStartPly: action.kind === 'wall' ? ply : state.historyStartPly, repetitionCounts: canonicalCounts(counts), outcome };
}
