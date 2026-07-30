import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import Ajv2020 from 'ajv/dist/2020.js';

const fixtureUrl = new URL('./fixtures/normal-duel-v1-cases.json', import.meta.url);
const schemaUrl = new URL('./fixtures/normal-duel-v1.schema.json', import.meta.url);
const gameLogicUrl = new URL('../js/game-logic.js', import.meta.url);
const aiUrl = new URL('../js/ai.js', import.meta.url);

const fixture = JSON.parse(readFileSync(fixtureUrl, 'utf8'));
const schema = JSON.parse(readFileSync(schemaUrl, 'utf8'));
const gameLogicSource = readFileSync(gameLogicUrl, 'utf8');
const aiSource = readFileSync(aiUrl, 'utf8');
const wallPattern = new RegExp(schema.$defs.wall.pattern);

function configFor(entry) {
  const config = fixture.configs[entry.configId];
  assert.ok(config, `${entry.id}: unknown config ${entry.configId}`);
  return config;
}

function parseWall(wall) {
  const match = wallPattern.exec(wall);
  if (!match) return null;
  return {
    orientation: wall[0],
    row: Number(wall.split('-')[1]),
    column: Number(wall.split('-')[2])
  };
}

function actionCode(config, action) {
  const cells = config.rows * config.columns;
  const anchors = (config.rows - 1) * (config.columns - 1);

  if (action.kind === 'pawn') {
    return action.to.r * config.columns + action.to.c;
  }

  const wall = parseWall(action.wall);
  assert.ok(wall, action.wall);
  const anchor = wall.row * (config.columns - 1) + wall.column;
  return cells + (wall.orientation === 'V' ? anchors : 0) + anchor;
}

function positionKey(config, position) {
  const square = ({ r, c }) => r * config.columns + c;
  const horizontal = [];
  const vertical = [];

  for (const wallText of position.walls) {
    const wall = parseWall(wallText);
    assert.ok(wall, wallText);
    const anchor = wall.row * (config.columns - 1) + wall.column;
    (wall.orientation === 'H' ? horizontal : vertical).push(anchor);
  }

  horizontal.sort((a, b) => a - b);
  vertical.sort((a, b) => a - b);

  return JSON.stringify([
    config.ruleset,
    config.rows,
    config.columns,
    square(config.start.A),
    square(config.start.B),
    config.goalRows.A,
    config.goalRows.B,
    config.initialStock.A,
    config.initialStock.B,
    config.jumpRule,
    config.repetitionThreshold,
    config.plyCap,
    config.firstPlayer,
    square(position.pawns.A),
    square(position.pawns.B),
    horizontal,
    vertical,
    position.stock.A,
    position.stock.B,
    position.turn
  ]);
}

function legacyContext(config) {
  const context = {
    CELL: 40,
    COLS: config.columns,
    CUR_MAP: 'duel',
    GOAL_2V2: {},
    ORDER_2V2: [],
    ROWS: config.rows,
    WW: 8,
    WW2: 8
  };
  vm.createContext(context);
  vm.runInContext(gameLogicSource, context);
  vm.runInContext(aiSource, context);
  return context;
}

function inBounds(config, coord) {
  return Number.isInteger(coord.r)
    && Number.isInteger(coord.c)
    && coord.r >= 0
    && coord.r < config.rows
    && coord.c >= 0
    && coord.c < config.columns;
}

function wallInBounds(config, wallText) {
  const wall = parseWall(wallText);
  return Boolean(wall)
    && wall.row >= 0
    && wall.row < config.rows - 1
    && wall.column >= 0
    && wall.column < config.columns - 1;
}

function contractWallLegal(context, config, position, wallText) {
  if (position.stock[position.turn] <= 0 || !wallInBounds(config, wallText)) {
    return false;
  }
  return Boolean(context.tryWall(
    wallText,
    new Set(position.walls),
    position.pawns.A,
    position.pawns.B,
    config.goalRows.A,
    config.goalRows.B
  ));
}

function expectedTurn(config, ply) {
  if (ply % 2 === 0) return config.firstPlayer;
  return config.firstPlayer === 'A' ? 'B' : 'A';
}

function compareKeyBytes(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function assertPosition(config, position, label) {
  assert.ok(inBounds(config, position.pawns.A), `${label}: A out of bounds`);
  assert.ok(inBounds(config, position.pawns.B), `${label}: B out of bounds`);
  assert.notDeepEqual(position.pawns.A, position.pawns.B, `${label}: pawns overlap`);

  const wallCodes = position.walls.map((wall) => {
    assert.ok(wallInBounds(config, wall), `${label}: invalid wall ${wall}`);
    return actionCode(config, { kind: 'wall', wall });
  });
  assert.deepEqual(
    wallCodes,
    [...new Set(wallCodes)].sort((a, b) => a - b),
    `${label}: walls must be unique and policy-code sorted`
  );

  for (const player of ['A', 'B']) {
    assert.ok(Number.isInteger(position.stock[player]), `${label}: stock`);
    assert.ok(position.stock[player] >= 0, `${label}: negative stock`);
    assert.ok(
      position.stock[player] <= config.initialStock[player],
      `${label}: stock exceeds initial`
    );
  }

  const spent = (config.initialStock.A - position.stock.A)
    + (config.initialStock.B - position.stock.B);
  assert.equal(spent, position.walls.length, `${label}: stock/wall mismatch`);

  const context = legacyContext(config);
  let builtWalls = new Set();
  for (const wall of position.walls) {
    const next = context.tryWall(
      wall,
      builtWalls,
      position.pawns.A,
      position.pawns.B,
      config.goalRows.A,
      config.goalRows.B
    );
    assert.ok(next, `${label}: incompatible or path-cutting wall ${wall}`);
    builtWalls = next;
  }
  assert.equal(builtWalls.size, position.walls.length, `${label}: wall count`);
}

function assertState(config, state, label) {
  assertPosition(config, state.position, `${label}/position`);
  assert.equal(state.position.turn, expectedTurn(config, state.ply), `${label}: turn`);
  assert.equal(state.positionKey, positionKey(config, state.position), `${label}: key`);
  assert.ok(state.historyStartPly <= state.ply, `${label}: history start`);

  const keys = state.repetitionCounts.map(({ positionKey: key }) => key);
  assert.deepEqual(
    keys,
    [...new Set(keys)].sort(compareKeyBytes),
    `${label}: repetition order`
  );
  assert.ok(keys.includes(state.positionKey), `${label}: missing current key`);

  const retained = state.repetitionCounts.reduce((sum, entry) => (
    sum + entry.count
  ), 0);
  assert.equal(
    retained,
    state.ply - state.historyStartPly + 1,
    `${label}: repetition count sum`
  );
}

function adjudicate(config, facts) {
  if (facts.goalWinner) {
    return { kind: 'win', winner: facts.goalWinner, reason: 'goal' };
  }
  if (facts.resultingPositionCount >= config.repetitionThreshold) {
    return { kind: 'draw', reason: 'threefold_repetition' };
  }
  if (facts.resultingPly >= config.plyCap) {
    return { kind: 'draw', reason: 'ply_cap' };
  }
  return { kind: 'ongoing' };
}

test('schema metadata, config ids, and required case ids are present', () => {
  assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema');
  assert.equal(
    schema.properties.fixtureFormat.const,
    fixture.fixtureFormat
  );
  assert.equal(fixture.ruleset, 'normal-duel-v1');

  const ids = new Set(fixture.cases.map((entry) => entry.id));
  assert.equal(ids.size, fixture.cases.length);
  for (const id of [
    'initial-9x9-a',
    'initial-9x9-b',
    'initial-7x7-a',
    'initial-7x7-b',
    'permissive-jump-open',
    'permissive-jump-straight-blocked',
    'permissive-jump-corner',
    'wall-overlap-rejected',
    'wall-crossing-rejected',
    'wall-duplicate-rejected',
    'wall-out-of-bounds-rejected',
    'wall-zero-stock-rejected',
    'wall-path-cut-rejected',
    'wall-resets-repetition',
    'pawn-preserves-repetition',
    'goal-beats-ply-cap',
    'threefold-beats-ply-cap',
    'ply-cap-boundary',
    'terminal-action-rejected',
    'hammer-feature-rejected'
  ]) {
    assert.ok(ids.has(id), id);
  }
});

test('the full fixture corpus validates against its Draft 2020-12 schema', () => {
  const ajv = new Ajv2020({ allErrors: true, strict: true });
  const validate = ajv.compile(schema);
  assert.equal(
    validate(fixture),
    true,
    JSON.stringify(validate.errors, null, 2)
  );
});

test('configuration and query positions obey contract bounds and ordering', () => {
  for (const [id, config] of Object.entries(fixture.configs)) {
    assert.equal(config.ruleset, fixture.ruleset, id);
    assert.equal(config.rows, config.columns, id);
    assert.ok([7, 9].includes(config.rows), id);
    assert.deepEqual(config.start.A, {
      r: config.rows - 1,
      c: Math.floor(config.columns / 2)
    }, id);
    assert.deepEqual(config.start.B, {
      r: 0,
      c: Math.floor(config.columns / 2)
    }, id);
    assert.deepEqual(config.goalRows, { A: 0, B: config.rows - 1 }, id);
    const policySize = config.rows * config.columns
      + 2 * (config.rows - 1) * (config.columns - 1);
    assert.equal(policySize, config.rows === 9 ? 209 : 121, id);
  }

  for (const entry of fixture.cases.filter(({ kind }) => kind === 'query')) {
    const config = configFor(entry);
    assertPosition(config, entry.position, entry.id);
    const context = legacyContext(config);
    const mover = entry.position.pawns[entry.position.turn];
    const opponent = entry.position.pawns[entry.position.turn === 'A' ? 'B' : 'A'];
    assert.ok(
      context.getMovesFrom(mover, opponent, new Set(entry.position.walls)).length > 0,
      `${entry.id}: ongoing position has no pawn action`
    );
  }
});

test('pawn query goldens match the current 1v1 AI move generator', () => {
  for (const entry of fixture.cases) {
    const destinations = entry.expect?.legalPawnDestinations;
    if (!destinations) continue;

    const config = configFor(entry);
    const context = legacyContext(config);
    const mover = entry.position.pawns[entry.position.turn];
    const opponent = entry.position.pawns[entry.position.turn === 'A' ? 'B' : 'A'];
    const actual = Array.from(context.getMovesFrom(
      mover,
      opponent,
      new Set(entry.position.walls)
    ), ({ r, c }) => ({ r, c })).sort((a, b) => (
      actionCode(config, { kind: 'pawn', to: a })
      - actionCode(config, { kind: 'pawn', to: b })
    ));

    assert.deepEqual(actual, destinations, entry.id);
  }
});

test('wall query goldens match contract validation and comparable legacy behavior', () => {
  for (const entry of fixture.cases) {
    if (entry.kind !== 'query' || entry.action?.kind !== 'wall') continue;

    const config = configFor(entry);
    const context = legacyContext(config);
    const legal = contractWallLegal(
      context,
      config,
      entry.position,
      entry.action.wall
    );
    assert.equal(legal, entry.expect.legal, entry.id);

    if (entry.expect.actionCode !== undefined) {
      assert.equal(actionCode(config, entry.action), entry.expect.actionCode, entry.id);
    }

    if (entry.expect.legacyComparable) {
      const legacy = Boolean(context.tryWall(
        entry.action.wall,
        new Set(entry.position.walls),
        entry.position.pawns.A,
        entry.position.pawns.B,
        config.goalRows.A,
        config.goalRows.B
      ));
      assert.equal(legacy, entry.expect.legal, `${entry.id}: legacy`);
    }

    if (entry.expect.pathsRemain !== undefined) {
      const candidateWalls = new Set(entry.position.walls);
      candidateWalls.add(entry.action.wall);
      const pathsRemain = context.hasPath(
        entry.position.pawns.A,
        candidateWalls,
        config.goalRows.A
      ) && context.hasPath(
        entry.position.pawns.B,
        candidateWalls,
        config.goalRows.B
      );
      assert.equal(pathsRemain, entry.expect.pathsRemain, `${entry.id}: paths`);
    }
  }
});

test('initial legal-action totals match complete legacy wall generation', () => {
  for (const entry of fixture.cases) {
    if (entry.expect?.legalActionCount === undefined) continue;
    const config = configFor(entry);
    const context = legacyContext(config);
    const position = entry.position;
    const mover = position.pawns[position.turn];
    const opponent = position.pawns[position.turn === 'A' ? 'B' : 'A'];
    let total = context.getMovesFrom(mover, opponent, new Set(position.walls)).length;

    for (const orientation of ['H', 'V']) {
      for (let row = 0; row < config.rows - 1; row += 1) {
        for (let column = 0; column < config.columns - 1; column += 1) {
          if (contractWallLegal(
            context,
            config,
            position,
            `${orientation}-${row}-${column}`
          )) {
            total += 1;
          }
        }
      }
    }

    assert.equal(total, entry.expect.legalActionCount, entry.id);
  }
});

test('reachable transition goldens satisfy canonical history and apply correctly', () => {
  for (const entry of fixture.cases.filter(({ kind }) => kind === 'transition')) {
    const config = configFor(entry);
    assertState(config, entry.state, `${entry.id}/state`);
    assertState(config, entry.nextState, `${entry.id}/nextState`);
    assert.equal(actionCode(config, entry.action), entry.expect.actionCode, entry.id);

    const context = legacyContext(config);
    const mover = entry.state.position.turn;
    let expectedPosition;

    if (entry.action.kind === 'wall') {
      assert.ok(contractWallLegal(
        context,
        config,
        entry.state.position,
        entry.action.wall
      ), entry.id);
      expectedPosition = {
        pawns: entry.state.position.pawns,
        walls: [...entry.state.position.walls, entry.action.wall].sort((left, right) => (
          actionCode(config, { kind: 'wall', wall: left })
          - actionCode(config, { kind: 'wall', wall: right })
        )),
        stock: {
          ...entry.state.position.stock,
          [mover]: entry.state.position.stock[mover] - 1
        },
        turn: mover === 'A' ? 'B' : 'A'
      };
    } else {
      const opponent = mover === 'A' ? 'B' : 'A';
      const legalMoves = context.getMovesFrom(
        entry.state.position.pawns[mover],
        entry.state.position.pawns[opponent],
        new Set(entry.state.position.walls)
      );
      assert.ok(
        legalMoves.some(({ r, c }) => (
          r === entry.action.to.r && c === entry.action.to.c
        )),
        entry.id
      );
      expectedPosition = {
        pawns: {
          ...entry.state.position.pawns,
          [mover]: entry.action.to
        },
        walls: entry.state.position.walls,
        stock: entry.state.position.stock,
        turn: opponent
      };
    }

    assert.deepEqual(entry.nextState.position, expectedPosition, entry.id);
    assert.equal(entry.nextState.ply, entry.state.ply + 1, entry.id);

    if (entry.action.kind === 'wall') {
      assert.equal(entry.nextState.historyStartPly, entry.nextState.ply, entry.id);
      assert.deepEqual(entry.nextState.repetitionCounts, [{
        positionKey: entry.nextState.positionKey,
        count: 1
      }], entry.id);
    } else {
      assert.equal(
        entry.nextState.historyStartPly,
        entry.state.historyStartPly,
        entry.id
      );
      const counts = new Map(entry.state.repetitionCounts.map((item) => (
        [item.positionKey, item.count]
      )));
      counts.set(
        entry.nextState.positionKey,
        (counts.get(entry.nextState.positionKey) ?? 0) + 1
      );
      const expectedCounts = [...counts]
        .sort(([left], [right]) => compareKeyBytes(left, right))
        .map(([key, count]) => ({ positionKey: key, count }));
      assert.deepEqual(entry.nextState.repetitionCounts, expectedCounts, entry.id);
    }
  }
});

test('adjudication goldens enforce goal, repetition, then ply-cap precedence', () => {
  for (const entry of fixture.cases.filter(({ kind }) => kind === 'adjudication')) {
    assert.deepEqual(
      adjudicate(configFor(entry), entry.facts),
      entry.expect.outcome,
      entry.id
    );
  }
});

test('rejection goldens pin terminal and unsupported-feature failures', () => {
  for (const entry of fixture.cases.filter(({ kind }) => kind === 'rejection')) {
    let error = null;
    if (entry.input.outcome && entry.input.outcome.kind !== 'ongoing') {
      error = 'terminal_state';
    } else if (entry.input.features && Object.values(entry.input.features).some(Boolean)) {
      error = 'unsupported_feature';
    }
    assert.equal(error, entry.expect.error, entry.id);
  }
});
