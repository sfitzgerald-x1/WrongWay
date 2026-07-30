import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';
import { NormalDuelConsumer } from '../js/normal-duel-browser-adapter.mjs';

const indexSource = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const gameLogicSource = readFileSync(new URL('../js/game-logic.js', import.meta.url), 'utf8');
const aiSource = readFileSync(new URL('../js/ai.js', import.meta.url), 'utf8');

const normalScope = Object.freeze({
  gameMode: 'pvc', map: 'duel', duelSize: 'standard', gameTimeMode: 'none',
  chaosMode: false, hammerMode: false, dropMode: false, isRanked: false,
  initialStock: 10, is2v2: false
});

function aiContext(consumer, dimensions = 9) {
  const context = {
    CELL: 40,
    COLS: dimensions,
    CUR_MAP: 'duel',
    GOAL_2V2: {},
    NormalDuelConsumer: consumer,
    ORDER_2V2: [],
    ROWS: dimensions,
    WW: dimensions - 1,
    WW2: dimensions - 1
  };
  vm.createContext(context);
  vm.runInContext(gameLogicSource, context);
  vm.runInContext(aiSource, context);
  return context;
}

test('browser adapter module loads before the Babel app and human consumers use its route', () => {
  const adapterIndex = indexSource.indexOf("window.normalDuelConsumerReady=import('./js/normal-duel-browser-adapter.mjs')");
  const babelIndex = indexSource.indexOf('<script type="text/babel" data-presets="react">');
  assert.ok(adapterIndex >= 0 && adapterIndex < babelIndex);
  assert.match(indexSource, /dataset\.normalDuelConsumer='loading'/);
  assert.match(indexSource, /dataset\.normalDuelConsumer=window\.__normalDuelConsumerReady\?'ready':'error'/);
  const mountIndex = indexSource.indexOf('root.render(React.createElement(ErrorBoundary');
  const gateIndex = indexSource.indexOf('window.normalDuelConsumerReady.then(function(consumer)');
  const failureIndex = indexSource.indexOf("message.textContent='Rules engine failed to load — reload the page'");
  assert.ok(gateIndex >= 0 && mountIndex > gateIndex && failureIndex > mountIndex);
  assert.match(indexSource, /normalDuelRoute\(state\)/);
  assert.match(indexSource, /route\.consumer\.legalMoves/);
  assert.match(indexSource, /const tryCurrentWall=/);
  assert.match(indexSource, /const next=tryCurrentWall\(key,S\.current\)/);
  assert.match(indexSource, /const v=!!tryCurrentWall\(previewKey,S\.current\)/);
  assert.match(indexSource, /d=aiEasy\(b,a,w,bl\.B,chaosItem,aiGoal,humGoal,duelRules\)/);
});

test('Duel AI receives an explicit A/B-aware reference rules object', () => {
  const calls = [];
  const consumer = {
    classify(scope) {
      calls.push(['classify', scope]);
      return { eligible: true, reason: null };
    },
    legalMoves(scope, snapshot) {
      calls.push(['moves', scope, snapshot]);
      return [{ r: 1, c: 4 }];
    },
    tryWall(scope, snapshot, wall) {
      calls.push(['wall', scope, snapshot, wall]);
      return new Set([...snapshot.walls, wall]);
    },
    moveTowardGoal(scope, snapshot, path) {
      calls.push(['toward', scope, snapshot, path]);
      return { r: 1, c: 4 };
    }
  };
  const context = aiContext(consumer);
  const rules = context.createDuelAiRules(normalScope, 8, 0);
  const ai = { r: 0, c: 4 };
  const human = { r: 8, c: 4 };
  const walls = new Set();
  assert.equal(rules.source, 'normal-duel-v1');
  assert.deepEqual(rules.moves(ai, human, walls, true), [{ r: 1, c: 4 }]);
  assert.deepEqual(rules.moves(ai, human, walls, false), [{ r: 1, c: 4 }]);
  assert.deepEqual([...rules.wall(ai, human, walls, 'H-0-0', true)], ['H-0-0']);
  assert.deepEqual(rules.toward(ai, human, walls, [ai, { r: 1, c: 4 }]), { r: 1, c: 4 });
  const moveSnapshots = calls.filter(([kind]) => kind === 'moves').map(([, , snapshot]) => snapshot);
  assert.equal(moveSnapshots[0].turn, 'B');
  assert.equal(moveSnapshots[1].turn, 'A');
  assert.deepEqual(JSON.parse(JSON.stringify(moveSnapshots[0].pA)), human);
  assert.deepEqual(JSON.parse(JSON.stringify(moveSnapshots[0].pB)), ai);
});

test('excluded Duel AI scopes retain the legacy implementation', () => {
  let referenceCalls = 0;
  const context = aiContext({
    classify() { return { eligible: false, reason: 'legacy_hammer_mode' }; },
    legalMoves() { referenceCalls += 1; throw new Error('unexpected reference route'); },
    tryWall() { referenceCalls += 1; throw new Error('unexpected reference route'); },
    moveTowardGoal() { referenceCalls += 1; throw new Error('unexpected reference route'); }
  });
  const rules = context.createDuelAiRules({ ...normalScope, hammerMode: true }, 8, 0);
  assert.equal(rules.source, 'legacy');
  assert.ok(rules.moves({ r: 0, c: 4 }, { r: 8, c: 4 }, new Set(), true).length > 0);
  assert.equal(referenceCalls, 0);
});

test('an explicit product scope fails closed when the adapter is unavailable', () => {
  const context = aiContext(undefined);
  assert.throws(
    () => context.createDuelAiRules(normalScope, 8, 0),
    /normal-duel-v1 adapter is unavailable/
  );
});

test('eligible Easy Duel obtains its winning move from the reference consumer', () => {
  let moveCalls = 0;
  const consumer = {
    classify() { return { eligible: true, reason: null }; },
    legalMoves(scope, snapshot) {
      moveCalls += 1;
      assert.equal(snapshot.turn, 'B');
      return [{ r: 8, c: 4 }];
    },
    tryWall() { throw new Error('winning move should not test walls'); },
    moveTowardGoal() { throw new Error('winning move should not need a path fallback'); }
  };
  const context = aiContext(consumer);
  const rules = context.createDuelAiRules(normalScope, 8, 0);
  const action = context.aiEasy(
    { r: 7, c: 4 }, { r: 0, c: 4 }, new Set(), 10, null, 8, 0, rules
  );
  assert.deepEqual(JSON.parse(JSON.stringify(action)), { type: 'move', pos: { r: 8, c: 4 } });
  assert.equal(moveCalls, 1);
});

test('reference routing preserves curated Hard AI decisions from the legacy path', () => {
  const cases = [
    {
      dimensions: 9, duelSize: 'standard',
      pA: { r: 8, c: 4 }, pB: { r: 0, c: 4 }, walls: []
    },
    {
      dimensions: 9, duelSize: 'standard',
      pA: { r: 4, c: 4 }, pB: { r: 3, c: 4 }, walls: ['H-2-4']
    },
    {
      dimensions: 7, duelSize: 'blitz',
      pA: { r: 6, c: 3 }, pB: { r: 0, c: 3 }, walls: []
    }
  ];
  for (const sample of cases) {
    const context = aiContext(NormalDuelConsumer, sample.dimensions);
    const scope = { ...normalScope, duelSize: sample.duelSize };
    const reference = context.createDuelAiRules(scope, sample.dimensions - 1, 0);
    const legacy = context.createDuelAiRules({ ...scope, initialStock: 12 }, sample.dimensions - 1, 0);
    const args = [
      sample.pB, sample.pA, new Set(sample.walls), 10, 10, [], null,
      sample.dimensions - 1, 0
    ];
    const expected = context.aiDuelHard(...args, legacy);
    const actual = context.aiDuelHard(...args, reference);
    assert.deepEqual(
      JSON.parse(JSON.stringify(actual)),
      JSON.parse(JSON.stringify(expected)),
      `${sample.duelSize} decision drifted`
    );
  }
});
