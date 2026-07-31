import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  applyAction,
  createInitialState,
  decodeAction,
  encodeAction,
  legalActionCodes,
  legalActions
} from '../js/normal-duel-engine.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import {
  CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
  sha256ArtifactFile
} from '../scripts/evaluation/candidate-artifact-manifest.mjs';
import {
  applyPinnedAntiStall,
  buildPinnedRecentAi,
  createPinnedHardBaseline,
  HARD_BASELINE_ID,
  HARD_BASELINE_TRUST_ROOT,
  rotatePinnedAction180,
  rotatePinnedCoordinate180,
  rotatePinnedWall180,
  verifyPinnedHardBaselineDirectory
} from '../scripts/evaluation/hard-baseline-46a871c7.mjs';
import {
  CANONICAL_STRENGTH_CONFIGURATION_SHA256,
  CANONICAL_STRENGTH_DEADLINE_MS,
  CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
  CANONICAL_STRENGTH_OBSERVER_CAP_MS,
  CANONICAL_STRENGTH_SEED,
  CALLER_SUPPLIED_CLOCK_PROFILE,
  DETERMINISTIC_REGRESSION_CLOCK_PROFILE,
  deterministicRegressionClockFactory,
  MINIMUM_STRENGTH_OPENING_PAIRS,
  normalDuelConfig,
  pairedClusterConfidenceIntervals,
  REAL_MONOTONIC_CLOCK_PROFILE,
  REGRESSION_MODE,
  runEvaluation,
  runMatch,
  summarizeEvaluation,
  STRENGTH_MODE,
  validateEnforcedStrengthOptions,
  validateOpeningBook
} from '../scripts/evaluation/normal-duel-strength.mjs';
import {
  CANONICAL_ENGINE_MEMORY_LIMIT_MIB,
  CANONICAL_ENGINE_V8_OLD_SPACE_MIB,
  createPinnedHardWorkerAdapter,
  createRecycledNormalDuelWasmCandidateAdapter,
  createWorkerEngineAdapter,
  getCandidateArtifactProvenance,
  getWorkerEngineIsolationProvenance,
  isAuthenticatedCandidateAdapter,
  isPinnedHardWorkerAdapter,
  isTrustedSubprocessDeadlineError,
  isTrustedSubprocessMemoryError,
  isWorkerEngineAdapter,
  NORMAL_DUEL_WASM_CANDIDATE_ID,
  RECYCLE_CHILD_AUDIT_SEAM,
  RECYCLE_TEARDOWN_FAULT_SEAM,
  SUPPORTED_WORKER_SESSION_BOUNDARIES,
  takeTrustedSubprocessDecision,
  takeTrustedSubprocessFailureTiming,
  takeTrustedSubprocessPendingTiming
} from '../scripts/evaluation/worker-engine-proxy.mjs';
import {
  generateBalancedOpeningBook,
  isVerifiedOpeningCorpus,
  openingArtifacts,
  verifyOpeningArtifacts
} from '../scripts/generate-normal-duel-balanced-openings.mjs';

const BASELINE_DIR = new URL('./fixtures/baselines/hard-46a871c7/', import.meta.url);
const BOOK_URL = new URL('./fixtures/normal-duel-balanced-openings-v1.json', import.meta.url);
const MANIFEST_URL = new URL('./fixtures/normal-duel-balanced-openings-v1.manifest.json', import.meta.url);
let canonicalCorpusCache = null;

function sha256(source) {
  return createHash('sha256').update(source).digest('hex');
}

function writeCandidateArtifact(directory, stem, entrySource, dependencies = {}) {
  const releaseDirectory = join(directory, stem);
  mkdirSync(releaseDirectory);
  const entry = `${stem}.mjs`;
  const records = [{ path: entry, source: entrySource }];
  for (const [path, source] of Object.entries(dependencies)) {
    records.push({ path, source });
  }
  records.sort((left, right) => left.path.localeCompare(right.path));
  for (const record of records) {
    writeFileSync(join(releaseDirectory, record.path), record.source);
  }
  const manifest = {
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    entry,
    files: records.map((record) => ({
      path: record.path,
      sha256: sha256(Buffer.from(record.source))
    }))
  };
  const manifestPath = join(releaseDirectory, 'manifest.json');
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return {
    entryPath: join(releaseDirectory, entry),
    manifestPath,
    manifest,
    dependencies: Object.fromEntries(
      records.filter((record) => record.path !== entry)
        .map((record) => [record.path, join(releaseDirectory, record.path)])
    )
  };
}

function verifiedCorpus(options = {}) {
  const artifacts = openingArtifacts(options);
  return verifyOpeningArtifacts(artifacts.bookText, artifacts.manifestText);
}

function canonicalEnforcedCorpus() {
  canonicalCorpusCache ??= verifiedCorpus({
    seed: CANONICAL_STRENGTH_SEED,
    count: MINIMUM_STRENGTH_OPENING_PAIRS,
    size: 9,
    firstPlayer: 'A'
  });
  return canonicalCorpusCache;
}

function scriptedEngine(id, select, { nodeBudget = true, observe = () => {} } = {}) {
  return Object.freeze({
    id,
    version: 'test-v1',
    capabilities: Object.freeze({ nodeBudget }),
    createSession(context) {
      let closed = false;
      return Object.freeze({
        selectAction(request) {
          assert.equal(closed, false);
          return select(request, context);
        },
        observe(transition) {
          assert.equal(closed, false);
          return observe(transition, context);
        },
        close() {
          closed = true;
        }
      });
    }
  });
}

function goalAction({ state, config }) {
  const player = state.position.turn;
  const goal = config.goalRows[player];
  const pawnActions = legalActions(config, state)
    .filter((action) => action.kind === 'pawn')
    .sort((left, right) =>
      Math.abs(left.to.r - goal) - Math.abs(right.to.r - goal)
      || encodeAction(config, left) - encodeAction(config, right));
  return {
    action: pawnActions[0],
    stats: { elapsedMs: 0.25, nodes: 7, depth: 2 }
  };
}

function oscillatingAction({ state, config }) {
  const player = state.position.turn;
  const pawn = state.position.pawns[player];
  const center = Math.floor(config.columns / 2);
  const targetColumn = pawn.c === center ? center - 1 : center;
  const action = legalActions(config, state).find((candidate) =>
    candidate.kind === 'pawn'
    && candidate.to.r === pawn.r
    && candidate.to.c === targetColumn);
  assert.ok(action, `${player} has the expected oscillation action`);
  return { action, stats: { elapsedMs: 0.1, nodes: 1, depth: 1 } };
}

function tickingClock(step = 0.01) {
  let value = 0;
  return { now: () => (value += step) };
}

function play(config, coordinates) {
  let state = createInitialState(config);
  const history = [];
  for (const to of coordinates) {
    const player = state.position.turn;
    const action = { kind: 'pawn', to };
    const before = state;
    state = applyAction(config, state, action);
    history.push({ player, action, stateBefore: before, stateAfter: state });
  }
  return { state, history };
}

function rotateStateForPinnedBOracle(state, config) {
  return {
    ...state,
    position: {
      ...state.position,
      pawns: {
        A: rotatePinnedCoordinate180(state.position.pawns.B, config),
        B: rotatePinnedCoordinate180(state.position.pawns.A, config)
      },
      walls: state.position.walls.map((wall) => rotatePinnedWall180(wall, config)),
      stock: {
        A: state.position.stock.B,
        B: state.position.stock.A
      },
      turn: state.position.turn === 'A' ? 'B' : 'A'
    }
  };
}

function rotateHistoryForPinnedBOracle(history, config) {
  return history.map((transition) => ({
    ...transition,
    player: transition.player === 'A' ? 'B' : 'A',
    action: rotatePinnedAction180(transition.action, config)
  }));
}

function seededState(config, seed, plies) {
  const random = createLcg32(seed);
  let state = createInitialState(config);
  const history = [];
  for (let index = 0; index < plies; index += 1) {
    const actions = legalActions(config, state);
    const action = actions[random() % actions.length];
    const before = state;
    state = applyAction(config, state, action);
    assert.equal(state.outcome.kind, 'ongoing');
    history.push({ player: before.position.turn, action, stateBefore: before, stateAfter: state });
  }
  return { state, history };
}

test('pinned Hard snapshots have explicit provenance and exact SHA-256', () => {
  const manifestSource = readFileSync(new URL('manifest.json', BASELINE_DIR));
  const manifest = JSON.parse(manifestSource.toString('utf8'));
  assert.equal(sha256(manifestSource), HARD_BASELINE_TRUST_ROOT.manifestSha256);
  assert.equal(manifest.baselineId, HARD_BASELINE_ID);
  assert.equal(manifest.baselineFormat, HARD_BASELINE_TRUST_ROOT.baselineVersion);
  assert.equal(manifest.sourceCommit, '46a871c7b061a33922bdb9c6d78355e2e9b6b607');
  assert.equal(manifest.license.status, 'no-standalone-license-file-at-source-commit');
  for (const record of manifest.sources.filter((source) => source.snapshot)) {
    const source = readFileSync(new URL(record.snapshot, BASELINE_DIR));
    assert.equal(sha256(source), record.snapshotSha256 ?? record.sha256, record.snapshot);
  }
  const aiSource = readFileSync(new URL('ai.js', BASELINE_DIR), 'utf8');
  assert.match(aiSource, /function aiDuelHard\(/);
  assert.match(aiSource, /const t0=Date\.now\(\);\s*const TIME=700;/);
  assert.match(aiSource, /recent\.length>=2[\s\S]*i=1;i<Math\.min\(recent\.length,5\)/);
  assert.doesNotMatch(aiSource, /createDuelAiRules/);
});

test('Hard trust root rejects a rewritten manifest that blesses a modified snapshot', () => {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), 'wrongway-hard-trust-'));
  try {
    for (const filename of [
      'ai.js',
      'game-logic.js',
      'index-hard-orchestration.txt',
      'manifest.json'
    ]) {
      copyFileSync(new URL(filename, BASELINE_DIR), join(temporaryDirectory, filename));
    }
    const aiPath = join(temporaryDirectory, 'ai.js');
    const modifiedAi = Buffer.concat([
      readFileSync(aiPath),
      Buffer.from('\n// adversarial rewrite\n')
    ]);
    writeFileSync(aiPath, modifiedAi);
    const manifestPath = join(temporaryDirectory, 'manifest.json');
    const rewrittenManifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    rewrittenManifest.sources.find((source) => source.snapshot === 'ai.js').sha256 =
      sha256(modifiedAi);
    writeFileSync(manifestPath, `${JSON.stringify(rewrittenManifest, null, 2)}\n`);
    assert.throws(
      () => verifyPinnedHardBaselineDirectory(temporaryDirectory),
      /manifest no longer matches the hard-coded trust root/
    );
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});

test('recentAi and anti-stall orchestration reproduce the pinned product semantics', () => {
  const recent = buildPinnedRecentAi(
    { r: 4, c: 4 },
    [
      { type: 'move', p: 'B', r: 0, c: 0 },
      { type: 'move', p: 'B', r: 1, c: 0 },
      { type: 'barricade', p: 'B', k: 'H-1-1' },
      { type: 'move', p: 'A', r: 8, c: 3 },
      { type: 'move', p: 'B', r: 2, c: 0 },
      { type: 'move', p: 'B', r: 3, c: 0 },
      { type: 'move', p: 'B', r: 4, c: 0 },
      { type: 'move', p: 'B', r: 5, c: 0 }
    ],
    'B'
  );
  assert.deepEqual(recent, [
    { r: 4, c: 4 },
    { r: 5, c: 0 },
    { r: 4, c: 0 },
    { r: 3, c: 0 },
    { r: 2, c: 0 },
    { r: 1, c: 0 }
  ]);

  const progress = { best: Infinity, stall: 2 };
  const state = {
    position: {
      pawns: { A: { r: 8, c: 4 }, B: { r: 1, c: 4 } },
      walls: []
    }
  };
  const runtime = {
    goalRows: { A: 0, B: 8 },
    bfsPath(position) {
      return position.c === 5
        ? [{ ...position }, { r: position.r + 1, c: position.c }, { r: position.r + 2, c: position.c }]
        : [{ ...position }, { r: position.r + 1, c: position.c }];
    },
    moveTowardGoal() {
      return { r: 2, c: 4 };
    }
  };
  const replaced = applyPinnedAntiStall({
    runtime,
    decision: { type: 'move', pos: { r: 1, c: 5 } },
    state,
    side: 'B',
    progress
  });
  assert.equal(replaced.replaced, true);
  assert.deepEqual(replaced.decision, { type: 'move', pos: { r: 2, c: 4 } });
  assert.deepEqual(progress, { best: Infinity, stall: 0 });
  applyPinnedAntiStall({
    runtime,
    decision: { type: 'barricade', walls: new Set(['H-1-1']) },
    state,
    side: 'B',
    progress
  });
  assert.equal(progress.stall, 0, 'wall proposals do not change the persistent stall count');
});

test('pinned Hard calls aiDuelHard directly and monotonic clock offsets do not drift an immediate decision', async () => {
  const config = normalDuelConfig({ size: 7 });
  const { state, history } = play(config, [
    { r: 6, c: 2 }, { r: 1, c: 3 },
    { r: 5, c: 2 }, { r: 2, c: 3 },
    { r: 5, c: 1 }, { r: 3, c: 3 },
    { r: 6, c: 1 }, { r: 4, c: 3 },
    { r: 6, c: 0 }, { r: 5, c: 3 },
    { r: 5, c: 0 }
  ]);
  assert.equal(state.position.turn, 'B');
  const baseline = createPinnedHardBaseline();
  const decideAt = async (offset) => {
    const session = baseline.createSession({ side: 'B', config, openingHistory: history });
    const clock = { now: () => offset };
    const response = await session.selectAction({ state, clock, deadlineAtMs: offset + 1_000 });
    const inspected = session.inspect();
    session.close();
    assert.deepEqual(inspected.progress, { best: Infinity, stall: 0 });
    return response.action;
  };
  assert.deepEqual(await decideAt(0), { kind: 'pawn', to: { r: 6, c: 3 } });
  assert.deepEqual(await decideAt(10_000), await decideAt(0));
});

test('pinned Hard uses the injected relative monotonic timeline for its 700 ms cutoff', async () => {
  const config = normalDuelConfig({ size: 7 });
  const state = createInitialState(config);
  const baseline = createPinnedHardBaseline();
  const decideAt = async (offset) => {
    let calls = 0;
    const values = [offset, offset, offset + 701, offset + 701];
    const clock = { now: () => values[calls++] ?? offset + 701 };
    const session = baseline.createSession({ side: 'A', config });
    const response = await session.selectAction({
      state,
      clock,
      deadlineAtMs: offset + 5_000
    });
    session.close();
    assert.ok(calls >= 3, 'the pinned Date.now cutoff consumed the injected clock');
    return response.action;
  };
  assert.deepEqual(await decideAt(10_000), await decideAt(0));
});

test('Hard on side A is exactly a 180-degree transform through the pinned B oracle', async () => {
  const config = normalDuelConfig({ size: 7 });
  const baseline = createPinnedHardBaseline();
  for (const [seed, plies] of [[11, 2], [29, 4], [47, 6]]) {
    const original = seededState(config, seed, plies);
    assert.equal(original.state.position.turn, 'A');
    const rotatedState = rotateStateForPinnedBOracle(original.state, config);
    const rotatedHistory = rotateHistoryForPinnedBOracle(original.history, config);
    assert.equal(rotatedState.position.turn, 'B');

    const sessionA = baseline.createSession({
      side: 'A',
      config,
      openingHistory: original.history
    });
    const sessionB = baseline.createSession({
      side: 'B',
      config,
      openingHistory: rotatedHistory
    });
    const makeClock = () => {
      let now = 0;
      return { now: () => (now += 25) };
    };
    const decisionA = await sessionA.selectAction({
      state: original.state,
      clock: makeClock(),
      deadlineAtMs: 10_000
    });
    const decisionB = await sessionB.selectAction({
      state: rotatedState,
      clock: makeClock(),
      deadlineAtMs: 10_000
    });
    sessionA.close();
    sessionB.close();
    assert.deepEqual(
      decisionA.action,
      rotatePinnedAction180(decisionB.action, config),
      `seed ${seed}`
    );
  }
});

test('opening book is byte-reproducible, unique, balanced, and covers 4–6 plies', () => {
  const bookText = readFileSync(BOOK_URL, 'utf8');
  const manifestText = readFileSync(MANIFEST_URL, 'utf8');
  const actualBook = JSON.parse(bookText);
  const actualManifest = JSON.parse(manifestText);
  const generated = openingArtifacts();
  const verified = verifyOpeningArtifacts(bookText, manifestText);
  assert.equal(bookText, generated.bookText);
  assert.equal(manifestText, generated.manifestText);
  assert.equal(actualManifest.sha256, sha256(bookText));
  assert.equal(actualBook.openings.length, 12);
  assert.equal(actualBook.configurationSha256, CANONICAL_STRENGTH_CONFIGURATION_SHA256);
  assert.equal(actualBook.generator.seed, CANONICAL_STRENGTH_SEED);
  assert.equal(actualBook.config.rows, 9);
  assert.equal(actualBook.config.columns, 9);
  assert.equal(actualBook.config.firstPlayer, 'A');
  assert.deepEqual(new Set(actualBook.openings.map((opening) => opening.targetPlies)), new Set([4, 5, 6]));
  assert.equal(new Set(actualBook.openings.map((opening) => opening.positionKey)).size, 12);
  for (const opening of actualBook.openings) {
    assert.equal(opening.actionCodes.length, opening.targetPlies);
    assert.equal(opening.diagnostics.balanced, true);
    assert.ok(Math.abs(opening.diagnostics.distanceDelta) <= 1);
  }
  assert.deepEqual(generateBalancedOpeningBook(), actualBook);
  assert.equal(verified.provenance.verified, true);
  assert.equal(verified.provenance.bookSha256, actualManifest.sha256);
  assert.equal(isVerifiedOpeningCorpus(verified.book, verified.provenance), true);
  assert.equal(Object.isFrozen(verified.book), true);
  assert.equal(Object.isFrozen(verified.book.openings[0].actionCodes), true);
  assert.equal(Object.isFrozen(verified.manifest.openings), true);
  assert.equal(Object.isFrozen(verified.provenance), true);
  assert.equal(isVerifiedOpeningCorpus(
    structuredClone(verified.book),
    structuredClone(verified.provenance)
  ), false);
  assert.equal(validateOpeningBook(actualBook).config.rows, 9);
  assert.throws(() => validateOpeningBook({
    ...actualBook,
    openings: [{ ...actualBook.openings[0], targetPlies: 3 }]
  }), /balanced 4–6-ply/);
  assert.throws(
    () => verifyOpeningArtifacts(bookText.replace('"actionCodes": [', '"actionCodes": [\n        0,'), manifestText),
    /differs from exact seeded generator output/
  );
  assert.throws(
    () => verifyOpeningArtifacts(bookText, manifestText.replace(actualManifest.sha256, '0'.repeat(64))),
    /differs from exact seeded generator output/
  );
  assert.throws(
    () => verifyOpeningArtifacts(
      bookText.replace(
        'normal-duel-balanced-opening-generator-1.0.0',
        'normal-duel-balanced-opening-generator-forged'
      ),
      manifestText
    ),
    /metadata does not identify this generator/
  );
  execFileSync(process.execPath, ['scripts/generate-normal-duel-balanced-openings.mjs', '--check'], {
    cwd: new URL('..', import.meta.url),
    stdio: 'pipe'
  });
});

test('match protocol adjudicates goal, threefold repetition, and ply cap', async () => {
  const goalEngineA = scriptedEngine('goal-a', goalAction);
  const goalEngineB = scriptedEngine('goal-b', goalAction);
  const goal = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'empty-goal', actionCodes: [] },
    engines: { A: goalEngineA, B: goalEngineB },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 1,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock()
  });
  assert.equal(goal.resultKind, 'win');
  assert.equal(goal.reason, 'goal');

  const repeatA = scriptedEngine('repeat-a', oscillatingAction);
  const repeatB = scriptedEngine('repeat-b', oscillatingAction);
  const repetition = await runMatch({
    config: normalDuelConfig({ size: 9 }),
    opening: { id: 'empty-repetition', actionCodes: [] },
    engines: { A: repeatA, B: repeatB },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 2,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock()
  });
  assert.equal(repetition.resultKind, 'draw');
  assert.equal(repetition.reason, 'threefold_repetition');
  assert.equal(repetition.finalPly, 8);

  const cap = await runMatch({
    config: normalDuelConfig({ size: 9, plyCap: 2 }),
    opening: { id: 'empty-cap', actionCodes: [] },
    engines: { A: repeatA, B: repeatB },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 3,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock()
  });
  assert.equal(cap.resultKind, 'draw');
  assert.equal(cap.reason, 'ply_cap');
  assert.equal(cap.finalPly, 2);
});

test('engine callbacks receive immutable copies and cannot corrupt authoritative play', async () => {
  const calls = [];
  const mutationAttempts = [];
  const malicious = Object.freeze({
    id: 'malicious',
    version: 'test-v1',
    capabilities: Object.freeze({ nodeBudget: true }),
    createSession(context) {
      for (const mutate of [
        () => context.openingHistory.push({}),
        () => { context.config.goalRows.A = 6; }
      ]) {
        try {
          mutate();
        } catch {
          mutationAttempts.push('blocked');
        }
      }
      return Object.freeze({
        selectAction(request) {
          calls.push(request.player);
          for (const mutate of [
            () => { request.state.position.turn = 'A'; },
            () => { request.state.position.pawns.A.r = 0; },
            () => { request.state.outcome.kind = 'win'; }
          ]) {
            try {
              mutate();
            } catch {
              mutationAttempts.push('blocked');
            }
          }
          return goalAction(request);
        },
        observe(transition) {
          for (const mutate of [
            () => { transition.stateAfter.position.turn = 'A'; },
            () => { transition.stateAfter.outcome.kind = 'win'; }
          ]) {
            try {
              mutate();
            } catch {
              mutationAttempts.push('blocked');
            }
          }
        },
        close() {}
      });
    }
  });
  const opponent = scriptedEngine('immutable-opponent', (request) => {
    calls.push(request.player);
    return goalAction(request);
  });
  const result = await runMatch({
    config: normalDuelConfig({ size: 7, plyCap: 2 }),
    opening: { id: 'immutable-boundary', actionCodes: [] },
    engines: { A: malicious, B: opponent },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 5,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock()
  });
  assert.deepEqual(calls, ['A', 'B']);
  assert.equal(result.resultKind, 'draw');
  assert.equal(result.reason, 'ply_cap');
  assert.equal(result.finalPly, 2);
  assert.ok(mutationAttempts.length >= 7);
});

test('session construction and validation errors forfeit the offending engine and clean up peers', async () => {
  let closed = false;
  const healthy = {
    id: 'healthy-session',
    version: 'test-v1',
    capabilities: { nodeBudget: true },
    createSession() {
      return {
        selectAction: goalAction,
        observe() {},
        close() { closed = true; }
      };
    }
  };
  const broken = {
    id: 'broken-session',
    version: 'test-v1',
    capabilities: { nodeBudget: true },
    createSession() {
      throw new Error('construction failed');
    }
  };
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'construction-forfeit', actionCodes: [] },
    engines: { A: healthy, B: broken },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 6,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock()
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.failedPlayer, 'B');
  assert.equal(result.winner, 'A');
  assert.equal(result.reason, 'crash');
  assert.equal(closed, true);
});

test('timing telemetry trusts the harness clock and labels engine timing as untrusted', async () => {
  const selfReporting = scriptedEngine('self-reporting', (request) => ({
    ...goalAction(request),
    stats: { nodes: 7, depth: 2, elapsedMs: 999_999 }
  }));
  const result = await runMatch({
    config: normalDuelConfig({ size: 7, plyCap: 1 }),
    opening: { id: 'trusted-timing', actionCodes: [] },
    engines: { A: selfReporting, B: scriptedEngine('timing-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 7,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock(2)
  });
  assert.deepEqual(result.telemetry.A.decisionMilliseconds, [2]);
  assert.deepEqual(result.telemetry.A.reportedDecisionMilliseconds, [999_999]);
});

test('null, crash, illegal, deadline, node-budget, and observer failures are losses', async () => {
  const opponent = scriptedEngine('opponent', goalAction);
  const cases = [
    ['null_action', scriptedEngine('null', () => null), STRENGTH_MODE, {}, tickingClock()],
    ['crash', scriptedEngine('crash', () => { throw new Error('boom'); }), STRENGTH_MODE, {}, tickingClock()],
    ['illegal_action', scriptedEngine('illegal', () => ({ kind: 'pawn', to: { r: 99, c: 99 } })), STRENGTH_MODE, {}, tickingClock()],
    ['deadline', scriptedEngine('slow', () => goalAction), STRENGTH_MODE, { perMoveDeadlineMs: 10 }, (() => {
      const values = [0, 11];
      return { now: () => values.shift() ?? 11 };
    })()],
    ['node_budget', scriptedEngine('budget', ({ state, config }) => ({
      ...goalAction({ state, config }),
      stats: { nodes: 101, depth: 1 }
    })), REGRESSION_MODE, { nodeBudget: 100 }, tickingClock()],
    ['crash', scriptedEngine('observer', goalAction, {
      async observe() { throw new Error('observer boom'); }
    }), REGRESSION_MODE, { nodeBudget: 100 }, tickingClock()]
  ];
  for (const [reason, offender, mode, options, clock] of cases) {
    const result = await runMatch({
      config: normalDuelConfig({ size: 7 }),
      opening: { id: `failure-${reason}-${offender.id}`, actionCodes: [] },
      engines: { A: offender, B: opponent },
      contenderSide: 'A',
      gameInPair: 0,
      seed: 10,
      mode,
      perMoveDeadlineMs: options.perMoveDeadlineMs ?? 900,
      nodeBudget: options.nodeBudget ?? null,
      clock
    });
    assert.equal(result.resultKind, 'forfeit', offender.id);
    assert.equal(result.reason, reason, offender.id);
    assert.equal(result.failedPlayer, 'A', offender.id);
    assert.equal(result.winner, 'B', offender.id);
  }
});

test('clock rollback fails closed instead of changing a decision deadline', async () => {
  const engineA = scriptedEngine('clock-a', goalAction);
  const engineB = scriptedEngine('clock-b', goalAction);
  const values = [5, 4];
  await assert.rejects(() => runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'clock-rollback', actionCodes: [] },
    engines: { A: engineA, B: engineB },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 99,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: 100,
    clock: { now: () => values.shift() ?? 4 }
  }), /clock moved backwards/);
});

test('enforced gate validation fixes strength, sample, isolation, and provenance requirements', async (t) => {
  const artifactDirectory = mkdtempSync(join(tmpdir(), 'wrongway-enforced-candidate-'));
  t.after(() => rmSync(artifactDirectory, { recursive: true, force: true }));
  let artifactIndex = 0;
  const isolated = async (id, options = {}) => {
    const artifact = writeCandidateArtifact(
      artifactDirectory,
      `candidate-${artifactIndex += 1}`,
      `
      export default {
        id: ${JSON.stringify(id)},
        version: 'test-v1',
        capabilities: {},
        createSession() {
          return { selectAction() {}, observe() {}, close() {} };
        }
      };
    `
    );
    return createWorkerEngineAdapter({
      ...options,
      moduleUrl: pathToFileURL(artifact.entryPath).href,
      candidateManifestPath: artifact.manifestPath,
      requireCanonicalMemoryIsolation:
        options.requireCanonicalMemoryIsolation ?? false
    });
  };
  const canonical = canonicalEnforcedCorpus();
  const contender = await isolated('isolated-contender');
  const baseline = await createPinnedHardWorkerAdapter();
  const valid = {
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    minimumOpeningPairs: MINIMUM_STRENGTH_OPENING_PAIRS,
    evaluatedOpeningCount: MINIMUM_STRENGTH_OPENING_PAIRS,
    evaluationSeed: CANONICAL_STRENGTH_SEED,
    contender,
    baseline,
    book: canonical.book,
    corpusProvenance: canonical.provenance
  };
  if (!isAuthenticatedCandidateAdapter(contender)
    || !isPinnedHardWorkerAdapter(baseline)) {
    assert.equal(
      contender.capabilities.memoryIsolation,
      'v8-old-space-only-ineligible-v1'
    );
    assert.equal(
      baseline.capabilities.memoryIsolation,
      'v8-old-space-only-ineligible-v1'
    );
    assert.throws(
      () => validateEnforcedStrengthOptions(valid),
      /canonical subprocess isolation/
    );
    await assert.rejects(
      isolated('canonical-isolation-required', {
        requireCanonicalMemoryIsolation: true
      }),
      /requires verified macOS .*taskpolicy support/
    );
    return;
  }
  const token = validateEnforcedStrengthOptions(valid);
  assert.equal(token.eligible, true);
  assert.equal(token.perMoveDeadlineMs, 900);
  assert.equal(token.configurationSha256, CANONICAL_STRENGTH_CONFIGURATION_SHA256);
  assert.equal(token.evaluationSeed, CANONICAL_STRENGTH_SEED);
  assert.equal(token.bookSha256, canonical.provenance.bookSha256);
  assert.equal(token.manifestSha256, canonical.provenance.manifestSha256);
  assert.equal(token.baselineId, 'hard-product-46a871c7');
  assert.equal(token.baselineVersion, 'wrongway-hard-product-baseline-v1');
  assert.equal(token.baselineSourceCommit,
    '46a871c7b061a33922bdb9c6d78355e2e9b6b607');
  assert.deepEqual(token.baselineTrustRoot, HARD_BASELINE_TRUST_ROOT);
  assert.equal(token.initializationCapMs, CANONICAL_STRENGTH_INITIALIZATION_CAP_MS);
  assert.equal(token.observerCapMs, CANONICAL_STRENGTH_OBSERVER_CAP_MS);
  assert.equal(token.activeTimeCharging, 'setup-observe-select-next-move-v1');
  assert.equal(token.integrityIsolation, 'node22-permission-readonly-subprocess-v1');
  assert.equal(token.stdioIsolation, 'null-device-v1');
  assert.equal(token.memoryIsolation, 'darwin-taskpolicy-rss-limit-v1');
  assert.equal(token.memoryLimitMiB, CANONICAL_ENGINE_MEMORY_LIMIT_MIB);
  assert.equal(token.v8OldSpaceMiB, CANONICAL_ENGINE_V8_OLD_SPACE_MIB);
  assert.equal(token.artifactIntegrity, 'content-addressed-hermetic-release-v1');
  assert.equal(token.moduleLoadIsolation, 'manifest-files-safe-builtins-v1');
  assert.equal(token.filesystemContentIsolation, 'manifest-files-only-v1');
  assert.equal(token.environmentIsolation, 'spawn-minimal-lang-c-utc-v1');
  assert.equal(token.networkIsolation, 'safe-builtins-no-network-db-v1');
  assert.equal(
    token.candidateArtifactProvenance.verification,
    'content-addressed-manifest-v1'
  );
  assert.equal(isAuthenticatedCandidateAdapter(contender), true);
  assert.equal(isPinnedHardWorkerAdapter(baseline), true);
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, mode: REGRESSION_MODE }),
    /requires monotonic-deadline strength mode/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, perMoveDeadlineMs: 899 }),
    /requires exactly 900 ms/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, evaluationSeed: 1 }),
    /requires canonical seed 1831565813/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, minimumOpeningPairs: 199 }),
    />= 200/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, evaluatedOpeningCount: 199 }),
    />= 200/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({
      ...valid,
      contender: {
        id: 'forged-isolation',
        version: 'test-v1',
        capabilities: { hardDeadlineIsolation: true },
        createSession() {}
      }
    }),
    /content-addressed candidate with canonical subprocess isolation/
  );
  const noncanonicalCaps = await isolated('noncanonical-caps', {
    initializationTimeoutMs: CANONICAL_STRENGTH_INITIALIZATION_CAP_MS + 1
  });
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, contender: noncanonicalCaps }),
    /content-addressed candidate with canonical subprocess isolation/
  );
  const unauthenticated = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(`
      export default {
        id: 'public-provenance-forgery',
        version: 'test-v1',
        candidateArtifactProvenance: ${JSON.stringify(
          getCandidateArtifactProvenance(contender)
        )},
        capabilities: {},
        createSession() {
          return { selectAction() {}, observe() {}, close() {} };
        }
      };
    `)}`
  });
  assert.equal(isAuthenticatedCandidateAdapter(unauthenticated), false);
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, contender: unauthenticated }),
    /content-addressed candidate with canonical subprocess isolation/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({
      ...valid,
      corpusProvenance: { ...canonical.provenance }
    }),
    /privately verified book/
  );
  assert.throws(
    () => validateEnforcedStrengthOptions({
      ...valid,
      book: {
        ...canonical.book,
        openings: canonical.book.openings.slice(0, 199)
      },
      evaluatedOpeningCount: 199
    }),
    /privately verified book/
  );

  const impostor = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(`
      export default {
        id: 'hard-product-46a871c7',
        version: 'wrongway-hard-product-baseline-v1',
        sourceCommit: '46a871c7b061a33922bdb9c6d78355e2e9b6b607',
        capabilities: {},
        createSession() {
          return { selectAction() {}, observe() {}, close() {} };
        }
      };
    `)}`
  });
  assert.throws(
    () => validateEnforcedStrengthOptions({ ...valid, baseline: impostor }),
    /privately branded pinned Hard subprocess baseline/
  );

  for (const alternate of [
    verifiedCorpus({ seed: 123, count: 12, size: 9, firstPlayer: 'A' }),
    verifiedCorpus({ seed: CANONICAL_STRENGTH_SEED, count: 12, size: 7, firstPlayer: 'A' }),
    verifiedCorpus({ seed: CANONICAL_STRENGTH_SEED, count: 12, size: 9, firstPlayer: 'B' })
  ]) {
    assert.throws(
      () => validateEnforcedStrengthOptions({
        ...valid,
        book: alternate.book,
        corpusProvenance: alternate.provenance,
        evaluatedOpeningCount: alternate.book.openings.length
      }),
      /canonical 9x9, first-player-A seeded corpus/
    );
  }
});

test('a forged public enforcement object cannot make fabricated winning results pass', () => {
  const contender = scriptedEngine('fabricated-contender', goalAction);
  const baseline = scriptedEngine('fabricated-baseline', goalAction);
  const emptyTelemetry = (engineId) => ({
    engineId,
    decisionMilliseconds: [],
    reportedDecisionMilliseconds: [],
    nodes: [],
    depths: [],
    actionMix: { pawn: 0, wall: 0 },
    antiStallReplacements: 0
  });
  const results = [];
  for (let pair = 0; pair < MINIMUM_STRENGTH_OPENING_PAIRS; pair += 1) {
    const pairId = `forged-${pair}`;
    results.push({
      pairId,
      contenderSide: 'A',
      engines: { A: contender.id, B: baseline.id },
      winner: 'A',
      resultKind: 'win',
      telemetry: {
        A: emptyTelemetry(contender.id),
        B: emptyTelemetry(baseline.id)
      }
    });
    results.push({
      pairId,
      contenderSide: 'B',
      engines: { A: baseline.id, B: contender.id },
      winner: 'B',
      resultKind: 'win',
      telemetry: {
        A: emptyTelemetry(baseline.id),
        B: emptyTelemetry(contender.id)
      }
    });
  }
  const summary = summarizeEvaluation(results, {
    contender,
    baseline,
    minimumOpeningPairs: MINIMUM_STRENGTH_OPENING_PAIRS,
    enforcement: { enforced: true, eligible: true }
  });
  assert.equal(summary.gate.criteriaMet, true);
  assert.equal(summary.gate.eligible, false);
  assert.equal(summary.gate.passed, false);
});

test('CLI rejects opening limits and noncanonical deadlines in enforced mode before loading engines', () => {
  for (const [args, message] of [
    [['--opening-limit', '200'], /--opening-limit is not allowed/],
    [['--deadline-ms', '899'], /requires --deadline-ms 900/],
    [['--seed', '1'], /requires --seed 1831565813/]
  ]) {
    const result = spawnSync(process.execPath, [
      'scripts/run-normal-duel-strength.mjs',
      '--candidate',
      './does-not-load.mjs',
      '--enforce-gate',
      ...args
    ], {
      cwd: new URL('..', import.meta.url),
      encoding: 'utf8'
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, message);
    assert.doesNotMatch(result.stderr, /does-not-load/);
  }
});

test('CLI requires an authenticated candidate manifest for enforced claims', () => {
  const result = spawnSync(process.execPath, [
    'scripts/run-normal-duel-strength.mjs',
    '--candidate',
    './does-not-load.mjs',
    '--enforce-gate'
  ], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8'
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /requires --candidate-manifest/);
  assert.doesNotMatch(result.stderr, /does-not-load/);
});

test('worker-backed strength decisions terminate a synchronously blocking engine', async () => {
  const blockingSource = `
    export default {
      id: 'blocking-worker',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() { while (true) {} },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const blocking = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(blockingSource)}`,
    initializationTimeoutMs: 1_000,
    observerTimeoutMs: 100
  });
  const startedAt = performance.now();
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'worker-hard-timeout', actionCodes: [] },
    engines: { A: blocking, B: scriptedEngine('worker-timeout-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 101,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: 50
  });
  const elapsed = performance.now() - startedAt;
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.reason, 'deadline');
  assert.equal(result.failedPlayer, 'A');
  assert.ok(elapsed < 1_000, `blocking worker should be terminated promptly, took ${elapsed} ms`);
});

test('candidate error codes cannot spoof a privately branded host deadline', async () => {
  const spoofSource = `
    export default {
      id: 'deadline-code-spoofer',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            const error = new Error('candidate deadline spoof');
            error.code = 'deadline_exceeded';
            throw error;
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const spoofer = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(spoofSource)}`
  });
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'deadline-code-spoof', actionCodes: [] },
    engines: { A: spoofer, B: scriptedEngine('deadline-spoof-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 200,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.failedPlayer, 'A');
  assert.equal(result.reason, 'crash');
  assert.equal(result.error, 'candidate deadline spoof');
});

test('candidate-selected exit 137 cannot spoof a private taskpolicy memory kill', async () => {
  const exitSource = `
    import process, { exit } from 'node:process';
    export default {
      id: 'memory-exit-spoofer',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            process.exitCode = 137;
            const probes = [
              () => exit(137),
              () => process.exit(137),
              () => process.reallyExit(137)
            ];
            let denied = 0;
            for (const probe of probes) {
              try { probe(); } catch (error) {
                if (error?.code === 'ERR_ACCESS_DENIED') denied += 1;
              }
            }
            if (denied !== probes.length) throw new Error('self-exit API escaped');
            throw new Error('self-exit 137 denied');
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const spoofer = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(exitSource)}`
  });
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'memory-exit-spoof', actionCodes: [] },
    engines: { A: spoofer, B: scriptedEngine('memory-spoof-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 207,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.failedPlayer, 'A');
  assert.equal(result.reason, 'crash');
  assert.equal(result.error, 'self-exit 137 denied');
});

test('subprocess session setup is charged against the canonical 900 ms move allowance', async () => {
  const slowSetupSource = `
    export default {
      id: 'slow-setup-subprocess',
      version: 'test-v1',
      capabilities: {},
      async createSession() {
        await new Promise((resolve) => setTimeout(resolve, 1_200));
        return {
          selectAction() {
            return { action: { kind: 'pawn', to: { r: 5, c: 3 } } };
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const slowSetup = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(slowSetupSource)}`
  });
  assert.equal(
    slowSetup.capabilities.initializationCapMs,
    CANONICAL_STRENGTH_INITIALIZATION_CAP_MS
  );
  const startedAt = performance.now();
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'setup-debt', actionCodes: [] },
    engines: { A: slowSetup, B: scriptedEngine('setup-debt-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 201,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  const elapsedMs = performance.now() - startedAt;
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.reason, 'deadline');
  assert.equal(result.failedPlayer, 'A');
  assert.ok(
    result.telemetry.A.chargedSetupMilliseconds[0] >= 800,
    `setup charge was ${result.telemetry.A.chargedSetupMilliseconds[0]} ms`
  );
  assert.equal(result.telemetry.A.chargedSelectMilliseconds[0], 0);
  assert.ok(elapsedMs < 1_600, `slow setup should be killed and reaped, took ${elapsedMs} ms`);
});

test('observer debt reduces the next subprocess decision allowance and uses parent timing', async () => {
  const observerDebtSource = `
    export default {
      id: 'observer-debt-subprocess',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          async selectAction() {
            await new Promise((resolve) => setTimeout(resolve, 300));
            return {
              action: { kind: 'pawn', to: { r: 1, c: 3 } },
              stats: { elapsedMs: 0 }
            };
          },
          async observe() {
            await new Promise((resolve) => setTimeout(resolve, 700));
          },
          close() {}
        };
      }
    };
  `;
  const observerDebt = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(observerDebtSource)}`
  });
  assert.equal(
    observerDebt.capabilities.observerCapMs,
    CANONICAL_STRENGTH_OBSERVER_CAP_MS
  );
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'observer-debt', actionCodes: [] },
    engines: { A: scriptedEngine('observer-debt-opponent', goalAction), B: observerDebt },
    contenderSide: 'B',
    gameInPair: 0,
    seed: 202,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  const charged = result.telemetry.B.decisionMilliseconds[0];
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.reason, 'deadline');
  assert.equal(result.failedPlayer, 'B');
  assert.ok(
    result.telemetry.B.chargedObserverMilliseconds[0] >= 600,
    `observer charge was ${result.telemetry.B.chargedObserverMilliseconds[0]} ms`
  );
  assert.ok(charged >= 850, `total trusted active-time charge was ${charged} ms`);
  assert.equal(result.telemetry.B.reportedDecisionMilliseconds.length, 0);
});

test('pinned Hard can make a healthy move after subprocess setup debt is applied', async () => {
  const hard = await createPinnedHardWorkerAdapter();
  const result = await runMatch({
    config: normalDuelConfig({ size: 7, plyCap: 1 }),
    opening: { id: 'healthy-hard-setup-debt', actionCodes: [] },
    engines: { A: hard, B: scriptedEngine('healthy-hard-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 203,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'draw');
  assert.equal(result.reason, 'ply_cap');
  assert.ok(result.telemetry.A.chargedSetupMilliseconds[0] > 0);
  assert.ok(result.telemetry.A.decisionMilliseconds[0]
    <= CANONICAL_STRENGTH_DEADLINE_MS);
});

test('taskpolicy memory profile kills external Buffer growth and reaps the engine', async () => {
  const allocationSource = `
    export default {
      id: 'external-memory-overage',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            const allocation = Buffer.alloc(112 * 1024 * 1024, 1);
            return allocation[0] === 1
              ? { kind: 'pawn', to: { r: 5, c: 3 } }
              : null;
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const engine = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(allocationSource)}`,
    memoryLimitMiB: 96,
    v8OldSpaceMiB: 64
  });
  if (engine.capabilities.memoryIsolation !== 'darwin-taskpolicy-rss-limit-v1') {
    assert.equal(
      engine.capabilities.memoryIsolation,
      'v8-old-space-only-ineligible-v1'
    );
    assert.equal(engine.capabilities.memoryLimitMiB, null);
    return;
  }
  assert.equal(engine.capabilities.memoryIsolation, 'darwin-taskpolicy-rss-limit-v1');
  assert.equal(engine.capabilities.memoryLimitMiB, 96);
  const startedAt = performance.now();
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'external-memory-overage', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('memory-overage-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 205,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.failedPlayer, 'A');
  assert.equal(result.reason, 'memory_limit');
  assert.ok(performance.now() - startedAt < 2_000);
});

test('terminal observer time is harvested into trusted telemetry before sessions close', async () => {
  const terminalObserverSource = `
    export default {
      id: 'terminal-observer-subprocess',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            return { action: { kind: 'pawn', to: { r: 0, c: 3 } } };
          },
          async observe() {
            await new Promise((resolve) => setTimeout(resolve, 120));
          },
          close() {}
        };
      }
    };
  `;
  const config = normalDuelConfig({ size: 7 });
  const openingPlay = play(config, [
    { r: 5, c: 3 }, { r: 0, c: 2 },
    { r: 4, c: 3 }, { r: 0, c: 1 },
    { r: 3, c: 3 }, { r: 0, c: 0 },
    { r: 2, c: 3 }, { r: 1, c: 0 },
    { r: 1, c: 3 }, { r: 2, c: 0 }
  ]);
  const terminalObserver = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(terminalObserverSource)}`
  });
  const result = await runMatch({
    config,
    opening: {
      id: 'terminal-observer',
      actionCodes: openingPlay.history.map(({ action }) => encodeAction(config, action))
    },
    engines: { A: terminalObserver, B: scriptedEngine('terminal-observer-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 204,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'win');
  assert.equal(result.winner, 'A');
  assert.ok(
    result.telemetry.A.chargedObserverMilliseconds.reduce((sum, value) => sum + value, 0) >= 100
  );
});

test('worker integrity permissions deny baseline writes, child processes, and nested workers while allowing WASM', async () => {
  const targetUrl = new URL('manifest.json', BASELINE_DIR);
  const targetPath = fileURLToPath(targetUrl);
  const before = readFileSync(targetUrl);
  const maliciousSource = `
    import { readFileSync, writeFileSync } from 'node:fs';
    import { spawnSync } from 'node:child_process';
    import { Worker } from 'node:worker_threads';

    export default {
      id: 'permission-attacker',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            let writeDenied = false;
            let childDenied = false;
            let workerDenied = false;
            try {
              const original = readFileSync(${JSON.stringify(targetPath)});
              writeFileSync(${JSON.stringify(targetPath)}, original);
            } catch (error) {
              writeDenied = error && error.code === 'ERR_ACCESS_DENIED';
            }
            try {
              const result = spawnSync(process.execPath, ['--version']);
              childDenied = result.error && result.error.code === 'ERR_ACCESS_DENIED';
            } catch (error) {
              childDenied = error && error.code === 'ERR_ACCESS_DENIED';
            }
            try {
              const nested = new Worker(
                new URL('data:text/javascript,while(true){}'),
                { type: 'module' }
              );
              nested.terminate();
            } catch (error) {
              workerDenied = error && error.code === 'ERR_ACCESS_DENIED';
            }
            let wasmAllowed = false;
            try {
              new WebAssembly.Module(Uint8Array.from([0,97,115,109,1,0,0,0]));
              wasmAllowed = true;
            } catch {}
            return writeDenied && childDenied && workerDenied && wasmAllowed
              ? { action: { kind: 'pawn', to: { r: 5, c: 3 } }, stats: { nodes: 1, depth: 1 } }
              : { action: { kind: 'pawn', to: { r: 99, c: 99 } }, stats: { nodes: 1, depth: 1 } };
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const attacker = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(maliciousSource)}`
  });
  assert.equal(
    attacker.capabilities.integrityIsolation,
    'node22-permission-readonly-subprocess-v1'
  );
  assert.equal(attacker.capabilities.stdioIsolation, 'null-device-v1');
  const result = await runMatch({
    config: normalDuelConfig({ size: 7, plyCap: 1 }),
    opening: { id: 'permission-integrity', actionCodes: [] },
    engines: { A: attacker, B: scriptedEngine('permission-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 102,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: 500
  });
  assert.equal(result.resultKind, 'draw');
  assert.equal(result.reason, 'ply_cap');
  assert.equal(sha256(readFileSync(targetUrl)), sha256(before));
  assert.equal(createPinnedHardBaseline().sourceCommit,
    '46a871c7b061a33922bdb9c6d78355e2e9b6b607');
});

test('strength CLI contains candidate output, signals, and private IPC', () => {
  const temporaryDirectory = mkdtempSync(join(tmpdir(), 'wrongway-cli-containment-'));
  const candidatePath = join(temporaryDirectory, 'candidate.mjs');
  const baselineManifest = new URL('manifest.json', BASELINE_DIR);
  const baselineSha256 = sha256(readFileSync(baselineManifest));
  try {
    writeFileSync(candidatePath, `
      import process, { kill } from 'node:process';
      import { writeSync } from 'node:fs';

      console.log('candidate-console-stdout');
      console.error('candidate-console-stderr');
      process.stdout.write('candidate-stream-stdout');
      process.stderr.write('candidate-stream-stderr');
      writeSync(1, 'candidate-fd-stdout');
      writeSync(2, 'candidate-fd-stderr');
      writeSync(1, 'O'.repeat(1_000_000));
      writeSync(2, 'E'.repeat(1_000_000));
      process._rawDebug('candidate-raw-debug');

      export default {
        id: 'output-signal-ipc-attacker',
        version: 'test-v1',
        capabilities: {},
        createSession() {
          return {
            selectAction() {
              const probes = [
                () => process.kill(process.ppid, 'SIGKILL'),
                () => kill(process.ppid, 'SIGKILL'),
                () => process._kill(process.ppid, 'SIGKILL'),
                () => process.abort(),
                () => process.send({ type: 'forged-response' }),
                () => process.disconnect()
              ];
              let denied = 0;
              for (const probe of probes) {
                try {
                  probe();
                } catch (error) {
                  if (error && error.code === 'ERR_ACCESS_DENIED') denied += 1;
                }
              }
              if (denied !== probes.length) {
                throw new Error('candidate containment probe was not denied');
              }
              throw new Error('contained-signal-and-ipc-ok');
            },
            observe() {},
            close() {}
          };
        }
      };
    `);
    const result = spawnSync(process.execPath, [
      'scripts/run-normal-duel-strength.mjs',
      '--candidate',
      candidatePath,
      '--mode',
      'strength',
      '--opening-limit',
      '1',
      '--minimum-opening-pairs',
      '1'
    ], {
      cwd: new URL('..', import.meta.url),
      encoding: 'utf8',
      timeout: 20_000
    });
    assert.equal(result.error, undefined);
    assert.equal(result.signal, null);
    assert.equal(result.status, 0, result.stderr);
    assert.equal(result.stderr, '');
    const report = JSON.parse(result.stdout);
    assert.equal(report.results.length, 2);
    assert.deepEqual(report.baselineProvenance.trustRoot, HARD_BASELINE_TRUST_ROOT);
    assert.equal(
      report.baselineProvenance.pinnedWorkerVerified,
      report.baselineIsolationProvenance.memoryIsolation
        === 'darwin-taskpolicy-rss-limit-v1'
    );
    assert.ok(
      report.results.some(({ error }) => error === 'contained-signal-and-ipc-ok'),
      'the candidate reached every denied signal/IPC probe'
    );
    assert.doesNotMatch(result.stdout, /candidate-(?:console|stream|fd|raw)/);
    assert.equal(sha256(readFileSync(baselineManifest)), baselineSha256);
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});

test('candidate artifact manifests bind labels, files, transitive loads, and every session', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-candidate-artifact-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const dependencySource = `export const release = 'content-addressed-release';\n`;
  const entrySource = `
    import { release } from './dependency.mjs';
    export default {
      id: 'claimed-human-label',
      version: release,
      capabilities: {},
      createSession() {
        return {
          selectAction() { return null; },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const artifact = writeCandidateArtifact(
    directory,
    'candidate',
    entrySource,
    { 'dependency.mjs': dependencySource }
  );
  const engine = await createWorkerEngineAdapter({
    moduleUrl: pathToFileURL(artifact.entryPath).href,
    candidateManifestPath: artifact.manifestPath,
    requireCanonicalMemoryIsolation: false
  });
  const provenance = getCandidateArtifactProvenance(engine);
  const canonicalIsolation =
    engine.capabilities.memoryIsolation === 'darwin-taskpolicy-rss-limit-v1';
  assert.equal(engine.id, 'claimed-human-label');
  assert.equal(engine.version, 'content-addressed-release');
  assert.equal(isAuthenticatedCandidateAdapter(engine), canonicalIsolation);
  assert.equal(provenance.verification, 'content-addressed-manifest-v1');
  assert.equal(provenance.manifestSha256, sha256ArtifactFile(artifact.manifestPath));
  assert.deepEqual(
    provenance.files.map(({ path }) => path),
    ['candidate.mjs', 'dependency.mjs']
  );

  const initial = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'artifact-initial', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('artifact-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 401,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(initial.reason, 'null_action');
  const reportBook = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const artifactReport = await runEvaluation({
    contender: engine,
    baseline: scriptedEngine('artifact-report-baseline', goalAction),
    book: { ...reportBook, openings: reportBook.openings.slice(0, 1) },
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    minimumOpeningPairs: 1
  });
  assert.equal(artifactReport.candidateArtifactVerified, canonicalIsolation);
  assert.deepEqual(artifactReport.candidateArtifactProvenance, provenance);
  assert.deepEqual(
    artifactReport.candidateIsolationProvenance,
    getWorkerEngineIsolationProvenance(engine)
  );
  assert.equal(
    artifactReport.candidateIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-game-v1'
  );
  assert.equal(
    artifactReport.candidateIsolationProvenance.sessionLifecycle,
    'stateful-session-per-game-v1'
  );

  const dependencyPath = artifact.dependencies['dependency.mjs'];
  writeFileSync(dependencyPath, `${dependencySource}// tampered after probe\n`);
  const dependencyTamper = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'artifact-dependency-tamper', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('artifact-opponent-2', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 402,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(dependencyTamper.reason, 'crash');
  assert.match(dependencyTamper.error, /artifact changed after adapter creation/);
  writeFileSync(dependencyPath, dependencySource);

  writeFileSync(artifact.entryPath, `${entrySource}// entry tamper\n`);
  const entryTamper = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'artifact-entry-tamper', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('artifact-opponent-3', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 403,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(entryTamper.reason, 'crash');
  assert.match(entryTamper.error, /artifact changed after adapter creation/);
  writeFileSync(artifact.entryPath, entrySource);

  const originalManifest = readFileSync(artifact.manifestPath, 'utf8');
  writeFileSync(artifact.manifestPath, `${originalManifest}\n`);
  const manifestTamper = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'artifact-manifest-tamper', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('artifact-opponent-4', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 404,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(manifestTamper.reason, 'crash');
  assert.match(manifestTamper.error, /manifest bytes changed/);
  writeFileSync(artifact.manifestPath, originalManifest);

  const lateUnlistedPath = join(dirname(artifact.manifestPath), 'late-unlisted.bin');
  writeFileSync(lateUnlistedPath, 'late mutation');
  const rootMutation = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'artifact-root-mutation', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('artifact-opponent-5', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 405,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(rootMutation.reason, 'crash');
  assert.match(rootMutation.error, /hermetic release contains an unlisted file/);
  rmSync(lateUnlistedPath);

  const unlisted = writeCandidateArtifact(
    directory,
    'unlisted',
    `import './unlisted-dependency.mjs';\n${entrySource}`,
    { 'unlisted-dependency.mjs': `export const ignored = true;\n` }
  );
  const entryRecord = unlisted.manifest.files.find((record) =>
    record.path === 'unlisted.mjs');
  writeFileSync(unlisted.manifestPath, `${JSON.stringify({
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    entry: 'unlisted.mjs',
    files: [entryRecord]
  }, null, 2)}\n`);
  await assert.rejects(
    createWorkerEngineAdapter({
      moduleUrl: pathToFileURL(unlisted.entryPath).href,
      candidateManifestPath: unlisted.manifestPath
    }),
    /hermetic release contains an unlisted file/i
  );

  for (const [name, manifest] of [
    ['traversal', {
      format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
      entry: '../candidate.mjs',
      files: [{ path: '../candidate.mjs', sha256: sha256ArtifactFile(artifact.entryPath) }]
    }],
    ['duplicate', {
      format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
      entry: 'candidate.mjs',
      files: [
        { path: 'candidate.mjs', sha256: sha256ArtifactFile(artifact.entryPath) },
        { path: 'candidate.mjs', sha256: sha256ArtifactFile(artifact.entryPath) }
      ]
    }]
  ]) {
    const manifestPath = join(directory, `${name}.manifest.json`);
    writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    await assert.rejects(
      createWorkerEngineAdapter({
        moduleUrl: pathToFileURL(artifact.entryPath).href,
        candidateManifestPath: manifestPath
      }),
      name === 'traversal' ? /canonical relative POSIX path/ : /strictly sorted/
    );
  }
});

test('authenticated candidate boundary sanitizes env and denies file/network escape', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-candidate-boundary-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const environmentName = 'WRONGWAY_CANDIDATE_BOUNDARY_SENTINEL';
  const priorEnvironment = process.env[environmentName];
  process.env[environmentName] = 'must-not-cross';
  t.after(() => {
    if (priorEnvironment === undefined) delete process.env[environmentName];
    else process.env[environmentName] = priorEnvironment;
  });

  const contained = writeCandidateArtifact(directory, 'contained', `
    import { readFileSync, writeSync } from 'node:fs';
    export default {
      id: 'authenticated-boundary-probe',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            if (process.env.${environmentName} !== undefined) {
              throw new Error('parent environment leaked');
            }
            new WebAssembly.Module(Uint8Array.from([0, 97, 115, 109, 1, 0, 0, 0]));
            for (const probe of [
              () => readFileSync('/etc/hosts'),
              () => writeSync(3, Buffer.from('forged-ipc'))
            ]) {
              let denied = false;
              try { probe(); } catch (error) {
                denied = error?.code === 'ERR_ACCESS_DENIED';
              }
              if (!denied) throw new Error('candidate boundary probe escaped');
            }
            return null;
          },
          observe() {},
          close() {}
        };
      }
    };
  `);
  const engine = await createWorkerEngineAdapter({
    moduleUrl: pathToFileURL(contained.entryPath).href,
    candidateManifestPath: contained.manifestPath
  });
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'authenticated-boundary', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('boundary-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 406,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.reason, 'null_action');

  const network = writeCandidateArtifact(directory, 'network', `
    import net from 'node:net';
    export default {
      id: String(net),
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return { selectAction() {}, observe() {}, close() {} };
      }
    };
  `);
  await assert.rejects(
    createWorkerEngineAdapter({
      moduleUrl: pathToFileURL(network.entryPath).href,
      candidateManifestPath: network.manifestPath
    }),
    /network module node:net/
  );

  for (const [stem, specifier, canonical] of [
    ['node-http-client', 'node:_http_client', 'node:_http_client'],
    ['bare-http-client', '_http_client', 'node:_http_client'],
    ['node-tls-wrap', 'node:_tls_wrap', 'node:_tls_wrap'],
    ['bare-tls-wrap', '_tls_wrap', 'node:_tls_wrap'],
    ['inspector-promises', 'node:inspector/promises', 'node:inspector/promises'],
    ['wasi', 'node:wasi', 'node:wasi'],
    ['sqlite', 'node:sqlite', 'node:sqlite'],
    ['child-process', 'node:child_process', 'node:child_process']
  ]) {
    const forbidden = writeCandidateArtifact(directory, stem, `
      import * as forbidden from ${JSON.stringify(specifier)};
      export default {
        id: String(forbidden),
        version: 'test-v1',
        capabilities: {},
        createSession() {
          return { selectAction() {}, observe() {}, close() {} };
        }
      };
    `);
    await assert.rejects(
      createWorkerEngineAdapter({
        moduleUrl: pathToFileURL(forbidden.entryPath).href,
        candidateManifestPath: forbidden.manifestPath
      }),
      (error) => {
        assert.match(error.message, /unapproved Node builtin/);
        assert.match(error.message, new RegExp(canonical.replace('/', '\\/')));
        return true;
      }
    );
  }
});

test('engine subprocess startup never inherits NODE_OPTIONS preload hooks', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-node-options-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const preloadPath = join(directory, 'preload.mjs');
  writeFileSync(
    preloadPath,
    'globalThis.__wrongwayUnmanifestedPreload = true;\n'
  );
  const priorNodeOptions = process.env.NODE_OPTIONS;
  process.env.NODE_OPTIONS = `--import=${pathToFileURL(preloadPath).href}`;
  t.after(() => {
    if (priorNodeOptions === undefined) delete process.env.NODE_OPTIONS;
    else process.env.NODE_OPTIONS = priorNodeOptions;
  });
  const source = `
    export default {
      id: globalThis.__wrongwayUnmanifestedPreload
        ? 'preload-observed'
        : 'minimal-env-clean',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return { selectAction() {}, observe() {}, close() {} };
      }
    };
  `;
  const engine = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(source)}`
  });
  assert.equal(engine.id, 'minimal-env-clean');
  assert.equal(
    getWorkerEngineIsolationProvenance(engine).environmentIsolation,
    'spawn-minimal-lang-c-utc-v1'
  );
});

test('subprocess descriptor normalization bounds identities and ignores unknown graphs', async () => {
  const hugeIdentitySource = `
    export default {
      id: 'x'.repeat(1_000_000),
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return { selectAction() {}, observe() {}, close() {} };
      }
    };
  `;
  await assert.rejects(
    createWorkerEngineAdapter({
      moduleUrl: `data:text/javascript,${encodeURIComponent(hugeIdentitySource)}`
    }),
    (error) => {
      assert.match(error.message, /descriptor\.id.*bounded string/);
      assert.ok(error.message.length < 256);
      return true;
    }
  );

  const accessorIdentitySource = `
    const descriptor = {
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return { selectAction() {}, observe() {}, close() {} };
      }
    };
    Object.defineProperty(descriptor, 'id', {
      enumerable: true,
      get() { throw new Error('getter-ran-' + 'x'.repeat(1_000_000)); }
    });
    export default descriptor;
  `;
  await assert.rejects(
    createWorkerEngineAdapter({
      moduleUrl: `data:text/javascript,${encodeURIComponent(accessorIdentitySource)}`
    }),
    (error) => {
      assert.match(error.message, /descriptor\.id must be an own data property/);
      assert.doesNotMatch(error.message, /getter-ran/);
      assert.ok(error.message.length < 256);
      return true;
    }
  );

  const hugeTrustRootSource = `
    export default {
      id: 'huge-trust-root',
      version: 'test-v1',
      baselineTrustRoot: {
        baselineId: 'x'.repeat(1_000_000),
        baselineVersion: 'v',
        sourceCommit: 'c',
        manifestSha256: 'm',
        gameLogicSha256: 'g',
        aiSha256: 'a',
        orchestrationSha256: 'o',
        originalIndexSha256: 'i'
      },
      capabilities: {},
      createSession() {
        return { selectAction() {}, observe() {}, close() {} };
      }
    };
  `;
  await assert.rejects(
    createWorkerEngineAdapter({
      moduleUrl: `data:text/javascript,${encodeURIComponent(hugeTrustRootSource)}`
    }),
    (error) => {
      assert.match(error.message, /baselineTrustRoot\.baselineId.*bounded string/);
      assert.ok(error.message.length < 256);
      return true;
    }
  );

  const unknownGraphSource = `
    const future = {};
    future.self = future;
    export default {
      id: 'unknown-graph-compatible',
      version: 'test-v1',
      futureDescriptorData: future,
      capabilities: { nodeBudget: false, futureCapability: future },
      createSession() {
        return {
          selectAction() {
            return {
              action: { kind: 'pawn', to: { r: 5, c: 3 } },
              stats: { depth: 1, futureTelemetry: future }
            };
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const compatible = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(unknownGraphSource)}`
  });
  const compatibleResult = await runMatch({
    config: normalDuelConfig({ size: 7, plyCap: 1 }),
    opening: { id: 'unknown-graph-compatible', actionCodes: [] },
    engines: { A: compatible, B: scriptedEngine('unknown-graph-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 301,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(compatibleResult.resultKind, 'draw');
  assert.equal(compatibleResult.reason, 'ply_cap');
  assert.deepEqual(compatible.capabilities.nodeBudget, false);
});

test('subprocess decision and error normalization contains cycles, proxies, and huge errors', async () => {
  const cases = [
    {
      id: 'cyclic-decision',
      body: `
        const result = {
          action: { kind: 'pawn', to: { r: 5, c: 3 } },
          stats: {}
        };
        result.loop = result;
        result.huge = 'x'.repeat(1_000_000);
        return result;
      `,
      expected: /decision contains an unsupported property/
    },
    {
      id: 'proxy-decision',
      body: `
        return new Proxy({}, {
          ownKeys() { throw new Error('proxy-trap-' + 'x'.repeat(1_000_000)); }
        });
      `,
      expected: /decision could not be inspected safely/
    },
    {
      id: 'accessor-decision',
      body: `
        const result = { stats: {} };
        Object.defineProperty(result, 'action', {
          enumerable: true,
          get() { throw new Error('action-getter-ran-' + 'x'.repeat(1_000_000)); }
        });
        return result;
      `,
      expected: /decision\.action must be an own data property/
    },
    {
      id: 'huge-thrown-error',
      body: `throw new Error('Z'.repeat(2_000_000));`,
      expected: /^Z{100}/
    }
  ];
  for (const entry of cases) {
    const source = `
      export default {
        id: ${JSON.stringify(entry.id)},
        version: 'test-v1',
        capabilities: {},
        createSession() {
          return {
            selectAction() { ${entry.body} },
            observe() {},
            close() {}
          };
        }
      };
    `;
    const engine = await createWorkerEngineAdapter({
      moduleUrl: `data:text/javascript,${encodeURIComponent(source)}`
    });
    const result = await runMatch({
      config: normalDuelConfig({ size: 7 }),
      opening: { id: entry.id, actionCodes: [] },
      engines: { A: engine, B: scriptedEngine(`${entry.id}-opponent`, goalAction) },
      contenderSide: 'A',
      gameInPair: 0,
      seed: 302,
      mode: STRENGTH_MODE,
      perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
    });
    assert.equal(result.resultKind, 'forfeit', entry.id);
    assert.equal(result.reason, 'crash', entry.id);
    assert.match(result.error, entry.expected, entry.id);
    assert.doesNotMatch(result.error, /getter-ran|proxy-trap/, entry.id);
    assert.ok(result.error.length <= 1_024, `${entry.id} error was not bounded`);
  }
});

test('captured runtime intrinsics keep IPC normalization strict after monkeypatching', async () => {
  const monkeypatchSource = `
    Object.getOwnPropertyDescriptor = () => ({ value: null });
    Object.getOwnPropertyDescriptors = () => ({});
    Object.getPrototypeOf = () => null;
    Object.hasOwn = () => true;
    Object.freeze = (value) => value;
    Object.values = () => [];
    Number.isFinite = () => true;
    Number.isSafeInteger = () => true;
    JSON.stringify = () => '{}';
    Buffer.byteLength = () => 0;
    String.prototype.slice = function slice() { return this; };
    RegExp.prototype.test = () => true;
    Set.prototype.has = () => false;
    Set.prototype.add = function add() { return this; };
    Object.prototype.toJSON = () => 'forged';

    export default {
      id: 'intrinsic-monkeypatch',
      version: 'test-v1',
      capabilities: {},
      createSession() {
        return {
          selectAction() {
            return {
              action: {
                kind: 'pawn',
                to: { r: 5, c: 3, payload: 'x'.repeat(1_000_000) }
              }
            };
          },
          observe() {},
          close() {}
        };
      }
    };
  `;
  const engine = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(monkeypatchSource)}`
  });
  const result = await runMatch({
    config: normalDuelConfig({ size: 7 }),
    opening: { id: 'intrinsic-monkeypatch', actionCodes: [] },
    engines: { A: engine, B: scriptedEngine('intrinsic-monkeypatch-opponent', goalAction) },
    contenderSide: 'A',
    gameInPair: 0,
    seed: 303,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.reason, 'crash');
  assert.match(result.error, /pawn action\.to contains an unsupported property/);
  assert.ok(result.error.length < 256);
});

test('evaluation pairs every opening across sides and reports strength, telemetry, and paired CIs', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 2) };
  const verified = verifyOpeningArtifacts(
    readFileSync(BOOK_URL, 'utf8'),
    readFileSync(MANIFEST_URL, 'utf8')
  );
  const contender = scriptedEngine('candidate', goalAction);
  const baseline = scriptedEngine('baseline', goalAction);
  const report = await runEvaluation({
    contender,
    baseline,
    book: smokeBook,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    minimumOpeningPairs: 200,
    corpusProvenance: verified.provenance,
    clockFactory: () => tickingClock()
  });
  assert.equal(report.results.length, 4);
  assert.equal(report.summary.openingPairs, 2);
  assert.equal(report.results[0].settings.mode, REGRESSION_MODE);
  assert.equal(report.results[0].settings.nodeBudget, 100);
  assert.equal(Number.isSafeInteger(report.results[0].settings.seed), true);
  assert.equal(report.summary.sideSplits.A.games, 2);
  assert.equal(report.summary.sideSplits.B.games, 2);
  assert.equal(report.summary.wins + report.summary.losses + report.summary.draws, 4);
  assert.equal(report.summary.telemetry.contender.nodes.reportedDecisions,
    report.summary.telemetry.contender.decisions);
  assert.equal(report.summary.telemetry.contender.timing.source, 'caller-supplied-clock-time');
  assert.equal(
    report.summary.telemetry.contender.timing.clockProfileId,
    CALLER_SUPPLIED_CLOCK_PROFILE.id
  );
  assert.ok(report.summary.telemetry.contender.untrustedSelfReportedTiming.reportedDecisions > 0);
  assert.ok(report.summary.telemetry.contender.actionMix.pawn > 0);
  assert.equal(report.summary.gate.sampleSizeMet, false);
  assert.equal(report.summary.gate.passed, false);
  assert.equal(report.summary.confidenceIntervals.winRate.clusters, 2);
  assert.equal(report.summary.confidenceIntervals.score.clusters, 2);
  assert.equal(report.corpusProvenance.bookSha256, verified.provenance.bookSha256);
  assert.equal(report.corpusProvenance.boundToEvaluatedBook, false);
  assert.deepEqual(report.book.evaluatedOpeningIds,
    smokeBook.openings.map((opening) => opening.id));
  assert.equal(report.book.openingCount, 2);
  assert.equal(report.book.configurationSha256, CANONICAL_STRENGTH_CONFIGURATION_SHA256);
  assert.equal(report.enforcement.eligible, false);
  assert.equal(report.baselineProvenance.id, baseline.id);
  assert.deepEqual(
    pairedClusterConfidenceIntervals(report.results, contender.id),
    report.summary.confidenceIntervals
  );
});

test('canonical 9x9 default regression evaluations repeat actions, outcomes, and telemetry', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 1) };

  async function evaluate() {
    const actions = [];
    const contender = scriptedEngine('deterministic-candidate', goalAction, {
      observe(transition, context) {
        actions.push({
          gameId: context.gameId,
          player: transition.player,
          actionCode: encodeAction(context.config, transition.action)
        });
      }
    });
    const report = await runEvaluation({
      contender,
      baseline: scriptedEngine('deterministic-baseline', goalAction),
      book: smokeBook,
      mode: REGRESSION_MODE,
      nodeBudget: 100,
      minimumOpeningPairs: 1
    });
    return { actions, report };
  }

  const first = await evaluate();
  const second = await evaluate();
  assert.deepEqual(second.actions, first.actions);
  assert.deepEqual(second.report.results, first.report.results);
  assert.deepEqual(second.report.summary, first.report.summary);
  assert.deepEqual(first.report.clockProfile, DETERMINISTIC_REGRESSION_CLOCK_PROFILE);
  assert.equal(
    first.report.summary.telemetry.contender.timing.source,
    'deterministic-logical-clock-time'
  );
  assert.equal(
    first.report.summary.telemetry.contender.timing.clockProfileId,
    DETERMINISTIC_REGRESSION_CLOCK_PROFILE.id
  );
  for (const result of first.report.results) {
    assert.deepEqual(result.settings.clockProfile, DETERMINISTIC_REGRESSION_CLOCK_PROFILE);
  }
});

test('canonical 9x9 pinned Hard near-cutoff roots repeat actions, results, and telemetry', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));

  async function evaluate(opening, hardSide) {
    const actions = [];
    const opponent = scriptedEngine('bounded-hard-opponent', () => null, {
      observe(transition) {
        actions.push({
          player: transition.player,
          actionCode: encodeAction(book.config, transition.action)
        });
      }
    });
    const opponentSide = hardSide === 'A' ? 'B' : 'A';
    const result = await runMatch({
      config: book.config,
      opening,
      engines: {
        [hardSide]: createPinnedHardBaseline(),
        [opponentSide]: opponent
      },
      contenderSide: hardSide,
      gameInPair: 0,
      seed: CANONICAL_STRENGTH_SEED,
      mode: REGRESSION_MODE,
      nodeBudget: 100
    });
    return { actions, result };
  }

  for (const opening of [book.openings[1], book.openings[2]]) {
    const hardSide = opening.targetPlies % 2 === 0 ? 'A' : 'B';
    const opponentSide = hardSide === 'A' ? 'B' : 'A';
    const first = await evaluate(opening, hardSide);
    const second = await evaluate(opening, hardSide);
    assert.equal(first.actions.length, 1, `${opening.id} bounds play after one Hard root`);
    assert.deepEqual(second.actions, first.actions, opening.id);
    assert.deepEqual(second.result, first.result, opening.id);
    assert.equal(first.result.resultKind, 'forfeit', opening.id);
    assert.equal(first.result.failedPlayer, opponentSide, opening.id);
    assert.deepEqual(
      first.result.settings.clockProfile,
      DETERMINISTIC_REGRESSION_CLOCK_PROFILE,
      opening.id
    );
    const elapsed = first.result.telemetry[hardSide].decisionMilliseconds[0];
    assert.ok(elapsed >= 700, `${opening.id} logical elapsed ${elapsed} ms reached cutoff`);
    assert.ok(elapsed <= 760, `${opening.id} logical elapsed ${elapsed} ms stayed cutoff-bound`);
    assert.equal(elapsed % 4, 0, opening.id);
    assert.equal(
      first.result.telemetry[hardSide].clockProfileId,
      DETERMINISTIC_REGRESSION_CLOCK_PROFILE.id,
      opening.id
    );
    assert.equal(
      first.result.telemetry[hardSide].timingSource,
      'deterministic-logical-clock-time',
      opening.id
    );
  }
});

test('canonical 9x9 pinned Hard reports a regression real safety-ceiling overrun distinctly', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const opening = book.openings[1];
  const hardSide = opening.targetPlies % 2 === 0 ? 'A' : 'B';
  const opponentSide = hardSide === 'A' ? 'B' : 'A';
  const result = await runMatch({
    config: book.config,
    opening,
    engines: {
      [hardSide]: createPinnedHardBaseline(),
      [opponentSide]: scriptedEngine('safety-ceiling-opponent', () => null)
    },
    contenderSide: hardSide,
    gameInPair: 0,
    seed: CANONICAL_STRENGTH_SEED,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: { now: () => 0 },
    clockProfile: {
      id: 'test-stalled-regression-clock-v1',
      source: 'test-stalled-logical-clock',
      deterministic: true,
      realSafetyCeilingMilliseconds: 1
    }
  });
  assert.equal(result.resultKind, 'forfeit');
  assert.equal(result.failedPlayer, hardSide);
  assert.equal(result.reason, 'clock_profile_exceeded');
  assert.match(result.error, /regression clock-profile safety ceiling/);
});

test('canonical 9x9 regression rejects worker adapters before play', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 1) };
  const worker = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(`
      export default {
        id: 'regression-worker',
        version: 'test-v1',
        capabilities: { nodeBudget: true },
        createSession() {
          return {
            selectAction() { return null; },
            observe() {},
            close() {}
          };
        }
      };
    `)}`
  });
  let sessionsStarted = 0;
  const inProcess = scriptedEngine('regression-in-process', goalAction);
  const countedInProcess = Object.freeze({
    ...inProcess,
    createSession(context) {
      sessionsStarted += 1;
      return inProcess.createSession(context);
    }
  });
  await assert.rejects(runEvaluation({
    contender: worker,
    baseline: countedInProcess,
    book: smokeBook,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    minimumOpeningPairs: 1
  }), /fixed-node regression mode requires in-process engine adapters/);
  assert.equal(sessionsStarted, 0);
});

test('deterministic regression clock factory gives every canonical 9x9 game a fresh clock', () => {
  const first = deterministicRegressionClockFactory();
  const second = deterministicRegressionClockFactory();
  assert.deepEqual(
    [first.now(), first.now(), first.now()],
    [0, 4, 8]
  );
  assert.deepEqual(
    [second.now(), second.now()],
    [0, 4]
  );
});

test('direct canonical 9x9 matches resolve mode-aware defaults and explicit clock provenance', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const opening = book.openings[0];
  const matchOptions = {
    config: book.config,
    opening,
    engines: {
      A: scriptedEngine('direct-clock-a', goalAction),
      B: scriptedEngine('direct-clock-b', () => null)
    },
    contenderSide: 'A',
    gameInPair: 0,
    seed: CANONICAL_STRENGTH_SEED
  };
  const strength = await runMatch({
    ...matchOptions,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS
  });
  assert.deepEqual(strength.settings.clockProfile, REAL_MONOTONIC_CLOCK_PROFILE);
  assert.equal(strength.telemetry.A.timingSource, 'trusted-harness-active-time');
  assert.equal(strength.telemetry.A.clockProfileId, REAL_MONOTONIC_CLOCK_PROFILE.id);

  const explicit = await runMatch({
    ...matchOptions,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clock: tickingClock(2)
  });
  assert.deepEqual(explicit.settings.clockProfile, CALLER_SUPPLIED_CLOCK_PROFILE);
  assert.equal(explicit.telemetry.A.timingSource, 'caller-supplied-clock-time');

  await assert.rejects(runMatch({
    ...matchOptions,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    clockProfile: { id: 'profile-without-clock' }
  }), /clockProfile requires an explicit clock/);
});

test('canonical 9x9 explicit clock overrides retain caller-supplied provenance', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 1) };
  const profile = Object.freeze({
    id: 'test-caller-logical-clock-v1',
    source: 'normal-duel-strength-test',
    deterministic: true,
    tickMilliseconds: 2
  });
  let clocksCreated = 0;
  const report = await runEvaluation({
    contender: scriptedEngine('explicit-clock-candidate', goalAction),
    baseline: scriptedEngine('explicit-clock-baseline', goalAction),
    book: smokeBook,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    minimumOpeningPairs: 1,
    clockFactory() {
      clocksCreated += 1;
      return tickingClock(2);
    },
    clockProfile: profile
  });
  assert.equal(clocksCreated, 2);
  assert.deepEqual(report.clockProfile, profile);
  for (const result of report.results) {
    assert.deepEqual(result.settings.clockProfile, profile);
  }

  const unprofiled = await runEvaluation({
    contender: scriptedEngine('unprofiled-clock-candidate', goalAction),
    baseline: scriptedEngine('unprofiled-clock-baseline', goalAction),
    book: smokeBook,
    mode: REGRESSION_MODE,
    nodeBudget: 100,
    minimumOpeningPairs: 1,
    clockFactory: () => tickingClock(2)
  });
  assert.deepEqual(unprofiled.clockProfile, CALLER_SUPPLIED_CLOCK_PROFILE);
});

test('enforced canonical 9x9 evaluation rejects clock overrides before play', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 1) };
  let sessionsStarted = 0;
  function neverStartedEngine(id) {
    const engine = scriptedEngine(id, goalAction);
    return Object.freeze({
      ...engine,
      createSession(context) {
        sessionsStarted += 1;
        return engine.createSession(context);
      }
    });
  }
  const common = {
    contender: neverStartedEngine('enforced-clock-candidate'),
    baseline: neverStartedEngine('enforced-clock-baseline'),
    book: smokeBook,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    enforceGate: true
  };
  await assert.rejects(runEvaluation({
    ...common,
    clockFactory: () => tickingClock()
  }), /enforced gate requires the default real monotonic clock/);
  await assert.rejects(runEvaluation({
    ...common,
    clockProfile: { id: 'forged-enforced-clock-profile' }
  }), /enforced gate requires the default real monotonic clock/);
  assert.equal(sessionsStarted, 0);
});

const WASM_CANDIDATE_TEMPLATE_PATH = fileURLToPath(
  new URL('../scripts/evaluation/normal-duel-wasm-candidate-adapter.mjs', import.meta.url)
);
const WASM_CANDIDATE_TEMPLATE_SOURCE = readFileSync(WASM_CANDIDATE_TEMPLATE_PATH);
const WASM_MAGIC_BYTES = Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]);
// Stands in for the generated wasm-bindgen module. `nodes` reports how many
// searches this process has served, so a per-decision child always reports 1
// while a reused per-game child counts up. The regression `nodeBudget` doubles
// as a fault-injection channel that needs no filesystem or environment access.
const STUB_WASM_SOURCE = Buffer.from(`
let calls = 0;

function respond(control) {
  calls += 1;
  if (control === 2) throw new Error('stub wasm select failure');
  if (control === 3) { for (;;) {} }
  if (control === 4) {
    const retained = [];
    for (let index = 0; index < 32; index += 1) {
      retained.push(Buffer.alloc(64 * 1024 * 1024, 1));
    }
    throw new Error('stub wasm retained ' + retained.length + ' blocks without a memory kill');
  }
  return JSON.stringify({
    actionCode: 1,
    action: { kind: 'pawn', to: { r: 7, c: 4 } },
    nodes: calls,
    completedDepth: 1
  });
}

export default async function initialize() {
  return {};
}

export function normalDuelVersion() {
  return 'stub-wasm-v1';
}

export function normalDuelSearchNodes(payload) {
  return respond(JSON.parse(payload).nodeBudget);
}

export function normalDuelSearchFor(payload) {
  JSON.parse(payload);
  return respond(undefined);
}
`, 'utf8');

function writeRecycledWasmRelease(parentDirectory, name, {
  entrySource = WASM_CANDIDATE_TEMPLATE_SOURCE,
  entryPath = 'adapter.mjs'
} = {}) {
  const releaseDirectory = join(parentDirectory, name);
  mkdirSync(releaseDirectory, { recursive: true });
  const records = [
    { path: entryPath, source: entrySource },
    { path: 'normal-duel-wasm.mjs', source: STUB_WASM_SOURCE },
    { path: 'normal-duel-wasm_bg.wasm', source: WASM_MAGIC_BYTES }
    // Manifests are sorted by code unit, not by locale collation.
  ].sort((left, right) => (left.path < right.path ? -1 : 1));
  for (const record of records) {
    writeFileSync(join(releaseDirectory, record.path), record.source);
  }
  const manifestPath = join(releaseDirectory, 'manifest.json');
  writeFileSync(manifestPath, `${JSON.stringify({
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    entry: entryPath,
    files: records.map((record) => ({ path: record.path, sha256: sha256(record.source) }))
  }, null, 2)}\n`);
  return {
    releaseDirectory,
    entryPath: join(releaseDirectory, entryPath),
    manifestPath
  };
}

async function recycledWasmCandidateBase(parentDirectory, name, options = {}) {
  const release = writeRecycledWasmRelease(parentDirectory, name, options);
  const base = await createWorkerEngineAdapter({
    moduleUrl: pathToFileURL(release.entryPath).href,
    candidateManifestPath: release.manifestPath
  });
  return { release, base };
}

async function authenticatedStandaloneCandidate(parentDirectory, name, descriptorSource) {
  const releaseParent = join(parentDirectory, name);
  mkdirSync(releaseParent, { recursive: true });
  const artifact = writeCandidateArtifact(releaseParent, 'adapter', `
    export async function createEngineAdapter() {
      return ${descriptorSource};
    }
  `);
  return createWorkerEngineAdapter({
    moduleUrl: pathToFileURL(artifact.entryPath).href,
    candidateManifestPath: artifact.manifestPath
  });
}

function recycledWasmSessionContext(config) {
  return Object.freeze({
    protocol: 'wrongway-normal-duel-strength-v1',
    mode: STRENGTH_MODE,
    gameId: 'recycled-wasm/0',
    side: 'A',
    config,
    seed: 11,
    nodeBudget: null,
    clockProfile: REAL_MONOTONIC_CLOCK_PROFILE,
    openingHistory: []
  });
}

function recycledWasmRequest(config, state, {
  mode = REGRESSION_MODE,
  nodeBudget = 1,
  wallClockBudgetMs = CANONICAL_STRENGTH_DEADLINE_MS
} = {}) {
  return Object.freeze({
    protocol: 'wrongway-normal-duel-strength-v1',
    mode,
    config,
    state,
    player: 'A',
    seed: 13,
    limits: Object.freeze({
      nodeBudget: mode === REGRESSION_MODE ? nodeBudget : null,
      deadlineAtMs: null,
      wallClockBudgetMs
    })
  });
}

function engineChildProcessIds() {
  const listed = spawnSync('/usr/bin/pgrep', ['-P', String(process.pid)], {
    encoding: 'utf8'
  });
  if (listed.error || listed.status === null || listed.status > 1) return null;
  return new Set(listed.stdout.split('\n').map((line) => line.trim()).filter(Boolean));
}

function assertNoSurvivingChildren(before, label) {
  if (before === null) return;
  const after = engineChildProcessIds();
  if (after === null) return;
  const survivors = [...after].filter((pid) => !before.has(pid));
  assert.deepEqual(survivors, [], `${label} left engine subprocesses alive: ${survivors}`);
}

// Canonical memory containment is a host property: only macOS with a verified
// /usr/sbin/taskpolicy provides it, and only then can a worker be eligible for
// enforced claims. Content-addressed artifact authentication is not a host
// property, so the recycler lifecycle below runs everywhere and only assertions
// specifically about canonical-worker eligibility read this.
function canonicalMemoryEligible(engine) {
  return engine.capabilities.memoryIsolation === 'darwin-taskpolicy-rss-limit-v1';
}

test('recycled normal-duel WASM admission refuses anything but the authenticated official release', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-admission-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base } = await recycledWasmCandidateBase(directory, 'genuine');
  assert.equal(base.id, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(base.capabilities.nodeBudget, true);
  assert.equal(base.capabilities.deadline, true);
  const canonicalMemory = canonicalMemoryEligible(base);
  // Enforced eligibility tracks the host; admission does not.
  assert.equal(isAuthenticatedCandidateAdapter(base), canonicalMemory);
  assert.equal(
    base.capabilities.memoryIsolation,
    canonicalMemory ? 'darwin-taskpolicy-rss-limit-v1' : 'v8-old-space-only-ineligible-v1'
  );

  for (const forged of [
    undefined,
    null,
    {},
    Object.freeze({
      id: NORMAL_DUEL_WASM_CANDIDATE_ID,
      version: base.version,
      capabilities: Object.freeze({ ...base.capabilities }),
      candidateArtifactProvenance: getCandidateArtifactProvenance(base),
      createSession() {}
    }),
    new Proxy(base, {})
  ]) {
    await assert.rejects(
      createRecycledNormalDuelWasmCandidateAdapter(forged),
      /authenticated official candidate/
    );
  }

  const unauthenticated = await createWorkerEngineAdapter({
    moduleUrl: `data:text/javascript,${encodeURIComponent(`
      export default {
        id: ${JSON.stringify(NORMAL_DUEL_WASM_CANDIDATE_ID)},
        version: 'stub-wasm-v1',
        capabilities: { nodeBudget: true, deadline: true },
        createSession() {
          return { selectAction() { return null; }, observe() {}, close() {} };
        }
      };
    `)}`
  });
  assert.equal(isAuthenticatedCandidateAdapter(unauthenticated), false);
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(unauthenticated),
    /authenticated official candidate/
  );

  const wrongId = await authenticatedStandaloneCandidate(directory, 'wrong-id', `{
    id: 'wrongway-normal-duel-wasm-search-lookalike',
    version: 'stub-wasm-v1',
    capabilities: { nodeBudget: true, deadline: true },
    createSession() {
      return { selectAction() { return null; }, observe() {}, close() {} };
    }
  }`);
  assert.equal(isAuthenticatedCandidateAdapter(wrongId), canonicalMemory);
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(wrongId),
    /exact candidate id wrongway-normal-duel-wasm-search/
  );

  const missingCapability = await authenticatedStandaloneCandidate(directory, 'no-deadline', `{
    id: ${JSON.stringify(NORMAL_DUEL_WASM_CANDIDATE_ID)},
    version: 'stub-wasm-v1',
    capabilities: { nodeBudget: true },
    createSession() {
      return { selectAction() { return null; }, observe() {}, close() {} };
    }
  }`);
  assert.equal(isAuthenticatedCandidateAdapter(missingCapability), canonicalMemory);
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(missingCapability),
    /nodeBudget and deadline capabilities/
  );

  const relabeled = await recycledWasmCandidateBase(directory, 'relabeled-entry', {
    entryPath: 'engine.mjs'
  });
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(relabeled.base),
    /authenticated release entry must be adapter\.mjs/
  );

  const impostorSource = Buffer.concat([
    WASM_CANDIDATE_TEMPLATE_SOURCE,
    Buffer.from('\n// byte-level impostor of the checked-in candidate template\n')
  ]);
  const impostor = await recycledWasmCandidateBase(directory, 'impostor', {
    entrySource: impostorSource
  });
  assert.equal(impostor.base.id, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(impostor.base.capabilities.deadline, true);
  assert.equal(isAuthenticatedCandidateAdapter(impostor.base), canonicalMemory);
  assert.notEqual(
    sha256ArtifactFile(impostor.release.entryPath),
    sha256ArtifactFile(WASM_CANDIDATE_TEMPLATE_PATH)
  );
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(impostor.base),
    /adapter\.mjs SHA-256 does not equal the checked-in/
  );

  const wrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);
  assert.equal(wrapper.capabilities.isolation, 'node-subprocess-per-decision-v1');

  // Recursive wrapping is refused: the already recycled wrapper is itself an
  // authenticated official adapter carrying the same private artifact binding,
  // so only its private per-decision lifecycle can tell it apart from a base.
  assert.equal(
    getWorkerEngineIsolationProvenance(wrapper).sessionLifecycle,
    'stateless-wasm-per-decision-v1'
  );
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(wrapper),
    /not an already recycled per-decision wrapper/
  );
  // Re-publishing the per-game public labels buys nothing, because a forged
  // object holds none of the private bindings admission actually reads.
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(new Proxy(wrapper, {
      get(target, property, receiver) {
        if (property === 'capabilities') {
          return Object.freeze({
            ...target.capabilities,
            isolation: 'node-subprocess-per-game-v1',
            sessionLifecycle: 'stateful-session-per-game-v1'
          });
        }
        return Reflect.get(target, property, receiver);
      }
    })),
    /authenticated official candidate/
  );
  // Wrapping the per-game base a second time stays legal: only nesting a
  // wrapper inside a wrapper is refused.
  const secondWrapperOfBase = await createRecycledNormalDuelWasmCandidateAdapter(base);
  assert.notEqual(secondWrapperOfBase, wrapper);
  assert.equal(
    secondWrapperOfBase.capabilities.sessionLifecycle,
    'stateless-wasm-per-decision-v1'
  );
});

test('recycled normal-duel WASM wrapper keeps private brands and reports per-decision lifecycle', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-brands-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base, release } = await recycledWasmCandidateBase(directory, 'genuine');
  const canonicalMemory = canonicalMemoryEligible(base);
  const wrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);

  // Wrapping neither grants nor removes enforced eligibility: the wrapper is a
  // canonical worker exactly when its per-game base is, so on a host without
  // verified taskpolicy support it stays ineligible for enforced claims.
  assert.equal(isWorkerEngineAdapter(base), canonicalMemory);
  assert.equal(isAuthenticatedCandidateAdapter(base), canonicalMemory);
  assert.equal(isWorkerEngineAdapter(wrapper), canonicalMemory);
  assert.equal(isAuthenticatedCandidateAdapter(wrapper), canonicalMemory);
  assert.equal(isPinnedHardWorkerAdapter(wrapper), false);
  assert.equal(wrapper.id, base.id);
  assert.equal(wrapper.version, base.version);
  assert.deepEqual(
    getCandidateArtifactProvenance(wrapper),
    getCandidateArtifactProvenance(base)
  );
  assert.equal(
    getCandidateArtifactProvenance(wrapper).manifestSha256,
    sha256ArtifactFile(release.manifestPath)
  );
  assert.deepEqual(wrapper.candidateArtifactProvenance, base.candidateArtifactProvenance);

  assert.equal(wrapper.capabilities.isolation, 'node-subprocess-per-decision-v1');
  assert.equal(wrapper.capabilities.sessionLifecycle, 'stateless-wasm-per-decision-v1');
  assert.equal(base.capabilities.isolation, 'node-subprocess-per-game-v1');
  assert.equal(base.capabilities.sessionLifecycle, 'stateful-session-per-game-v1');
  for (const key of [
    'nodeBudget',
    'deadline',
    'hardDeadlineIsolation',
    'integrityIsolation',
    'stdioIsolation',
    'memoryIsolation',
    'memoryLimitMiB',
    'v8OldSpaceMiB',
    'memoryPreflight',
    'artifactIntegrity',
    'moduleLoadIsolation',
    'filesystemContentIsolation',
    'environmentIsolation',
    'networkIsolation',
    'activeTimeCharging',
    'initializationCapMs',
    'observerCapMs'
  ]) {
    assert.equal(wrapper.capabilities[key], base.capabilities[key], key);
  }
  assert.equal(
    wrapper.capabilities.memoryLimitMiB,
    canonicalMemory ? CANONICAL_ENGINE_MEMORY_LIMIT_MIB : null
  );
  assert.equal(wrapper.capabilities.v8OldSpaceMiB, CANONICAL_ENGINE_V8_OLD_SPACE_MIB);

  // Private provenance reports the exact subprocess isolation as well as the
  // exact session lifecycle, and those two fields are the only difference
  // between a recycled wrapper and its per-game base.
  const wrapperProvenance = getWorkerEngineIsolationProvenance(wrapper);
  const baseProvenance = getWorkerEngineIsolationProvenance(base);
  assert.equal(wrapperProvenance.subprocessIsolation, 'node-subprocess-per-decision-v1');
  assert.equal(wrapperProvenance.sessionLifecycle, 'stateless-wasm-per-decision-v1');
  assert.equal(baseProvenance.subprocessIsolation, 'node-subprocess-per-game-v1');
  assert.equal(baseProvenance.sessionLifecycle, 'stateful-session-per-game-v1');
  assert.deepEqual(
    { ...wrapperProvenance, subprocessIsolation: null, sessionLifecycle: null },
    { ...baseProvenance, subprocessIsolation: null, sessionLifecycle: null }
  );

  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const baseSession = base.createSession(context);
  try {
    await baseSession.ready();
    const first = takeTrustedSubprocessDecision(
      await baseSession.selectAction(recycledWasmRequest(config, state))
    );
    const second = takeTrustedSubprocessDecision(
      await baseSession.selectAction(recycledWasmRequest(config, state))
    );
    assert.equal(first.value.stats.nodes, 1);
    assert.equal(second.value.stats.nodes, 2, 'base sessions must stay per-game and stateful');
  } finally {
    await baseSession.close();
  }

  const hard = await createPinnedHardWorkerAdapter();
  assert.equal(isPinnedHardWorkerAdapter(hard), canonicalMemory);
  assert.equal(hard.capabilities.isolation, 'node-subprocess-per-game-v1');
  assert.equal(hard.capabilities.sessionLifecycle, 'stateful-session-per-game-v1');
  assert.equal(
    getWorkerEngineIsolationProvenance(hard).subprocessIsolation,
    'node-subprocess-per-game-v1'
  );
  assert.equal(
    getWorkerEngineIsolationProvenance(hard).sessionLifecycle,
    'stateful-session-per-game-v1'
  );
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(hard),
    /authenticated official candidate/
  );
});

test('recycled normal-duel WASM sessions are lazy, stateless, and reaped before each decision returns', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-lifecycle-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base, release } = await recycledWasmCandidateBase(directory, 'genuine');
  const wrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const transition = Object.freeze({
    player: 'A',
    action: Object.freeze({ kind: 'pawn', to: Object.freeze({ r: 7, c: 4 }) }),
    stateBefore: state,
    stateAfter: state,
    source: 'engine'
  });

  // While the release is tampered every child creation fails, so a session that
  // can still be created, made ready, and observed proves it holds no child.
  const strayPath = join(release.releaseDirectory, 'stray.bin');
  writeFileSync(strayPath, 'unlisted late mutation');
  const session = wrapper.createSession(context);
  assert.equal(takeTrustedSubprocessPendingTiming(session), null);
  assert.deepEqual(await session.ready(), {
    id: base.id,
    version: base.version,
    sourceCommit: base.sourceCommit ?? null,
    baselineTrustRoot: base.baselineTrustRoot ?? null
  });
  assert.equal(await session.observe(transition), undefined);
  let setupFailure = null;
  try {
    await session.selectAction(recycledWasmRequest(config, state));
  } catch (error) {
    setupFailure = error;
  }
  assert.ok(setupFailure, 'a tampered release must fail the per-decision child setup');
  assert.match(setupFailure.message, /hermetic release contains an unlisted file/);
  assert.equal(isTrustedSubprocessDeadlineError(setupFailure), false);
  assert.equal(isTrustedSubprocessMemoryError(setupFailure), false);
  const setupTiming = takeTrustedSubprocessFailureTiming(setupFailure);
  assert.ok(setupTiming, 'setup failures must keep trusted parent timing');
  assert.equal(setupTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(Number.isFinite(setupTiming.chargedActiveMs) && setupTiming.chargedActiveMs >= 0);
  rmSync(strayPath);

  const before = engineChildProcessIds();
  const startedAt = performance.now();
  const firstDecision = takeTrustedSubprocessDecision(
    await session.selectAction(recycledWasmRequest(config, state, { mode: STRENGTH_MODE }))
  );
  const elapsedMs = performance.now() - startedAt;
  assertNoSurvivingChildren(before, 'a completed recycled decision');
  assert.ok(firstDecision, 'the wrapper must re-brand a trusted decision');
  assert.equal(firstDecision.timing.source, 'trusted-parent-subprocess-clock');
  assert.equal(firstDecision.value.stats.nodes, 1);
  assert.ok(firstDecision.timing.setupMs > 0, 'per-decision setup must be charged');
  assert.ok(firstDecision.timing.selectMs > 0);
  assert.equal(firstDecision.timing.observerMs, 0);
  assert.equal(
    firstDecision.timing.chargedActiveMs,
    firstDecision.timing.setupMs
    + firstDecision.timing.observerMs
    + firstDecision.timing.selectMs
  );
  // Setup, select, and teardown all happen inside the measured call, so the
  // charge can neither exceed it nor leave the teardown uncharged.
  assert.ok(
    firstDecision.timing.chargedActiveMs <= elapsedMs + 1,
    `charged ${firstDecision.timing.chargedActiveMs} ms of a ${elapsedMs} ms decision`
  );
  assert.ok(
    firstDecision.timing.chargedActiveMs >= elapsedMs - 5,
    `charged ${firstDecision.timing.chargedActiveMs} ms of a ${elapsedMs} ms decision`
  );

  assert.equal(await session.observe(transition), undefined);
  const secondDecision = takeTrustedSubprocessDecision(
    await session.selectAction(recycledWasmRequest(config, state))
  );
  assertNoSurvivingChildren(before, 'a second completed recycled decision');
  assert.equal(
    secondDecision.value.stats.nodes,
    1,
    'each decision must run in a fresh WASM subprocess'
  );

  const concurrent = session.selectAction(recycledWasmRequest(config, state));
  await assert.rejects(
    session.selectAction(recycledWasmRequest(config, state)),
    /does not support concurrent decisions/
  );
  assert.ok(takeTrustedSubprocessDecision(await concurrent));

  await session.close();
  await session.close();
  await assert.rejects(
    session.selectAction(recycledWasmRequest(config, state)),
    /session is closed/
  );
  await assert.rejects(session.observe(transition), /session is closed/);
  assertNoSurvivingChildren(before, 'a closed recycled session');
});

test('recycled normal-duel WASM decision failures keep trusted timing and private classification', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-failures-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base } = await recycledWasmCandidateBase(directory, 'genuine');
  const canonicalMemory = canonicalMemoryEligible(base);
  const wrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const before = engineChildProcessIds();

  const crashSession = wrapper.createSession(context);
  let crash = null;
  try {
    await crashSession.selectAction(recycledWasmRequest(config, state, { nodeBudget: 2 }));
  } catch (error) {
    crash = error;
  }
  await crashSession.close();
  assertNoSurvivingChildren(before, 'a crashed recycled decision');
  assert.ok(crash);
  assert.equal(crash.message, 'stub wasm select failure');
  assert.equal(isTrustedSubprocessDeadlineError(crash), false);
  assert.equal(isTrustedSubprocessMemoryError(crash), false);
  const crashTiming = takeTrustedSubprocessFailureTiming(crash);
  assert.ok(crashTiming);
  assert.equal(crashTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(crashTiming.setupMs > 0, `setup charge was ${crashTiming?.setupMs} ms`);
  assert.ok(crashTiming.selectMs > 0, 'teardown must be charged into selectMs');

  const deadlineSession = wrapper.createSession(context);
  const blocking = deadlineSession.selectAction(
    recycledWasmRequest(config, state, { nodeBudget: 3, wallClockBudgetMs: 400 })
  );
  const closing = deadlineSession.close();
  let deadline = null;
  try {
    await blocking;
  } catch (error) {
    deadline = error;
  }
  await closing;
  assertNoSurvivingChildren(before, 'a deadline-killed recycled decision');
  assert.ok(deadline, 'a blocking decision must be terminated, not returned');
  assert.equal(
    isTrustedSubprocessDeadlineError(deadline),
    true,
    `deadline classification lost: ${deadline?.message}`
  );
  assert.equal(isTrustedSubprocessMemoryError(deadline), false);
  const deadlineTiming = takeTrustedSubprocessFailureTiming(deadline);
  assert.ok(deadlineTiming);
  assert.equal(deadlineTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(deadlineTiming.chargedActiveMs > 0);

  // Only a canonical worker has an RSS ceiling to exceed, so the memory kill is
  // the one classification that is genuinely host-conditional. Running the
  // over-allocating decision anywhere else would just allocate gigabytes and
  // prove nothing about containment.
  if (!canonicalMemory) {
    assert.equal(wrapper.capabilities.memoryIsolation, 'v8-old-space-only-ineligible-v1');
    assert.equal(wrapper.capabilities.memoryLimitMiB, null);
    assert.equal(
      wrapper.capabilities.memoryPreflight,
      'unsupported-platform-or-failed-preflight'
    );
    return;
  }
  const memorySession = wrapper.createSession(context);
  let memory = null;
  try {
    await memorySession.selectAction(
      recycledWasmRequest(config, state, { nodeBudget: 4, wallClockBudgetMs: 3_000 })
    );
  } catch (error) {
    memory = error;
  }
  await memorySession.close();
  assertNoSurvivingChildren(before, 'a memory-killed recycled decision');
  assert.ok(memory, 'an over-allocating decision must be killed, not returned');
  assert.equal(
    isTrustedSubprocessMemoryError(memory),
    true,
    `memory classification lost: ${memory?.message}`
  );
  const memoryTiming = takeTrustedSubprocessFailureTiming(memory);
  assert.ok(memoryTiming);
  assert.equal(memoryTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(memoryTiming.setupMs > 0);
  assert.ok(memoryTiming.selectMs > 0);
});

test('recycled normal-duel WASM teardown failures fail closed in every decision branch', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-teardown-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base, release } = await recycledWasmCandidateBase(directory, 'genuine');
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const before = engineChildProcessIds();

  // The seam runs after the real inner close, so it reproduces a rejected
  // teardown — an unproven reap — without ever leaving a child alive.
  let injectedFaults = 0;
  const rejectClose = () => {
    injectedFaults += 1;
    const error = new Error(`simulated close rejection ${injectedFaults}`);
    error.name = 'EngineSubprocessCloseError';
    throw error;
  };
  const faulty = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    [RECYCLE_TEARDOWN_FAULT_SEAM]: rejectClose
  });
  // Installing the seam changes neither admission nor any published brand.
  assert.equal(faulty.id, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(faulty.capabilities.isolation, 'node-subprocess-per-decision-v1');
  assert.equal(faulty.capabilities.sessionLifecycle, 'stateless-wasm-per-decision-v1');
  assert.deepEqual(
    getCandidateArtifactProvenance(faulty),
    getCandidateArtifactProvenance(base)
  );

  const healthySession = faulty.createSession(context);
  let unreapedHealthy = null;
  try {
    await healthySession.selectAction(
      recycledWasmRequest(config, state, { mode: STRENGTH_MODE })
    );
  } catch (error) {
    unreapedHealthy = error;
  }
  await healthySession.close();
  assertNoSurvivingChildren(before, 'an unreaped healthy recycled decision');
  assert.ok(unreapedHealthy, 'a decision whose child was not proven reaped must not return');
  assert.equal(unreapedHealthy.name, 'EngineSubprocessRecycleError');
  assert.equal(unreapedHealthy.code, 'subprocess_recycle_failed');
  assert.equal(unreapedHealthy.reaped, false);
  assert.match(unreapedHealthy.message, /could not reap its per-decision subprocess/);
  assert.match(unreapedHealthy.message, /simulated close rejection 1/);
  assert.equal(unreapedHealthy.cause, unreapedHealthy.teardownFailure);
  assert.equal(unreapedHealthy.cause.name, 'EngineSubprocessCloseError');
  assert.equal(unreapedHealthy.operationFailure, null);
  assert.equal(unreapedHealthy.unprovenClassification, null);
  assert.equal(isTrustedSubprocessDeadlineError(unreapedHealthy), false);
  assert.equal(isTrustedSubprocessMemoryError(unreapedHealthy), false);
  // The trusted decision brand was consumed by the wrapper, so the move the
  // live child produced is unreachable rather than merely unreported.
  assert.equal(takeTrustedSubprocessDecision(unreapedHealthy), null);
  const healthyTiming = takeTrustedSubprocessFailureTiming(unreapedHealthy);
  assert.ok(healthyTiming, 'an unproven reap still carries trusted parent timing');
  assert.equal(healthyTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(healthyTiming.setupMs > 0);
  assert.ok(healthyTiming.selectMs > 0, 'teardown must be charged into selectMs');
  assert.equal(
    healthyTiming.chargedActiveMs,
    healthyTiming.setupMs + healthyTiming.observerMs + healthyTiming.selectMs
  );

  const crashSession = faulty.createSession(context);
  let unreapedCrash = null;
  try {
    await crashSession.selectAction(recycledWasmRequest(config, state, { nodeBudget: 2 }));
  } catch (error) {
    unreapedCrash = error;
  }
  await crashSession.close();
  assertNoSurvivingChildren(before, 'an unreaped crashed recycled decision');
  assert.ok(unreapedCrash);
  assert.equal(unreapedCrash.name, 'EngineSubprocessRecycleError');
  assert.equal(unreapedCrash.reaped, false);
  assert.match(unreapedCrash.message, /simulated close rejection 2/);
  assert.match(unreapedCrash.message, /\(after .*stub wasm select failure\)/);
  assert.equal(unreapedCrash.cause.name, 'EngineSubprocessCloseError');
  assert.equal(unreapedCrash.operationFailure.message, 'stub wasm select failure');
  assert.equal(unreapedCrash.unprovenClassification, null);
  const crashTiming = takeTrustedSubprocessFailureTiming(unreapedCrash);
  assert.ok(crashTiming, 'the decision failure timing is carried by the recycle crash');
  assert.equal(crashTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(crashTiming.setupMs > 0);
  assert.ok(crashTiming.selectMs > 0);

  const deadlineSession = faulty.createSession(context);
  let unreapedDeadline = null;
  try {
    await deadlineSession.selectAction(
      recycledWasmRequest(config, state, { nodeBudget: 3, wallClockBudgetMs: 400 })
    );
  } catch (error) {
    unreapedDeadline = error;
  }
  await deadlineSession.close();
  assertNoSurvivingChildren(before, 'an unreaped deadline-killed recycled decision');
  assert.ok(unreapedDeadline, 'a blocking decision must be terminated, not returned');
  assert.equal(unreapedDeadline.name, 'EngineSubprocessRecycleError');
  assert.equal(unreapedDeadline.reaped, false);
  assert.match(unreapedDeadline.message, /simulated close rejection 3/);
  // While cleanup is unproven the child may still be running, so the deadline
  // verdict is no longer claimed as the outcome...
  assert.equal(isTrustedSubprocessDeadlineError(unreapedDeadline), false);
  assert.equal(isTrustedSubprocessMemoryError(unreapedDeadline), false);
  // ...but it is kept verbatim as context, brand and all.
  assert.equal(isTrustedSubprocessDeadlineError(unreapedDeadline.operationFailure), true);
  assert.equal(unreapedDeadline.unprovenClassification, 'deadline_exceeded');
  assert.equal(unreapedDeadline.cause.name, 'EngineSubprocessCloseError');
  const deadlineTiming = takeTrustedSubprocessFailureTiming(unreapedDeadline);
  assert.ok(deadlineTiming);
  assert.equal(deadlineTiming.source, 'trusted-parent-subprocess-clock');
  assert.ok(deadlineTiming.chargedActiveMs > 0);
  assert.equal(injectedFaults, 3);

  // A setup that never produced a child has nothing to reap, so the seam is not
  // reached and the setup failure is surfaced unchanged rather than as a
  // fabricated unreaped-child crash.
  const strayPath = join(release.releaseDirectory, 'stray.bin');
  writeFileSync(strayPath, 'unlisted late mutation');
  const setupSession = faulty.createSession(context);
  let setupFailure = null;
  try {
    await setupSession.selectAction(recycledWasmRequest(config, state));
  } catch (error) {
    setupFailure = error;
  }
  await setupSession.close();
  rmSync(strayPath);
  assertNoSurvivingChildren(before, 'a failed recycled setup');
  assert.ok(setupFailure);
  assert.equal(injectedFaults, 3);
  assert.notEqual(setupFailure.name, 'EngineSubprocessRecycleError');
  assert.match(setupFailure.message, /hermetic release contains an unlisted file/);
  assert.ok(takeTrustedSubprocessFailureTiming(setupFailure));

  // The unmodified wrapper of the same base still returns decisions, so the
  // fail-closed path above is the seam's doing and not a broken wrapper.
  const healthyWrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);
  const healthyWrapperSession = healthyWrapper.createSession(context);
  const decision = takeTrustedSubprocessDecision(
    await healthyWrapperSession.selectAction(recycledWasmRequest(config, state))
  );
  await healthyWrapperSession.close();
  assertNoSurvivingChildren(before, 'a reaped recycled decision');
  assert.ok(decision);
  assert.equal(decision.value.stats.nodes, 1);
});

const LATEGAME_FIXTURE_URL = new URL(
  './fixtures/normal-duel-recycled-lategame-9x9-007-v1.json',
  import.meta.url
);
const LATEGAME_FINAL_POSITION_KEY = JSON.stringify([
  'normal-duel-v1', 9, 9, 76, 4, 0, 8, 10, 10, 'permissive-adjacent-exit-v1', 3, 200, 'A', 25, 63,
  [6, 8, 10, 15, 20, 41, 43, 45, 51, 53, 57, 59, 61, 63], [11, 13, 22, 38, 42, 55], 0, 0, 'A'
]);

// The real-release regression in scripts/check-normal-duel-wasm-candidate.mjs
// replays this fixture through the recycler, which needs a built WASM candidate.
// This test needs none, so the fixture itself stays guarded on every host.
test('canonical 9x9 late-game recycler fixture reconstructs its pinned checkpoint anchors', () => {
  const fixture = JSON.parse(readFileSync(LATEGAME_FIXTURE_URL, 'utf8'));
  assert.equal(fixture.format, 'normal-duel-recycled-lategame-replay-v1');
  assert.equal(fixture.source.format, 'normal-duel-hardplus-exhibition-replay-v1');
  assert.equal(
    fixture.source.sha256,
    '1906c7587d117ff88fcd18ead35b82219723e2d95e74dc9f7f1d197ded4ad47b'
  );
  assert.equal(fixture.source.actionDerivation, 'canonical-legal-action-code-match-v1');
  const config = fixture.config;
  assert.deepEqual(config, normalDuelConfig({ size: 9, firstPlayer: 'A', plyCap: 200 }));
  assert.equal(fixture.actionCount, 74);
  assert.equal(fixture.actionCodes.length, 74);
  assert.equal(
    sha256(JSON.stringify(fixture.actionCodes)),
    '2623de1808d6edd89188e372ffed8d68465216201189f69c59dd0700ddf1f9b1'
  );
  assert.deepEqual(fixture.checkpointInvariants, {
    turn: 'A',
    outcome: { kind: 'ongoing' },
    wallCount: 20,
    stock: { A: 0, B: 0 },
    historyStartPly: 52,
    maximumRepetitionCount: 1
  });
  assert.deepEqual(fixture.checkpoints.map(({ ply }) => ply), [58, 62, 66, 70, 74]);
  assert.deepEqual(
    fixture.checkpoints.map(({ repetitionEntryCount }) => repetitionEntryCount),
    [7, 11, 15, 19, 23]
  );

  // Codes are trusted only as far as canonical legality allows, and every anchor
  // is compared against the engine's own transition rather than the fixture's.
  let state = createInitialState(config);
  const reached = [];
  for (const [index, code] of fixture.actionCodes.entries()) {
    assert.ok(
      legalActionCodes(config, state).includes(code),
      `late-game ply ${index + 1} code ${code} is not legal in the reconstructed state`
    );
    state = applyAction(config, state, decodeAction(config, code));
    assert.equal(state.ply, index + 1);
    const checkpoint = fixture.checkpoints.find(({ ply }) => ply === state.ply);
    if (checkpoint === undefined) continue;
    const invariants = fixture.checkpointInvariants;
    assert.equal(state.position.turn, invariants.turn);
    assert.deepEqual(state.outcome, invariants.outcome);
    assert.equal(state.position.walls.length, invariants.wallCount);
    assert.deepEqual(state.position.stock, invariants.stock);
    assert.equal(state.historyStartPly, invariants.historyStartPly);
    assert.equal(state.repetitionCounts.length, checkpoint.repetitionEntryCount);
    assert.deepEqual(
      [...new Set(state.repetitionCounts.map(({ count }) => count))],
      [invariants.maximumRepetitionCount]
    );
    assert.equal(
      sha256(JSON.stringify(state.repetitionCounts)),
      checkpoint.repetitionCountsSha256,
      `late-game ply ${checkpoint.ply} repetition history hash changed`
    );
    assert.equal(state.positionKey, checkpoint.positionKey);
    reached.push(state.ply);
  }
  assert.deepEqual(reached, [58, 62, 66, 70, 74]);
  assert.equal(state.ply, 74);
  assert.equal(state.positionKey, LATEGAME_FINAL_POSITION_KEY);

  // Small and frozen: the fixture carries no replay telemetry and no position
  // snapshot, so nothing it records can substitute for a reconstructed state.
  assert.deepEqual(Object.keys(fixture).sort(), [
    'actionCodes',
    'actionCodesSha256',
    'actionCount',
    'checkpointInvariants',
    'checkpoints',
    'config',
    'description',
    'format',
    'source'
  ]);
  for (const checkpoint of fixture.checkpoints) {
    assert.deepEqual(Object.keys(checkpoint).sort(), [
      'ply',
      'positionKey',
      'repetitionCountsSha256',
      'repetitionEntryCount'
    ]);
  }
});

// The audit trail of `count` consecutive healthy decisions in one fresh session:
// one opened and one reaped record per child, numbered from one.
function auditRecords(count) {
  return Array.from({ length: count }, (ignored, index) => [
    { phase: 'opened', generation: index + 1, selection: index + 1 },
    { phase: 'reaped', generation: index + 1, selection: index + 1 }
  ]).flat();
}

test('recycled normal-duel WASM child audit observes fresh children and changes nothing', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-audit-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base, release } = await recycledWasmCandidateBase(directory, 'genuine');
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const before = engineChildProcessIds();

  const trail = [];
  const audited = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    [RECYCLE_CHILD_AUDIT_SEAM]: (record) => {
      trail.push(record);
    }
  });
  // Installing an observer changes neither admission nor any published brand.
  const unaudited = await createRecycledNormalDuelWasmCandidateAdapter(base);
  assert.equal(audited.id, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.deepEqual(audited.capabilities, unaudited.capabilities);
  assert.deepEqual(
    getCandidateArtifactProvenance(audited),
    getCandidateArtifactProvenance(base)
  );
  assert.deepEqual(
    getWorkerEngineIsolationProvenance(audited),
    getWorkerEngineIsolationProvenance(unaudited)
  );
  assert.equal(isAuthenticatedCandidateAdapter(audited), canonicalMemoryEligible(base));

  const session = audited.createSession(context);
  const nodes = [];
  for (let index = 0; index < 3; index += 1) {
    const decision = takeTrustedSubprocessDecision(
      await session.selectAction(recycledWasmRequest(config, state))
    );
    assert.ok(decision, 'an audited decision must still carry its trusted brand');
    nodes.push(decision.value.stats.nodes);
    assert.equal(decision.timing.source, 'trusted-parent-subprocess-clock');
    // The awaited decision has already returned, so its reaped record was
    // emitted before the caller could see the move.
    assert.deepEqual(trail, auditRecords(index + 1));
  }
  assert.deepEqual(nodes, [1, 1, 1], 'auditing must not make a child outlive its decision');
  assertNoSurvivingChildren(before, 'audited recycled decisions');

  // Records are immutable, carry nothing but the two ordinals and a phase, and
  // hold no handle to the base, the manifest, the child, or its close.
  for (const record of trail) {
    assert.ok(Object.isFrozen(record));
    assert.deepEqual(Object.keys(record).sort(), ['generation', 'phase', 'selection']);
    assert.equal(Object.getPrototypeOf(record), Object.prototype);
    assert.throws(() => { record.phase = 'reaped'; }, TypeError);
    assert.throws(() => { record.child = base; }, TypeError);
    assert.throws(() => { delete record.generation; }, TypeError);
  }
  assert.deepEqual(trail[0], { phase: 'opened', generation: 1, selection: 1 });

  // A child that never became ready opens no generation, so the trail keeps
  // exactly one opened record per real child even though the session consumed a
  // selection ordinal for the failed decision.
  const strayPath = join(release.releaseDirectory, 'stray.bin');
  writeFileSync(strayPath, 'unlisted late mutation');
  await assert.rejects(
    session.selectAction(recycledWasmRequest(config, state)),
    /hermetic release contains an unlisted file/
  );
  assert.deepEqual(trail, auditRecords(3));
  rmSync(strayPath);
  assert.ok(takeTrustedSubprocessDecision(
    await session.selectAction(recycledWasmRequest(config, state))
  ));
  assert.deepEqual(trail.slice(-2), [
    { phase: 'opened', generation: 4, selection: 5 },
    { phase: 'reaped', generation: 4, selection: 5 }
  ]);
  await session.close();

  // Generations are per wrapper and never reused, so a second session continues
  // the same monotonic child sequence while restarting its own selection count.
  const secondSession = audited.createSession(context);
  assert.ok(takeTrustedSubprocessDecision(
    await secondSession.selectAction(recycledWasmRequest(config, state))
  ));
  await secondSession.close();
  assert.deepEqual(trail.slice(-2), [
    { phase: 'opened', generation: 5, selection: 1 },
    { phase: 'reaped', generation: 5, selection: 1 }
  ]);
  assert.equal(new Set(trail.map(({ generation }) => generation)).size, 5);
  assert.deepEqual(
    trail.map(({ phase }) => phase),
    Array.from({ length: 5 }, () => ['opened', 'reaped']).flat()
  );

  // A faulting or asynchronous collector is swallowed: the decision, its trusted
  // timing, and its reaping are exactly what an unobserved decision produces.
  const observedRecordCount = trail.length;
  const faultyObservations = [];
  const faulty = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    [RECYCLE_CHILD_AUDIT_SEAM]: (record) => {
      faultyObservations.push(record.phase);
      try {
        record.phase = 'tampered';
      } catch {
        faultyObservations.push('immutable');
      }
      if (record.phase === 'reaped') {
        return Promise.reject(new Error('collector rejected after the reap'));
      }
      throw new Error('collector threw during the audit');
    }
  });
  const faultySession = faulty.createSession(context);
  const faultyDecision = takeTrustedSubprocessDecision(
    await faultySession.selectAction(recycledWasmRequest(config, state, { mode: STRENGTH_MODE }))
  );
  await faultySession.close();
  // Give a swallowed rejection a turn to surface as an unhandled rejection.
  await new Promise((resolve) => { setTimeout(resolve, 10); });
  assertNoSurvivingChildren(before, 'a recycled decision with a faulting audit collector');
  assert.ok(faultyDecision, 'a faulting collector must not cost the decision its brand');
  assert.equal(faultyDecision.value.stats.nodes, 1);
  assert.equal(faultyDecision.timing.source, 'trusted-parent-subprocess-clock');
  assert.ok(faultyDecision.timing.setupMs > 0);
  assert.ok(faultyDecision.timing.selectMs > 0);
  assert.equal(
    faultyDecision.timing.chargedActiveMs,
    faultyDecision.timing.setupMs
    + faultyDecision.timing.observerMs
    + faultyDecision.timing.selectMs
  );
  assert.deepEqual(faultyObservations, ['opened', 'immutable', 'reaped', 'immutable']);
  assert.equal(
    trail.length,
    observedRecordCount,
    'each wrapper audits only its own children'
  );

  // The seam is reachable only through its exported symbol, and it never relaxes
  // admission: a same-named string key does nothing and a forged base is still
  // refused with an observer installed.
  const stringKeyed = [];
  const ignoredSeam = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    'recycled-normal-duel-wasm-child-audit': (record) => stringKeyed.push(record),
    [RECYCLE_CHILD_AUDIT_SEAM]: 'not-a-function'
  });
  const ignoredSession = ignoredSeam.createSession(context);
  assert.ok(takeTrustedSubprocessDecision(
    await ignoredSession.selectAction(recycledWasmRequest(config, state))
  ));
  await ignoredSession.close();
  assert.deepEqual(stringKeyed, [], 'only the exported symbol may install an observer');
  assert.equal(trail.length, observedRecordCount);
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(Object.freeze({
      id: base.id,
      version: base.version,
      capabilities: Object.freeze({ ...base.capabilities }),
      candidateArtifactProvenance: base.candidateArtifactProvenance,
      createSession() {}
    }), { [RECYCLE_CHILD_AUDIT_SEAM]: (record) => trail.push(record) }),
    /authenticated official candidate/
  );
  await assert.rejects(
    createRecycledNormalDuelWasmCandidateAdapter(audited, {
      [RECYCLE_CHILD_AUDIT_SEAM]: (record) => trail.push(record)
    }),
    /not an already recycled per-decision wrapper/
  );
  assertNoSurvivingChildren(before, 'audit seam admission refusals');
});

test('recycled normal-duel WASM child audit reports an unproven reap as reap_failed', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-audit-fault-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base } = await recycledWasmCandidateBase(directory, 'genuine');
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const context = recycledWasmSessionContext(config);
  const before = engineChildProcessIds();

  // Both seams together: the real inner close still runs first, then the
  // injected fault makes the reap unproven, and only then is the phase emitted.
  const trail = [];
  const faulty = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    [RECYCLE_TEARDOWN_FAULT_SEAM]: () => {
      throw new Error('simulated close rejection');
    },
    [RECYCLE_CHILD_AUDIT_SEAM]: (record) => {
      trail.push(record);
    }
  });
  const session = faulty.createSession(context);
  let unreaped = null;
  try {
    await session.selectAction(recycledWasmRequest(config, state));
  } catch (error) {
    unreaped = error;
  }
  await session.close();
  assertNoSurvivingChildren(before, 'an audited unreaped recycled decision');
  assert.deepEqual(trail, [
    { phase: 'opened', generation: 1, selection: 1 },
    { phase: 'reap_failed', generation: 1, selection: 1 }
  ]);
  assert.ok(trail.every((record) => Object.isFrozen(record)));

  // Auditing neither rescues nor reclassifies the fail-closed decision.
  assert.ok(unreaped, 'an unproven reap must still fail the decision closed');
  assert.equal(unreaped.name, 'EngineSubprocessRecycleError');
  assert.equal(unreaped.code, 'subprocess_recycle_failed');
  assert.equal(unreaped.reaped, false);
  assert.equal(unreaped.cause, unreaped.teardownFailure);
  assert.equal(unreaped.operationFailure, null);
  assert.equal(unreaped.unprovenClassification, null);
  assert.equal(takeTrustedSubprocessDecision(unreaped), null);
  const timing = takeTrustedSubprocessFailureTiming(unreaped);
  assert.ok(timing);
  assert.equal(timing.source, 'trusted-parent-subprocess-clock');
  assert.ok(timing.selectMs > 0, 'teardown stays charged into selectMs');

  // A failing decision behind a proven reap is still audited as reaped: the
  // phase reports the child, never the move.
  const crashTrail = [];
  const audited = await createRecycledNormalDuelWasmCandidateAdapter(base, {
    [RECYCLE_CHILD_AUDIT_SEAM]: (record) => {
      crashTrail.push(record);
    }
  });
  const crashSession = audited.createSession(context);
  await assert.rejects(
    crashSession.selectAction(recycledWasmRequest(config, state, { nodeBudget: 2 })),
    /stub wasm select failure/
  );
  const healthy = takeTrustedSubprocessDecision(
    await crashSession.selectAction(recycledWasmRequest(config, state))
  );
  await crashSession.close();
  assertNoSurvivingChildren(before, 'an audited crashed recycled decision');
  assert.ok(healthy);
  assert.equal(healthy.value.stats.nodes, 1);
  assert.deepEqual(crashTrail, [
    { phase: 'opened', generation: 1, selection: 1 },
    { phase: 'reaped', generation: 1, selection: 1 },
    { phase: 'opened', generation: 2, selection: 2 },
    { phase: 'reaped', generation: 2, selection: 2 }
  ]);
});

const WORKER_ENGINE_PROXY_MODULE_URL =
  new URL('../scripts/evaluation/worker-engine-proxy.mjs', import.meta.url).href;

// Reproduce a host without verified canonical memory containment — every
// non-macOS CI runner — by fixing `process.platform` before the proxy module
// runs its taskpolicy preflight. Running this in a child process keeps the
// module-level preflight result of the main test process untouched, so the
// non-canonical recycler lifecycle is covered even on a canonical host.
function simulatedNonCanonicalHostRecyclerReport(release, { context, request }) {
  const source = `
    Object.defineProperty(process, 'platform', { value: 'linux', configurable: true });
    const proxy = await import(${JSON.stringify(WORKER_ENGINE_PROXY_MODULE_URL)});
    const context = ${JSON.stringify(context)};
    const request = ${JSON.stringify(request)};

    async function rejection(run) {
      try {
        await run();
        return null;
      } catch (error) {
        return error.message;
      }
    }

    const base = await proxy.createWorkerEngineAdapter({
      moduleUrl: ${JSON.stringify(pathToFileURL(release.entryPath).href)},
      candidateManifestPath: ${JSON.stringify(release.manifestPath)}
    });
    const report = {
      platform: process.platform,
      baseId: base.id,
      baseIsolation: base.capabilities.isolation,
      baseSessionLifecycle: base.capabilities.sessionLifecycle,
      baseMemoryIsolation: base.capabilities.memoryIsolation,
      baseMemoryLimitMiB: base.capabilities.memoryLimitMiB,
      baseMemoryPreflight: base.capabilities.memoryPreflight,
      baseV8OldSpaceMiB: base.capabilities.v8OldSpaceMiB,
      baseWorkerEligible: proxy.isWorkerEngineAdapter(base),
      baseAuthenticatedEligible: proxy.isAuthenticatedCandidateAdapter(base),
      baseIsolationProvenance: proxy.getWorkerEngineIsolationProvenance(base),
      supportedBoundaries: proxy.SUPPORTED_WORKER_SESSION_BOUNDARIES
    };

    const wrapper = await proxy.createRecycledNormalDuelWasmCandidateAdapter(base);
    report.admitted = true;
    report.wrapperId = wrapper.id;
    report.wrapperIsolation = wrapper.capabilities.isolation;
    report.wrapperSessionLifecycle = wrapper.capabilities.sessionLifecycle;
    report.wrapperMemoryIsolation = wrapper.capabilities.memoryIsolation;
    report.wrapperMemoryLimitMiB = wrapper.capabilities.memoryLimitMiB;
    report.wrapperArtifactIntegrity = wrapper.capabilities.artifactIntegrity;
    report.wrapperFilesystemContentIsolation = wrapper.capabilities.filesystemContentIsolation;
    report.wrapperNetworkIsolation = wrapper.capabilities.networkIsolation;
    report.wrapperWorkerEligible = proxy.isWorkerEngineAdapter(wrapper);
    report.wrapperAuthenticatedEligible = proxy.isAuthenticatedCandidateAdapter(wrapper);
    report.wrapperPinnedEligible = proxy.isPinnedHardWorkerAdapter(wrapper);
    report.wrapperProvenance = proxy.getCandidateArtifactProvenance(wrapper);
    report.wrapperIsolationProvenance = proxy.getWorkerEngineIsolationProvenance(wrapper);

    const session = wrapper.createSession(context);
    report.pendingTimingBeforeUse = proxy.takeTrustedSubprocessPendingTiming(session);
    report.ready = await session.ready();
    report.observeReturnedUndefined = (await session.observe(null)) === undefined;
    report.nodes = [];
    for (let index = 0; index < 2; index += 1) {
      const decision = proxy.takeTrustedSubprocessDecision(await session.selectAction(request));
      report.nodes.push(decision === null ? null : decision.value.stats.nodes);
      report.timingSource = decision?.timing.source ?? null;
      report.setupMs = decision?.timing.setupMs ?? null;
      report.selectMs = decision?.timing.selectMs ?? null;
      report.observerMs = decision?.timing.observerMs ?? null;
      report.chargedActiveMs = decision?.timing.chargedActiveMs ?? null;
    }
    await session.close();
    await session.close();
    report.closedSelect = await rejection(() => session.selectAction(request));
    report.wrapperOfWrapper = await rejection(
      () => proxy.createRecycledNormalDuelWasmCandidateAdapter(wrapper)
    );
    report.arbitraryWorker = await rejection(
      () => proxy.createRecycledNormalDuelWasmCandidateAdapter(Object.freeze({
        id: base.id,
        version: base.version,
        capabilities: Object.freeze({ ...base.capabilities }),
        candidateArtifactProvenance: base.candidateArtifactProvenance,
        createSession() {}
      }))
    );
    process.stdout.write(JSON.stringify(report));
  `;
  const result = spawnSync(process.execPath, ['--input-type=module', '--eval', source], {
    cwd: fileURLToPath(new URL('..', import.meta.url)),
    encoding: 'utf8',
    timeout: 60_000
  });
  assert.equal(result.error, undefined);
  assert.equal(result.signal, null);
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stderr, '');
  return JSON.parse(result.stdout);
}

test('recycled normal-duel WASM admission separates artifact authentication from canonical memory eligibility', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-noncanonical-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const release = writeRecycledWasmRelease(directory, 'genuine');
  const config = normalDuelConfig({ size: 9 });
  const state = createInitialState(config);
  const report = simulatedNonCanonicalHostRecyclerReport(release, {
    context: recycledWasmSessionContext(config),
    request: recycledWasmRequest(config, state)
  });

  // The simulated host produces the ineligible V8 fallback memory profile, so
  // neither the base nor its wrapper may claim canonical-worker eligibility.
  assert.equal(report.platform, 'linux');
  assert.equal(report.baseMemoryIsolation, 'v8-old-space-only-ineligible-v1');
  assert.equal(report.baseMemoryLimitMiB, null);
  assert.equal(report.baseMemoryPreflight, 'unsupported-platform-or-failed-preflight');
  assert.equal(report.baseV8OldSpaceMiB, CANONICAL_ENGINE_V8_OLD_SPACE_MIB);
  assert.equal(report.baseWorkerEligible, false);
  assert.equal(report.baseAuthenticatedEligible, false);
  assert.equal(report.wrapperWorkerEligible, false);
  assert.equal(report.wrapperAuthenticatedEligible, false);
  assert.equal(report.wrapperPinnedEligible, false);
  assert.equal(report.wrapperMemoryIsolation, 'v8-old-space-only-ineligible-v1');
  assert.equal(report.wrapperMemoryLimitMiB, null);

  // Artifact authentication is host-independent, so the official template-backed
  // release is still admitted and still rebranded per-decision.
  assert.equal(report.admitted, true);
  assert.equal(report.baseId, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(report.baseIsolation, 'node-subprocess-per-game-v1');
  assert.equal(report.baseSessionLifecycle, 'stateful-session-per-game-v1');
  assert.equal(report.wrapperId, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(report.wrapperIsolation, 'node-subprocess-per-decision-v1');
  assert.equal(report.wrapperSessionLifecycle, 'stateless-wasm-per-decision-v1');
  assert.equal(
    report.baseIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-game-v1'
  );
  assert.equal(
    report.baseIsolationProvenance.sessionLifecycle,
    'stateful-session-per-game-v1'
  );
  assert.equal(
    report.wrapperIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-decision-v1'
  );
  assert.equal(
    report.wrapperIsolationProvenance.sessionLifecycle,
    'stateless-wasm-per-decision-v1'
  );
  // Both reported pairs come from the module's one supported table, which the
  // relaxed memory profile does not widen.
  assert.deepEqual(report.supportedBoundaries, [
    {
      subprocessIsolation: 'node-subprocess-per-game-v1',
      sessionLifecycle: 'stateful-session-per-game-v1'
    },
    {
      subprocessIsolation: 'node-subprocess-per-decision-v1',
      sessionLifecycle: 'stateless-wasm-per-decision-v1'
    }
  ]);

  // The private artifact binding and its provenance are preserved verbatim.
  assert.equal(report.wrapperArtifactIntegrity, 'content-addressed-hermetic-release-v1');
  assert.equal(report.wrapperFilesystemContentIsolation, 'manifest-files-only-v1');
  assert.equal(report.wrapperNetworkIsolation, 'safe-builtins-no-network-db-v1');
  assert.equal(report.wrapperProvenance.verification, 'content-addressed-manifest-v1');
  assert.equal(report.wrapperProvenance.entry, 'adapter.mjs');
  assert.equal(
    report.wrapperProvenance.manifestSha256,
    sha256ArtifactFile(release.manifestPath)
  );

  // The production per-decision lifecycle really ran on the simulated host.
  assert.equal(report.pendingTimingBeforeUse, null);
  assert.equal(report.ready.id, NORMAL_DUEL_WASM_CANDIDATE_ID);
  assert.equal(report.observeReturnedUndefined, true);
  assert.deepEqual(report.nodes, [1, 1], 'each decision must run in a fresh WASM subprocess');
  assert.equal(report.timingSource, 'trusted-parent-subprocess-clock');
  assert.ok(report.setupMs > 0, `per-decision setup charge was ${report.setupMs} ms`);
  assert.ok(report.selectMs > 0);
  assert.equal(report.observerMs, 0);
  assert.equal(
    report.chargedActiveMs,
    report.setupMs + report.observerMs + report.selectMs
  );
  assert.match(report.closedSelect, /session is closed/);

  // Relaxing the memory profile does not relax anything else about admission.
  assert.match(report.wrapperOfWrapper, /not an already recycled per-decision wrapper/);
  assert.match(report.arbitraryWorker, /authenticated official candidate/);
});

test('CLI rejects per-decision recycling outside strength mode or without a manifest', () => {
  for (const [args, message] of [
    [
      ['--mode', 'regression', '--node-budget', '10'],
      /--recycle-normal-duel-wasm-per-decision requires strength mode/
    ],
    [[], /--recycle-normal-duel-wasm-per-decision requires --candidate-manifest/],
    [
      ['--mode', 'strength'],
      /--recycle-normal-duel-wasm-per-decision requires --candidate-manifest/
    ]
  ]) {
    const result = spawnSync(process.execPath, [
      'scripts/run-normal-duel-strength.mjs',
      '--candidate',
      './does-not-load.mjs',
      '--recycle-normal-duel-wasm-per-decision',
      ...args
    ], {
      cwd: new URL('..', import.meta.url),
      encoding: 'utf8'
    });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, message);
    // Rejected while parsing arguments, before any engine is loaded.
    assert.doesNotMatch(result.stderr, /does-not-load/);
    assert.equal(result.stdout, '');
  }
});

test('CLI recycles only the candidate and leaves the default per-game lifecycle unchanged', (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-cli-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const release = writeRecycledWasmRelease(directory, 'genuine');
  const strengthRun = (releasePaths, extra) => spawnSync(process.execPath, [
    'scripts/run-normal-duel-strength.mjs',
    '--candidate',
    releasePaths.entryPath,
    '--candidate-manifest',
    releasePaths.manifestPath,
    '--mode',
    'strength',
    '--opening-limit',
    '1',
    '--minimum-opening-pairs',
    '1',
    ...extra
  ], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
    timeout: 120_000
  });

  const recycledRun = strengthRun(release, ['--recycle-normal-duel-wasm-per-decision']);
  assert.equal(recycledRun.status, 0, recycledRun.stderr);
  const recycled = JSON.parse(recycledRun.stdout);
  const defaultRun = strengthRun(release, []);
  assert.equal(defaultRun.status, 0, defaultRun.stderr);
  const unrecycled = JSON.parse(defaultRun.stdout);

  assert.equal(recycled.results.length, 2);
  assert.equal(unrecycled.results.length, 2);
  assert.equal(
    recycled.candidateIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-decision-v1'
  );
  assert.equal(
    recycled.candidateIsolationProvenance.sessionLifecycle,
    'stateless-wasm-per-decision-v1'
  );
  // The default lifecycle is unchanged: the same release without the flag is
  // still one per-game child.
  assert.equal(
    unrecycled.candidateIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-game-v1'
  );
  assert.equal(
    unrecycled.candidateIsolationProvenance.sessionLifecycle,
    'stateful-session-per-game-v1'
  );
  // Only the candidate is wrapped; the pinned Hard baseline stays per-game.
  for (const report of [recycled, unrecycled]) {
    assert.equal(
      report.baselineIsolationProvenance.subprocessIsolation,
      'node-subprocess-per-game-v1'
    );
    assert.equal(
      report.baselineIsolationProvenance.sessionLifecycle,
      'stateful-session-per-game-v1'
    );
    assert.equal(report.baselineProvenance.id, 'hard-product-46a871c7');
  }
  // Recycled and non-recycled reports are provenance-distinct, and those two
  // fields are the only isolation difference between them.
  assert.notDeepEqual(
    recycled.candidateIsolationProvenance,
    unrecycled.candidateIsolationProvenance
  );
  assert.deepEqual(
    { ...recycled.candidateIsolationProvenance, subprocessIsolation: null, sessionLifecycle: null },
    {
      ...unrecycled.candidateIsolationProvenance,
      subprocessIsolation: null,
      sessionLifecycle: null
    }
  );
  assert.deepEqual(
    recycled.candidateArtifactProvenance,
    unrecycled.candidateArtifactProvenance
  );
  assert.equal(
    recycled.candidateArtifactProvenance.manifestSha256,
    sha256ArtifactFile(release.manifestPath)
  );

  // The flag routes through admission, so a byte-level impostor of the official
  // release is refused and no report is written.
  const impostor = writeRecycledWasmRelease(directory, 'impostor', {
    entrySource: Buffer.concat([
      WASM_CANDIDATE_TEMPLATE_SOURCE,
      Buffer.from('\n// CLI-level impostor of the checked-in candidate template\n')
    ])
  });
  const impostorRun = strengthRun(impostor, ['--recycle-normal-duel-wasm-per-decision']);
  assert.notEqual(impostorRun.status, 0);
  assert.match(impostorRun.stderr, /adapter\.mjs SHA-256 does not equal the checked-in/);
  assert.equal(impostorRun.stdout, '');
});

test('enforced validation binds the exact candidate isolation and session lifecycle', async (t) => {
  const directory = mkdtempSync(join(tmpdir(), 'wrongway-recycled-enforced-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { base } = await recycledWasmCandidateBase(directory, 'genuine');
  const wrapper = await createRecycledNormalDuelWasmCandidateAdapter(base);

  // Exactly two internally supported pairs, each exact and frozen.
  assert.equal(Object.isFrozen(SUPPORTED_WORKER_SESSION_BOUNDARIES), true);
  for (const boundary of SUPPORTED_WORKER_SESSION_BOUNDARIES) {
    assert.equal(Object.isFrozen(boundary), true);
  }
  assert.deepEqual(SUPPORTED_WORKER_SESSION_BOUNDARIES, [
    {
      subprocessIsolation: 'node-subprocess-per-game-v1',
      sessionLifecycle: 'stateful-session-per-game-v1'
    },
    {
      subprocessIsolation: 'node-subprocess-per-decision-v1',
      sessionLifecycle: 'stateless-wasm-per-decision-v1'
    }
  ]);

  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const recycledReport = await runEvaluation({
    contender: wrapper,
    baseline: scriptedEngine('recycled-report-baseline', goalAction),
    book: { ...book, openings: book.openings.slice(0, 1) },
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    minimumOpeningPairs: 1
  });
  assert.deepEqual(
    recycledReport.candidateIsolationProvenance,
    getWorkerEngineIsolationProvenance(wrapper)
  );
  assert.equal(
    recycledReport.candidateIsolationProvenance.subprocessIsolation,
    'node-subprocess-per-decision-v1'
  );
  assert.equal(
    recycledReport.candidateIsolationProvenance.sessionLifecycle,
    'stateless-wasm-per-decision-v1'
  );
  assert.equal(recycledReport.baselineIsolationProvenance, null);

  const canonical = canonicalEnforcedCorpus();
  const hard = await createPinnedHardWorkerAdapter();
  const enforcedOptions = (contender) => ({
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    minimumOpeningPairs: MINIMUM_STRENGTH_OPENING_PAIRS,
    evaluatedOpeningCount: MINIMUM_STRENGTH_OPENING_PAIRS,
    evaluationSeed: CANONICAL_STRENGTH_SEED,
    contender,
    baseline: hard,
    book: canonical.book,
    corpusProvenance: canonical.provenance
  });

  // A public copy of the wrapper that relabels itself per-game holds none of
  // the private bindings, so enforcement refuses it on either host.
  const relabeled = Object.freeze({
    id: wrapper.id,
    version: wrapper.version,
    sourceCommit: wrapper.sourceCommit,
    baselineTrustRoot: wrapper.baselineTrustRoot,
    candidateArtifactProvenance: wrapper.candidateArtifactProvenance,
    capabilities: Object.freeze({
      ...wrapper.capabilities,
      isolation: 'node-subprocess-per-game-v1',
      sessionLifecycle: 'stateful-session-per-game-v1'
    }),
    createSession(context) {
      return wrapper.createSession(context);
    }
  });
  assert.equal(relabeled.capabilities.hardDeadlineIsolation, true);
  assert.equal(getWorkerEngineIsolationProvenance(relabeled), null);
  assert.throws(
    () => validateEnforcedStrengthOptions(enforcedOptions(relabeled)),
    /content-addressed candidate with canonical subprocess isolation/
  );

  if (!isAuthenticatedCandidateAdapter(base) || !isPinnedHardWorkerAdapter(hard)) {
    // Without verified taskpolicy support no adapter can be enforced-eligible,
    // so the token binding below is asserted only on a canonical host.
    assert.equal(isAuthenticatedCandidateAdapter(wrapper), false);
    for (const contender of [base, wrapper]) {
      assert.throws(
        () => validateEnforcedStrengthOptions(enforcedOptions(contender)),
        /content-addressed candidate with canonical subprocess isolation/
      );
    }
    return;
  }

  const baseToken = validateEnforcedStrengthOptions(enforcedOptions(base));
  const wrapperToken = validateEnforcedStrengthOptions(enforcedOptions(wrapper));
  assert.equal(baseToken.candidateIsolation, 'node-subprocess-per-game-v1');
  assert.equal(baseToken.candidateSessionLifecycle, 'stateful-session-per-game-v1');
  assert.equal(wrapperToken.candidateIsolation, 'node-subprocess-per-decision-v1');
  assert.equal(wrapperToken.candidateSessionLifecycle, 'stateless-wasm-per-decision-v1');
  // The historical 900 ms canonical gate accepts either explicitly reported
  // lifecycle: recycling is never silently required, and the token differs in
  // exactly the two lifecycle fields.
  for (const token of [baseToken, wrapperToken]) {
    assert.equal(token.enforced, true);
    assert.equal(token.eligible, true);
    assert.equal(token.perMoveDeadlineMs, CANONICAL_STRENGTH_DEADLINE_MS);
  }
  assert.notDeepEqual({ ...wrapperToken }, { ...baseToken });
  const withoutLifecycle = (token) => ({
    ...token,
    candidateIsolation: null,
    candidateSessionLifecycle: null
  });
  assert.deepEqual(withoutLifecycle(wrapperToken), withoutLifecycle(baseToken));

  // A private pair that is neither supported pair is unreachable by
  // construction: the only two writers of private isolation metadata are the
  // proxy's own adapter factory and its recycler, each of which publishes one
  // exact pair, and the private brand check refuses any other shape before
  // enforcement reads it. Nothing here adds a seam to forge one.
  for (const engine of [base, wrapper]) {
    const provenance = getWorkerEngineIsolationProvenance(engine);
    assert.ok(SUPPORTED_WORKER_SESSION_BOUNDARIES.some((boundary) =>
      boundary.subprocessIsolation === provenance.subprocessIsolation
      && boundary.sessionLifecycle === provenance.sessionLifecycle));
    assert.equal(engine.capabilities.isolation, provenance.subprocessIsolation);
    assert.equal(engine.capabilities.sessionLifecycle, provenance.sessionLifecycle);
  }
});

test('canonical 9x9 strength evaluations retain the real monotonic clock profile', async () => {
  const book = JSON.parse(readFileSync(BOOK_URL, 'utf8'));
  const smokeBook = { ...book, openings: book.openings.slice(0, 1) };
  const report = await runEvaluation({
    contender: scriptedEngine('strength-clock-candidate', goalAction),
    baseline: scriptedEngine('strength-clock-baseline', goalAction),
    book: smokeBook,
    mode: STRENGTH_MODE,
    perMoveDeadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    minimumOpeningPairs: 1
  });
  assert.deepEqual(report.clockProfile, REAL_MONOTONIC_CLOCK_PROFILE);
  for (const result of report.results) {
    assert.deepEqual(result.settings.clockProfile, REAL_MONOTONIC_CLOCK_PROFILE);
  }
});
