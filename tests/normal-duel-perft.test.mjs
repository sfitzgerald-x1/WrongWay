import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  applyAction, applyLegalAction, createInitialState, encodeAction, legalActions, validateConfig, validateState
} from '../js/normal-duel-engine.mjs';
import {
  MAX_PERFT_DEPTH, MAX_PERFT_NODES, MAX_PERFT_NODES_HARD_CAP,
  perft, perftReport, seededWallState, stateFromActionCodes
} from '../js/normal-duel-perft.mjs';
import { LCG32_ALGORITHM, createLcg32 } from '../js/lcg32.mjs';
import {
  FIXTURE_FORMAT, GENERATOR_VERSION, createPerftManifest, generatePerftFixture,
  renderPerftFixture, renderPerftManifest
} from '../scripts/generate-normal-duel-perft.mjs';

const fixtureText = readFileSync(new URL('./fixtures/normal-duel-perft-v1.json', import.meta.url), 'utf8');
const manifestText = readFileSync(new URL('./fixtures/normal-duel-perft-v1.manifest.json', import.meta.url), 'utf8');
const fixture = JSON.parse(fixtureText); const manifest = JSON.parse(manifestText);
const fixtureKeys = new Set(['fixtureFormat', 'ruleset', 'source', 'generator', 'semantics', 'configs', 'cases']);
const corpusGeneratorKeys = new Set(['name', 'version', 'seededAlgorithm', 'seededSelection']);
const semanticsKeys = new Set(['leafDefinition', 'deduplication', 'actionOrder', 'countType', 'divideIndexing', 'nodeBudget']);
const caseKeys = new Set(['id', 'kind', 'configId', 'actionCodes', 'generator', 'perftOptions', 'provenance', 'state', 'expect']);
const generatorKeys = new Set(['algorithm', 'seed', 'plies']);
const expectedKeys = new Set(['depth', 'leavesByDepth', 'nodeVisits', 'rootActionCodes', 'divide']);
const reportKeys = new Set(['depth', 'leavesByDepth', 'nodeVisits', 'divide']);
const divideEntryKeys = new Set(['action', 'actionCode', 'childLeavesByDepth']);
const manifestKeys = new Set(['fixtureFormat', 'generatorVersion', 'prng', 'caseCount', 'sha256', 'cases']);
const manifestCaseKeys = new Set(['id', 'kind', 'configId', 'config', 'provenance', 'perftOptions', 'state', 'expect']);
const manifestExpectedKeys = new Set(['depth', 'rootActionCount', 'nodeVisits', 'leavesByDepth']);

function assertExactKeys(value, keys, label) {
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), `${label} keys`);
}
function assertEngineCode(fn, code, label) {
  assert.throws(fn, (error) => error?.code === code, label);
}
function stateFor(entry) {
  const config = fixture.configs[entry.configId];
  if (entry.kind === 'initial') return createInitialState(config);
  return entry.kind === 'action-codes'
    ? stateFromActionCodes(config, entry.actionCodes)
    : seededWallState(config, entry.generator);
}
function divideVectors(report) {
  return report.divide.map(({ actionCode, childLeavesByDepth }) => [actionCode, [...childLeavesByDepth]]);
}
function optionsFor(entry) { return entry.perftOptions ?? undefined; }

/*
 * Deliberately test-only oracle: all transitions flow through public
 * applyAction, so it does not share perft's applyLegalAction fast path.
 */
function applyActionOnlyExactPerft(config, state, depth) {
  if (depth === 0) return 1;
  if (state.outcome.kind !== 'ongoing') return 0;
  const actions = legalActions(config, state);
  if (depth === 1) return actions.length;
  return actions.reduce((total, action) => total + applyActionOnlyExactPerft(config, applyAction(config, state, action), depth - 1), 0);
}

function assertProvenance(entry, config) {
  assert.deepEqual(entry.provenance.config, config, `${entry.id} provenance config`);
  if (entry.kind === 'initial') {
    assertExactKeys(entry.provenance, new Set(['mode', 'config']), `${entry.id} provenance`);
    assert.equal(entry.provenance.mode, 'initial-state-v1');
    return;
  }
  if (entry.kind === 'action-codes') {
    assertExactKeys(entry.provenance, new Set(['mode', 'config', 'actionCodes']), `${entry.id} provenance`);
    assert.equal(entry.provenance.mode, 'explicit-action-codes-v1');
    assert.deepEqual(entry.provenance.actionCodes, entry.actionCodes, `${entry.id} provenance actions`);
    return;
  }
  assertExactKeys(entry.provenance, new Set(['mode', 'config', 'algorithm', 'seed', 'plies', 'selection']), `${entry.id} provenance`);
  assert.equal(entry.provenance.mode, 'seeded-wall-only-lcg32-v1');
  assert.equal(entry.provenance.algorithm, entry.generator.algorithm, `${entry.id} provenance algorithm`);
  assert.equal(entry.provenance.seed, entry.generator.seed, `${entry.id} provenance seed`);
  assert.equal(entry.provenance.plies, entry.generator.plies, `${entry.id} provenance plies`);
  assert.equal(entry.provenance.selection, fixture.generator.seededSelection, `${entry.id} provenance selection`);
}

function validateFixture() {
  assertExactKeys(fixture, fixtureKeys, 'fixture');
  assert.equal(fixture.fixtureFormat, FIXTURE_FORMAT);
  assert.equal(fixture.ruleset, 'normal-duel-v1');
  assert.equal(fixture.source, 'js/normal-duel-engine.mjs legalActions/applyLegalAction');
  assertExactKeys(fixture.generator, corpusGeneratorKeys, 'corpus generator');
  assert.deepEqual(fixture.generator, {
    name: 'normal-duel-perft', version: GENERATOR_VERSION, seededAlgorithm: 'lcg32-v1',
    seededSelection: 'initialize from uint32 seed; advance before each selection; filter canonical ascending legal actions to walls only; choose walls[state%walls.length]; modulo bias is intentional and no rejection sampling is used'
  });
  assertExactKeys(fixture.semantics, semanticsKeys, 'semantics');
  assert.equal(fixture.semantics.leafDefinition, 'exact-depth-v1: P(s,0)=1; P(terminal,d>0)=0; P(ongoing,d)=sum(P(child,d-1))');
  assert.equal(fixture.semantics.deduplication, 'none; count every legal action-tree occurrence');
  assert.equal(fixture.semantics.actionOrder, 'ascending canonical policy action code');
  assert.equal(fixture.semantics.countType, 'safe integer only; traversal rejects counts above Number.MAX_SAFE_INTEGER');
  assert.equal(fixture.semantics.divideIndexing, 'divide[i].childLeavesByDepth[d]=P(apply(rootAction[i]),d); root leavesByDepth[d+1]=sum_i childLeavesByDepth[d]');
  assert.equal(fixture.semantics.nodeBudget, 'state visits use a conservative MAX_PERFT_NODES default; explicit maxNodes may opt in up to MAX_PERFT_NODES_HARD_CAP, and traversal rejects before exceeding the selected deterministic cap');
  assert.equal(fixture.cases.length, 9, 'freeze a compact but discriminating set of cases');
  assertExactKeys(manifest, manifestKeys, 'manifest');
  assert.equal(manifest.fixtureFormat, fixture.fixtureFormat, 'manifest fixture format');
  assert.equal(manifest.generatorVersion, GENERATOR_VERSION, 'manifest generator version');
  assert.equal(manifest.caseCount, fixture.cases.length, 'manifest case count');

  const ids = new Set(); const covered = new Set();
  for (const entry of fixture.cases) {
    assertExactKeys(entry, caseKeys, entry.id);
    assert.equal(typeof entry.id, 'string'); assert.ok(!ids.has(entry.id), `duplicate case ${entry.id}`); ids.add(entry.id);
    assert.ok(['initial', 'action-codes', 'seeded-walls'].includes(entry.kind), `${entry.id} kind`);
    const config = fixture.configs[entry.configId]; assert.ok(config, `${entry.id} config`); validateConfig(config);
    covered.add(`${config.rows}x${config.columns}:${entry.kind}`);
    assert.deepEqual(validateState(config, entry.state), entry.state, `${entry.id} canonical GameState`);
    assertExactKeys(entry.expect, expectedKeys, `${entry.id} expected`);
    assert.ok(entry.perftOptions === null || (entry.perftOptions !== null && typeof entry.perftOptions === 'object'
      && !Array.isArray(entry.perftOptions)), `${entry.id} perft options`);
    if (entry.perftOptions !== null) {
      assertExactKeys(entry.perftOptions, new Set(['maxNodes']), `${entry.id} perft options`);
      assert.ok(Number.isSafeInteger(entry.perftOptions.maxNodes)
        && entry.perftOptions.maxNodes > MAX_PERFT_NODES
        && entry.perftOptions.maxNodes <= MAX_PERFT_NODES_HARD_CAP, `${entry.id} bounded perft opt-in`);
    }
    assert.ok(entry.provenance !== null && typeof entry.provenance === 'object' && !Array.isArray(entry.provenance), `${entry.id} provenance`);
    assertProvenance(entry, config);
    assert.ok(Number.isInteger(entry.expect.depth) && entry.expect.depth >= 2 && entry.expect.depth <= MAX_PERFT_DEPTH, `${entry.id} depth`);
    assert.equal(entry.expect.leavesByDepth.length, entry.expect.depth + 1, `${entry.id} exact-depth vector`);
    assert.equal(entry.expect.leavesByDepth[0], 1, `${entry.id} root zero-depth leaf`);
    for (const count of entry.expect.leavesByDepth) assert.ok(Number.isSafeInteger(count) && count >= 0, `${entry.id} count`);
    const caseBudget = entry.perftOptions?.maxNodes ?? MAX_PERFT_NODES;
    assert.ok(Number.isSafeInteger(entry.expect.nodeVisits) && entry.expect.nodeVisits > 0
      && entry.expect.nodeVisits <= caseBudget && caseBudget <= MAX_PERFT_NODES_HARD_CAP, `${entry.id} bounded visit count`);
    assert.ok(entry.expect.rootActionCodes.every(Number.isSafeInteger), `${entry.id} root action codes`);
    assert.equal(entry.expect.divide.length, entry.expect.rootActionCodes.length, `${entry.id} divide size`);
    assert.deepEqual(entry.expect.divide.map(([code]) => code), entry.expect.rootActionCodes, `${entry.id} divide code order`);
    for (const [code, childLeavesByDepth] of entry.expect.divide) {
      assert.ok(Number.isSafeInteger(code), `${entry.id} divide code`);
      assert.equal(childLeavesByDepth.length, entry.expect.depth, `${entry.id} child depth vector`);
      assert.equal(childLeavesByDepth[0], 1, `${entry.id} child zero-depth leaf`);
      for (const count of childLeavesByDepth) assert.ok(Number.isSafeInteger(count) && count >= 0, `${entry.id} child count`);
    }
    if (entry.kind === 'initial') {
      assert.equal(entry.actionCodes, null, `${entry.id} action codes`);
      assert.equal(entry.generator, null, `${entry.id} generator`);
    }
    if (entry.kind === 'action-codes') {
      assert.ok(Array.isArray(entry.actionCodes) && entry.actionCodes.length > 0, `${entry.id} action codes`);
      assert.equal(entry.generator, null, `${entry.id} generator`);
    }
    if (entry.kind === 'seeded-walls') {
      assert.equal(entry.actionCodes, null, `${entry.id} action codes`);
      assertExactKeys(entry.generator, generatorKeys, `${entry.id} generator`);
      assert.equal(entry.generator.algorithm, fixture.generator.seededAlgorithm);
      assert.ok(Number.isInteger(entry.generator.seed) && entry.generator.seed >= 0 && entry.generator.seed <= 0xffff_ffff);
      assert.ok(Number.isInteger(entry.generator.plies) && entry.generator.plies > 0);
    }
  }
  assert.ok(covered.has('9x9:initial') && covered.has('7x7:initial'), 'both board sizes have initial geometry coverage');
  assert.ok(covered.has('9x9:seeded-walls') && covered.has('7x7:seeded-walls'), 'both board sizes have seeded coverage');
  assert.equal(manifest.cases.length, fixture.cases.length, 'manifest case entries');
  for (const [index, entry] of fixture.cases.entries()) {
    const manifestCase = manifest.cases[index];
    assertExactKeys(manifestCase, manifestCaseKeys, `${entry.id} manifest case`);
    assert.equal(manifestCase.id, entry.id, `${entry.id} manifest id`);
    assert.equal(manifestCase.kind, entry.kind, `${entry.id} manifest kind`);
    assert.equal(manifestCase.configId, entry.configId, `${entry.id} manifest config id`);
    assert.deepEqual(manifestCase.config, fixture.configs[entry.configId], `${entry.id} manifest config`);
    assert.deepEqual(manifestCase.provenance, entry.provenance, `${entry.id} manifest provenance`);
    assert.deepEqual(manifestCase.perftOptions, entry.perftOptions, `${entry.id} manifest perft options`);
    assert.deepEqual(manifestCase.state, entry.state, `${entry.id} manifest state`);
    assertExactKeys(manifestCase.expect, manifestExpectedKeys, `${entry.id} manifest expectation`);
    assert.deepEqual(manifestCase.expect, {
      depth: entry.expect.depth,
      rootActionCount: entry.expect.rootActionCodes.length,
      nodeVisits: entry.expect.nodeVisits,
      leavesByDepth: entry.expect.leavesByDepth
    }, `${entry.id} manifest expectation`);
  }
}

test('perft corpus freezes full canonical root action and child-depth divide vectors', () => {
  validateFixture();
  for (const entry of fixture.cases) {
    const config = fixture.configs[entry.configId]; const state = stateFor(entry);
    assert.deepEqual(state, entry.state, `${entry.id} generator/action replay drifted`);
    const report = perftReport(config, state, entry.expect.depth, optionsFor(entry));
    assertExactKeys(report, reportKeys, `${entry.id} report surface`);
    assert.deepEqual(report.leavesByDepth, entry.expect.leavesByDepth, entry.id);
    assert.equal(report.nodeVisits, entry.expect.nodeVisits, `${entry.id} counted state visits`);
    assert.deepEqual(report.divide.map(({ actionCode }) => actionCode), entry.expect.rootActionCodes, `${entry.id} root order`);
    assert.deepEqual(divideVectors(report), entry.expect.divide, `${entry.id} child exact-depth divide`);
    assert.deepEqual(report.divide.map(({ actionCode }) => actionCode),
      legalActions(config, state).map((action) => encodeAction(config, action)), `${entry.id} canonical engine order`);
    assert.deepEqual(report.divide.map(({ actionCode }) => actionCode),
      [...report.divide.map(({ actionCode }) => actionCode)].sort((left, right) => left - right), `${entry.id} ascending codes`);
    for (const divideEntry of report.divide) {
      assertExactKeys(divideEntry, divideEntryKeys, `${entry.id} divide surface`);
      assertExactKeys(divideEntry.action,
        divideEntry.action.kind === 'pawn' ? new Set(['kind', 'to']) : new Set(['kind', 'wall']),
        `${entry.id} divide action surface`);
      if (divideEntry.action.kind === 'pawn') {
        assertExactKeys(divideEntry.action.to, new Set(['r', 'c']), `${entry.id} pawn destination surface`);
      }
    }
    for (let depth = 1; depth <= report.depth; depth += 1) {
      assert.equal(report.leavesByDepth[depth], report.divide.reduce(
        (total, entry) => total + entry.childLeavesByDepth[depth - 1], 0
      ), `${entry.id} divide root/child indexing at ${depth}`);
    }
    assert.ok(Object.isFrozen(report) && Object.isFrozen(report.divide) && Object.isFrozen(report.leavesByDepth), `${entry.id} immutable report`);
    assert.ok(report.divide.every((entry) => Object.isFrozen(entry) && Object.isFrozen(entry.childLeavesByDepth)), `${entry.id} immutable child vectors`);
    for (let depth = 0; depth <= entry.expect.depth; depth += 1) {
      assert.equal(perft(config, state, depth, optionsFor(entry)), entry.expect.leavesByDepth[depth], `${entry.id} scalar perft at depth ${depth}`);
    }
  }
});

test('applyLegalAction matches applyAction for trusted canonical actions without mutating inputs', () => {
  for (const entry of fixture.cases.filter(({ id }) => id !== 'terminal-goal-win-7x7-a')) {
    const config = fixture.configs[entry.configId]; const state = stateFor(entry);
    if (state.outcome.kind !== 'ongoing') continue;
    const actions = legalActions(config, state);
    const samples = [actions[0], actions.find(({ kind }) => kind === 'wall')].filter(Boolean);
    for (const action of samples) {
      const stateBefore = structuredClone(state); const actionBefore = structuredClone(action);
      const trusted = applyLegalAction(config, state, action);
      assert.deepEqual(trusted, applyAction(config, state, action), `${entry.id}/${action.kind}`);
      assert.deepEqual(state, stateBefore, `${entry.id}/${action.kind} state remains immutable`);
      assert.deepEqual(action, actionBefore, `${entry.id}/${action.kind} action remains immutable`);
    }
  }
});

test('applyLegalAction rejects fabricated pawn and wall candidates without rebuilding all actions', () => {
  const config = fixture.configs.standardA; const initial = createInitialState(config);
  assertEngineCode(() => applyLegalAction(config, initial, { kind: 'pawn', to: { r: 0, c: 4 } }), 'illegal_action',
    'in-bounds fabricated goal pawn is rejected');

  const afterHorizontal = applyAction(config, initial, { kind: 'wall', wall: 'H-0-0' });
  assertEngineCode(() => applyLegalAction(config, afterHorizontal, { kind: 'wall', wall: 'H-0-0' }), 'illegal_action',
    'duplicate wall is rejected');
  assertEngineCode(() => applyLegalAction(config, afterHorizontal, { kind: 'wall', wall: 'V-0-0' }), 'illegal_action',
    'crossing wall is rejected');

  const stocklessConfig = fixture.configs.blitzEndgameA;
  assertEngineCode(() => applyLegalAction(stocklessConfig, createInitialState(stocklessConfig), { kind: 'wall', wall: 'H-0-0' }), 'illegal_action',
    'wall is rejected when the mover has no stock');
});

test('applyAction-only exact perft oracle agrees with cheap frozen cases', () => {
  const oracleCaseIds = new Set([
    'initial-geometry-7x7-a',
    'curated-remaining-walls-7x7-a',
    'seeded-wall-exhausted-9x9-a',
    'terminal-goal-win-7x7-a'
  ]);
  for (const entry of fixture.cases.filter(({ id }) => oracleCaseIds.has(id))) {
    assert.equal(entry.perftOptions, null, `${entry.id} stays in the cheap default-budget oracle set`);
    const config = fixture.configs[entry.configId];
    assert.equal(applyActionOnlyExactPerft(config, stateFor(entry), entry.expect.depth), entry.expect.leavesByDepth.at(-1), entry.id);
  }
});

test('fixtures include reachable remaining-wall depth-three branches at both CI scales', () => {
  const entry = fixture.cases.find(({ id }) => id === 'curated-remaining-walls-7x7-a');
  const config = fixture.configs[entry.configId]; const state = stateFor(entry); const report = perftReport(config, state, entry.expect.depth, optionsFor(entry));
  assert.equal(state.position.stock.A > 0 && state.position.stock.B > 0, true, 'both sides retain walls');
  assert.ok(report.leavesByDepth.at(-1) >= 1_000 && report.leavesByDepth.at(-1) < 10_000, 'depth-three count stays CI-sized');
  assert.ok(report.divide.some(({ action }) => action.kind === 'wall'), 'root has wall branching');
  assert.ok(report.divide.some(({ action }) => legalActions(config, applyAction(config, state, action)).some((child) => child.kind === 'wall')),
    'at least one deeper ply has wall branching');

  const wallRich = fixture.cases.find(({ id }) => id === 'seeded-remaining-walls-9x9-a');
  const wallConfig = fixture.configs[wallRich.configId]; const wallState = stateFor(wallRich);
  assert.deepEqual(wallState.position.stock, { A: 1, B: 1 }, '9x9 probe deliberately retains one wall each');
  assert.deepEqual(wallRich.perftOptions, { maxNodes: 4096 }, '9x9 wall-rich probe declares its bounded opt-in');
  assert.ok(wallRich.expect.nodeVisits <= wallRich.perftOptions.maxNodes, '9x9 probe remains below its explicit evaluation budget');
  assert.deepEqual(wallRich.expect.leavesByDepth, [1, 62, 3594, 20893], '9x9 wall-rich exact-depth counts');
  const rootWall = legalActions(wallConfig, wallState).find(({ kind }) => kind === 'wall');
  assert.ok(rootWall, '9x9 root is wall-heavy');
  const rootPawn = legalActions(wallConfig, wallState).find(({ kind }) => kind === 'pawn');
  assert.ok(rootPawn, '9x9 root also permits a pawn line that retains both walls');
  const afterRootPawn = applyAction(wallConfig, wallState, rootPawn);
  const replyPawn = legalActions(wallConfig, afterRootPawn).find(({ kind }) => kind === 'pawn');
  assert.ok(replyPawn, '9x9 first deeper layer retains a pawn reply');
  assert.ok(legalActions(wallConfig, applyAction(wallConfig, afterRootPawn, replyPawn))
    .some(({ kind }) => kind === 'wall'), '9x9 second deeper layer retains wall branching');
});

test('perft reports preserve explicit depth-zero and depth-one accounting', () => {
  const entry = fixture.cases.find(({ id }) => id === 'initial-geometry-7x7-a');
  const config = fixture.configs[entry.configId]; const state = stateFor(entry);
  const depthZero = perftReport(config, state, 0);
  assert.deepEqual(depthZero, { depth: 0, leavesByDepth: [1], nodeVisits: 1, divide: [] });

  const depthOne = perftReport(config, state, 1);
  assertExactKeys(depthOne, reportKeys, 'depth-one report surface');
  assert.deepEqual(depthOne.leavesByDepth, [1, entry.expect.leavesByDepth[1]]);
  assert.equal(depthOne.nodeVisits, 1, 'depth-one root children are algebraic leaf occurrences');
  assert.equal(depthOne.divide.length, entry.expect.rootActionCodes.length, 'depth-one divide covers every root action');
  assert.ok(depthOne.divide.every(({ childLeavesByDepth }) => childLeavesByDepth.length === 1 && childLeavesByDepth[0] === 1),
    'depth-one divide stores only algebraic child depth zero');
});

test('terminal artifacts use exact-depth semantics, including early terminal children', () => {
  const terminal = fixture.cases.find(({ id }) => id === 'terminal-goal-win-7x7-a');
  const terminalReport = perftReport(fixture.configs[terminal.configId], stateFor(terminal), terminal.expect.depth, optionsFor(terminal));
  assert.deepEqual(terminalReport.leavesByDepth, [1, 0, 0, 0, 0]);
  assert.deepEqual(terminalReport.divide, []);

  const ongoing = fixture.cases.find(({ id }) => id === 'ongoing-early-goal-child-7x7-a');
  const config = fixture.configs[ongoing.configId]; const state = stateFor(ongoing); const report = perftReport(config, state, ongoing.expect.depth, optionsFor(ongoing));
  assert.equal(state.outcome.kind, 'ongoing');
  const winningChild = report.divide.find(({ actionCode }) => actionCode === 3);
  assert.deepEqual(winningChild.childLeavesByDepth, [1, 0, 0]);
  assert.deepEqual(applyAction(config, state, winningChild.action).outcome, { kind: 'win', winner: 'A', reason: 'goal' });
});

test('first-player B seeded root is ongoing and has a frozen non-empty divide', () => {
  const entry = fixture.cases.find(({ id }) => id === 'seeded-first-player-b-9x9');
  const config = fixture.configs[entry.configId]; const state = stateFor(entry); const report = perftReport(config, state, entry.expect.depth, optionsFor(entry));
  assert.equal(config.firstPlayer, 'B');
  assert.equal(state.outcome.kind, 'ongoing');
  assert.equal(state.position.turn, 'B');
  assert.ok(report.divide.length > 0);
  assert.deepEqual(divideVectors(report), entry.expect.divide);
});

test('seeded wall construction has specified advance-then-modulo selection and safe terminal/exhaustion rejection', () => {
  assert.equal(fixture.generator.seededAlgorithm, LCG32_ALGORITHM, 'fixture names the shared LCG contract');
  const stream = createLcg32(0);
  assert.deepEqual([stream(), stream(), stream(), stream()], [1013904223, 1196435762, 3519870697, 2868466484],
    'shared LCG advances before each deterministic selection');
  for (const entry of fixture.cases.filter(({ kind }) => kind === 'seeded-walls')) {
    const config = fixture.configs[entry.configId];
    const first = seededWallState(config, entry.generator); const second = seededWallState(config, entry.generator);
    assert.deepEqual(second, first, entry.id);
    assert.deepEqual(first, entry.state, entry.id);
    assert.equal(first.ply, entry.generator.plies, entry.id);
    assert.equal(first.position.stock.A + first.position.stock.B,
      config.initialStock.A + config.initialStock.B - entry.generator.plies, `${entry.id} retained stock`);
  }
  const firstAdvanced = seededWallState(fixture.configs.blitzA, { seed: 0, plies: 1 });
  assert.deepEqual(firstAdvanced.position.walls, ['H-1-1'], 'LCG advances before wall-only modulo selection');
  const stockless = { ...fixture.configs.blitzA, initialStock: { A: 0, B: 0 } };
  assert.deepEqual(seededWallState(stockless, { seed: 0, plies: 0 }), createInitialState(stockless));
  assert.throws(() => seededWallState(stockless, { seed: 0, plies: 1 }), /exceeds total initial wall stock/);
  assert.throws(() => seededWallState(fixture.configs.blitzA, { seed: 0, plies: 21 }), /exceeds total initial wall stock/);
  const capped = { ...fixture.configs.blitzA, plyCap: 1 };
  assert.throws(() => seededWallState(capped, { seed: 0, plies: 2 }), /state became terminal at ply 1/);
});

test('perft has a deterministic bounded-node guard before broad traversal', () => {
  const config = fixture.configs.blitzA; const initial = createInitialState(config);
  assert.equal(MAX_PERFT_NODES, 400);
  assert.ok(MAX_PERFT_NODES_HARD_CAP > MAX_PERFT_NODES);
  assert.throws(() => perftReport(config, initial, 3), /node budget exceeded \(400 counted state visits\)/,
    'the default cap promptly stops a broad depth-three traversal');
  assert.throws(() => perftReport(config, initial, 2, { maxNodes: 2 }), /node budget exceeded \(2 counted state visits\)/);
  assert.deepEqual(perftReport(config, initial, 2, { maxNodes: MAX_PERFT_NODES + 1 }).leavesByDepth, [1, 75, 5357],
    'an explicit, bounded opt-in may exceed the conservative default');
  assert.throws(() => perftReport(config, initial, 2, { maxNodes: MAX_PERFT_NODES_HARD_CAP + 1 }),
    new RegExp(`maxNodes exceeds MAX_PERFT_NODES_HARD_CAP \\(${MAX_PERFT_NODES_HARD_CAP}\\)`));
  assert.throws(() => perft(config, initial, -1), /non-negative safe integer/);
  assert.throws(() => perftReport(config, initial, MAX_PERFT_DEPTH + 1), /exceeds MAX_PERFT_DEPTH/);
});

test('perft generator and SHA-256 manifest exactly reproduce the frozen artifacts', () => {
  assert.deepEqual(generatePerftFixture(), fixture);
  assert.equal(renderPerftFixture(), fixtureText);
  assert.equal(renderPerftManifest(fixtureText), manifestText);
  assert.deepEqual(createPerftManifest(fixtureText), manifest);
  assert.equal(manifest.sha256, createHash('sha256').update(fixtureText, 'utf8').digest('hex'));
});
