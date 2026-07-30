import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import * as engine from '../js/normal-duel-engine.mjs';
import {
  CORPUS_FORMAT, CORPUS_PATH, EXPLICIT_SELECTION_MODE, GENERATOR_VERSION, MANIFEST_PATH, PRNG,
  SEEDED_SELECTION_MODE, createCorpusManifest, createLcg32, generateTrajectoryRecords,
  renderCorpusManifest, renderTrajectoryCorpus, selectSeededAction
} from '../scripts/generate-normal-duel-trajectories.mjs';

const RECORD_KEYS = new Set([
  'corpusFormat', 'generatorVersion', 'selectionMode', 'prng', 'trajectoryId', 'seed', 'step',
  'configuration', 'state', 'legalActionCodes', 'selectedAction', 'selectedActionCode',
  'nextState', 'outcome', 'endReason'
]);
const MANIFEST_KEYS = new Set([
  'corpusFormat', 'generatorVersion', 'selectionModes', 'lineCount', 'sha256', 'trajectories'
]);
const SEEDED_SELECTION_KEYS = new Set(['prng', 'seed', 'transition', 'selection']);
const EXPLICIT_SELECTION_KEYS = new Set(['prng', 'seed', 'selection']);
const MANIFEST_TRAJECTORY_KEYS = new Set([
  'trajectoryId', 'selectionMode', 'seed', 'prng', 'board', 'firstPlayer', 'plyCap',
  'configuration', 'configurationSha256', 'selectedActionCodes', 'endReason'
]);
const END_REASONS = new Set(['continue', 'sample_limit', 'goal', 'threefold_repetition', 'ply_cap']);
const EXPECTED_FINAL_REASONS = Object.freeze({
  'sample-standard-9x9-a': 'sample_limit',
  'sample-blitz-7x7-b': 'sample_limit',
  'terminal-goal-7x7-a': 'goal',
  'terminal-threefold-9x9-b': 'threefold_repetition',
  'terminal-ply-cap-9x9-a': 'ply_cap'
});
const PINNED_ACTION_CODES = Object.freeze({
  'sample-standard-9x9-a': [153, 5, 67, 91, 58, 4],
  'sample-blitz-7x7-b': [56, 46, 2, 64, 3, 39],
  'terminal-goal-7x7-a': [38, 2, 31, 3, 24, 2, 17, 3, 10, 2, 3],
  'terminal-threefold-9x9-b': [5, 75, 4, 76, 5, 75, 4, 76],
  'terminal-ply-cap-9x9-a': [67, 13]
});
const frozenText = readFileSync(CORPUS_PATH, 'utf8');
const manifestText = readFileSync(MANIFEST_PATH, 'utf8');

function exactKeys(value, keys, message) {
  assert.deepEqual(new Set(Object.keys(value)), keys, message);
}
function parseStrictJsonl(text) {
  assert.equal(text.includes('\r'), false, 'JSONL uses LF only');
  assert.equal(text.endsWith('\n'), true, 'JSONL has exactly one final LF');
  const lines = text.slice(0, -1).split('\n');
  assert.ok(lines.length > 0, 'JSONL is non-empty');
  for (const line of lines) assert.notEqual(line, '', 'JSONL has no blank lines');
  return lines.map((line, index) => {
    let record;
    assert.doesNotThrow(() => { record = JSON.parse(line); }, `line ${index + 1} is valid JSON`);
    exactKeys(record, RECORD_KEYS, `line ${index + 1} has no unknown or omitted fields`);
    return record;
  });
}

test('normal-duel-v1 frozen trajectory corpus is strictly shaped and exactly replayable', () => {
  const records = parseStrictJsonl(frozenText);
  assert.deepEqual(new Set(records.map((record) => `${record.configuration.rows}x${record.configuration.columns}`)), new Set(['9x9', '7x7']));
  assert.deepEqual(new Set(records.map((record) => record.configuration.firstPlayer)), new Set(['A', 'B']));
  assert.equal(records.some((record) => record.selectedAction.kind === 'wall'), true);
  assert.equal(records.some((record) => record.selectedAction.kind === 'pawn'), true);
  assert.deepEqual(new Set(records.filter((record) => record.outcome.kind !== 'ongoing').map((record) => record.endReason)), new Set(['goal', 'threefold_repetition', 'ply_cap']));
  for (const trajectoryId of ['sample-standard-9x9-a', 'sample-blitz-7x7-b']) {
    const sample = records.filter((record) => record.trajectoryId === trajectoryId);
    assert.equal(sample.length, 6, `${trajectoryId} has six linked records`);
    assert.equal(sample.filter((record) => record.selectedAction.kind === 'wall').length >= 2, true, `${trajectoryId} includes at least two walls`);
    assert.deepEqual(sample.map((record) => record.step), [0, 1, 2, 3, 4, 5]);
  }
  for (const [trajectoryId, selectedActionCodes] of Object.entries(PINNED_ACTION_CODES)) {
    assert.deepEqual(
      records.filter((record) => record.trajectoryId === trajectoryId).map((record) => record.selectedActionCode),
      selectedActionCodes,
      `${trajectoryId} selection trace is pinned`
    );
  }

  const previousByTrajectory = new Map();
  const randomByTrajectory = new Map();
  const lastStepByTrajectory = new Map();
  for (const record of records) lastStepByTrajectory.set(record.trajectoryId, record.step);
  for (const record of records) {
    assert.equal(record.corpusFormat, CORPUS_FORMAT);
    assert.equal(record.generatorVersion, GENERATOR_VERSION);
    assert.equal(new Set([SEEDED_SELECTION_MODE, EXPLICIT_SELECTION_MODE]).has(record.selectionMode), true);
    assert.equal(typeof record.trajectoryId, 'string');
    if (record.selectionMode === SEEDED_SELECTION_MODE) {
      assert.equal(record.prng, PRNG);
      assert.equal(Number.isSafeInteger(record.seed) && record.seed >= 0 && record.seed <= 0xffffffff, true);
    } else {
      assert.equal(record.prng, null);
      assert.equal(record.seed, null);
    }
    assert.equal(Number.isSafeInteger(record.step) && record.step >= 0, true);
    assert.equal(END_REASONS.has(record.endReason), true);
    assert.deepEqual(engine.validateConfig(record.configuration), record.configuration);
    assert.deepEqual(engine.validateState(record.configuration, record.state), record.state);
    assert.equal(record.legalActionCodes.every((code) => Number.isSafeInteger(code)), true);
    assert.equal(Number.isSafeInteger(record.selectedActionCode), true);
    assert.deepEqual(record.legalActionCodes, engine.legalActionCodes(record.configuration, record.state));
    assert.deepEqual(record.legalActionCodes, [...new Set(record.legalActionCodes)].sort((left, right) => left - right));
    assert.deepEqual(record.legalActionCodes.map((code) => engine.decodeAction(record.configuration, code)), engine.legalActions(record.configuration, record.state));
    assert.equal(record.selectedActionCode, engine.encodeAction(record.configuration, record.selectedAction));
    assert.equal(record.legalActionCodes.includes(record.selectedActionCode), true);
    if (record.selectionMode === SEEDED_SELECTION_MODE) {
      let nextRandom = randomByTrajectory.get(record.trajectoryId);
      if (!nextRandom) {
        nextRandom = createLcg32(record.seed);
        randomByTrajectory.set(record.trajectoryId, nextRandom);
      }
      const expected = selectSeededAction(engine.legalActions(record.configuration, record.state), nextRandom, record.step);
      assert.equal(record.selectedActionCode, engine.encodeAction(record.configuration, expected), 'seeded selection follows the pinned cadence and LCG');
    }
    const nextState = engine.applyAction(record.configuration, record.state, record.selectedAction);
    assert.deepEqual(record.nextState, nextState);
    assert.deepEqual(record.outcome, nextState.outcome);
    const isLast = record.step === lastStepByTrajectory.get(record.trajectoryId);
    const expectedEndReason = isLast ? EXPECTED_FINAL_REASONS[record.trajectoryId] : 'continue';
    assert.equal(record.endReason, expectedEndReason, `${record.trajectoryId} step ${record.step} has the pinned end reason`);
    const previous = previousByTrajectory.get(record.trajectoryId);
    if (previous) {
      assert.equal(record.step, previous.step + 1);
      assert.deepEqual(record.state, previous.nextState);
      assert.equal(previous.endReason, 'continue');
    } else {
      assert.equal(record.step, 0);
      assert.deepEqual(record.state, engine.createInitialState(record.configuration), `${record.trajectoryId} starts from the configured initial state`);
    }
    previousByTrajectory.set(record.trajectoryId, record);
  }
  for (const last of previousByTrajectory.values()) assert.notEqual(last.endReason, 'continue');
});

test('normal-duel-v1 trajectory corpus and manifest regenerate byte-identically with full replay identity', () => {
  const records = parseStrictJsonl(frozenText);
  assert.deepEqual(generateTrajectoryRecords(), records);
  assert.equal(renderTrajectoryCorpus(), frozenText);
  assert.equal(renderCorpusManifest(), manifestText);
  const manifest = JSON.parse(manifestText);
  assert.equal(manifestText.includes('\r'), false, 'manifest uses LF only');
  assert.equal(manifestText.endsWith('\n'), true, 'manifest has a final LF');
  exactKeys(manifest, MANIFEST_KEYS, 'manifest has no unknown or omitted fields');
  exactKeys(manifest.selectionModes[SEEDED_SELECTION_MODE], SEEDED_SELECTION_KEYS, 'seeded selection specification is exact');
  exactKeys(manifest.selectionModes[EXPLICIT_SELECTION_MODE], EXPLICIT_SELECTION_KEYS, 'explicit selection specification is exact');
  assert.equal(manifest.selectionModes[SEEDED_SELECTION_MODE].prng, PRNG);
  assert.equal(manifest.selectionModes[SEEDED_SELECTION_MODE].seed, 'uint32');
  assert.equal(manifest.selectionModes[EXPLICIT_SELECTION_MODE].prng, null);
  assert.equal(manifest.selectionModes[EXPLICIT_SELECTION_MODE].seed, null);
  for (const trajectory of manifest.trajectories) {
    exactKeys(trajectory, MANIFEST_TRAJECTORY_KEYS, `${trajectory.trajectoryId} manifest identity is exact`);
    assert.equal(new Set([SEEDED_SELECTION_MODE, EXPLICIT_SELECTION_MODE]).has(trajectory.selectionMode), true);
    if (trajectory.selectionMode === SEEDED_SELECTION_MODE) {
      assert.equal(trajectory.prng, PRNG);
      assert.equal(Number.isSafeInteger(trajectory.seed) && trajectory.seed >= 0 && trajectory.seed <= 0xffffffff, true);
    } else {
      assert.equal(trajectory.prng, null);
      assert.equal(trajectory.seed, null);
    }
    assert.deepEqual(engine.validateConfig(trajectory.configuration), trajectory.configuration);
    assert.equal(trajectory.plyCap, trajectory.configuration.plyCap);
    assert.equal(
      trajectory.configurationSha256,
      createHash('sha256').update(JSON.stringify(trajectory.configuration), 'utf8').digest('hex')
    );
    assert.deepEqual(trajectory.selectedActionCodes, PINNED_ACTION_CODES[trajectory.trajectoryId]);
  }
  assert.deepEqual(manifest, createCorpusManifest(frozenText));
  assert.equal(manifest.lineCount, records.length);
  assert.equal(manifest.sha256, createHash('sha256').update(frozenText, 'utf8').digest('hex'));
});

test('seeded trajectory selection rejects invalid seeds and empty cadence candidates', () => {
  assert.throws(() => createLcg32(-1), /uint32 seed/);
  assert.throws(() => createLcg32(0x1_0000_0000), /uint32 seed/);
  assert.throws(
    () => selectSeededAction([{ kind: 'pawn', to: { r: 0, c: 0 } }], () => 0, 0),
    /no wall candidates at cadence step 0/
  );
});

test('generated JSON artifacts are protected by the repository LF policy', () => {
  const attributes = new Set(
    readFileSync(new URL('../.gitattributes', import.meta.url), 'utf8').trimEnd().split('\n')
  );
  assert.equal(attributes.has('*.jsonl text eol=lf'), true);
  assert.equal(attributes.has('*.json text eol=lf'), true);
});
