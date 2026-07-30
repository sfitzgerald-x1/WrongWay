/**
 * Deterministically generates the frozen normal-duel-v1 JSONL replay corpus.
 *
 * Seeded selection semantics (seeded-lcg32-v1): initialize state from the uint32 seed; advance it
 * before every selection with
 * `state = (state * 1664525 + 1013904223) mod 2^32`; filter the engine's
 * canonical ascending legal-action list by the cadence-selected kind; then
 * choose `candidates[state % candidates.length]`. Modulo bias is intentional;
 * there is no rejection sampling. Each seeded sample selects wall, pawn, pawn
 * on a repeating three-ply cadence.
 * Explicit selection semantics (explicit-actions-v1): dedicated terminal probes
 * use explicit legal action sequences so goal,
 * threefold-repetition, and ply-cap adjudication stay small and reproducible.
 *
 * `node scripts/generate-normal-duel-trajectories.mjs --check` verifies both
 * checked-in artifacts, `--write` regenerates them, and no arguments print the
 * canonical JSONL.
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { LCG32_ALGORITHM, createLcg32 } from '../js/lcg32.mjs';
import * as engine from '../js/normal-duel-engine.mjs';

export const CORPUS_FORMAT = 'normal-duel-v1-trajectories-1.1.0';
export const GENERATOR_VERSION = 'normal-duel-trajectory-generator-1.1.0';
export const PRNG = LCG32_ALGORITHM;
export { createLcg32 };
export const SEEDED_SELECTION_MODE = 'seeded-lcg32-v1';
export const EXPLICIT_SELECTION_MODE = 'explicit-actions-v1';
export const CORPUS_PATH = new URL('../tests/fixtures/normal-duel-v1-trajectories.jsonl', import.meta.url);
export const MANIFEST_PATH = new URL('../tests/fixtures/normal-duel-v1-trajectories.manifest.json', import.meta.url);

const baseConfig = (rows, columns, firstPlayer = 'A', plyCap = 200) => ({
  ruleset: 'normal-duel-v1', rows, columns,
  start: { A: { r: rows - 1, c: Math.floor(columns / 2) }, B: { r: 0, c: Math.floor(columns / 2) } },
  goalRows: { A: 0, B: rows - 1 }, initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap, firstPlayer
});

const PLANS = Object.freeze([
  Object.freeze({ id: 'sample-standard-9x9-a', selectionMode: SEEDED_SELECTION_MODE, seed: 0x1a2b3c4d, config: baseConfig(9, 9, 'A') }),
  Object.freeze({ id: 'sample-blitz-7x7-b', selectionMode: SEEDED_SELECTION_MODE, seed: 0x5e6f7788, config: baseConfig(7, 7, 'B') }),
  Object.freeze({
    id: 'terminal-goal-7x7-a', selectionMode: EXPLICIT_SELECTION_MODE, config: baseConfig(7, 7, 'A'),
    prefix: [
      { kind: 'pawn', to: { r: 5, c: 3 } }, { kind: 'pawn', to: { r: 0, c: 2 } },
      { kind: 'pawn', to: { r: 4, c: 3 } }, { kind: 'pawn', to: { r: 0, c: 3 } },
      { kind: 'pawn', to: { r: 3, c: 3 } }, { kind: 'pawn', to: { r: 0, c: 2 } },
      { kind: 'pawn', to: { r: 2, c: 3 } }, { kind: 'pawn', to: { r: 0, c: 3 } },
      { kind: 'pawn', to: { r: 1, c: 3 } }, { kind: 'pawn', to: { r: 0, c: 2 } }
    ], selectedAction: { kind: 'pawn', to: { r: 0, c: 3 } }
  }),
  Object.freeze({
    id: 'terminal-threefold-9x9-b', selectionMode: EXPLICIT_SELECTION_MODE, config: baseConfig(9, 9, 'B'),
    prefix: [
      { kind: 'pawn', to: { r: 0, c: 5 } }, { kind: 'pawn', to: { r: 8, c: 3 } },
      { kind: 'pawn', to: { r: 0, c: 4 } }, { kind: 'pawn', to: { r: 8, c: 4 } },
      { kind: 'pawn', to: { r: 0, c: 5 } }, { kind: 'pawn', to: { r: 8, c: 3 } },
      { kind: 'pawn', to: { r: 0, c: 4 } }
    ], selectedAction: { kind: 'pawn', to: { r: 8, c: 4 } }
  }),
  Object.freeze({
    id: 'terminal-ply-cap-9x9-a', selectionMode: EXPLICIT_SELECTION_MODE, config: baseConfig(9, 9, 'A', 2),
    prefix: [{ kind: 'pawn', to: { r: 7, c: 4 } }], selectedAction: { kind: 'pawn', to: { r: 1, c: 4 } }
  })
]);

function clone(value) { return structuredClone(value); }
export function selectSeededAction(actions, nextRandom, step) {
  if (!Number.isSafeInteger(step) || step < 0) throw new TypeError(`seeded selection step must be a non-negative integer; received ${String(step)}`);
  const kind = step % 3 === 0 ? 'wall' : 'pawn';
  const candidates = actions.filter((action) => action.kind === kind);
  if (candidates.length === 0) {
    throw new RangeError(`seeded-lcg32-v1 has no ${kind} candidates at cadence step ${step}`);
  }
  return candidates[nextRandom() % candidates.length];
}
function endReason(nextState, isLast) {
  if (nextState.outcome.kind === 'ongoing') return isLast ? 'sample_limit' : 'continue';
  return nextState.outcome.reason;
}
function recordFor(plan, step, state, selectedAction, nextState, isLast) {
  const legalActionCodes = engine.legalActionCodes(plan.config, state);
  return {
    corpusFormat: CORPUS_FORMAT,
    generatorVersion: GENERATOR_VERSION,
    selectionMode: plan.selectionMode,
    prng: plan.selectionMode === SEEDED_SELECTION_MODE ? PRNG : null,
    trajectoryId: plan.id,
    seed: plan.selectionMode === SEEDED_SELECTION_MODE ? plan.seed : null,
    step,
    configuration: clone(plan.config),
    state: clone(state),
    legalActionCodes,
    selectedAction: clone(selectedAction),
    selectedActionCode: engine.encodeAction(plan.config, selectedAction),
    nextState: clone(nextState),
    outcome: clone(nextState.outcome),
    endReason: endReason(nextState, isLast)
  };
}
/** Return canonical replay records in their stable line order. */
export function generateTrajectoryRecords() {
  const records = [];
  for (const plan of PLANS) {
    if (plan.selectionMode === SEEDED_SELECTION_MODE) {
      const nextRandom = createLcg32(plan.seed); let state = engine.createInitialState(plan.config);
      for (let step = 0; step < 6; step += 1) {
        const action = selectSeededAction(engine.legalActions(plan.config, state), nextRandom, step);
        const nextState = engine.applyAction(plan.config, state, action);
        records.push(recordFor(plan, step, state, action, nextState, step === 5));
        state = nextState;
      }
    } else if (plan.selectionMode === EXPLICIT_SELECTION_MODE) {
      const actions = [...(plan.prefix ?? []), plan.selectedAction];
      let state = engine.createInitialState(plan.config);
      for (let step = 0; step < actions.length; step += 1) {
        const nextState = engine.applyAction(plan.config, state, actions[step]);
        records.push(recordFor(plan, step, state, actions[step], nextState, step === actions.length - 1));
        state = nextState;
      }
    } else {
      throw new TypeError(`Unknown trajectory selection mode: ${String(plan.selectionMode)}`);
    }
  }
  return records;
}

export function renderTrajectoryCorpus() {
  return `${generateTrajectoryRecords().map((record) => JSON.stringify(record)).join('\n')}\n`;
}
export function createCorpusManifest(corpus = renderTrajectoryCorpus()) {
  const records = corpus.trimEnd().split('\n').map((line) => JSON.parse(line));
  return {
    corpusFormat: CORPUS_FORMAT,
    generatorVersion: GENERATOR_VERSION,
    selectionModes: {
      [SEEDED_SELECTION_MODE]: {
        prng: PRNG,
        seed: 'uint32',
        transition: 'state=(state*1664525+1013904223) mod 2^32',
        selection: 'initialize from the uint32 seed; advance before every selection; filter the engine canonical ascending legal-action list to wall when step%3===0 and pawn otherwise; choose candidates[state%candidates.length]; modulo bias is intentional and no rejection sampling is used'
      },
      [EXPLICIT_SELECTION_MODE]: {
        prng: null,
        seed: null,
        selection: 'use the checked-in explicit legal action sequence for each terminal probe'
      }
    },
    lineCount: records.length,
    sha256: createHash('sha256').update(corpus, 'utf8').digest('hex'),
    trajectories: records.filter((record) => record.step === 0).map((record) => ({
      trajectoryId: record.trajectoryId,
      selectionMode: record.selectionMode,
      seed: record.seed,
      prng: record.prng,
      board: `${record.configuration.rows}x${record.configuration.columns}`,
      firstPlayer: record.configuration.firstPlayer,
      plyCap: record.configuration.plyCap,
      configuration: clone(record.configuration),
      configurationSha256: createHash('sha256').update(JSON.stringify(record.configuration), 'utf8').digest('hex'),
      selectedActionCodes: records.filter((candidate) => candidate.trajectoryId === record.trajectoryId)
        .map((candidate) => candidate.selectedActionCode),
      endReason: records.filter((candidate) => candidate.trajectoryId === record.trajectoryId).at(-1).endReason
    }))
  };
}
export function renderCorpusManifest() { return `${JSON.stringify(createCorpusManifest(), null, 2)}\n`; }

function main() {
  const corpus = renderTrajectoryCorpus(); const manifest = renderCorpusManifest();
  if (process.argv.includes('--write')) {
    writeFileSync(CORPUS_PATH, corpus, 'utf8');
    writeFileSync(MANIFEST_PATH, manifest, 'utf8');
    return;
  }
  if (process.argv.includes('--check')) {
    const corpusMatches = readFileSync(CORPUS_PATH, 'utf8') === corpus;
    const manifestMatches = readFileSync(MANIFEST_PATH, 'utf8') === manifest;
    if (!corpusMatches || !manifestMatches) {
      process.stderr.write('normal-duel trajectory corpus or manifest is stale; regenerate from this script.\n');
      process.exitCode = 1;
    }
    return;
  }
  process.stdout.write(corpus);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
