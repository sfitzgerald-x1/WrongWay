/**
 * Deterministically generates the frozen normal-duel-v1 perft corpus.
 *
 * Seeded wall plans use lcg32-v1: initialize from the uint32 seed, advance
 * before every selection with `state = (state * 1664525 + 1013904223) mod
 * 2^32`, filter the rules engine's canonical ascending legal-action list to
 * walls, then select `walls[state % walls.length]`. Modulo bias is intentional
 * and there is no rejection sampling. Curated and terminal plans instead
 * replay their explicit, canonical action-code sequences through the engine.
 *
 * `node scripts/generate-normal-duel-perft.mjs --check` verifies both checked-
 * in artifacts; `--write` regenerates them; no arguments print the corpus.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import * as engine from '../js/normal-duel-engine.mjs';
import { perftReport, seededWallState, stateFromActionCodes } from '../js/normal-duel-perft.mjs';

export const FIXTURE_FORMAT = 'normal-duel-perft-v1-1.4.0';
export const GENERATOR_VERSION = 'normal-duel-perft-generator-1.4.0';
export const PRNG = 'lcg32-v1';
export const FIXTURE_PATH = new URL('../tests/fixtures/normal-duel-perft-v1.json', import.meta.url);
export const MANIFEST_PATH = new URL('../tests/fixtures/normal-duel-perft-v1.manifest.json', import.meta.url);

const baseConfig = (size, firstPlayer = 'A', initialStock = 10) => ({
  ruleset: 'normal-duel-v1', rows: size, columns: size,
  start: { A: { r: size - 1, c: Math.floor(size / 2) }, B: { r: 0, c: Math.floor(size / 2) } },
  goalRows: { A: 0, B: size - 1 }, initialStock: { A: initialStock, B: initialStock },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap: 200, firstPlayer
});

const CONFIGS = Object.freeze({
  standardA: baseConfig(9),
  standardB: baseConfig(9, 'B'),
  blitzA: baseConfig(7),
  blitzEndgameA: baseConfig(7, 'A', 0)
});

// A legal 7x7 wall sequence intentionally leaves A with one and B with two
// walls. Its depth-three subtree has 3,861 exact leaves and wall actions at
// both the root and deeper plies.
const CURATED_7X7_REMAINING_WALLS = Object.freeze([
  49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81
]);
const ENDGAME_PREFIX = Object.freeze([38, 2, 31, 3, 24, 2, 17, 3, 10, 2]);

export const PLANS = Object.freeze([
  Object.freeze({ id: 'initial-geometry-9x9-a', kind: 'initial', configId: 'standardA', depth: 2 }),
  Object.freeze({ id: 'initial-geometry-7x7-a', kind: 'initial', configId: 'blitzA', depth: 2 }),
  Object.freeze({
    id: 'curated-remaining-walls-7x7-a', kind: 'action-codes', configId: 'blitzA', depth: 3,
    actionCodes: CURATED_7X7_REMAINING_WALLS
  }),
  // This deliberately retains one wall for each side. It is the bounded
  // 9x9 depth-three probe that exercises wall-heavy branching at the root and
  // two deeper action layers without opening the full initial-board tree.
  Object.freeze({
    id: 'seeded-remaining-walls-9x9-a', kind: 'seeded-walls', configId: 'standardA', depth: 3,
    generator: { algorithm: PRNG, seed: 25, plies: 18 }, perftOptions: { maxNodes: 4096 }
  }),
  Object.freeze({
    id: 'seeded-wall-exhausted-9x9-a', kind: 'seeded-walls', configId: 'standardA', depth: 4,
    generator: { algorithm: PRNG, seed: 0x1a2b3c4d, plies: 20 }
  }),
  Object.freeze({
    id: 'seeded-wall-exhausted-7x7-a', kind: 'seeded-walls', configId: 'blitzA', depth: 4,
    generator: { algorithm: PRNG, seed: 0xcafebabe, plies: 20 }
  }),
  Object.freeze({
    id: 'ongoing-early-goal-child-7x7-a', kind: 'action-codes', configId: 'blitzEndgameA', depth: 3,
    actionCodes: ENDGAME_PREFIX
  }),
  Object.freeze({
    id: 'terminal-goal-win-7x7-a', kind: 'action-codes', configId: 'blitzEndgameA', depth: 4,
    actionCodes: [...ENDGAME_PREFIX, 3]
  }),
  Object.freeze({
    id: 'seeded-first-player-b-9x9', kind: 'seeded-walls', configId: 'standardB', depth: 4,
    generator: { algorithm: PRNG, seed: 0x01020304, plies: 20 }
  })
]);

function clone(value) { return structuredClone(value); }
function provenanceForPlan(plan, config) {
  if (plan.kind === 'initial') {
    return { mode: 'initial-state-v1', config: clone(config) };
  }
  if (plan.kind === 'action-codes') {
    return { mode: 'explicit-action-codes-v1', config: clone(config), actionCodes: [...plan.actionCodes] };
  }
  if (plan.kind === 'seeded-walls') {
    return {
      mode: 'seeded-wall-only-lcg32-v1', config: clone(config),
      algorithm: plan.generator.algorithm, seed: plan.generator.seed, plies: plan.generator.plies,
      selection: 'initialize from uint32 seed; advance before each selection; filter canonical ascending legal actions to walls only; choose walls[state%walls.length]; modulo bias is intentional and no rejection sampling is used'
    };
  }
  throw new TypeError(`unknown plan kind ${plan.kind}`);
}
function stateForPlan(plan) {
  const config = CONFIGS[plan.configId];
  if (!config) throw new TypeError(`unknown config ${plan.configId}`);
  if (plan.kind === 'initial') return engine.createInitialState(config);
  if (plan.kind === 'action-codes') return stateFromActionCodes(config, plan.actionCodes);
  if (plan.kind === 'seeded-walls') return seededWallState(config, plan.generator);
  throw new TypeError(`unknown plan kind ${plan.kind}`);
}
function caseForPlan(plan) {
  const config = CONFIGS[plan.configId]; const state = stateForPlan(plan);
  const report = perftReport(config, state, plan.depth, plan.perftOptions);
  return {
    id: plan.id,
    kind: plan.kind,
    configId: plan.configId,
    actionCodes: plan.kind === 'action-codes' ? [...plan.actionCodes] : null,
    generator: plan.kind === 'seeded-walls' ? clone(plan.generator) : null,
    perftOptions: plan.perftOptions ? clone(plan.perftOptions) : null,
    provenance: provenanceForPlan(plan, config),
    state: clone(state),
    expect: {
      depth: plan.depth,
      leavesByDepth: [...report.leavesByDepth],
      nodeVisits: report.nodeVisits,
      rootActionCodes: report.divide.map(({ actionCode }) => actionCode),
      divide: report.divide.map(({ actionCode, childLeavesByDepth }) => [actionCode, [...childLeavesByDepth]])
    }
  };
}

/** Return the stable, complete perft fixture object. */
export function generatePerftFixture() {
  return {
    fixtureFormat: FIXTURE_FORMAT,
    ruleset: 'normal-duel-v1',
    source: 'js/normal-duel-engine.mjs legalActions/applyLegalAction',
    generator: {
      name: 'normal-duel-perft', version: GENERATOR_VERSION, seededAlgorithm: PRNG,
      seededSelection: 'initialize from uint32 seed; advance before each selection; filter canonical ascending legal actions to walls only; choose walls[state%walls.length]; modulo bias is intentional and no rejection sampling is used'
    },
    semantics: {
      leafDefinition: 'exact-depth-v1: P(s,0)=1; P(terminal,d>0)=0; P(ongoing,d)=sum(P(child,d-1))',
      deduplication: 'none; count every legal action-tree occurrence',
      actionOrder: 'ascending canonical policy action code',
      countType: 'safe integer only; traversal rejects counts above Number.MAX_SAFE_INTEGER',
      divideIndexing: 'divide[i].childLeavesByDepth[d]=P(apply(rootAction[i]),d); root leavesByDepth[d+1]=sum_i childLeavesByDepth[d]',
      nodeBudget: 'state visits use a conservative MAX_PERFT_NODES default; explicit maxNodes may opt in up to MAX_PERFT_NODES_HARD_CAP, and traversal rejects before exceeding the selected deterministic cap'
    },
    configs: clone(CONFIGS),
    cases: PLANS.map(caseForPlan)
  };
}
export function renderPerftFixture() { return `${JSON.stringify(generatePerftFixture(), null, 2)}\n`; }
export function createPerftManifest(fixtureText = renderPerftFixture()) {
  const fixture = JSON.parse(fixtureText);
  return {
    fixtureFormat: FIXTURE_FORMAT,
    generatorVersion: GENERATOR_VERSION,
    prng: {
      name: PRNG,
      transition: 'state=(state*1664525+1013904223) mod 2^32',
      selection: 'initialize from uint32 seed; advance before each selection; filter canonical ascending legal actions to walls only; choose walls[state%walls.length]; modulo bias is intentional and no rejection sampling is used'
    },
    caseCount: fixture.cases.length,
    sha256: createHash('sha256').update(fixtureText, 'utf8').digest('hex'),
    cases: fixture.cases.map((entry) => ({
      id: entry.id,
      kind: entry.kind,
      configId: entry.configId,
      config: clone(fixture.configs[entry.configId]),
      provenance: clone(entry.provenance),
      perftOptions: clone(entry.perftOptions),
      state: clone(entry.state),
      expect: {
        depth: entry.expect.depth,
        rootActionCount: entry.expect.rootActionCodes.length,
        nodeVisits: entry.expect.nodeVisits,
        leavesByDepth: [...entry.expect.leavesByDepth]
      }
    }))
  };
}
export function renderPerftManifest(fixtureText = renderPerftFixture()) {
  return `${JSON.stringify(createPerftManifest(fixtureText), null, 2)}\n`;
}

function main() {
  const fixture = renderPerftFixture(); const manifest = renderPerftManifest(fixture);
  if (process.argv.includes('--write')) {
    writeFileSync(FIXTURE_PATH, fixture, 'utf8');
    writeFileSync(MANIFEST_PATH, manifest, 'utf8');
    return;
  }
  if (process.argv.includes('--check')) {
    const fixtureMatches = readFileSync(FIXTURE_PATH, 'utf8') === fixture;
    const manifestMatches = readFileSync(MANIFEST_PATH, 'utf8') === manifest;
    if (!fixtureMatches || !manifestMatches) {
      process.stderr.write('normal-duel perft fixture or manifest is stale; regenerate from this script.\n');
      process.exitCode = 1;
    }
    return;
  }
  process.stdout.write(fixture);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
