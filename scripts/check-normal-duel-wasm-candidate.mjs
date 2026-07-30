#!/usr/bin/env node

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createInitialState, legalActionCodes } from '../js/normal-duel-engine.mjs';
import { loadCandidateArtifactManifest } from './evaluation/candidate-artifact-manifest.mjs';
import {
  createWorkerEngineAdapter,
  getCandidateArtifactProvenance,
  takeTrustedSubprocessDecision
} from './evaluation/worker-engine-proxy.mjs';
import {
  REGRESSION_MODE,
  STRENGTH_MODE
} from './evaluation/normal-duel-strength.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RELEASE = resolve(ROOT, 'rust/target/wasm-candidate/release');
const ADAPTER = resolve(RELEASE, 'adapter.mjs');
const MANIFEST = resolve(RELEASE, 'manifest.json');
const BUILD = resolve(ROOT, 'scripts/build-normal-duel-wasm-candidate.mjs');
const perft = JSON.parse(
  readFileSync(resolve(ROOT, 'tests/fixtures/normal-duel-perft-v1.json'), 'utf8')
);
const searchParity = JSON.parse(
  readFileSync(resolve(ROOT, 'tests/fixtures/normal-duel-wasm-search-nodes-v1.json'), 'utf8')
);

assert.deepEqual(readdirSync(RELEASE).sort(), [
  'adapter.mjs',
  'manifest.json',
  'normal-duel-wasm.mjs',
  'normal-duel-wasm_bg.wasm'
]);
const firstManifest = readFileSync(MANIFEST, 'utf8');
const rebuild = spawnSync(process.execPath, [BUILD], {
  cwd: ROOT,
  encoding: 'utf8'
});
assert.equal(rebuild.status, 0, rebuild.stderr || rebuild.stdout);
assert.equal(
  readFileSync(MANIFEST, 'utf8'),
  firstManifest,
  'rebuilding must produce an identical content-addressed manifest'
);
const binding = loadCandidateArtifactManifest({
  manifestPath: MANIFEST,
  moduleUrl: pathToFileURL(ADAPTER).href
});
assert.equal(binding.provenance.entry, 'adapter.mjs');
assert.deepEqual(binding.provenance.files.map(({ path }) => path), [
  'adapter.mjs',
  'normal-duel-wasm.mjs',
  'normal-duel-wasm_bg.wasm'
]);
const adapterSource = readFileSync(ADAPTER, 'utf8');
assert.match(adapterSource, /node:fs\/promises/);
assert.match(adapterSource, /normal-duel-wasm_bg\.wasm/);
assert.doesNotMatch(adapterSource, /package\.json|\bfetch\s*\(/);

const directWasm = await import(pathToFileURL(resolve(RELEASE, 'normal-duel-wasm.mjs')).href);
await directWasm.default({
  module_or_path: readFileSync(resolve(RELEASE, 'normal-duel-wasm_bg.wasm'))
});
const directConfig = perft.configs.standardA;
const directState = createInitialState(directConfig);
const directStarted = performance.now();
const directReport = JSON.parse(directWasm.normalDuelSearchFor(JSON.stringify({
  config: directConfig,
  state: directState,
  timeBudgetMs: 1,
  options: {
    maxDepth: 64,
    transpositionCapacity: 262_144,
    aspirationWindow: 64
  }
})));
const directElapsedMs = performance.now() - directStarted;
assert.ok(legalActionCodes(directConfig, directState).includes(directReport.actionCode));
assert.equal(directReport.stopped, true);
assert.ok(directElapsedMs < 250, `1 ms WASM deadline took ${directElapsedMs} ms`);

const directNodeReport = JSON.parse(directWasm.normalDuelSearchNodes(JSON.stringify(searchParity.request)));
assert.deepEqual(
  directNodeReport,
  searchParity.report,
  'generated WASM fixed-node search must match the committed native parity fixture'
);

const adapter = await createWorkerEngineAdapter({
  moduleUrl: pathToFileURL(ADAPTER).href,
  candidateManifestPath: MANIFEST
});
try {
  assert.equal(adapter.id, 'wrongway-normal-duel-wasm-search');
  assert.equal(typeof adapter.version, 'string');
  assert.deepEqual(adapter.capabilities.nodeBudget, true);
  assert.deepEqual(adapter.capabilities.deadline, true);
  const provenance = getCandidateArtifactProvenance(adapter);
  assert.ok(provenance, 'worker adapter must retain verified artifact provenance');
  assert.equal(provenance.verification, 'content-addressed-manifest-v1');
  assert.equal(provenance.rootPolicy, 'hermetic-release-directory-v1');
  assert.equal(provenance.entry, 'adapter.mjs');
  assert.deepEqual(provenance.files.map(({ path }) => path), [
    'adapter.mjs',
    'normal-duel-wasm.mjs',
    'normal-duel-wasm_bg.wasm'
  ]);
  const config = perft.configs.standardA;
  const state = createInitialState(config);
  const session = adapter.createSession({ side: 'A', config, openingHistory: [] });
  try {
    await session.ready();
    const wrappedDecision = await session.selectAction({
      mode: REGRESSION_MODE,
      config,
      state,
      limits: { nodeBudget: 1, wallClockBudgetMs: 1_000 }
    });
    const trustedDecision = takeTrustedSubprocessDecision(wrappedDecision);
    assert.ok(trustedDecision, 'worker decision must retain its trusted payload');
    const decision = trustedDecision.value;
    assert.ok(Number.isSafeInteger(decision.action?.to?.r) || typeof decision.action?.wall === 'string');
    assert.equal(decision.stats.nodes <= 1, true);
    assert.equal(Number.isSafeInteger(decision.stats.depth), true);
  } finally {
    await session.close();
  }

  const strengthSession = adapter.createSession({ side: 'A', config, openingHistory: [] });
  try {
    await strengthSession.ready();
    const requestedStrengthBudgetMs = 1_000;
    const wrappedDecision = await strengthSession.selectAction({
      mode: STRENGTH_MODE,
      config,
      state,
      limits: { nodeBudget: null, wallClockBudgetMs: requestedStrengthBudgetMs }
    });
    const trustedDecision = takeTrustedSubprocessDecision(wrappedDecision);
    assert.ok(trustedDecision, 'deadline search must retain its trusted payload');
    assert.equal(trustedDecision.timing.source, 'trusted-parent-subprocess-clock');
    assert.ok(
      trustedDecision.timing.chargedActiveMs <= requestedStrengthBudgetMs,
      `trusted active time ${trustedDecision.timing.chargedActiveMs} ms exceeded ${requestedStrengthBudgetMs} ms`
    );
    const decision = trustedDecision.value;
    assert.ok(Number.isSafeInteger(decision.action?.to?.r) || typeof decision.action?.wall === 'string');
    assert.equal(Number.isSafeInteger(decision.stats.nodes), true);
    assert.equal(Number.isSafeInteger(decision.stats.depth), true);
  } finally {
    await strengthSession.close();
  }
} finally {
  // Sessions are closed above; the probe subprocess is already terminated by
  // createWorkerEngineAdapter after descriptor inspection.
}

process.stdout.write('hermetic normal-duel WASM candidate verified and worker-loadable\n');
