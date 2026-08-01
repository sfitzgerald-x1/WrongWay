#!/usr/bin/env node

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  applyAction,
  createInitialState,
  decodeAction,
  encodeAction,
  legalActionCodes
} from '../js/normal-duel-engine.mjs';
import { loadCandidateArtifactManifest } from './evaluation/candidate-artifact-manifest.mjs';
import {
  createRecycledNormalDuelWasmCandidateAdapter,
  createWorkerEngineAdapter,
  getCandidateArtifactProvenance,
  getWorkerEngineIsolationProvenance,
  RECYCLE_CHILD_AUDIT_SEAM,
  takeTrustedSubprocessDecision
} from './evaluation/worker-engine-proxy.mjs';
import { CANONICAL_STRENGTH_CONFIG } from './evaluation/normal-duel-strength-constants.mjs';
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
/*
 * Cumulative late-game WASM memory-containment regression.
 *
 * One persistent per-game candidate subprocess exhausted the 512 MiB canonical
 * memory profile at ply 74 of the canonical 9x9 exhibition below, so the
 * regression replays that game's late plies through the per-decision recycler
 * against the freshly built real release. Every expectation is anchored to a
 * literal here: the fixture's action codes are pinned by their SHA-256, each
 * checkpoint state is recomputed from `createInitialState` by the canonical
 * engine rather than read from the fixture, and the final position key is pinned
 * outright, so the fixture can assert no state the engine does not reproduce.
 */
const LATEGAME_FIXTURE = resolve(
  ROOT,
  'tests/fixtures/normal-duel-recycled-lategame-9x9-007-v1.json'
);
const LATEGAME_FORMAT = 'normal-duel-recycled-lategame-replay-v1';
const LATEGAME_SOURCE_FORMAT = 'normal-duel-hardplus-exhibition-replay-v1';
const LATEGAME_SOURCE_SHA256 =
  '1906c7587d117ff88fcd18ead35b82219723e2d95e74dc9f7f1d197ded4ad47b';
const LATEGAME_ACTION_CODES_SHA256 =
  '2623de1808d6edd89188e372ffed8d68465216201189f69c59dd0700ddf1f9b1';
const LATEGAME_ACTION_COUNT = 74;
const LATEGAME_CHECKPOINT_PLIES = [58, 62, 66, 70, 74];
const LATEGAME_REPETITION_ENTRY_COUNTS = [7, 11, 15, 19, 23];
const LATEGAME_CHECKPOINT_INVARIANTS = {
  turn: 'A',
  outcome: { kind: 'ongoing' },
  wallCount: 20,
  stock: { A: 0, B: 0 },
  historyStartPly: 52,
  maximumRepetitionCount: 1
};
const LATEGAME_FINAL_POSITION_KEY = JSON.stringify([
  'normal-duel-v1', 9, 9, 76, 4, 0, 8, 10, 10, 'permissive-adjacent-exit-v1', 3, 200, 'A', 25, 63,
  [6, 8, 10, 15, 20, 41, 43, 45, 51, 53, 57, 59, 61, 63], [11, 13, 22, 38, 42, 55], 0, 0, 'A'
]);
const LATEGAME_BUDGET_MS = 1_000;
const sha256Text = (value) => createHash('sha256').update(value).digest('hex');

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
  timeBudgetMs: 1
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
const defaultHorizonReport = JSON.parse(
  directWasm.normalDuelSearchNodes(JSON.stringify(searchParity.defaultHorizon.request))
);
assert.deepEqual(
  defaultHorizonReport,
  searchParity.defaultHorizon.report,
  'generated WASM omitted-options horizon search must match the committed native parity fixture'
);
assert.ok(
  defaultHorizonReport.diagnostics.immediateGoalHorizonHits > 0,
  'generated WASM horizon fixture must exercise immediate-goal diagnostics'
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
    assert.ok(decision.stats.depth >= 1, 'strength smoke must complete at least depth one');
  } finally {
    await strengthSession.close();
  }

  const lategame = JSON.parse(readFileSync(LATEGAME_FIXTURE, 'utf8'));
  assert.equal(lategame.format, LATEGAME_FORMAT);
  assert.equal(lategame.source.format, LATEGAME_SOURCE_FORMAT);
  assert.equal(lategame.source.sha256, LATEGAME_SOURCE_SHA256);
  assert.equal(lategame.source.actionDerivation, 'canonical-legal-action-code-match-v1');
  assert.deepEqual(lategame.config, CANONICAL_STRENGTH_CONFIG);
  assert.equal(lategame.actionCount, LATEGAME_ACTION_COUNT);
  assert.equal(lategame.actionCodes.length, LATEGAME_ACTION_COUNT);
  assert.equal(
    sha256Text(JSON.stringify(lategame.actionCodes)),
    LATEGAME_ACTION_CODES_SHA256,
    'the fixture action codes must hash to the pinned canonical late-game sequence'
  );
  assert.deepEqual(lategame.checkpointInvariants, LATEGAME_CHECKPOINT_INVARIANTS);
  assert.deepEqual(lategame.checkpoints.map(({ ply }) => ply), LATEGAME_CHECKPOINT_PLIES);
  assert.deepEqual(
    lategame.checkpoints.map(({ repetitionEntryCount }) => repetitionEntryCount),
    LATEGAME_REPETITION_ENTRY_COUNTS
  );

  // Replay the pinned codes through the canonical engine. Codes are trusted only
  // as far as membership in `legalActionCodes` allows, and each checkpoint state
  // is the engine's own transition result rather than anything the fixture
  // recorded about the position.
  const lategameConfig = lategame.config;
  const lategameCheckpoints = [];
  let lategameState = createInitialState(lategameConfig);
  for (const [index, code] of lategame.actionCodes.entries()) {
    assert.ok(
      legalActionCodes(lategameConfig, lategameState).includes(code),
      `late-game ply ${index + 1} code ${code} is not legal in the reconstructed state`
    );
    lategameState = applyAction(
      lategameConfig,
      lategameState,
      decodeAction(lategameConfig, code)
    );
    assert.equal(lategameState.ply, index + 1);
    const checkpoint = lategame.checkpoints.find(({ ply }) => ply === lategameState.ply);
    if (checkpoint === undefined) continue;
    const invariants = LATEGAME_CHECKPOINT_INVARIANTS;
    assert.equal(lategameState.position.turn, invariants.turn);
    assert.deepEqual(lategameState.outcome, invariants.outcome);
    assert.equal(lategameState.position.walls.length, invariants.wallCount);
    assert.deepEqual(lategameState.position.stock, invariants.stock);
    assert.equal(lategameState.historyStartPly, invariants.historyStartPly);
    assert.equal(lategameState.repetitionCounts.length, checkpoint.repetitionEntryCount);
    assert.deepEqual(
      [...new Set(lategameState.repetitionCounts.map(({ count }) => count))],
      [invariants.maximumRepetitionCount],
      `late-game ply ${checkpoint.ply} must hold no repeated position`
    );
    assert.equal(
      sha256Text(JSON.stringify(lategameState.repetitionCounts)),
      checkpoint.repetitionCountsSha256,
      `late-game ply ${checkpoint.ply} repetition history hash changed`
    );
    assert.equal(lategameState.positionKey, checkpoint.positionKey);
    lategameCheckpoints.push(lategameState);
  }
  assert.equal(lategameCheckpoints.length, LATEGAME_CHECKPOINT_PLIES.length);
  assert.equal(lategameState.ply, LATEGAME_ACTION_COUNT);
  assert.equal(
    lategameState.positionKey,
    LATEGAME_FINAL_POSITION_KEY,
    'the reconstructed late-game final position must equal the pinned position key'
  );

  // Recycle the real manifest-authenticated release itself, and audit the child
  // lifecycle instead of any host-dependent resident-memory or process reading.
  const auditTrail = [];
  const recycled = await createRecycledNormalDuelWasmCandidateAdapter(adapter, {
    [RECYCLE_CHILD_AUDIT_SEAM]: (record) => {
      auditTrail.push(record);
    }
  });
  assert.equal(
    getCandidateArtifactProvenance(recycled).verification,
    'content-addressed-manifest-v1'
  );
  assert.equal(recycled.capabilities.isolation, 'node-subprocess-per-decision-v1');
  assert.equal(recycled.capabilities.sessionLifecycle, 'stateless-wasm-per-decision-v1');
  const recycledIsolation = getWorkerEngineIsolationProvenance(recycled);
  assert.equal(recycledIsolation.subprocessIsolation, 'node-subprocess-per-decision-v1');
  assert.equal(recycledIsolation.sessionLifecycle, 'stateless-wasm-per-decision-v1');

  // Exactly one outer session, made ready once, serves every checkpoint: the
  // containment claim is that the children behind it are fresh, not that the
  // caller restarts anything.
  const lategameSession = recycled.createSession({
    side: 'A',
    config: lategameConfig,
    openingHistory: []
  });
  try {
    await lategameSession.ready();
    for (const [index, checkpointState] of lategameCheckpoints.entries()) {
      const selection = index + 1;
      const { ply } = checkpointState;
      const wrapped = await lategameSession.selectAction({
        mode: STRENGTH_MODE,
        config: lategameConfig,
        state: checkpointState,
        limits: { nodeBudget: null, wallClockBudgetMs: LATEGAME_BUDGET_MS }
      });
      const trusted = takeTrustedSubprocessDecision(wrapped);
      assert.ok(trusted, `late-game ply ${ply} lost its trusted subprocess brand`);
      const { timing } = trusted;
      assert.equal(timing.source, 'trusted-parent-subprocess-clock');
      for (const component of ['setupMs', 'observerMs', 'selectMs', 'chargedActiveMs']) {
        assert.ok(
          Number.isFinite(timing[component]) && timing[component] >= 0,
          `late-game ply ${ply} ${component} was ${timing[component]}`
        );
      }
      assert.equal(
        timing.chargedActiveMs,
        timing.setupMs + timing.observerMs + timing.selectMs,
        `late-game ply ${ply} charged active time is not its component sum`
      );
      assert.ok(
        timing.chargedActiveMs <= LATEGAME_BUDGET_MS,
        `late-game ply ${ply} charged ${timing.chargedActiveMs} ms of a ${LATEGAME_BUDGET_MS} ms budget`
      );
      // The move itself is not pinned: a real search may legitimately choose
      // differently between builds, so only its legality here is required.
      const move = trusted.value;
      assert.ok(Number.isSafeInteger(move.stats.nodes) && move.stats.nodes >= 0);
      assert.ok(
        Number.isSafeInteger(move.stats.depth) && move.stats.depth >= 1,
        `late-game ply ${ply} completed depth ${move.stats.depth}`
      );
      assert.ok(
        legalActionCodes(lategameConfig, checkpointState)
          .includes(encodeAction(lategameConfig, move.action)),
        `late-game ply ${ply} returned an illegal action ${JSON.stringify(move.action)}`
      );
      // The awaited decision has already returned, so a `reaped` record now
      // present is proof that its child was reaped first.
      assert.equal(auditTrail.length, 2 * selection);
      assert.deepEqual(auditTrail.slice(-2), [
        { phase: 'opened', generation: selection, selection },
        { phase: 'reaped', generation: selection, selection }
      ], `late-game ply ${ply} did not open and reap exactly one fresh child`);
    }
  } finally {
    await lategameSession.close();
  }
  assert.deepEqual(
    auditTrail,
    LATEGAME_CHECKPOINT_PLIES.flatMap((ignoredPly, index) => [
      { phase: 'opened', generation: index + 1, selection: index + 1 },
      { phase: 'reaped', generation: index + 1, selection: index + 1 }
    ]),
    'the five late-game decisions must each open and reap their own child in order'
  );
  assert.equal(
    new Set(auditTrail.map(({ generation }) => generation)).size,
    LATEGAME_CHECKPOINT_PLIES.length,
    'every audited per-decision child must have a distinct generation'
  );
  assert.ok(auditTrail.every((record) => Object.isFrozen(record)));
} finally {
  // Sessions are closed above; the probe subprocess is already terminated by
  // createWorkerEngineAdapter after descriptor inspection.
}

process.stdout.write('hermetic normal-duel WASM candidate verified and worker-loadable\n');
