#!/usr/bin/env node
/**
 * Record exactly one non-enforced canonical 9x9 exhibition and emit a replay
 * artifact for docs/exhibitions/replay.html.
 *
 * Side A is the authenticated Hard+ WASM candidate release; side B is the
 * privately branded pinned Hard subprocess baseline. The rules engine, never a
 * bot, applies every action; this script only records what the shared
 * `runMatch` protocol played, so the exhibition uses the same deadline,
 * isolation, and adjudication paths as the strength harness.
 *
 * Transitions are captured by wrapping the baseline descriptor: it observes
 * every applied transition, which yields the full per-ply action list without
 * touching the candidate's trusted-timing path.
 *
 * The candidate descriptor is wrapped too, but only to lower what it *asks* for:
 * each decision receives a cloned request whose `limits.wallClockBudgetMs` is
 * clamped down, while the harness's outer contract (`perMoveDeadlineMs`,
 * `deadlineAtMs`, and the guarded clock) passes through untouched. The pinned
 * Hard baseline's request limits are never rewritten.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  applyAction,
  createInitialState,
  decodeAction
} from '../js/normal-duel-engine.mjs';
import { verifyOpeningArtifacts } from './generate-normal-duel-balanced-openings.mjs';
import {
  runMatch,
  STRENGTH_MODE,
  validateOpeningBook
} from './evaluation/normal-duel-strength.mjs';
import {
  createPinnedHardWorkerAdapter,
  createRecycledNormalDuelWasmCandidateAdapter,
  createWorkerEngineAdapter,
  getCandidateArtifactProvenance,
  getWorkerEngineIsolationProvenance,
  isAuthenticatedCandidateAdapter,
  isPinnedHardWorkerAdapter
} from './evaluation/worker-engine-proxy.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const RELEASE = resolve(ROOT, 'rust/target/wasm-candidate/release');
const BOOK = resolve(ROOT, 'tests/fixtures/normal-duel-balanced-openings-v1.json');
const BOOK_MANIFEST = resolve(ROOT, 'tests/fixtures/normal-duel-balanced-openings-v1.manifest.json');
const OUTPUT_DIRECTORY = resolve(ROOT, 'docs/exhibitions');
export const EXHIBITION_ARTIFACT_FORMAT = 'normal-duel-hardplus-exhibition-replay-v1';
const DEFAULT_OPENING_ID = 'balanced-9x9-007';
const DEFAULT_DEADLINE_MS = 15_000;
/**
 * What the Hard+ candidate is asked for per decision. The harness still enforces
 * the full `DEFAULT_DEADLINE_MS`; this only shrinks the requested allotment so the
 * candidate leaves headroom under the outer deadline.
 */
const DEFAULT_CANDIDATE_REQUESTED_BUDGET_MS = 14_000;
/**
 * The Hard+ WASM adapter reserves this much of the requested budget for cold JIT,
 * JSON conversion, IPC, and the final budget poll before handing a search budget
 * to Rust (`STRENGTH_OVERHEAD_MARGIN_MS` in the released adapter).
 */
const CANDIDATE_ADAPTER_OVERHEAD_MARGIN_MS = 100;
/** The exhibition is pinned to this repository commit, not to a moving branch. */
const PINNED_REPOSITORY_COMMIT = 'bb35d8c';

function fail(message) {
  throw new Error(`run-hardplus-exhibition: ${message}`);
}

function integer(raw, name, minimum = 0) {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) fail(`${name} must be an integer >= ${minimum}`);
  return value;
}

function parseArguments(argv) {
  const options = {
    openingId: DEFAULT_OPENING_ID,
    deadlineMs: DEFAULT_DEADLINE_MS,
    candidateRequestedBudgetMs: DEFAULT_CANDIDATE_REQUESTED_BUDGET_MS,
    seed: null,
    out: null,
    candidate: resolve(RELEASE, 'adapter.mjs'),
    candidateManifest: resolve(RELEASE, 'manifest.json'),
    allowCommitDrift: false,
    recycleNormalDuelWasmPerDecision: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--opening') options.openingId = argv[++index];
    else if (argument === '--deadline-ms') options.deadlineMs = integer(argv[++index], 'deadline-ms', 1);
    else if (argument === '--candidate-requested-budget-ms') {
      options.candidateRequestedBudgetMs = integer(argv[++index], 'candidate-requested-budget-ms', 1);
    } else if (argument === '--seed') options.seed = integer(argv[++index], 'seed');
    else if (argument === '--out') options.out = resolve(argv[++index]);
    else if (argument === '--candidate') options.candidate = resolve(argv[++index]);
    else if (argument === '--candidate-manifest') options.candidateManifest = resolve(argv[++index]);
    else if (argument === '--allow-commit-drift') options.allowCommitDrift = true;
    else if (argument === '--recycle-normal-duel-wasm-per-decision') {
      options.recycleNormalDuelWasmPerDecision = true;
    }
    else fail(`unknown argument ${argument}`);
  }
  options.out ??= resolve(OUTPUT_DIRECTORY, `hardplus-exhibition-${options.openingId}.json`);
  return options;
}

function git(...args) {
  return execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
}

function repositoryProvenance(allowCommitDrift) {
  const headCommit = git('rev-parse', 'HEAD');
  const matchesPin = headCommit.startsWith(PINNED_REPOSITORY_COMMIT);
  if (!matchesPin && !allowCommitDrift) {
    fail(
      `exhibition is pinned to commit ${PINNED_REPOSITORY_COMMIT} but HEAD is ${headCommit};`
      + ' pass --allow-commit-drift to record from a different commit'
    );
  }
  return Object.freeze({
    headCommit,
    pinnedRepositoryCommit: PINNED_REPOSITORY_COMMIT,
    headMatchesPinnedCommit: matchesPin,
    worktreeDirty: git('status', '--porcelain').length > 0
  });
}

/** Replay the committed opening so the artifact carries its actions too. */
function openingPlies(config, opening) {
  let state = createInitialState(config);
  const initialPosition = state.position;
  const plies = [];
  for (const code of opening.actionCodes) {
    const actor = state.position.turn;
    const action = decodeAction(config, code);
    state = applyAction(config, state, action);
    plies.push({
      phase: 'opening',
      ply: state.ply,
      actor,
      actionCode: code,
      action,
      position: state.position,
      outcome: state.outcome
    });
  }
  return { initialPosition, plies };
}

/**
 * Wrap an engine descriptor so every applied transition is recorded. Only
 * `observe` is intercepted; `selectAction` returns the underlying value by
 * reference so trusted subprocess decisions stay intact.
 */
function recordingEngine(engine, onTransition) {
  return Object.freeze({
    ...engine,
    async createSession(context) {
      const session = await engine.createSession(context);
      const wrapper = {
        selectAction: (request) => session.selectAction(request),
        async observe(transition) {
          onTransition(transition);
          return session.observe(transition);
        },
        close: () => session.close()
      };
      if (typeof session.ready === 'function') wrapper.ready = () => session.ready();
      return Object.freeze(wrapper);
    }
  });
}

/**
 * Wrap an engine descriptor so each decision request asks for at most
 * `requestedBudgetMs` of wall-clock time. The harness's outer contract is left
 * alone: `perMoveDeadlineMs` stays whatever `runMatch` was given, and the request
 * clone still carries the original `deadlineAtMs` and guarded `clock` by
 * reference, so deadline enforcement and trusted timing are unchanged. The budget
 * is only ever clamped downward, so a smaller harness budget still wins.
 */
function requestedBudgetEngine(engine, requestedBudgetMs) {
  const clampedLimits = (limits) => {
    const budget = limits?.wallClockBudgetMs;
    // Non-strength modes leave the budget null; nothing to clamp there.
    if (!Number.isFinite(budget) || budget <= 0) return limits;
    return Object.freeze({ ...limits, wallClockBudgetMs: Math.min(budget, requestedBudgetMs) });
  };
  return Object.freeze({
    ...engine,
    async createSession(context) {
      const session = await engine.createSession(context);
      const wrapper = {
        selectAction: (request) => session.selectAction(Object.freeze({
          ...request,
          limits: clampedLimits(request.limits)
        })),
        observe: (transition) => session.observe(transition),
        close: () => session.close()
      };
      if (typeof session.ready === 'function') wrapper.ready = () => session.ready();
      return Object.freeze(wrapper);
    }
  });
}

/** Read one side's telemetry in decision order, one sample per decision. */
function telemetryReader(telemetry) {
  let index = 0;
  return () => {
    const at = index;
    index += 1;
    return Object.freeze({
      decisionIndex: at,
      nodes: telemetry.nodes[at] ?? null,
      depth: telemetry.depths[at] ?? null,
      chargedActiveMs: telemetry.decisionMilliseconds[at] ?? null,
      chargedSetupMs: telemetry.chargedSetupMilliseconds[at] ?? null,
      chargedObserverMs: telemetry.chargedObserverMilliseconds[at] ?? null,
      chargedSelectMs: telemetry.chargedSelectMilliseconds[at] ?? null,
      reportedElapsedMs: telemetry.reportedDecisionMilliseconds[at] ?? null,
      timingSource: telemetry.timingSource,
      clockProfileId: telemetry.clockProfileId
    });
  };
}

function sampleCounts(telemetry, plies) {
  return Object.freeze({
    plies,
    decisionSamples: telemetry.decisionMilliseconds.length,
    nodeSamples: telemetry.nodes.length,
    depthSamples: telemetry.depths.length,
    reportedSamples: telemetry.reportedDecisionMilliseconds.length
  });
}

const options = parseArguments(process.argv.slice(2));
const repository = repositoryProvenance(options.allowCommitDrift);
const verified = verifyOpeningArtifacts(
  readFileSync(BOOK, 'utf8'),
  readFileSync(BOOK_MANIFEST, 'utf8')
);
const { book, config } = validateOpeningBook(verified.book);
if (config.rows !== 9 || config.columns !== 9) fail('exhibition requires the canonical 9x9 configuration');
const opening = book.openings.find((entry) => entry.id === options.openingId);
if (!opening) fail(`opening ${options.openingId} is not in the verified book`);

const isolatedCandidate = await createWorkerEngineAdapter({
  moduleUrl: pathToFileURL(options.candidate).href,
  candidateManifestPath: options.candidateManifest
});
if (!isAuthenticatedCandidateAdapter(isolatedCandidate)) {
  fail('candidate must be a content-addressed, authenticated subprocess adapter');
}
// Only the candidate is ever recycled. The pinned Hard baseline below keeps its
// per-game subprocess exactly as in every other run.
const candidate = options.recycleNormalDuelWasmPerDecision
  ? await createRecycledNormalDuelWasmCandidateAdapter(isolatedCandidate)
  : isolatedCandidate;
const baseline = await createPinnedHardWorkerAdapter();
if (!isPinnedHardWorkerAdapter(baseline)) fail('baseline must be the pinned Hard subprocess adapter');

const transitions = [];
const seed = options.seed ?? book.generator.seed;
// Never ask for more than the harness deadline actually allows.
const candidateRequestedBudgetMs = Math.min(options.candidateRequestedBudgetMs, options.deadlineMs);
const result = await runMatch({
  config,
  opening,
  engines: {
    A: requestedBudgetEngine(candidate, candidateRequestedBudgetMs),
    B: recordingEngine(baseline, (transition) => transitions.push(transition))
  },
  contenderSide: 'A',
  gameInPair: 0,
  seed,
  mode: STRENGTH_MODE,
  perMoveDeadlineMs: options.deadlineMs
});

const roles = Object.freeze({ A: 'candidate-hard-plus', B: 'pinned-hard-baseline' });
const readers = { A: telemetryReader(result.telemetry.A), B: telemetryReader(result.telemetry.B) };
const replayed = openingPlies(config, opening);
const gamePlies = transitions.map((transition) => ({
  phase: 'game',
  ply: transition.stateAfter.ply,
  actor: transition.player,
  actionCode: null,
  action: transition.action,
  position: transition.stateAfter.position,
  outcome: transition.stateAfter.outcome,
  telemetry: readers[transition.player]()
}));
const plies = [
  ...replayed.plies.map((ply) => ({ ...ply, telemetry: null })),
  ...gamePlies
];
const playedBySide = {
  A: gamePlies.filter((ply) => ply.actor === 'A').length,
  B: gamePlies.filter((ply) => ply.actor === 'B').length
};

const artifact = {
  format: EXHIBITION_ARTIFACT_FORMAT,
  exhibition: {
    kind: 'non-enforced-canonical-9x9-exhibition-v1',
    enforced: false,
    mode: result.settings.mode,
    /** Retained under its original name so existing replay readers keep working. */
    perTurnDeadlineMs: options.deadlineMs,
    outerPerTurnDeadlineMs: result.settings.perMoveDeadlineMs ?? options.deadlineMs,
    candidateRequestedBudgetMs,
    candidateAdapterOverheadMarginMs: CANDIDATE_ADAPTER_OVERHEAD_MARGIN_MS,
    candidateEffectiveRustBudgetMs:
      candidateRequestedBudgetMs - CANDIDATE_ADAPTER_OVERHEAD_MARGIN_MS,
    budgetNote:
      'The harness enforced outerPerTurnDeadlineMs per turn for both sides. Only the'
      + ' Hard+ candidate (A) received a cloned request with limits.wallClockBudgetMs'
      + ' lowered to candidateRequestedBudgetMs; its outer deadlineAtMs and clock were'
      + ' unchanged, and the pinned Hard baseline (B) kept the harness limits verbatim.'
      + ' The candidate adapter reserves candidateAdapterOverheadMarginMs from the'
      + ' requested budget, so Rust searched for candidateEffectiveRustBudgetMs'
      + ' (= requested - margin) nominally; per-decision setup/observer active-time'
      + ' debt and the deadline clamp can lower the actual search budget further.',
    seed,
    clockProfile: result.settings.clockProfile,
    openingId: opening.id,
    openingTargetPlies: opening.targetPlies,
    roles,
    sides: {
      A: { role: roles.A, engineId: candidate.id, version: candidate.version },
      B: {
        role: roles.B,
        engineId: baseline.id,
        version: baseline.version,
        sourceCommit: baseline.sourceCommit ?? null
      }
    }
  },
  repository,
  config,
  book: {
    format: book.bookFormat,
    generator: book.generator,
    configurationSha256: book.configurationSha256,
    openingCount: book.openings.length
  },
  corpusProvenance: verified.provenance,
  candidate: {
    id: candidate.id,
    version: candidate.version,
    authenticated: true,
    artifactProvenance: getCandidateArtifactProvenance(candidate),
    isolationProvenance: getWorkerEngineIsolationProvenance(candidate),
    capabilities: candidate.capabilities
  },
  baseline: {
    id: baseline.id,
    version: baseline.version,
    sourceCommit: baseline.sourceCommit ?? null,
    trustRoot: baseline.baselineTrustRoot,
    pinnedWorkerVerified: true,
    isolationProvenance: getWorkerEngineIsolationProvenance(baseline)
  },
  outcome: {
    resultKind: result.resultKind,
    winner: result.winner,
    winnerRole: result.winner ? roles[result.winner] : null,
    reason: result.reason,
    failedPlayer: result.failedPlayer,
    error: result.error,
    startingPly: result.startingPly,
    finalPly: result.finalPly,
    finalOutcome: result.finalOutcome
  },
  initialPosition: replayed.initialPosition,
  plies,
  telemetry: {
    A: result.telemetry.A,
    B: result.telemetry.B,
    alignment: {
      note: 'Per-ply telemetry is index-aligned per side in decision order;'
        + ' a sample count below the played-ply count leaves that field null.',
      A: sampleCounts(result.telemetry.A, playedBySide.A),
      B: sampleCounts(result.telemetry.B, playedBySide.B)
    },
    caveats: [
      'The pinned Hard baseline descriptor is wrapped to record transitions, so its'
      + ' trailing setup/observer active-time flush is not charged; candidate (A)'
      + ' telemetry uses the unwrapped trusted subprocess path.',
      'The pinned Hard baseline reports depth but not node counts.'
    ]
  }
};

mkdirSync(dirname(options.out), { recursive: true });
writeFileSync(options.out, `${JSON.stringify(artifact, null, 2)}\n`);
process.stdout.write(
  `${options.out}\n`
  + `${artifact.exhibition.openingId}: ${result.resultKind}`
  + `${result.winner ? ` for ${result.winner} (${roles[result.winner]})` : ''}`
  + `${result.reason ? ` by ${result.reason}` : ''}`
  + ` at ply ${result.finalPly} (${plies.length} recorded plies)\n`
);
