/**
 * Deterministic, engine-agnostic normal-duel match and strength protocol.
 *
 * Engine descriptor:
 *   { id, version, capabilities, createSession(context) }
 * Session:
 *   { selectAction(request), observe(transition), close() }
 *
 * `selectAction` may return an Action directly or
 * `{ action, stats: { elapsedMs?, nodes?, depth?, ... } }`. The rules engine,
 * never either bot, validates and applies every action and adjudicates goal,
 * threefold repetition, and the configured ply cap.
 */
import { performance } from 'node:perf_hooks';

import {
  applyAction,
  createInitialState,
  decodeAction,
  validateConfig
} from '../../js/normal-duel-engine.mjs';
import { createLcg32 } from '../../js/lcg32.mjs';
import {
  assertVerifiedOpeningCorpus,
  isVerifiedOpeningCorpus
} from '../generate-normal-duel-balanced-openings.mjs';
import {
  CANONICAL_STRENGTH_CONFIG,
  CANONICAL_STRENGTH_CONFIGURATION_SHA256,
  CANONICAL_STRENGTH_DEADLINE_MS,
  CANONICAL_STRENGTH_GENERATOR_ALGORITHM,
  CANONICAL_STRENGTH_GENERATOR_VERSION,
  CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
  CANONICAL_STRENGTH_OBSERVER_CAP_MS,
  CANONICAL_STRENGTH_SEED,
  MINIMUM_STRENGTH_OPENING_PAIRS,
  normalDuelConfig
} from './normal-duel-strength-constants.mjs';
import {
  CANONICAL_ENGINE_MEMORY_LIMIT_MIB,
  CANONICAL_ENGINE_V8_OLD_SPACE_MIB,
  getCandidateArtifactProvenance,
  getWorkerEngineIsolationProvenance,
  isAuthenticatedCandidateAdapter,
  isTrustedSubprocessDeadlineError,
  isTrustedSubprocessMemoryError,
  isPinnedHardWorkerAdapter,
  isWorkerEngineAdapter,
  takeTrustedSubprocessDecision,
  takeTrustedSubprocessFailureTiming,
  takeTrustedSubprocessPendingTiming
} from './worker-engine-proxy.mjs';
import {
  isPinnedHardClockProfileExceededError
} from './hard-baseline-46a871c7.mjs';

export {
  CANONICAL_STRENGTH_CONFIG,
  CANONICAL_STRENGTH_CONFIGURATION_SHA256,
  CANONICAL_STRENGTH_DEADLINE_MS,
  CANONICAL_STRENGTH_GENERATOR_ALGORITHM,
  CANONICAL_STRENGTH_GENERATOR_VERSION,
  CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
  CANONICAL_STRENGTH_OBSERVER_CAP_MS,
  CANONICAL_STRENGTH_SEED,
  MINIMUM_STRENGTH_OPENING_PAIRS,
  normalDuelConfig
};

export const STRENGTH_PROTOCOL = 'normal-duel-strength-protocol-v1';
export const REGRESSION_MODE = 'fixed-node-budget-v1';
export const STRENGTH_MODE = 'monotonic-deadline-v1';
export const DETERMINISTIC_REGRESSION_CLOCK_PROFILE = Object.freeze({
  id: 'normal-duel-regression-logical-clock-v1',
  source: 'fixed-logical-tick',
  deterministic: true,
  tickMilliseconds: 4,
  startsAtMilliseconds: 0,
  calibration: '12-canonical-9x9-pinned-hard-roots-v1',
  realSafetyCeilingMilliseconds: 10_000,
  productEquivalent: false
});
export const REAL_MONOTONIC_CLOCK_PROFILE = Object.freeze({
  id: 'real-monotonic-performance-now-v1',
  source: 'performance.now',
  deterministic: false,
  tickMilliseconds: null,
  startsAtMilliseconds: null,
  calibration: null,
  realSafetyCeilingMilliseconds: null,
  productEquivalent: true
});
export const CALLER_SUPPLIED_CLOCK_PROFILE = Object.freeze({
  id: 'caller-supplied-clock-unprofiled-v1',
  source: 'caller-supplied',
  deterministic: null,
  tickMilliseconds: null,
  startsAtMilliseconds: null,
  calibration: null,
  realSafetyCeilingMilliseconds: null,
  productEquivalent: false
});
const ENFORCEMENT_TOKENS = new WeakSet();
const ENFORCEMENT_RESULT_SETS = new WeakMap();
const PROTOCOL_ERRORS = new WeakSet();

function fail(message) {
  const error = new TypeError(`${STRENGTH_PROTOCOL}: ${message}`);
  PROTOCOL_ERRORS.add(error);
  throw error;
}

function other(player) {
  return player === 'A' ? 'B' : 'A';
}

function deepFreeze(value, seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) deepFreeze(child, seen);
  return Object.freeze(value);
}

function immutableSnapshot(value) {
  return deepFreeze(structuredClone(value));
}

export function deterministicRegressionClockFactory() {
  let now = DETERMINISTIC_REGRESSION_CLOCK_PROFILE.startsAtMilliseconds;
  return Object.freeze({
    now() {
      const current = now;
      now += DETERMINISTIC_REGRESSION_CLOCK_PROFILE.tickMilliseconds;
      return current;
    }
  });
}

function realMonotonicClockFactory() {
  return Object.freeze({ now: () => performance.now() });
}

function checkedClockProfile(profile, name = 'clockProfile') {
  try {
    if (!profile || typeof profile !== 'object' || Array.isArray(profile)
      || typeof profile.id !== 'string' || profile.id.length === 0) {
      fail(`${name} must be an object with a non-empty id`);
    }
    return immutableSnapshot(profile);
  } catch (error) {
    if (error && typeof error === 'object' && PROTOCOL_ERRORS.has(error)) throw error;
    fail(`${name} must be a structured-cloneable object with a non-empty id`);
  }
}

function timingSourceForClockProfile(profile) {
  if (profile.deterministic === true) return 'deterministic-logical-clock-time';
  if (profile.id === REAL_MONOTONIC_CLOCK_PROFILE.id) {
    return 'trusted-harness-active-time';
  }
  return 'caller-supplied-clock-time';
}

function failureReason(error) {
  if (isPinnedHardClockProfileExceededError(error)) return 'clock_profile_exceeded';
  if (isTrustedSubprocessDeadlineError(error)) return 'deadline';
  if (isTrustedSubprocessMemoryError(error)) return 'memory_limit';
  return 'crash';
}

function assertRegressionEnginesAreInProcess(mode, engines) {
  if (mode !== REGRESSION_MODE) return;
  for (const engine of engines) {
    if (getWorkerEngineIsolationProvenance(engine) !== null) {
      fail('fixed-node regression mode requires in-process engine adapters');
    }
  }
}

function cloneAction(action) {
  return action.kind === 'pawn'
    ? { kind: 'pawn', to: { r: action.to.r, c: action.to.c } }
    : { kind: 'wall', wall: action.wall };
}

function integer(value, name, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) fail(`${name} must be an integer >= ${minimum}`);
  return value;
}

function finite(value, name, minimum = 0) {
  if (!Number.isFinite(value) || value < minimum) fail(`${name} must be finite and >= ${minimum}`);
  return value;
}

export function assertEngineDescriptor(engine, name = 'engine') {
  if (!engine || typeof engine !== 'object'
    || typeof engine.id !== 'string' || engine.id.length === 0
    || typeof engine.version !== 'string' || engine.version.length === 0
    || typeof engine.createSession !== 'function') {
    fail(`${name} must provide non-empty id/version and createSession()`);
  }
  return engine;
}

export function validateEnforcedStrengthOptions({
  mode,
  perMoveDeadlineMs,
  minimumOpeningPairs,
  evaluatedOpeningCount,
  evaluationSeed,
  contender,
  baseline,
  book,
  corpusProvenance
}) {
  if (mode !== STRENGTH_MODE) {
    fail('enforced gate requires monotonic-deadline strength mode');
  }
  if (perMoveDeadlineMs !== CANONICAL_STRENGTH_DEADLINE_MS) {
    fail(`enforced gate requires exactly ${CANONICAL_STRENGTH_DEADLINE_MS} ms per move`);
  }
  if (evaluationSeed !== CANONICAL_STRENGTH_SEED) {
    fail(`enforced gate requires canonical seed ${CANONICAL_STRENGTH_SEED}`);
  }
  assertEngineDescriptor(contender, 'contender');
  if (contender.capabilities?.hardDeadlineIsolation !== true
    || !isWorkerEngineAdapter(contender)
    || !isAuthenticatedCandidateAdapter(contender)) {
    fail(
      'enforced gate requires a content-addressed candidate with canonical subprocess isolation'
    );
  }
  assertEngineDescriptor(baseline, 'baseline');
  if (!isPinnedHardWorkerAdapter(baseline)) {
    fail('enforced gate requires the privately branded pinned Hard subprocess baseline');
  }
  assertVerifiedOpeningCorpus(book, corpusProvenance);
  if (book.configurationSha256 !== CANONICAL_STRENGTH_CONFIGURATION_SHA256
    || corpusProvenance.configurationSha256 !== CANONICAL_STRENGTH_CONFIGURATION_SHA256
    || JSON.stringify(book.config) !== JSON.stringify(CANONICAL_STRENGTH_CONFIG)
    || book.generator.seed !== CANONICAL_STRENGTH_SEED
    || corpusProvenance.generatorSeed !== CANONICAL_STRENGTH_SEED
    || book.generator.version !== CANONICAL_STRENGTH_GENERATOR_VERSION
    || corpusProvenance.generatorVersion !== CANONICAL_STRENGTH_GENERATOR_VERSION
    || book.generator.algorithm !== CANONICAL_STRENGTH_GENERATOR_ALGORITHM
    || corpusProvenance.generatorAlgorithm !== CANONICAL_STRENGTH_GENERATOR_ALGORITHM) {
    fail('enforced gate requires the canonical 9x9, first-player-A seeded corpus');
  }
  integer(minimumOpeningPairs, 'minimumOpeningPairs', MINIMUM_STRENGTH_OPENING_PAIRS);
  integer(evaluatedOpeningCount, 'evaluatedOpeningCount', MINIMUM_STRENGTH_OPENING_PAIRS);
  if (evaluatedOpeningCount < minimumOpeningPairs) {
    fail(`enforced gate requires at least ${minimumOpeningPairs} evaluated opening pairs`);
  }
  if (book.openings.length !== evaluatedOpeningCount
    || book.generator.openingCount !== evaluatedOpeningCount
    || corpusProvenance.generatorOpeningCount !== evaluatedOpeningCount) {
    fail('enforced gate must evaluate the complete privately verified opening corpus');
  }
  const token = Object.freeze({
    enforced: true,
    eligible: true,
    minimumOpeningPairs,
    evaluatedOpeningCount,
    perMoveDeadlineMs,
    evaluationSeed,
    configurationSha256: CANONICAL_STRENGTH_CONFIGURATION_SHA256,
    bookSha256: corpusProvenance.bookSha256,
    manifestSha256: corpusProvenance.manifestSha256,
    baselineId: baseline.id,
    baselineVersion: baseline.version,
    baselineSourceCommit: baseline.sourceCommit,
    baselineTrustRoot: baseline.baselineTrustRoot,
    candidateArtifactProvenance: getCandidateArtifactProvenance(contender),
    artifactIntegrity: 'content-addressed-hermetic-release-v1',
    moduleLoadIsolation: 'manifest-files-safe-builtins-v1',
    filesystemContentIsolation: 'manifest-files-only-v1',
    environmentIsolation: 'spawn-minimal-lang-c-utc-v1',
    networkIsolation: 'safe-builtins-no-network-db-v1',
    hardDeadlineIsolation: true,
    integrityIsolation: 'node22-permission-readonly-subprocess-v1',
    stdioIsolation: 'null-device-v1',
    memoryIsolation: 'darwin-taskpolicy-rss-limit-v1',
    memoryLimitMiB: CANONICAL_ENGINE_MEMORY_LIMIT_MIB,
    v8OldSpaceMiB: CANONICAL_ENGINE_V8_OLD_SPACE_MIB,
    memoryPreflight: 'darwin-taskpolicy-preflight-96-112-v1',
    activeTimeCharging: 'setup-observe-select-next-move-v1',
    initializationCapMs: CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
    observerCapMs: CANONICAL_STRENGTH_OBSERVER_CAP_MS,
    corpusVerified: true
  });
  ENFORCEMENT_TOKENS.add(token);
  return token;
}

export function validateOpeningBook(book) {
  if (!book || book.bookFormat !== 'normal-duel-balanced-opening-book-v1'
    || book.generator?.algorithm !== 'lcg32-v1'
    || !Array.isArray(book.openings) || book.openings.length === 0) {
    fail('book must be a non-empty normal-duel-balanced-opening-book-v1 using lcg32-v1');
  }
  const config = validateConfig(book.config);
  const ids = new Set();
  const positions = new Set();
  for (const opening of book.openings) {
    if (!opening || typeof opening.id !== 'string' || opening.id.length === 0
      || ids.has(opening.id)
      || !Number.isSafeInteger(opening.targetPlies)
      || opening.targetPlies < 4 || opening.targetPlies > 6
      || !Array.isArray(opening.actionCodes)
      || opening.actionCodes.length !== opening.targetPlies
      || typeof opening.positionKey !== 'string' || opening.positionKey.length === 0
      || positions.has(opening.positionKey)
      || opening.diagnostics?.balanced !== true
      || !Number.isFinite(opening.diagnostics?.distanceDelta)
      || Math.abs(opening.diagnostics.distanceDelta) > 1) {
      fail(`opening ${String(opening?.id)} violates the balanced 4–6-ply book contract`);
    }
    for (const code of opening.actionCodes) integer(code, `${opening.id} action code`);
    replayOpening(config, opening);
    ids.add(opening.id);
    positions.add(opening.positionKey);
  }
  return Object.freeze({ book, config });
}

function assertSession(session, engineId) {
  if (!session || typeof session.selectAction !== 'function'
    || typeof session.observe !== 'function'
    || typeof session.close !== 'function') {
    fail(`${engineId} createSession() must provide selectAction(), observe(), and close()`);
  }
}

function replayOpening(config, opening) {
  if (!opening || typeof opening.id !== 'string' || !Array.isArray(opening.actionCodes)) {
    fail('opening must provide id and actionCodes');
  }
  let state = createInitialState(config);
  const history = [];
  for (let index = 0; index < opening.actionCodes.length; index += 1) {
    const code = opening.actionCodes[index];
    integer(code, `${opening.id} action code`);
    const player = state.position.turn;
    const action = decodeAction(config, code);
    const before = state;
    state = applyAction(config, state, action);
    if (state.outcome.kind !== 'ongoing' && index !== opening.actionCodes.length - 1) {
      fail(`${opening.id} continues after a terminal opening action`);
    }
    history.push(Object.freeze({
      player,
      action: Object.freeze(cloneAction(action)),
      stateBefore: before,
      stateAfter: state
    }));
  }
  if (state.outcome.kind !== 'ongoing') fail(`${opening.id} must end in an ongoing state`);
  if (opening.positionKey && state.positionKey !== opening.positionKey) {
    fail(`${opening.id} positionKey differs from replay`);
  }
  return Object.freeze({ state, history: Object.freeze(history) });
}

function monotonic(clock) {
  if (!clock || typeof clock.now !== 'function') fail('clock must provide now()');
  return finite(clock.now(), 'clock.now()');
}

function guardMonotonicClock(clock) {
  let previous = -Infinity;
  return Object.freeze({
    now() {
      const value = monotonic(clock);
      if (value < previous) fail(`clock moved backwards from ${previous} to ${value}`);
      previous = value;
      return value;
    }
  });
}

async function settleDecision(session, request, wallTimeoutMs) {
  let timer = null;
  try {
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve({ kind: 'wall-timeout' }), wallTimeoutMs);
    });
    const decision = Promise.resolve()
      .then(() => session.selectAction(request))
      .then(
        (value) => ({ kind: 'value', value }),
        (error) => ({ kind: 'error', error })
      );
    return await Promise.race([decision, timeout]);
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

function decisionEnvelope(raw) {
  if (raw && typeof raw === 'object' && Object.hasOwn(raw, 'action')) {
    return { action: raw.action, stats: raw.stats ?? {} };
  }
  return { action: raw, stats: {} };
}

function failureResult({ state, opening, assignments, failedPlayer, reason, error, telemetry }) {
  return Object.freeze({
    protocol: STRENGTH_PROTOCOL,
    openingId: opening.id,
    pairId: opening.pairId ?? opening.id,
    gameInPair: assignments.gameInPair,
    contenderSide: assignments.contenderSide,
    engines: Object.freeze({ A: assignments.A.id, B: assignments.B.id }),
    settings: assignments.settings,
    winner: other(failedPlayer),
    resultKind: 'forfeit',
    reason,
    failedPlayer,
    error: error ? String(error.message ?? error) : null,
    startingPly: opening.actionCodes.length,
    finalPly: state.ply,
    finalOutcome: state.outcome,
    telemetry
  });
}

function newTelemetry(engine, clockProfile) {
  return {
    engineId: engine.id,
    clockProfileId: clockProfile.id,
    timingSource: timingSourceForClockProfile(clockProfile),
    decisionMilliseconds: [],
    chargedSetupMilliseconds: [],
    chargedObserverMilliseconds: [],
    chargedSelectMilliseconds: [],
    reportedDecisionMilliseconds: [],
    nodes: [],
    depths: [],
    actionMix: { pawn: 0, wall: 0 },
    antiStallReplacements: 0
  };
}

function recordTrustedTiming(telemetry, timing, fallbackSelectMs = 0) {
  const setupMs = Number.isFinite(timing?.setupMs) ? Math.max(0, timing.setupMs) : 0;
  const observerMs = Number.isFinite(timing?.observerMs) ? Math.max(0, timing.observerMs) : 0;
  const selectMs = Number.isFinite(timing?.selectMs)
    ? Math.max(0, timing.selectMs)
    : Math.max(0, fallbackSelectMs);
  const chargedActiveMs = Number.isFinite(timing?.chargedActiveMs)
    ? Math.max(0, timing.chargedActiveMs)
    : setupMs + observerMs + selectMs;
  telemetry.decisionMilliseconds.push(chargedActiveMs);
  telemetry.chargedSetupMilliseconds.push(setupMs);
  telemetry.chargedObserverMilliseconds.push(observerMs);
  telemetry.chargedSelectMilliseconds.push(selectMs);
}

function recordStats(telemetry, stats, measuredElapsed, trustedTiming = null) {
  recordTrustedTiming(telemetry, trustedTiming, measuredElapsed);
  if (Number.isFinite(stats?.elapsedMs) && stats.elapsedMs >= 0) {
    telemetry.reportedDecisionMilliseconds.push(stats.elapsedMs);
  }
  if (Number.isSafeInteger(stats?.nodes) && stats.nodes >= 0) telemetry.nodes.push(stats.nodes);
  if (Number.isFinite(stats?.depth) && stats.depth >= 0) telemetry.depths.push(stats.depth);
  if (stats?.antiStallReplaced === true) telemetry.antiStallReplacements += 1;
}

function frozenTelemetry(raw, sessions = {}) {
  for (const side of ['A', 'B']) {
    const pendingTiming = takeTrustedSubprocessPendingTiming(sessions[side]);
    if (pendingTiming) recordTrustedTiming(raw[side], pendingTiming);
  }
  return Object.freeze({
    A: Object.freeze({
      ...raw.A,
      decisionMilliseconds: Object.freeze([...raw.A.decisionMilliseconds]),
      chargedSetupMilliseconds: Object.freeze([...raw.A.chargedSetupMilliseconds]),
      chargedObserverMilliseconds: Object.freeze([...raw.A.chargedObserverMilliseconds]),
      chargedSelectMilliseconds: Object.freeze([...raw.A.chargedSelectMilliseconds]),
      reportedDecisionMilliseconds: Object.freeze([...raw.A.reportedDecisionMilliseconds]),
      nodes: Object.freeze([...raw.A.nodes]),
      depths: Object.freeze([...raw.A.depths]),
      actionMix: Object.freeze({ ...raw.A.actionMix })
    }),
    B: Object.freeze({
      ...raw.B,
      decisionMilliseconds: Object.freeze([...raw.B.decisionMilliseconds]),
      chargedSetupMilliseconds: Object.freeze([...raw.B.chargedSetupMilliseconds]),
      chargedObserverMilliseconds: Object.freeze([...raw.B.chargedObserverMilliseconds]),
      chargedSelectMilliseconds: Object.freeze([...raw.B.chargedSelectMilliseconds]),
      reportedDecisionMilliseconds: Object.freeze([...raw.B.reportedDecisionMilliseconds]),
      nodes: Object.freeze([...raw.B.nodes]),
      depths: Object.freeze([...raw.B.depths]),
      actionMix: Object.freeze({ ...raw.B.actionMix })
    })
  });
}

/**
 * Play one opening from a fixed side assignment. Any null, crash, illegal
 * action, deadline miss, or node-budget protocol violation is an immediate
 * loss for the offending engine.
 */
export async function runMatch(options) {
  const {
    config,
    opening,
    engines,
    contenderSide,
    gameInPair,
    seed,
    mode = STRENGTH_MODE,
    perMoveDeadlineMs = CANONICAL_STRENGTH_DEADLINE_MS,
    nodeBudget = null,
    clock = null,
    clockProfile = null,
    wallTimeoutGraceMs = 100
  } = options;
  const checkedConfig = validateConfig(config);
  if (!engines || !engines.A || !engines.B) fail('engines must assign A and B');
  assertEngineDescriptor(engines.A, 'engines.A');
  assertEngineDescriptor(engines.B, 'engines.B');
  if (contenderSide !== 'A' && contenderSide !== 'B') fail('contenderSide must be A or B');
  integer(gameInPair, 'gameInPair');
  integer(seed, 'seed');
  if (mode !== STRENGTH_MODE && mode !== REGRESSION_MODE) fail(`unsupported mode ${mode}`);
  if (mode === STRENGTH_MODE) finite(perMoveDeadlineMs, 'perMoveDeadlineMs', 1);
  if (mode === REGRESSION_MODE) integer(nodeBudget, 'nodeBudget', 1);
  assertRegressionEnginesAreInProcess(mode, [engines.A, engines.B]);
  finite(wallTimeoutGraceMs, 'wallTimeoutGraceMs');
  if (clock === null && clockProfile !== null) {
    fail('clockProfile requires an explicit clock');
  }
  const selectedClock = clock
    ?? (mode === REGRESSION_MODE
      ? deterministicRegressionClockFactory()
      : realMonotonicClockFactory());
  const selectedClockProfile = checkedClockProfile(
    clockProfile
      ?? (clock !== null
        ? CALLER_SUPPLIED_CLOCK_PROFILE
        : mode === REGRESSION_MODE
          ? DETERMINISTIC_REGRESSION_CLOCK_PROFILE
          : REAL_MONOTONIC_CLOCK_PROFILE)
  );

  const replayed = replayOpening(checkedConfig, opening);
  let state = replayed.state;
  const guardedClock = guardMonotonicClock(selectedClock);
  const assignments = Object.freeze({
    A: engines.A,
    B: engines.B,
    contenderSide,
    gameInPair,
    settings: Object.freeze({
      mode,
      seed,
      nodeBudget: mode === REGRESSION_MODE ? nodeBudget : null,
      perMoveDeadlineMs: mode === STRENGTH_MODE ? perMoveDeadlineMs : null,
      clockProfile: selectedClockProfile
    })
  });
  const sessionSeed = createLcg32(seed);
  const sessions = {};
  const telemetry = {
    A: newTelemetry(engines.A, selectedClockProfile),
    B: newTelemetry(engines.B, selectedClockProfile)
  };
  const engineConfig = immutableSnapshot(checkedConfig);

  try {
    for (const side of ['A', 'B']) {
      try {
        sessions[side] = await engines[side].createSession(immutableSnapshot({
          protocol: STRENGTH_PROTOCOL,
          mode,
          gameId: `${opening.id}/${gameInPair}`,
          side,
          config: checkedConfig,
          seed: sessionSeed(),
          nodeBudget: mode === REGRESSION_MODE ? nodeBudget : null,
          clockProfile: selectedClockProfile,
          openingHistory: replayed.history
        }));
        assertSession(sessions[side], engines[side].id);
        if (typeof sessions[side].ready === 'function') await sessions[side].ready();
      } catch (error) {
        const trustedTiming = takeTrustedSubprocessFailureTiming(error);
        if (trustedTiming) recordTrustedTiming(telemetry[side], trustedTiming);
        return failureResult({
          state,
          opening,
          assignments,
          failedPlayer: side,
          reason: failureReason(error),
          error,
          telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
    }

    while (state.outcome.kind === 'ongoing') {
      const player = state.position.turn;
      const startedAt = monotonic(guardedClock);
      const deadlineAtMs = mode === STRENGTH_MODE ? startedAt + perMoveDeadlineMs : Infinity;
      const wallTimeoutMs = mode === STRENGTH_MODE
        ? Math.ceil(perMoveDeadlineMs + wallTimeoutGraceMs)
        : 60_000;
      const settled = await settleDecision(sessions[player], Object.freeze({
        protocol: STRENGTH_PROTOCOL,
        mode,
        state: immutableSnapshot(state),
        config: engineConfig,
        player,
        seed: sessionSeed(),
        clock: guardedClock,
        deadlineAtMs,
        limits: Object.freeze({
          nodeBudget: mode === REGRESSION_MODE ? nodeBudget : null,
          deadlineAtMs: mode === STRENGTH_MODE ? deadlineAtMs : null,
          wallClockBudgetMs: mode === STRENGTH_MODE ? perMoveDeadlineMs : null
        })
      }), wallTimeoutMs);
      const endedAt = monotonic(guardedClock);
      if (settled.kind === 'wall-timeout') {
        recordTrustedTiming(telemetry[player], null, Math.max(0, endedAt - startedAt));
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'deadline', telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      if (settled.kind === 'error') {
        const trustedTiming = takeTrustedSubprocessFailureTiming(settled.error);
        recordTrustedTiming(
          telemetry[player],
          trustedTiming,
          Math.max(0, endedAt - startedAt)
        );
        const reason = failureReason(settled.error);
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason, error: settled.error, telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      if (mode === STRENGTH_MODE && endedAt > deadlineAtMs) {
        recordTrustedTiming(telemetry[player], null, Math.max(0, endedAt - startedAt));
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'deadline', telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      let action;
      let stats;
      let trustedDecision = null;
      let timingRecorded = false;
      try {
        trustedDecision = takeTrustedSubprocessDecision(settled.value);
        const engineValue = trustedDecision ? trustedDecision.value : settled.value;
        ({ action, stats } = decisionEnvelope(immutableSnapshot(engineValue)));
        recordStats(
          telemetry[player],
          stats,
          Math.max(0, endedAt - startedAt),
          trustedDecision?.timing ?? null
        );
        timingRecorded = true;
      } catch (error) {
        if (!timingRecorded) {
          recordTrustedTiming(
            telemetry[player],
            trustedDecision?.timing ?? null,
            Math.max(0, endedAt - startedAt)
          );
        }
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'crash', error, telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      if (action === null || action === undefined) {
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'null_action', telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      if (mode === REGRESSION_MODE
        && engines[player].capabilities?.nodeBudget === true
        && (!Number.isSafeInteger(stats?.nodes) || stats.nodes < 0 || stats.nodes > nodeBudget)) {
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'node_budget', telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      let next;
      let trustedAction;
      try {
        trustedAction = immutableSnapshot(action);
        next = applyAction(checkedConfig, state, trustedAction);
      } catch (error) {
        return failureResult({
          state, opening, assignments, failedPlayer: player,
          reason: 'illegal_action', error, telemetry: frozenTelemetry(telemetry, sessions)
        });
      }
      telemetry[player].actionMix[trustedAction.kind] += 1;
      const transition = immutableSnapshot({
        player,
        action: cloneAction(trustedAction),
        stateBefore: state,
        stateAfter: next,
        source: 'engine'
      });
      for (const side of ['A', 'B']) {
        try {
          await sessions[side].observe(transition);
        } catch (error) {
          const trustedTiming = takeTrustedSubprocessFailureTiming(error);
          if (trustedTiming) recordTrustedTiming(telemetry[side], trustedTiming);
          return failureResult({
            state: next,
            opening,
            assignments,
            failedPlayer: side,
            reason: failureReason(error),
            error,
            telemetry: frozenTelemetry(telemetry, sessions)
          });
        }
      }
      state = next;
    }

    return Object.freeze({
      protocol: STRENGTH_PROTOCOL,
      openingId: opening.id,
      pairId: opening.pairId ?? opening.id,
      gameInPair,
      contenderSide,
      engines: Object.freeze({ A: engines.A.id, B: engines.B.id }),
      settings: assignments.settings,
      winner: state.outcome.kind === 'win' ? state.outcome.winner : null,
      resultKind: state.outcome.kind,
      reason: state.outcome.reason ?? null,
      failedPlayer: null,
      error: null,
      startingPly: opening.actionCodes.length,
      finalPly: state.ply,
      finalOutcome: state.outcome,
      telemetry: frozenTelemetry(telemetry, sessions)
    });
  } finally {
    for (const side of ['A', 'B']) {
      if (!sessions[side]) continue;
      try {
        await sessions[side].close();
      } catch {
        // A cleanup failure cannot rewrite an already-adjudicated match.
      }
    }
  }
}

function resultPoint(result, contenderId, drawValue) {
  if (result.winner === null) return drawValue;
  return result.engines[result.winner] === contenderId ? 1 : 0;
}

function clusterInterval(values) {
  const count = values.length;
  if (count === 0) {
    return Object.freeze({ clusters: 0, mean: null, standardError: null, lower: null, upper: null });
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / count;
  if (count === 1) {
    return Object.freeze({ clusters: 1, mean, standardError: null, lower: 0, upper: 1 });
  }
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (count - 1);
  const standardError = Math.sqrt(variance / count);
  return Object.freeze({
    clusters: count,
    mean,
    standardError,
    lower: Math.max(0, mean - 1.96 * standardError),
    upper: Math.min(1, mean + 1.96 * standardError)
  });
}

export function pairedClusterConfidenceIntervals(results, contenderId) {
  const pairs = new Map();
  for (const result of results) {
    const pair = pairs.get(result.pairId) ?? [];
    pair.push(result);
    pairs.set(result.pairId, pair);
  }
  const winClusters = [];
  const scoreClusters = [];
  for (const [pairId, pair] of pairs) {
    if (pair.length !== 2
      || new Set(pair.map((result) => result.contenderSide)).size !== 2) {
      fail(`pair ${pairId} must contain exactly one contender game on each side`);
    }
    winClusters.push(pair.reduce((sum, result) => sum + resultPoint(result, contenderId, 0), 0) / 2);
    scoreClusters.push(pair.reduce((sum, result) => sum + resultPoint(result, contenderId, 0.5), 0) / 2);
  }
  return Object.freeze({
    method: 'paired-opening-cluster-normal-95-v1',
    winRate: clusterInterval(winClusters),
    score: clusterInterval(scoreClusters)
  });
}

function percentile(sorted, fraction) {
  if (sorted.length === 0) return null;
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

function summarizeEngineTelemetry(results, engineId) {
  const elapsed = [];
  const setupElapsed = [];
  const observerElapsed = [];
  const selectElapsed = [];
  const reportedElapsed = [];
  const nodes = [];
  const depths = [];
  const actionMix = { pawn: 0, wall: 0 };
  const clockProfileIds = new Set();
  const timingSources = new Set();
  let antiStallReplacements = 0;
  for (const result of results) {
    for (const side of ['A', 'B']) {
      const telemetry = result.telemetry[side];
      if (telemetry.engineId !== engineId) continue;
      clockProfileIds.add(telemetry.clockProfileId);
      timingSources.add(telemetry.timingSource);
      elapsed.push(...telemetry.decisionMilliseconds);
      setupElapsed.push(...(telemetry.chargedSetupMilliseconds ?? []));
      observerElapsed.push(...(telemetry.chargedObserverMilliseconds ?? []));
      selectElapsed.push(...(telemetry.chargedSelectMilliseconds ?? []));
      reportedElapsed.push(...telemetry.reportedDecisionMilliseconds);
      nodes.push(...telemetry.nodes);
      depths.push(...telemetry.depths);
      actionMix.pawn += telemetry.actionMix.pawn;
      actionMix.wall += telemetry.actionMix.wall;
      antiStallReplacements += telemetry.antiStallReplacements;
    }
  }
  const sortedElapsed = [...elapsed].sort((left, right) => left - right);
  const sortedReportedElapsed = [...reportedElapsed].sort((left, right) => left - right);
  const totalMs = elapsed.reduce((sum, value) => sum + value, 0);
  const reportedTotalMs = reportedElapsed.reduce((sum, value) => sum + value, 0);
  const totalNodes = nodes.reduce((sum, value) => sum + value, 0);
  const clockProfileId = clockProfileIds.size === 1
    ? [...clockProfileIds][0]
    : 'mixed';
  const timingSource = timingSources.size === 1
    ? [...timingSources][0]
    : 'mixed-clock-time';
  return Object.freeze({
    decisions: elapsed.length,
    timing: Object.freeze({
      source: timingSource,
      clockProfileId,
      totalMs,
      meanMs: elapsed.length ? totalMs / elapsed.length : null,
      p50Ms: percentile(sortedElapsed, 0.5),
      p95Ms: percentile(sortedElapsed, 0.95),
      maxMs: sortedElapsed.at(-1) ?? null
    }),
    chargedActiveTime: Object.freeze({
      setupMs: setupElapsed.reduce((sum, value) => sum + value, 0),
      observerMs: observerElapsed.reduce((sum, value) => sum + value, 0),
      selectMs: selectElapsed.reduce((sum, value) => sum + value, 0)
    }),
    untrustedSelfReportedTiming: Object.freeze({
      reportedDecisions: reportedElapsed.length,
      totalMs: reportedTotalMs,
      meanMs: reportedElapsed.length ? reportedTotalMs / reportedElapsed.length : null,
      p50Ms: percentile(sortedReportedElapsed, 0.5),
      p95Ms: percentile(sortedReportedElapsed, 0.95),
      maxMs: sortedReportedElapsed.at(-1) ?? null
    }),
    nodes: Object.freeze({
      reportedDecisions: nodes.length,
      total: totalNodes,
      mean: nodes.length ? totalNodes / nodes.length : null
    }),
    depth: Object.freeze({
      reportedDecisions: depths.length,
      mean: depths.length ? depths.reduce((sum, value) => sum + value, 0) / depths.length : null,
      max: depths.length ? Math.max(...depths) : null
    }),
    actionMix: Object.freeze(actionMix),
    antiStallReplacements
  });
}

export function summarizeEvaluation(
  results,
  {
    contender,
    baseline,
    minimumOpeningPairs = MINIMUM_STRENGTH_OPENING_PAIRS,
    enforcement = Object.freeze({ enforced: false, eligible: false })
  }
) {
  assertEngineDescriptor(contender, 'contender');
  assertEngineDescriptor(baseline, 'baseline');
  const games = results.length;
  const wins = results.filter((result) =>
    result.winner !== null && result.engines[result.winner] === contender.id).length;
  const losses = results.filter((result) =>
    result.winner !== null && result.engines[result.winner] === baseline.id).length;
  const draws = games - wins - losses;
  const sideSplits = {};
  for (const side of ['A', 'B']) {
    const subset = results.filter((result) => result.contenderSide === side);
    const sideWins = subset.filter((result) =>
      result.winner !== null && result.engines[result.winner] === contender.id).length;
    const sideLosses = subset.filter((result) =>
      result.winner !== null && result.engines[result.winner] === baseline.id).length;
    sideSplits[side] = Object.freeze({
      games: subset.length,
      wins: sideWins,
      losses: sideLosses,
      draws: subset.length - sideWins - sideLosses,
      winRate: subset.length ? sideWins / subset.length : null,
      score: subset.length ? (sideWins + 0.5 * (subset.length - sideWins - sideLosses)) / subset.length : null
    });
  }
  const confidenceIntervals = pairedClusterConfidenceIntervals(results, contender.id);
  const pairCount = new Set(results.map((result) => result.pairId)).size;
  const failures = {};
  for (const result of results) {
    if (result.resultKind !== 'forfeit') continue;
    failures[result.reason] = (failures[result.reason] ?? 0) + 1;
  }
  const enforcementEligible = ENFORCEMENT_TOKENS.has(enforcement)
    && ENFORCEMENT_RESULT_SETS.get(results) === enforcement;
  return Object.freeze({
    protocol: STRENGTH_PROTOCOL,
    contender: Object.freeze({ id: contender.id, version: contender.version }),
    baseline: Object.freeze({
      id: baseline.id,
      version: baseline.version,
      sourceCommit: baseline.sourceCommit ?? null
    }),
    openingPairs: pairCount,
    games,
    wins,
    losses,
    draws,
    winRate: games ? wins / games : null,
    score: games ? (wins + 0.5 * draws) / games : null,
    sideSplits: Object.freeze(sideSplits),
    confidenceIntervals,
    failures: Object.freeze(failures),
    telemetry: Object.freeze({
      contender: summarizeEngineTelemetry(results, contender.id),
      baseline: summarizeEngineTelemetry(results, baseline.id)
    }),
    gate: Object.freeze({
      enforced: enforcementEligible,
      eligible: enforcementEligible,
      canonicalDeadlineMs: enforcementEligible ? enforcement.perMoveDeadlineMs : null,
      canonicalInitializationCapMs: enforcementEligible
        ? enforcement.initializationCapMs
        : null,
      canonicalObserverCapMs: enforcementEligible ? enforcement.observerCapMs : null,
      canonicalConfigurationSha256: enforcementEligible
        ? enforcement.configurationSha256
        : null,
      canonicalSeed: enforcementEligible ? enforcement.evaluationSeed : null,
      requiredOpeningPairs: minimumOpeningPairs,
      requiredGames: minimumOpeningPairs * 2,
      requiredWinRate: 0.66,
      requiresPairedScoreLowerBoundAbove: 0.5,
      sampleSizeMet: pairCount >= minimumOpeningPairs && games >= minimumOpeningPairs * 2,
      winRateMet: games > 0 && wins / games >= 0.66,
      pairedSuperiorityMet: confidenceIntervals.score.lower !== null
        && confidenceIntervals.score.lower > 0.5,
      criteriaMet: pairCount >= minimumOpeningPairs
        && games >= minimumOpeningPairs * 2
        && wins / games >= 0.66
        && confidenceIntervals.score.lower > 0.5,
      passed: enforcementEligible
        && pairCount >= minimumOpeningPairs
        && games >= minimumOpeningPairs * 2
        && wins / games >= 0.66
        && confidenceIntervals.score.lower > 0.5
    })
  });
}

/**
 * Evaluate each opening twice, swapping contender and baseline sides. Openings
 * are sequential by default so reports and failure reproduction are stable.
 */
export async function runEvaluation({
  contender,
  baseline,
  book,
  config = book?.config,
  mode = STRENGTH_MODE,
  seed = book?.generator?.seed ?? 0,
  perMoveDeadlineMs = CANONICAL_STRENGTH_DEADLINE_MS,
  nodeBudget = null,
  minimumOpeningPairs = MINIMUM_STRENGTH_OPENING_PAIRS,
  clockFactory = null,
  clockProfile = null,
  enforceGate = false,
  corpusProvenance = null
}) {
  assertEngineDescriptor(contender, 'contender');
  assertEngineDescriptor(baseline, 'baseline');
  if (contender.id === baseline.id) fail('contender and baseline ids must differ');
  if (mode !== STRENGTH_MODE && mode !== REGRESSION_MODE) fail(`unsupported mode ${mode}`);
  assertRegressionEnginesAreInProcess(mode, [contender, baseline]);
  const validatedBook = validateOpeningBook(book);
  const checkedConfig = validateConfig(config ?? validatedBook.config);
  if (JSON.stringify(checkedConfig) !== JSON.stringify(validatedBook.config)) {
    fail('evaluation config must exactly match the opening-book config');
  }
  integer(seed, 'seed');
  if (enforceGate && (clockFactory !== null || clockProfile !== null)) {
    fail('enforced gate requires the default real monotonic clock');
  }
  if (clockFactory !== null && typeof clockFactory !== 'function') {
    fail('clockFactory must be a function or null');
  }
  if (clockFactory === null && clockProfile !== null) {
    fail('clockProfile requires an explicit clockFactory');
  }
  const selectedClockFactory = clockFactory
    ?? (mode === REGRESSION_MODE
      ? deterministicRegressionClockFactory
      : realMonotonicClockFactory);
  const selectedClockProfile = checkedClockProfile(
    clockProfile
      ?? (clockFactory !== null
        ? CALLER_SUPPLIED_CLOCK_PROFILE
        : mode === REGRESSION_MODE
          ? DETERMINISTIC_REGRESSION_CLOCK_PROFILE
          : REAL_MONOTONIC_CLOCK_PROFILE)
  );
  const enforcement = enforceGate
    ? validateEnforcedStrengthOptions({
      mode,
      perMoveDeadlineMs,
      minimumOpeningPairs,
      evaluatedOpeningCount: book.openings.length,
      evaluationSeed: seed,
      contender,
      baseline,
      book,
      corpusProvenance
    })
    : Object.freeze({ enforced: false, eligible: false });
  const corpusBindingVerified = isVerifiedOpeningCorpus(book, corpusProvenance);
  const nextSeed = createLcg32(seed);
  const results = [];
  for (const opening of book.openings) {
    for (const contenderSide of ['A', 'B']) {
      const baselineSide = other(contenderSide);
      const engines = {
        [contenderSide]: contender,
        [baselineSide]: baseline
      };
      results.push(await runMatch({
        config: checkedConfig,
        opening: Object.freeze({ ...opening, pairId: opening.id }),
        engines,
        contenderSide,
        gameInPair: contenderSide === 'A' ? 0 : 1,
        seed: nextSeed(),
        mode,
        perMoveDeadlineMs,
        nodeBudget,
        clock: selectedClockFactory({ opening, contenderSide }),
        clockProfile: selectedClockProfile
      }));
    }
  }
  const frozenResults = Object.freeze(results);
  if (ENFORCEMENT_TOKENS.has(enforcement)) {
    ENFORCEMENT_RESULT_SETS.set(frozenResults, enforcement);
  }
  const summary = summarizeEvaluation(frozenResults, {
    contender,
    baseline,
    minimumOpeningPairs,
    enforcement
  });
  return Object.freeze({
    protocol: STRENGTH_PROTOCOL,
    mode,
    seed,
    perMoveDeadlineMs: mode === STRENGTH_MODE ? perMoveDeadlineMs : null,
    clockProfile: selectedClockProfile,
    book: Object.freeze({
      format: book.bookFormat,
      generatorVersion: book.generator.version,
      generatorAlgorithm: book.generator.algorithm,
      generatorSeed: book.generator.seed,
      configurationSha256: book.configurationSha256,
      openingCount: book.openings.length,
      evaluatedOpeningIds: Object.freeze(book.openings.map((opening) => opening.id))
    }),
    baselineProvenance: Object.freeze({
      id: baseline.id,
      version: baseline.version,
      sourceCommit: baseline.sourceCommit ?? null,
      trustRoot: baseline.baselineTrustRoot
        ? immutableSnapshot(baseline.baselineTrustRoot)
        : null,
      pinnedWorkerVerified: isPinnedHardWorkerAdapter(baseline)
    }),
    candidateArtifactProvenance: getCandidateArtifactProvenance(contender)
      ? immutableSnapshot(getCandidateArtifactProvenance(contender))
      : null,
    candidateArtifactVerified: isAuthenticatedCandidateAdapter(contender),
    candidateIsolationProvenance: getWorkerEngineIsolationProvenance(contender)
      ? immutableSnapshot(getWorkerEngineIsolationProvenance(contender))
      : null,
    baselineIsolationProvenance: getWorkerEngineIsolationProvenance(baseline)
      ? immutableSnapshot(getWorkerEngineIsolationProvenance(baseline))
      : null,
    enforcement: enforcement && ENFORCEMENT_TOKENS.has(enforcement)
      ? immutableSnapshot(enforcement)
      : Object.freeze({ enforced: false, eligible: false }),
    corpusProvenance: corpusProvenance
      ? immutableSnapshot({
        ...corpusProvenance,
        verified: corpusBindingVerified,
        boundToEvaluatedBook: corpusBindingVerified
      })
      : Object.freeze({ verified: false }),
    results: frozenResults,
    summary
  });
}
