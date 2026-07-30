/**
 * Compatibility-named engine proxy backed by a terminable Node subprocess.
 *
 * One permission-restricted child process is created per engine per game.
 * Candidate stdio is connected to the null device; only the parent emits reports.
 */
import { execFile, fork } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { accessSync, constants as fsConstants } from 'node:fs';
import { dirname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  candidateArtifactRuntimePolicy,
  loadCandidateArtifactManifest,
  sha256ArtifactFile,
  verifyCandidateArtifactManifest
} from './candidate-artifact-manifest.mjs';
import {
  HARD_BASELINE_TRUST_ROOT
} from './hard-baseline-46a871c7.mjs';
import {
  CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
  CANONICAL_STRENGTH_OBSERVER_CAP_MS
} from './normal-duel-strength-constants.mjs';

const RUNTIME_PATH = fileURLToPath(new URL('./worker-engine-runtime.mjs', import.meta.url));
const PINNED_HARD_MODULE_URL = new URL('./hard-baseline-46a871c7.mjs', import.meta.url).href;
const TASKPOLICY_PATH = '/usr/sbin/taskpolicy';
export const CANONICAL_ENGINE_MEMORY_LIMIT_MIB = 512;
export const CANONICAL_ENGINE_V8_OLD_SPACE_MIB = 128;
const MEMORY_PREFLIGHT_LIMIT_MIB = 96;
const MEMORY_PREFLIGHT_ALLOCATION_MIB = 112;
const MEMORY_PREFLIGHT_PROFILE = 'darwin-taskpolicy-preflight-96-112-v1';
const ENGINE_SUBPROCESS_ENVIRONMENT = Object.freeze({
  LANG: 'C',
  LC_ALL: 'C',
  TZ: 'UTC'
});
const SUBPROCESS_ADAPTERS = new WeakMap();
const AUTHENTICATED_ARTIFACT_ADAPTERS = new WeakMap();
const PINNED_HARD_SUBPROCESS_ADAPTERS = new WeakSet();
const TRUSTED_DECISIONS = new WeakMap();
const TRUSTED_FAILURE_TIMING = new WeakMap();
const TRUSTED_PENDING_TIMING = new WeakMap();
const TRUSTED_DEADLINE_ERRORS = new WeakSet();
const TRUSTED_MEMORY_ERRORS = new WeakSet();
let taskpolicyPreflightPromise = null;

export const PINNED_HARD_WORKER_ID = HARD_BASELINE_TRUST_ROOT.baselineId;
export const PINNED_HARD_WORKER_VERSION = HARD_BASELINE_TRUST_ROOT.baselineVersion;
export const PINNED_HARD_SOURCE_COMMIT = HARD_BASELINE_TRUST_ROOT.sourceCommit;

function exactCanonicalIsolation(metadata) {
  return metadata?.initializationCapMs === CANONICAL_STRENGTH_INITIALIZATION_CAP_MS
    && metadata?.observerCapMs === CANONICAL_STRENGTH_OBSERVER_CAP_MS
    && metadata?.activeTimeCharging === 'setup-observe-select-next-move-v1'
    && metadata?.permissionProfile === 'node22-permission-readonly-subprocess-v1'
    && metadata?.stdioIsolation === 'null-device-v1'
    && metadata?.environmentIsolation === 'spawn-minimal-lang-c-utc-v1'
    && metadata?.memoryIsolation === 'darwin-taskpolicy-rss-limit-v1'
    && metadata?.memoryLimitMiB === CANONICAL_ENGINE_MEMORY_LIMIT_MIB
    && metadata?.v8OldSpaceMiB === CANONICAL_ENGINE_V8_OLD_SPACE_MIB
    && metadata?.memoryPreflight === MEMORY_PREFLIGHT_PROFILE;
}

function exactAuthenticatedBoundary(metadata) {
  return metadata?.artifactIntegrity === 'content-addressed-hermetic-release-v1'
    && metadata?.moduleLoadIsolation === 'manifest-files-safe-builtins-v1'
    && metadata?.filesystemContentIsolation === 'manifest-files-only-v1'
    && metadata?.environmentIsolation === 'spawn-minimal-lang-c-utc-v1'
    && metadata?.networkIsolation === 'safe-builtins-no-network-db-v1';
}

export function isWorkerEngineAdapter(engine) {
  return Boolean(engine && typeof engine === 'object'
    && exactCanonicalIsolation(SUBPROCESS_ADAPTERS.get(engine)));
}

export function isPinnedHardWorkerAdapter(engine) {
  return isWorkerEngineAdapter(engine)
    && PINNED_HARD_SUBPROCESS_ADAPTERS.has(engine)
    && engine.id === PINNED_HARD_WORKER_ID
    && engine.version === PINNED_HARD_WORKER_VERSION
    && engine.sourceCommit === PINNED_HARD_SOURCE_COMMIT
    && trustRootsEqual(engine.baselineTrustRoot, HARD_BASELINE_TRUST_ROOT);
}

export function isAuthenticatedCandidateAdapter(engine) {
  return isWorkerEngineAdapter(engine)
    && exactAuthenticatedBoundary(SUBPROCESS_ADAPTERS.get(engine))
    && AUTHENTICATED_ARTIFACT_ADAPTERS.has(engine);
}

export function getCandidateArtifactProvenance(engine) {
  const privateBinding = AUTHENTICATED_ARTIFACT_ADAPTERS.get(engine);
  if (privateBinding) return privateBinding.provenance;
  return SUBPROCESS_ADAPTERS.get(engine)?.artifactProvenance ?? null;
}

export function getWorkerEngineIsolationProvenance(engine) {
  const metadata = engine && typeof engine === 'object'
    ? SUBPROCESS_ADAPTERS.get(engine)
    : null;
  if (!metadata) return null;
  return Object.freeze({
    permissionProfile: metadata.permissionProfile,
    stdioIsolation: metadata.stdioIsolation,
    memoryIsolation: metadata.memoryIsolation,
    memoryLimitMiB: metadata.memoryLimitMiB,
    v8OldSpaceMiB: metadata.v8OldSpaceMiB,
    memoryPreflight: metadata.memoryPreflight,
    artifactIntegrity: metadata.artifactIntegrity,
    moduleLoadIsolation: metadata.moduleLoadIsolation,
    filesystemContentIsolation: metadata.filesystemContentIsolation,
    environmentIsolation: metadata.environmentIsolation,
    networkIsolation: metadata.networkIsolation
  });
}

export function takeTrustedSubprocessDecision(value) {
  if (!value || typeof value !== 'object') return null;
  const record = TRUSTED_DECISIONS.get(value) ?? null;
  if (record) TRUSTED_DECISIONS.delete(value);
  return record;
}

export function takeTrustedSubprocessFailureTiming(error) {
  if (!error || typeof error !== 'object') return null;
  const timing = TRUSTED_FAILURE_TIMING.get(error) ?? null;
  if (timing) TRUSTED_FAILURE_TIMING.delete(error);
  return timing;
}

export function takeTrustedSubprocessPendingTiming(session) {
  const take = session && typeof session === 'object'
    ? TRUSTED_PENDING_TIMING.get(session)
    : null;
  return typeof take === 'function' ? take() : null;
}

export function isTrustedSubprocessDeadlineError(error) {
  return Boolean(error && typeof error === 'object'
    && TRUSTED_DEADLINE_ERRORS.has(error));
}

export function isTrustedSubprocessMemoryError(error) {
  return Boolean(error && typeof error === 'object'
    && TRUSTED_MEMORY_ERRORS.has(error));
}

function workerError(record, fallback = 'engine subprocess failed') {
  const error = new Error(record?.message ?? fallback);
  error.name = record?.name ?? 'EngineSubprocessError';
  if (record?.code) error.remoteCode = record.code;
  if (record?.stack) error.stack = record.stack;
  return error;
}

function timeoutError(message) {
  const error = new Error(message);
  error.name = 'EngineSubprocessTimeoutError';
  error.code = 'deadline_exceeded';
  TRUSTED_DEADLINE_ERRORS.add(error);
  return error;
}

function memoryLimitError(message) {
  const error = new Error(message);
  error.name = 'EngineSubprocessMemoryError';
  error.code = 'memory_limit_exceeded';
  TRUSTED_MEMORY_ERRORS.add(error);
  return error;
}

function runTaskpolicyPreflight() {
  if (taskpolicyPreflightPromise) return taskpolicyPreflightPromise;
  taskpolicyPreflightPromise = new Promise((resolve) => {
    if (process.platform !== 'darwin') {
      resolve(false);
      return;
    }
    try {
      accessSync(TASKPOLICY_PATH, fsConstants.X_OK);
    } catch {
      resolve(false);
      return;
    }
    const source = [
      `const size=${MEMORY_PREFLIGHT_ALLOCATION_MIB}*1024*1024;`,
      'const value=Buffer.alloc(size, 1);',
      'setTimeout(()=>process.stdout.write(String(value[0])), 250);'
    ].join('');
    execFile(TASKPOLICY_PATH, [
      '-m',
      String(MEMORY_PREFLIGHT_LIMIT_MIB),
      process.execPath,
      '--max-old-space-size=64',
      '--eval',
      source
    ], {
      encoding: 'utf8',
      timeout: 5_000,
      maxBuffer: 1_024
    }, (error) => {
      const killedByLimit = Boolean(error) && error.signal === 'SIGKILL';
      resolve(killedByLimit);
    });
  });
  return taskpolicyPreflightPromise;
}

function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive integer`);
  }
  return value;
}

async function memoryIsolationProfile({
  memoryLimitMiB,
  v8OldSpaceMiB,
  requireCanonicalMemoryIsolation
}) {
  positiveInteger(memoryLimitMiB, 'memoryLimitMiB');
  positiveInteger(v8OldSpaceMiB, 'v8OldSpaceMiB');
  if (v8OldSpaceMiB >= memoryLimitMiB) {
    throw new TypeError('v8OldSpaceMiB must be smaller than memoryLimitMiB');
  }
  const taskpolicyVerified = await runTaskpolicyPreflight();
  if (requireCanonicalMemoryIsolation && !taskpolicyVerified) {
    throw new Error(
      'canonical engine memory isolation requires verified macOS /usr/sbin/taskpolicy support'
    );
  }
  return Object.freeze(taskpolicyVerified
    ? {
      kind: 'darwin-taskpolicy-rss-limit-v1',
      memoryLimitMiB,
      v8OldSpaceMiB,
      preflight: MEMORY_PREFLIGHT_PROFILE
    }
    : {
      kind: 'v8-old-space-only-ineligible-v1',
      memoryLimitMiB: null,
      v8OldSpaceMiB,
      preflight: 'unsupported-platform-or-failed-preflight'
    });
}

function positiveFinite(value, name) {
  if (!Number.isFinite(value) || value <= 0) {
    throw new TypeError(`${name} must be finite and > 0`);
  }
  return value;
}

function freezeTiming({ setupMs = 0, observerMs = 0, selectMs = 0 }) {
  return Object.freeze({
    source: 'trusted-parent-subprocess-clock',
    setupMs,
    observerMs,
    selectMs,
    chargedActiveMs: setupMs + observerMs + selectMs
  });
}

function brandFailure(error, timing) {
  if (error && typeof error === 'object' && !TRUSTED_FAILURE_TIMING.has(error)) {
    TRUSTED_FAILURE_TIMING.set(error, freezeTiming(timing));
  }
  return error;
}

function brandDecision(value, timing) {
  const wrapper = Object.freeze({});
  TRUSTED_DECISIONS.set(wrapper, Object.freeze({
    value,
    timing: freezeTiming(timing)
  }));
  return wrapper;
}

function trustRootsEqual(left, right) {
  if (!left || !right) return false;
  return Object.keys(HARD_BASELINE_TRUST_ROOT)
    .every((key) => left[key] === right[key]);
}

function entryOnlyArtifactProvenance(moduleUrl) {
  try {
    const url = new URL(moduleUrl);
    if (url.protocol !== 'file:') {
      return Object.freeze({
        verification: 'unverified-module-url-v1',
        entryUrl: moduleUrl,
        entrySha256: null
      });
    }
    const filename = fileURLToPath(url);
    return Object.freeze({
      verification: 'parent-entry-hash-only-v1',
      entryUrl: moduleUrl,
      entrySha256: sha256ArtifactFile(filename)
    });
  } catch {
    return Object.freeze({
      verification: 'unverified-module-url-v1',
      entryUrl: moduleUrl,
      entrySha256: null
    });
  }
}

function readOnlyPermissionExecArgv(artifactPolicy) {
  const argumentsList = ['--permission', '--no-addons'];
  if (!artifactPolicy) {
    argumentsList.splice(1, 0, '--allow-fs-read=*');
    return argumentsList;
  }
  const allowedDirectories = new Set([
    dirname(RUNTIME_PATH),
    dirname(artifactPolicy.manifest.filename)
  ]);
  for (const filename of [...allowedDirectories].sort()) {
    argumentsList.push(`--allow-fs-read=${filename}`);
  }
  return argumentsList;
}

function startSubprocess({
  mode,
  moduleUrl,
  loadMode,
  context,
  initializationCapMs,
  artifactPolicy,
  memoryProfile
}) {
  const startedAt = performance.now();
  const channelId = randomBytes(24).toString('hex');
  const nodeExecArgv = [
    `--max-old-space-size=${memoryProfile.v8OldSpaceMiB}`,
    ...readOnlyPermissionExecArgv(artifactPolicy)
  ];
  const useTaskpolicy = memoryProfile.kind === 'darwin-taskpolicy-rss-limit-v1';
  const child = fork(RUNTIME_PATH, [], {
    execPath: useTaskpolicy ? TASKPOLICY_PATH : process.execPath,
    execArgv: useTaskpolicy
      ? [
        '-m',
        String(memoryProfile.memoryLimitMiB),
        process.execPath,
        ...nodeExecArgv
      ]
      : nodeExecArgv,
    cwd: artifactPolicy?.root ?? process.cwd(),
    env: ENGINE_SUBPROCESS_ENVIRONMENT,
    serialization: 'advanced',
    stdio: ['ignore', 'ignore', 'ignore', 'ipc']
  });

  let exited = false;
  let exitRecord = null;
  let terminatePromise = null;
  let readySettled = false;
  let readyTimer;
  let disconnectTimer;
  let resolveReady;
  let rejectReady;
  const pending = new Map();
  const exitPromise = new Promise((resolve) => {
    child.once('exit', (code, signal) => {
      exited = true;
      exitRecord = Object.freeze({ code, signal });
      resolve(exitRecord);
    });
  });
  const ready = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
    readyTimer = setTimeout(() => {
      const elapsed = performance.now() - startedAt;
      const error = brandFailure(
        timeoutError(`engine subprocess initialization exceeded ${initializationCapMs} ms`),
        { setupMs: elapsed }
      );
      readySettled = true;
      reject(error);
      void terminate();
    }, initializationCapMs);
  });

  const rejectAll = (error) => {
    if (!readySettled) {
      readySettled = true;
      clearTimeout(readyTimer);
      rejectReady(brandFailure(error, {
        setupMs: performance.now() - startedAt
      }));
    }
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    pending.clear();
  };

  const terminate = () => {
    if (terminatePromise) return terminatePromise;
    terminatePromise = (async () => {
      if (!exited) child.kill('SIGKILL');
      await exitPromise;
    })();
    return terminatePromise;
  };

  child.on('message', (message) => {
    if (!message || message.channelId !== channelId) {
      rejectAll(new Error('engine subprocess sent a message on an invalid IPC channel'));
      void terminate();
      return;
    }
    if (message.type === 'ready') {
      if (readySettled) {
        rejectAll(new Error('engine subprocess sent a duplicate ready message'));
        void terminate();
        return;
      }
      const elapsedMs = performance.now() - startedAt;
      if (elapsedMs > initializationCapMs) {
        const error = brandFailure(
          timeoutError(`engine subprocess initialization exceeded ${initializationCapMs} ms`),
          { setupMs: elapsedMs }
        );
        readySettled = true;
        clearTimeout(readyTimer);
        rejectReady(error);
        void terminate();
        return;
      }
      readySettled = true;
      clearTimeout(readyTimer);
      resolveReady(Object.freeze({
        descriptor: message.descriptor,
        elapsedMs
      }));
      return;
    }
    if (message.type === 'fatal') {
      rejectAll(workerError(message.error));
      void terminate();
      return;
    }
    if (message.type !== 'response') {
      rejectAll(new Error(`engine subprocess sent unsupported message ${String(message.type)}`));
      void terminate();
      return;
    }
    const request = pending.get(message.id);
    if (!request) {
      rejectAll(new Error('engine subprocess sent an unknown or late response'));
      void terminate();
      return;
    }
    pending.delete(message.id);
    clearTimeout(request.timer);
    if (message.error) request.reject(workerError(message.error));
    else request.resolve(message.value);
  });
  child.on('error', (error) => {
    rejectAll(error);
    void terminate();
  });
  child.on('disconnect', () => {
    if (!exited) {
      disconnectTimer = setTimeout(() => {
        if (exited) return;
        rejectAll(new Error('engine subprocess closed its IPC channel'));
        void terminate();
      }, 50);
      disconnectTimer.unref();
    }
  });
  exitPromise.then(({ code, signal }) => {
    clearTimeout(disconnectTimer);
    if (pending.size > 0 || !readySettled || code !== 0) {
      const unexpected = memoryProfile.kind === 'darwin-taskpolicy-rss-limit-v1'
        && signal === 'SIGKILL'
        ? memoryLimitError(
          `engine subprocess exceeded the ${memoryProfile.memoryLimitMiB} MiB memory profile`
        )
        : new Error(
          `engine subprocess exited unexpectedly (code=${String(code)}, signal=${String(signal)})`
        );
      rejectAll(unexpected);
    }
  });

  let nextRequestId = 1;
  const rpc = async (method, payload, timeoutMs) => {
    await ready;
    if (exited) {
      throw new Error(
        `engine subprocess is no longer running (${JSON.stringify(exitRecord)})`
      );
    }
    const id = nextRequestId;
    nextRequestId += 1;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(timeoutError(`${method} exceeded ${timeoutMs} ms`));
        void terminate();
      }, Math.max(1, Math.ceil(timeoutMs)));
      pending.set(id, { resolve, reject, timer });
      try {
        child.send({
          type: 'rpc',
          channelId,
          id,
          method,
          payload
        }, undefined, undefined, (error) => {
          if (!error) return;
          const request = pending.get(id);
          if (!request) return;
          pending.delete(id);
          clearTimeout(timer);
          request.reject(error);
          void terminate();
        });
      } catch (error) {
        pending.delete(id);
        clearTimeout(timer);
        reject(error);
        void terminate();
      }
    });
  };

  try {
    child.send({
      type: 'initialize',
      channelId,
      mode,
      moduleUrl,
      loadMode,
      context,
      artifactPolicy
    }, undefined, undefined, (error) => {
      if (!error) return;
      rejectAll(error);
      void terminate();
    });
  } catch (error) {
    rejectAll(error);
    void terminate();
  }

  return Object.freeze({
    ready,
    rpc,
    terminate
  });
}

async function probe({
  moduleUrl,
  loadMode,
  initializationCapMs,
  artifactPolicy,
  memoryProfile
}) {
  const host = startSubprocess({
    mode: 'probe',
    moduleUrl,
    loadMode,
    context: null,
    initializationCapMs,
    artifactPolicy,
    memoryProfile
  });
  try {
    return (await host.ready).descriptor;
  } finally {
    await host.terminate();
  }
}

export async function createWorkerEngineAdapter({
  moduleUrl,
  loadMode = Object.freeze({ kind: 'auto' }),
  initializationTimeoutMs = CANONICAL_STRENGTH_INITIALIZATION_CAP_MS,
  observerTimeoutMs = CANONICAL_STRENGTH_OBSERVER_CAP_MS,
  candidateManifestPath = null,
  requireCanonicalMemoryIsolation = false,
  memoryLimitMiB = CANONICAL_ENGINE_MEMORY_LIMIT_MIB,
  v8OldSpaceMiB = CANONICAL_ENGINE_V8_OLD_SPACE_MIB
}) {
  if (typeof moduleUrl !== 'string' || moduleUrl.length === 0) {
    throw new TypeError('moduleUrl must be a non-empty URL string');
  }
  positiveFinite(initializationTimeoutMs, 'initializationTimeoutMs');
  positiveFinite(observerTimeoutMs, 'observerTimeoutMs');
  const memoryProfile = await memoryIsolationProfile({
    memoryLimitMiB,
    v8OldSpaceMiB,
    requireCanonicalMemoryIsolation
  });
  const artifactBinding = candidateManifestPath === null
    ? null
    : loadCandidateArtifactManifest({ manifestPath: candidateManifestPath, moduleUrl });
  const effectiveModuleUrl = artifactBinding
    ? pathToFileURL(artifactBinding.entryFilename).href
    : moduleUrl;
  if (artifactBinding) verifyCandidateArtifactManifest(artifactBinding);
  const artifactPolicy = artifactBinding
    ? candidateArtifactRuntimePolicy(artifactBinding)
    : null;
  const artifactProvenance = artifactBinding
    ? artifactBinding.provenance
    : entryOnlyArtifactProvenance(moduleUrl);
  const metadata = await probe({
    moduleUrl: effectiveModuleUrl,
    loadMode,
    initializationCapMs: initializationTimeoutMs,
    artifactPolicy,
    memoryProfile
  });
  const isolationMetadata = Object.freeze({
    initializationCapMs: initializationTimeoutMs,
    observerCapMs: observerTimeoutMs,
    activeTimeCharging: 'setup-observe-select-next-move-v1',
    permissionProfile: 'node22-permission-readonly-subprocess-v1',
    stdioIsolation: 'null-device-v1',
    memoryIsolation: memoryProfile.kind,
    memoryLimitMiB: memoryProfile.memoryLimitMiB,
    v8OldSpaceMiB: memoryProfile.v8OldSpaceMiB,
    memoryPreflight: memoryProfile.preflight,
    artifactIntegrity: artifactBinding
      ? 'content-addressed-hermetic-release-v1'
      : null,
    moduleLoadIsolation: artifactBinding
      ? 'manifest-files-safe-builtins-v1'
      : null,
    filesystemContentIsolation: artifactBinding
      ? 'manifest-files-only-v1'
      : null,
    environmentIsolation: 'spawn-minimal-lang-c-utc-v1',
    networkIsolation: artifactBinding
      ? 'safe-builtins-no-network-db-v1'
      : null,
    artifactProvenance
  });

  const adapter = Object.freeze({
    id: metadata.id,
    version: metadata.version,
    sourceCommit: metadata.sourceCommit,
    baselineTrustRoot: metadata.baselineTrustRoot
      ? Object.freeze({ ...metadata.baselineTrustRoot })
      : null,
    candidateArtifactProvenance: artifactProvenance,
    capabilities: Object.freeze({
      ...metadata.capabilities,
      hardDeadlineIsolation: true,
      isolation: 'node-subprocess-per-game-v1',
      integrityIsolation: isolationMetadata.permissionProfile,
      stdioIsolation: isolationMetadata.stdioIsolation,
      memoryIsolation: isolationMetadata.memoryIsolation,
      memoryLimitMiB: isolationMetadata.memoryLimitMiB,
      v8OldSpaceMiB: isolationMetadata.v8OldSpaceMiB,
      memoryPreflight: isolationMetadata.memoryPreflight,
      artifactIntegrity: isolationMetadata.artifactIntegrity,
      moduleLoadIsolation: isolationMetadata.moduleLoadIsolation,
      filesystemContentIsolation: isolationMetadata.filesystemContentIsolation,
      environmentIsolation: isolationMetadata.environmentIsolation,
      networkIsolation: isolationMetadata.networkIsolation,
      activeTimeCharging: isolationMetadata.activeTimeCharging,
      initializationCapMs: isolationMetadata.initializationCapMs,
      observerCapMs: isolationMetadata.observerCapMs,
      filesystemRead: true,
      filesystemWrite: false,
      childProcess: false,
      nestedWorker: false,
      nativeAddons: false
    }),
    createSession(context) {
      const sessionArtifactPolicy = artifactBinding
        ? candidateArtifactRuntimePolicy(artifactBinding)
        : null;
      const host = startSubprocess({
        mode: 'session',
        moduleUrl: effectiveModuleUrl,
        loadMode,
        context,
        initializationCapMs: initializationTimeoutMs,
        artifactPolicy: sessionArtifactPolicy,
        memoryProfile
      });
      let closed = false;
      let setupMs = 0;
      let observerDebtMs = 0;
      const sessionReady = host.ready.then((readyRecord) => {
        const actual = readyRecord.descriptor;
        if (actual.id !== metadata.id || actual.version !== metadata.version
          || actual.sourceCommit !== metadata.sourceCommit
          || JSON.stringify(actual.baselineTrustRoot) !== JSON.stringify(metadata.baselineTrustRoot)) {
          void host.terminate();
          throw brandFailure(
            new Error('engine descriptor identity changed between probe and session'),
            { setupMs: readyRecord.elapsedMs }
          );
        }
        setupMs = readyRecord.elapsedMs;
        return actual;
      });
      const session = Object.freeze({
        ready: () => sessionReady,
        async selectAction(request) {
          if (closed) throw new Error('engine subprocess session is closed');
          await sessionReady;
          const budgetMs = positiveFinite(
            request.limits?.wallClockBudgetMs,
            'request.limits.wallClockBudgetMs'
          );
          const chargedSetupMs = setupMs;
          const chargedObserverMs = observerDebtMs;
          setupMs = 0;
          observerDebtMs = 0;
          const remainingMs = budgetMs - chargedSetupMs - chargedObserverMs;
          if (remainingMs <= 0) {
            throw brandFailure(
              timeoutError('engine active-time debt exhausted the move deadline'),
              {
                setupMs: chargedSetupMs,
                observerMs: chargedObserverMs,
                selectMs: 0
              }
            );
          }
          const { clock: _clock, deadlineAtMs: _deadlineAtMs, ...serializable } = request;
          const selectStartedAt = performance.now();
          try {
            const value = await host.rpc(
              'selectAction',
              {
                ...serializable,
                limits: {
                  ...serializable.limits,
                  wallClockBudgetMs: remainingMs
                }
              },
              remainingMs
            );
            const selectMs = performance.now() - selectStartedAt;
            if (selectMs > remainingMs) {
              throw timeoutError('engine decision exceeded its remaining active-time deadline');
            }
            return brandDecision(value, {
              setupMs: chargedSetupMs,
              observerMs: chargedObserverMs,
              selectMs
            });
          } catch (error) {
            throw brandFailure(error, {
              setupMs: chargedSetupMs,
              observerMs: chargedObserverMs,
              selectMs: performance.now() - selectStartedAt
            });
          }
        },
        async observe(transition) {
          if (closed) throw new Error('engine subprocess session is closed');
          await sessionReady;
          const observerStartedAt = performance.now();
          try {
            const value = await host.rpc('observe', transition, observerTimeoutMs);
            const elapsedMs = performance.now() - observerStartedAt;
            observerDebtMs += elapsedMs;
            if (elapsedMs > observerTimeoutMs) {
              const chargedSetupMs = setupMs;
              const chargedObserverMs = observerDebtMs;
              setupMs = 0;
              observerDebtMs = 0;
              throw brandFailure(
                timeoutError(`observe exceeded ${observerTimeoutMs} ms`),
                {
                  setupMs: chargedSetupMs,
                  observerMs: chargedObserverMs,
                  selectMs: 0
                }
              );
            }
            return value;
          } catch (error) {
            const trustedTiming = takeTrustedSubprocessFailureTiming(error);
            if (trustedTiming) {
              throw brandFailure(error, trustedTiming);
            }
            const elapsedMs = performance.now() - observerStartedAt;
            observerDebtMs += elapsedMs;
            const chargedSetupMs = setupMs;
            const chargedObserverMs = observerDebtMs;
            setupMs = 0;
            observerDebtMs = 0;
            throw brandFailure(error, {
              setupMs: chargedSetupMs,
              observerMs: chargedObserverMs,
              selectMs: 0
            });
          }
        },
        async close() {
          if (closed) return;
          closed = true;
          await host.terminate();
        }
      });
      TRUSTED_PENDING_TIMING.set(session, () => {
        const pendingSetupMs = setupMs;
        const pendingObserverMs = observerDebtMs;
        setupMs = 0;
        observerDebtMs = 0;
        if (pendingSetupMs <= 0 && pendingObserverMs <= 0) return null;
        return freezeTiming({
          setupMs: pendingSetupMs,
          observerMs: pendingObserverMs,
          selectMs: 0
        });
      });
      return session;
    }
  });
  SUBPROCESS_ADAPTERS.set(adapter, isolationMetadata);
  if (artifactBinding) AUTHENTICATED_ARTIFACT_ADAPTERS.set(adapter, artifactBinding);
  return adapter;
}

export async function createPinnedHardWorkerAdapter(options = {}) {
  const adapter = await createWorkerEngineAdapter({
    ...options,
    moduleUrl: PINNED_HARD_MODULE_URL,
    loadMode: Object.freeze({
      kind: 'named-factory',
      exportName: 'createPinnedHardBaseline'
    })
  });
  if (adapter.id !== PINNED_HARD_WORKER_ID
    || adapter.version !== PINNED_HARD_WORKER_VERSION
    || adapter.sourceCommit !== PINNED_HARD_SOURCE_COMMIT
    || !trustRootsEqual(adapter.baselineTrustRoot, HARD_BASELINE_TRUST_ROOT)) {
    throw new TypeError('pinned Hard subprocess identity or trust root changed');
  }
  PINNED_HARD_SUBPROCESS_ADAPTERS.add(adapter);
  return adapter;
}
