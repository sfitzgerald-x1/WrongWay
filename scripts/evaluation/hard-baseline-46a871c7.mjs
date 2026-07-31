/**
 * Exact, evaluation-only adapter for the Hard product behavior at 46a871c7.
 *
 * The unmodified source snapshots execute in a capability-minimal VM. The only
 * browser global deliberately replaced is `Date.now`: it delegates to the
 * request's monotonic clock, preserving the product's relative 700 ms cutoff
 * without depending on wall-clock time. This module must never be imported by
 * the product client.
 */
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SNAPSHOT_DIR = resolve(HERE, '../../tests/fixtures/baselines/hard-46a871c7');
const REQUIRED_SOURCES = Object.freeze(['game-logic.js', 'ai.js']);
const INVOKE_HARD = new vm.Script('aiDuelHard(...__hardArgs)', {
  filename: 'hard-product-46a871c7/invoke-aiDuelHard.js'
});
const REGRESSION_MODE = 'fixed-node-budget-v1';
const DEFAULT_REGRESSION_REAL_SAFETY_CEILING_MS = 10_000;
const MAX_REGRESSION_REAL_SAFETY_CEILING_MS = 60_000;
const CLOCK_PROFILE_EXCEEDED_ERRORS = new WeakSet();

export const HARD_BASELINE_ID = 'hard-product-46a871c7';
export const HARD_BASELINE_TRUST_ROOT = Object.freeze({
  baselineId: HARD_BASELINE_ID,
  baselineVersion: 'wrongway-hard-product-baseline-v1',
  sourceCommit: '46a871c7b061a33922bdb9c6d78355e2e9b6b607',
  manifestSha256: '55591ccaeb32cdfff4c30f23723b5e98d13ef304b5fd6eaa1d7ca761148f0df6',
  gameLogicSha256: 'db8dc06882f015d78460882e0ab629f660d0a4c1c7600d986a0e0a979139e75a',
  aiSha256: '07ec05bb99602e36a879095ec159728bbf0ba520338d0a20a904888f2b57a777',
  orchestrationSha256: '8dd628e49934e3bb401833d6b69a8854ed6f1dffaee945fbc399bc2a4d2e72bc',
  originalIndexSha256: '87468038df741428fe25f8532f8871459f181554a6dbadf4e151b6bf44621f4c'
});

export class HardBaselineError extends Error {
  constructor(code, message, options) {
    super(message, options);
    this.name = 'HardBaselineError';
    this.code = code;
  }
}

export function isPinnedHardClockProfileExceededError(error) {
  return error instanceof Error && CLOCK_PROFILE_EXCEEDED_ERRORS.has(error);
}

function fail(code, message, options) {
  const error = new HardBaselineError(code, message, options);
  if (code === 'clock_profile_exceeded') CLOCK_PROFILE_EXCEEDED_ERRORS.add(error);
  throw error;
}

function sha256(source) {
  return createHash('sha256').update(source).digest('hex');
}

function readAndVerifySnapshots(snapshotDirectory = SNAPSHOT_DIR) {
  const manifestSource = readFileSync(resolve(snapshotDirectory, 'manifest.json'));
  if (sha256(manifestSource) !== HARD_BASELINE_TRUST_ROOT.manifestSha256) {
    fail('baseline_integrity', 'baseline manifest no longer matches the hard-coded trust root');
  }
  const manifest = JSON.parse(manifestSource.toString('utf8'));
  if (manifest.baselineId !== HARD_BASELINE_ID
    || manifest.baselineFormat !== HARD_BASELINE_TRUST_ROOT.baselineVersion
    || manifest.sourceCommit !== HARD_BASELINE_TRUST_ROOT.sourceCommit) {
    fail('invalid_baseline_manifest', 'baseline identity or source commit changed');
  }
  const expectedRecords = Object.freeze({
    'game-logic.js': Object.freeze({
      path: 'js/game-logic.js',
      sha256: HARD_BASELINE_TRUST_ROOT.gameLogicSha256
    }),
    'ai.js': Object.freeze({
      path: 'js/ai.js',
      sha256: HARD_BASELINE_TRUST_ROOT.aiSha256
    }),
    'index-hard-orchestration.txt': Object.freeze({
      path: 'index.html',
      sha256: HARD_BASELINE_TRUST_ROOT.originalIndexSha256,
      snapshotSha256: HARD_BASELINE_TRUST_ROOT.orchestrationSha256
    })
  });
  if (!Array.isArray(manifest.sources) || manifest.sources.length !== 3) {
    fail('invalid_baseline_manifest', 'baseline source records changed');
  }
  const snapshots = new Map();
  for (const record of manifest.sources) {
    const expectedRecord = expectedRecords[record.snapshot];
    if (!expectedRecord || record.path !== expectedRecord.path
      || record.sha256 !== expectedRecord.sha256
      || (expectedRecord.snapshotSha256
        ? record.snapshotSha256 !== expectedRecord.snapshotSha256
        : Object.hasOwn(record, 'snapshotSha256'))) {
      fail('invalid_baseline_manifest', `trust-root record changed for ${String(record.snapshot)}`);
    }
    const source = readFileSync(resolve(snapshotDirectory, record.snapshot));
    const expectedSnapshotSha = expectedRecord.snapshotSha256 ?? expectedRecord.sha256;
    if (sha256(source) !== expectedSnapshotSha) {
      fail('baseline_integrity', `${record.snapshot} no longer matches its pinned SHA-256`);
    }
    if (REQUIRED_SOURCES.includes(record.snapshot)) {
      snapshots.set(record.snapshot, source.toString('utf8'));
    }
  }
  for (const filename of REQUIRED_SOURCES) {
    if (!snapshots.has(filename)) fail('invalid_baseline_manifest', `missing ${filename} provenance`);
  }
  return Object.freeze({ manifest: Object.freeze(manifest), snapshots });
}

export function verifyPinnedHardBaselineDirectory(snapshotDirectory) {
  if (typeof snapshotDirectory !== 'string' || snapshotDirectory.length === 0) {
    fail('invalid_baseline_directory', 'snapshotDirectory must be a non-empty path');
  }
  const verified = readAndVerifySnapshots(resolve(snapshotDirectory));
  return Object.freeze({
    baselineId: verified.manifest.baselineId,
    sourceCommit: verified.manifest.sourceCommit,
    trustRoot: HARD_BASELINE_TRUST_ROOT
  });
}

function finiteMonotonic(clock) {
  if (!clock || typeof clock.now !== 'function') {
    fail('invalid_clock', 'clock must provide now()');
  }
  const value = clock.now();
  if (!Number.isFinite(value) || value < 0) fail('invalid_clock', 'clock.now() must be finite and non-negative');
  return value;
}

function cloneCoord(coord) {
  return { r: coord.r, c: coord.c };
}

export function rotatePinnedCoordinate180(coord, config) {
  return {
    r: config.rows - 1 - coord.r,
    c: config.columns - 1 - coord.c
  };
}

export function rotatePinnedWall180(wall, config) {
  const match = /^([HV])-(\d+)-(\d+)$/.exec(wall);
  if (!match) fail('invalid_wall', `cannot rotate malformed wall ${String(wall)}`);
  const [, orientation, rowText, columnText] = match;
  const row = Number(rowText);
  const column = Number(columnText);
  return `${orientation}-${config.rows - 2 - row}-${config.columns - 2 - column}`;
}

export function rotatePinnedAction180(action, config) {
  return action.kind === 'pawn'
    ? { kind: 'pawn', to: rotatePinnedCoordinate180(action.to, config) }
    : { kind: 'wall', wall: rotatePinnedWall180(action.wall, config) };
}

function other(player) {
  return player === 'A' ? 'B' : 'A';
}

function exactWallDifference(before, afterLike) {
  if (!afterLike || typeof afterLike[Symbol.iterator] !== 'function') {
    fail('invalid_baseline_decision', 'Hard barricade decision did not return an iterable wall set');
  }
  const after = [...afterLike];
  const additions = after.filter((wall) => !before.has(wall));
  const removals = [...before].filter((wall) => !after.includes(wall));
  if (additions.length !== 1 || removals.length !== 0 || after.length !== before.size + 1) {
    fail('invalid_baseline_decision', 'Hard barricade decision must add exactly one wall');
  }
  return additions[0];
}

function productHistoryEntry(player, action) {
  return action.kind === 'pawn'
    ? { type: 'move', p: player, r: action.to.r, c: action.to.c }
    : { type: 'barricade', p: player, k: action.wall };
}

function decisionFromOracleFrame(decision, config, rotated) {
  if (!rotated || decision === null || decision === undefined) return decision;
  if (decision.type === 'move') {
    return { type: 'move', pos: rotatePinnedCoordinate180(decision.pos, config) };
  }
  if (decision.type === 'barricade') {
    return {
      type: 'barricade',
      walls: new Set([...decision.walls].map((wall) => rotatePinnedWall180(wall, config)))
    };
  }
  return decision;
}

export function buildPinnedRecentAi(current, history, player) {
  const recent = [cloneCoord(current)];
  for (let index = history.length - 1; index >= 0 && recent.length < 6; index -= 1) {
    const move = history[index];
    if (move.type === 'move' && move.p === player) recent.push({ r: move.r, c: move.c });
  }
  return recent;
}

/**
 * Product-exact anti-stall wrapper from index.html at the pinned commit.
 * `progress` is intentionally mutable session state, matching aiProgRef.current.
 */
export function applyPinnedAntiStall({ runtime, decision, state, side, progress, chaosItem = null }) {
  if (!decision || decision.type !== 'move') return Object.freeze({ decision, replaced: false });
  const opponent = other(side);
  const position = state.position;
  const aiPosition = position.pawns[side];
  const humanPosition = position.pawns[opponent];
  const walls = new Set(position.walls);
  const aiGoal = runtime.goalRows[side];
  const beforePath = runtime.bfsPath(aiPosition, walls, aiGoal);
  const beforeDistance = beforePath ? beforePath.length - 1 : Infinity;
  const afterPath = runtime.bfsPath(decision.pos, walls, aiGoal);
  const afterDistance = afterPath ? afterPath.length - 1 : Infinity;
  const isItem = Boolean(chaosItem
    && decision.pos.r === chaosItem.r
    && decision.pos.c === chaosItem.c);
  if (afterDistance < beforeDistance || isItem) progress.stall = 0;
  else progress.stall += 1;
  if (progress.stall >= 3 && afterDistance >= beforeDistance && !isItem && beforePath) {
    const forced = runtime.moveTowardGoal(aiPosition, humanPosition, walls, beforePath);
    if (forced) {
      progress.stall = 0;
      return Object.freeze({ decision: { type: 'move', pos: forced }, replaced: true });
    }
  }
  return Object.freeze({ decision, replaced: false });
}

function createRuntime(snapshots) {
  let activeClock = { now: () => performance.now() };
  const dateCapability = Object.freeze({
    now() {
      return finiteMonotonic(activeClock);
    }
  });
  const sandbox = Object.create(null);
  Object.defineProperties(sandbox, {
    ROWS: { value: 9, writable: true, enumerable: true },
    COLS: { value: 9, writable: true, enumerable: true },
    CELL: { value: 1, writable: false, enumerable: true },
    WW: { value: 1, writable: false, enumerable: true },
    CUR_MAP: { value: 'duel', writable: false, enumerable: true },
    Date: { value: dateCapability, writable: false, enumerable: true },
    __hardArgs: { value: null, writable: true, enumerable: false }
  });
  const context = vm.createContext(sandbox, {
    name: HARD_BASELINE_ID,
    codeGeneration: { strings: false, wasm: false }
  });
  for (const filename of REQUIRED_SOURCES) {
    new vm.Script(snapshots.get(filename), {
      filename: `hard-product-46a871c7/${filename}`
    }).runInContext(context, { timeout: 1_000 });
  }
  for (const name of ['aiDuelHard', 'bfsPath', 'moveTowardGoal']) {
    if (typeof sandbox[name] !== 'function') fail('baseline_load', `snapshot did not define ${name}`);
  }

  return Object.freeze({
    setBoard(config) {
      sandbox.ROWS = config.rows;
      sandbox.COLS = config.columns;
    },
    setClock(clock) {
      activeClock = clock;
    },
    decide(args, timeoutMs, timeoutCode = 'deadline_exceeded') {
      sandbox.__hardArgs = args;
      try {
        return INVOKE_HARD.runInContext(context, { timeout: timeoutMs });
      } catch (error) {
        if (error?.code === 'ERR_SCRIPT_EXECUTION_TIMEOUT') {
          fail(
            timeoutCode,
            timeoutCode === 'clock_profile_exceeded'
              ? `pinned Hard exceeded the ${timeoutMs} ms regression clock-profile safety ceiling`
              : 'pinned Hard decision exceeded the outer deadline',
            { cause: error }
          );
        }
        throw error;
      } finally {
        sandbox.__hardArgs = null;
      }
    },
    bfsPath(position, walls, goalRow) {
      return sandbox.bfsPath(position, walls, goalRow);
    },
    moveTowardGoal(aiPosition, humanPosition, walls, path) {
      return sandbox.moveTowardGoal(aiPosition, humanPosition, walls, path);
    }
  });
}

function assertSessionInput({ side, config, openingHistory }) {
  if (side !== 'A' && side !== 'B') fail('invalid_session', 'side must be A or B');
  if (!config || config.ruleset !== 'normal-duel-v1'
    || !((config.rows === 9 && config.columns === 9) || (config.rows === 7 && config.columns === 7))) {
    fail('unsupported_config', 'pinned Hard supports normal 7x7/9x9 Duel only');
  }
  if (!Array.isArray(openingHistory)) fail('invalid_session', 'openingHistory must be an array');
}

/**
 * Build the baseline descriptor consumed by the strength harness.
 *
 * A descriptor creates fresh state per game. This is where the product's
 * `aiProgRef.current={best:Infinity,stall:0}` reset is pinned.
 */
export function createPinnedHardBaseline() {
  const { manifest, snapshots } = readAndVerifySnapshots();
  return Object.freeze({
    id: HARD_BASELINE_ID,
    version: manifest.baselineFormat,
    sourceCommit: manifest.sourceCommit,
    baselineTrustRoot: HARD_BASELINE_TRUST_ROOT,
    capabilities: Object.freeze({
      nodeBudget: false,
      deadline: true,
      deterministicClock: true
    }),
    createSession({
      side,
      config,
      mode = null,
      clockProfile = null,
      openingHistory = []
    }) {
      assertSessionInput({ side, config, openingHistory });
      const configuredSafetyCeiling = clockProfile?.realSafetyCeilingMilliseconds
        ?? DEFAULT_REGRESSION_REAL_SAFETY_CEILING_MS;
      if (mode === REGRESSION_MODE
        && (!Number.isSafeInteger(configuredSafetyCeiling)
          || configuredSafetyCeiling < 1
          || configuredSafetyCeiling > MAX_REGRESSION_REAL_SAFETY_CEILING_MS)) {
        fail(
          'invalid_clock_profile',
          `regression real safety ceiling must be an integer from 1 to ${MAX_REGRESSION_REAL_SAFETY_CEILING_MS} ms`
        );
      }
      const runtime = createRuntime(snapshots);
      runtime.setBoard(config);
      // Runtime is frozen, so keep goal rows beside it for the orchestration
      // wrapper rather than mutating the VM capability surface.
      const productRuntime = Object.freeze({
        bfsPath: runtime.bfsPath,
        moveTowardGoal: runtime.moveTowardGoal,
        goalRows: Object.freeze({ A: config.goalRows.A, B: config.goalRows.B })
      });
      const history = openingHistory.map(({ player, action }) => productHistoryEntry(player, action));
      const progress = { best: Infinity, stall: 0 };
      let closed = false;
      let lastClockValue = -Infinity;

      return Object.freeze({
        async selectAction({ state, clock = { now: () => performance.now() }, deadlineAtMs = Infinity }) {
          if (closed) fail('session_closed', 'baseline session is closed');
          if (state.position.turn !== side) fail('wrong_turn', `baseline ${side} asked to move on ${state.position.turn}'s turn`);
          const monotonicClock = Object.freeze({
            now() {
              const value = finiteMonotonic(clock);
              if (value < lastClockValue) {
                fail('invalid_clock', `clock moved backwards from ${lastClockValue} to ${value}`);
              }
              lastClockValue = value;
              return value;
            }
          });
          runtime.setClock(monotonicClock);
          const startedAt = finiteMonotonic(monotonicClock);
          const regressionClock = mode === REGRESSION_MODE;
          const timeoutMs = regressionClock
            ? configuredSafetyCeiling
            : Number.isFinite(deadlineAtMs)
              ? Math.max(1, Math.ceil(deadlineAtMs - startedAt))
              : 1_000;
          const opponent = other(side);
          const position = state.position;
          const walls = new Set(position.walls);
          const useRotatedBOracle = side === 'A';
          const toOracleCoordinate = (coord) => useRotatedBOracle
            ? rotatePinnedCoordinate180(coord, config)
            : cloneCoord(coord);
          const oracleWalls = useRotatedBOracle
            ? new Set([...walls].map((wall) => rotatePinnedWall180(wall, config)))
            : walls;
          const recentAi = buildPinnedRecentAi(position.pawns[side], history, side)
            .map(toOracleCoordinate);
          const raw = runtime.decide([
            toOracleCoordinate(position.pawns[side]),
            toOracleCoordinate(position.pawns[opponent]),
            oracleWalls,
            position.stock[side],
            position.stock[opponent],
            recentAi,
            null,
            config.goalRows.B,
            config.goalRows.A
          ], timeoutMs, regressionClock ? 'clock_profile_exceeded' : 'deadline_exceeded');
          const oracleState = useRotatedBOracle
            ? {
              position: {
                pawns: {
                  A: toOracleCoordinate(position.pawns[opponent]),
                  B: toOracleCoordinate(position.pawns[side])
                },
                walls: [...oracleWalls]
              }
            }
            : state;
          const antiStall = applyPinnedAntiStall({
            runtime: productRuntime,
            decision: raw,
            state: oracleState,
            side: 'B',
            progress
          });
          const decision = decisionFromOracleFrame(
            antiStall.decision,
            config,
            useRotatedBOracle
          );
          let action = null;
          if (decision?.type === 'move') action = { kind: 'pawn', to: cloneCoord(decision.pos) };
          else if (decision?.type === 'barricade') {
            action = { kind: 'wall', wall: exactWallDifference(walls, decision.walls) };
          } else if (decision !== null && decision !== undefined) {
            fail('invalid_baseline_decision', `unknown Hard decision type ${String(decision.type)}`);
          }
          const endedAt = finiteMonotonic(monotonicClock);
          return Object.freeze({
            action,
            stats: Object.freeze({
              elapsedMs: Math.max(0, endedAt - startedAt),
              nodes: null,
              depth: 3,
              antiStallReplaced: antiStall.replaced,
              recentAiCount: recentAi.length
            })
          });
        },
        observe({ player, action }) {
          if (closed) fail('session_closed', 'baseline session is closed');
          history.push(productHistoryEntry(player, action));
        },
        inspect() {
          return Object.freeze({
            progress: Object.freeze({ best: progress.best, stall: progress.stall }),
            history: Object.freeze(history.map((entry) => Object.freeze({ ...entry })))
          });
        },
        close() {
          closed = true;
        }
      });
    }
  });
}
