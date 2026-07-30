/**
 * Hermetic Node adapter for the browser-target normal-duel WebAssembly package.
 *
 * This module is copied verbatim to a candidate release as `adapter.mjs`.
 * Keep its imports relative to that release: the authenticated subprocess
 * permits reads only for files named by the candidate manifest.
 */
import { readFile } from 'node:fs/promises';

import initialize, * as wasm from './normal-duel-wasm.mjs';

export const ENGINE_ID = 'wrongway-normal-duel-wasm-search';

// The parent deadline includes cold WASM JIT, JSON conversion, IPC, and the
// engine's last budget poll. Leave enough headroom for a first move in a fresh
// per-game subprocess instead of claiming that time as search.
const STRENGTH_OVERHEAD_MARGIN_MS = 100;
const REGRESSION_MODE = 'fixed-node-budget-v1';
const STRENGTH_MODE = 'monotonic-deadline-v1';

let initialized = null;
// Kept as a tiny seam so the source-template unit test can fix the same clock
// used by production without altering the one-argument adapter contract.
const currentNow = () => performance.now();

function fail(message) {
  throw new TypeError(`normal-duel-wasm candidate: ${message}`);
}

function safePositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${name} must be a positive safe integer`);
  return value;
}

function safeNonNegativeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 0) fail(`${name} must be a non-negative safe integer`);
  return value;
}

function parsedCanonicalAction(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    fail('WASM search returned an invalid action');
  }
  if (value.kind === 'pawn'
    && Object.keys(value).length === 2
    && value.to && typeof value.to === 'object' && !Array.isArray(value.to)
    && Object.keys(value.to).length === 2) {
    const r = safeNonNegativeInteger(value.to.r, 'search action.to.r');
    const c = safeNonNegativeInteger(value.to.c, 'search action.to.c');
    return Object.freeze({ kind: 'pawn', to: Object.freeze({ r, c }) });
  }
  if (value.kind === 'wall'
    && Object.keys(value).length === 2
    && typeof value.wall === 'string' && value.wall.length > 0) {
    return Object.freeze({ kind: 'wall', wall: value.wall });
  }
  fail('WASM search returned an invalid action');
}

/**
 * Build the one-argument JSON payload accepted by the WASM exports. Strength
 * searches reserve a little wall-clock budget for JSON, IPC, and process work.
 * Canonical candidate calls intentionally omit `options`: the strict boundary
 * then applies the single Rust `SearchOptions::default` definition.
 *
 * The subprocess runtime supplies `limits.deadlineAtMs` from its own monotonic
 * clock immediately before invoking `selectAction`. Use it when present rather
 * than starting a second full budget after the request reaches this adapter.
 */
export function searchInvocationForRequest(request) {
  if (!request || typeof request !== 'object' || !request.config || !request.state) {
    fail('selectAction request must include config and state');
  }
  if (request.mode === REGRESSION_MODE) {
    const nodeBudget = safePositiveInteger(request.limits?.nodeBudget, 'request.limits.nodeBudget');
    return Object.freeze({
      exportName: 'normalDuelSearchNodes',
      payload: JSON.stringify({ config: request.config, state: request.state, nodeBudget })
    });
  }
  if (request.mode === STRENGTH_MODE) {
    const requestedMs = request.limits?.wallClockBudgetMs;
    if (!Number.isFinite(requestedMs) || requestedMs < 1) {
      fail('request.limits.wallClockBudgetMs must be a positive finite number');
    }
    const suppliedDeadlineMs = request.limits?.deadlineAtMs;
    if (suppliedDeadlineMs !== null && suppliedDeadlineMs !== undefined
      && !Number.isFinite(suppliedDeadlineMs)) {
      fail('request.limits.deadlineAtMs must be a finite number or null');
    }
    // Derive the canonical integer budget from the requested allotment, then
    // clamp it to the child deadline. Rounding the remaining fractional time
    // upward avoids losing a whole search millisecond to sub-millisecond
    // adapter setup while preserving almost the full overhead margin.
    const requestedSearchMs = Math.floor(requestedMs) - STRENGTH_OVERHEAD_MARGIN_MS;
    const deadlineSearchMs = Number.isFinite(suppliedDeadlineMs)
      ? Math.ceil(suppliedDeadlineMs - currentNow()) - STRENGTH_OVERHEAD_MARGIN_MS
      : Infinity;
    const timeBudgetMs = Math.max(
      1,
      Math.min(requestedSearchMs, deadlineSearchMs)
    );
    return Object.freeze({
      exportName: 'normalDuelSearchFor',
      payload: JSON.stringify({ config: request.config, state: request.state, timeBudgetMs })
    });
  }
  fail(`unsupported request mode ${String(request.mode)}`);
}

function parsedSearchReport(source) {
  let report;
  try {
    report = JSON.parse(source);
  } catch {
    fail('WASM search returned invalid JSON');
  }
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    fail('WASM search returned an invalid report');
  }
  if (report.actionCode !== null) {
    safeNonNegativeInteger(report.actionCode, 'search actionCode');
    report.action = parsedCanonicalAction(report.action);
  } else if (report.action !== null) {
    fail('terminal search report must have a null action');
  }
  safeNonNegativeInteger(report.nodes, 'search nodes');
  safeNonNegativeInteger(report.completedDepth, 'search completedDepth');
  return report;
}

async function initializeWasm() {
  if (initialized === null) {
    initialized = readFile(new URL('./normal-duel-wasm_bg.wasm', import.meta.url))
      .then((bytes) => initialize({ module_or_path: bytes }));
  }
  await initialized;
}

export async function createEngineAdapter() {
  await initializeWasm();
  const version = wasm.normalDuelVersion();
  if (typeof version !== 'string' || version.length === 0) fail('WASM version is missing');
  return Object.freeze({
    id: ENGINE_ID,
    version,
    capabilities: Object.freeze({ nodeBudget: true, deadline: true }),
    createSession() {
      return Object.freeze({
        async selectAction(request) {
          const invocation = searchInvocationForRequest(request);
          const search = wasm[invocation.exportName];
          if (typeof search !== 'function') {
            fail(`WASM package does not export ${invocation.exportName}`);
          }
          const report = parsedSearchReport(search(invocation.payload));
          if (report.actionCode === null) return null;
          return Object.freeze({
            action: report.action,
            stats: Object.freeze({ nodes: report.nodes, depth: report.completedDepth })
          });
        },
        async observe() {},
        async close() {}
      });
    }
  });
}
