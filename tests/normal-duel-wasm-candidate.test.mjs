import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  REGRESSION_MODE,
  STRENGTH_MODE
} from '../scripts/evaluation/normal-duel-strength.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const config = Object.freeze({ rows: 9, columns: 9 });
const templateSource = readFileSync(
  resolve(ROOT, 'scripts/evaluation/normal-duel-wasm-candidate-adapter.mjs'),
  'utf8'
).replace(
  "import initialize, * as wasm from './normal-duel-wasm.mjs';",
  'const initialize = async () => {}; const wasm = Object.create(null);'
).replace(
  'const currentNow = () => performance.now();',
  'const currentNow = () => 10_000.25;'
);
const adapterTemplate = await import(
  `data:text/javascript;base64,${Buffer.from(templateSource).toString('base64')}`
);

test('WASM candidate routes regression and strength requests to their bounded exports', () => {
  const base = { config, state: { position: {} } };
  const regression = adapterTemplate.searchInvocationForRequest({
    ...base,
    mode: REGRESSION_MODE,
    limits: { nodeBudget: 7 }
  });
  assert.equal(regression.exportName, 'normalDuelSearchNodes');
  assert.deepEqual(JSON.parse(regression.payload), {
    config,
    state: { position: {} },
    nodeBudget: 7
  });

  const strength = adapterTemplate.searchInvocationForRequest({
    ...base,
    mode: STRENGTH_MODE,
    limits: { wallClockBudgetMs: 125.9 }
  });
  assert.equal(strength.exportName, 'normalDuelSearchFor');
  assert.deepEqual(JSON.parse(strength.payload), {
    config,
    state: { position: {} },
    timeBudgetMs: 25
  });

  const canonicalStrength = adapterTemplate.searchInvocationForRequest({
    ...base,
    mode: STRENGTH_MODE,
    limits: { wallClockBudgetMs: 900, deadlineAtMs: 10_900 }
  });
  assert.deepEqual(JSON.parse(canonicalStrength.payload), {
    config,
    state: { position: {} },
    timeBudgetMs: 800
  });

  const oneMillisecondLate = adapterTemplate.searchInvocationForRequest({
    ...base,
    mode: STRENGTH_MODE,
    limits: { wallClockBudgetMs: 900, deadlineAtMs: 10_899 }
  });
  assert.deepEqual(JSON.parse(oneMillisecondLate.payload), {
    config,
    state: { position: {} },
    timeBudgetMs: 799
  });

  const directStrength = adapterTemplate.searchInvocationForRequest({
    ...base,
    mode: STRENGTH_MODE,
    limits: { wallClockBudgetMs: 900 }
  });
  assert.deepEqual(JSON.parse(directStrength.payload), {
    config,
    state: { position: {} },
    timeBudgetMs: 800
  });
});

test('WASM candidate caps strength search at the child monotonic deadline', () => {
  const deadlineAtMs = 10_060.25;
  const invocation = adapterTemplate.searchInvocationForRequest({
    config,
    state: { position: {} },
    mode: STRENGTH_MODE,
    limits: {
      wallClockBudgetMs: 1_000,
      deadlineAtMs
    }
  });
  const { timeBudgetMs } = JSON.parse(invocation.payload);
  assert.equal(timeBudgetMs, 1);

  assert.throws(() => adapterTemplate.searchInvocationForRequest({
    config,
    state: { position: {} },
    mode: STRENGTH_MODE,
    limits: { wallClockBudgetMs: 100, deadlineAtMs: Number.NaN }
  }), /deadlineAtMs must be a finite number or null/);
});
