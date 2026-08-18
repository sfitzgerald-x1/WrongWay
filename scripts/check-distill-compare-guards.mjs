#!/usr/bin/env node
/**
 * Self-checks for the R0 screen guards in `distill-compare.mjs`.
 *
 * Run: node scripts/check-distill-compare-guards.mjs
 *
 * The guards run BEFORE `loadWasm()`, so each case drives the real script with
 * `WW_DISTILL_WASM_DIR` pointed at a directory that does not exist. That makes the
 * outcome unambiguous without playing a single game:
 *
 *   exit 3  -- the anchor gate refused
 *   exit 2  -- an input was rejected (bad env, unreadable ledger)
 *   exit 1  -- THE GATE LET IT THROUGH and the script then failed to load the engine
 *
 * Testing the script rather than an exported helper is deliberate: the defect these
 * cases exist for was a gate that was present and passable, and only the real argv ->
 * ledger -> gate path can show that.
 *
 * Every case below is a bug that a review found in the first cut. Verified against the
 * pre-fix file: it fails 7 of them.
 */

import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT = path.join(HERE, 'distill-compare.mjs');
const dir = mkdtempSync(path.join(tmpdir(), 'ww-anchor-'));
const FAILURES = [];

const HEALTHY = { base: 'P', scorePct: 51.2, elo: 8, games: 200, decided: 180, ts: '2026-01-01' };

/** Run the script with `ledger` on disk. `raw` writes the file verbatim. */
function run({ ledger, raw, args = [], env = {} }) {
  const file = path.join(dir, `ledger-${Math.random().toString(36).slice(2)}.json`);
  writeFileSync(file, raw !== undefined ? raw : JSON.stringify(ledger ?? {}, null, 2));
  const r = spawnSync(process.execPath, [
    SCRIPT, '--a', 's1', '--b', 's2', '--base', 'P', '--ledger', file, ...args,
  ], {
    encoding: 'utf8',
    env: { ...process.env, WW_DISTILL_WASM_DIR: path.join(dir, 'no-such-wasm-dir'), ...env },
  });
  return { code: r.status, out: `${r.stdout || ''}${r.stderr || ''}` };
}

function refuses(label, opts) {
  const { code, out } = run(opts);
  if (code === 2 || code === 3) return;
  FAILURES.push(`${label}: exit ${code}, want a refusal (2 or 3)`
    + `${code === 1 ? ' -- THE GATE PASSED IT' : ''}\n      ${out.trim().split('\n')[0] || ''}`);
}

function passesGate(label, opts) {
  const { code, out } = run(opts);
  if (code === 1) return;   // reached loadWasm, which is as far as this harness goes
  FAILURES.push(`${label}: exit ${code}, want the gate to pass (1)\n      ${out.trim().split('\n')[0] || ''}`);
}

// --- the gate does its job ----------------------------------------------------

refuses('empty ledger: student-vs-student with no anchor', { ledger: {} });

passesGate('both arms properly anchored against the base', {
  ledger: { s1: { vsBase: HEALTHY }, s2: { vsBase: { ...HEALTHY, scorePct: 48.9, elo: -8 } } },
});

passesGate('legitimate anchor run (one arm IS the base)', {
  ledger: {}, args: ['--b', 'P'],
});

// --- HIGH-1: the anchor must be against THIS base -----------------------------

refuses('anchors recorded against a DIFFERENT parent', {
  ledger: {
    s1: { vsBase: { ...HEALTHY, base: 'OLD-PARENT' } },
    s2: { vsBase: { ...HEALTHY, base: 'OLD-PARENT' } },
  },
});

// --- HIGH-2: a key is not a measurement ---------------------------------------

refuses('vsBase is `true`', { ledger: { s1: { vsBase: true }, s2: { vsBase: true } } });
refuses('vsBase is `{}`', { ledger: { s1: { vsBase: {} }, s2: { vsBase: {} } } });
refuses('vsBase has no game count', {
  ledger: { s1: { vsBase: { ...HEALTHY, games: undefined } }, s2: { vsBase: HEALTHY } },
});

// --- HIGH-3: a student cannot be a parent -------------------------------------

refuses('--base names an arm that is itself a recorded student', {
  ledger: { s2: { vsBase: { ...HEALTHY, base: 'P' } } },
  args: ['--base', 's2'],
});

// --- MEDIUM-1/2: an anchor showing collapse is not permission ------------------

refuses('one arm is 530 Elo below its parent', {
  ledger: {
    s1: { vsBase: { ...HEALTHY, scorePct: 4.5, elo: -530 } },
    s2: { vsBase: HEALTHY },
  },
});

refuses('one arm was annihilated 0-0-200 (saturated, elo null)', {
  ledger: {
    s1: { vsBase: { ...HEALTHY, scorePct: 0, elo: null } },
    s2: { vsBase: HEALTHY },
  },
});

refuses('one arm is beyond the breakage threshold in the FAVOURABLE direction', {
  ledger: {
    s1: { vsBase: { ...HEALTHY, scorePct: 96.0, elo: 552 } },
    s2: { vsBase: HEALTHY },
  },
});

// --- MEDIUM-3: the threshold itself must be a number ---------------------------

refuses('WW_BREAKAGE_ELO is a plausible typo ("2x145" -> NaN)', {
  ledger: { s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } },
  env: { WW_BREAKAGE_ELO: '2x145' },
});

// Empty is UNSET, not zero. `export WW_BREAKAGE_ELO=` and an unset variable expand
// identically in shell, and the pre-fix `Number('' ?? d)` read that as 0 -- a threshold
// of zero fires SUSPECTED BREAKAGE on every run including a dead-even 50%, which is the
// same as having no tripwire, just noisier. Falling back to the default is the safe
// reading, and it is what the corpus-scale gate does with an empty env too.
passesGate('WW_BREAKAGE_ELO is empty -> falls back to the default, not 0', {
  ledger: { s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } },
  env: { WW_BREAKAGE_ELO: '' },
});

refuses('WW_POOL_SPREAD_HEALTHY is negative', {
  ledger: { s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } },
  env: { WW_POOL_SPREAD_HEALTHY: '-1' },
});

// --- a malformed ledger refuses rather than crashing ---------------------------

refuses('ledger file is literal null', { raw: 'null' });
refuses('ledger file is not JSON', { raw: '{ not json' });
refuses('ledger entry is null', { raw: JSON.stringify({ s1: null, s2: null }) });

// -------------------------------------------------------------------------------

rmSync(dir, { recursive: true, force: true });
if (FAILURES.length) {
  console.error(`FAIL (${FAILURES.length})`);
  for (const f of FAILURES) console.error(`  ${f}`);
  process.exit(1);
}
console.log('all distill-compare guard checks passed: an unanchored, stale-anchored,'
  + ' self-anchored or collapsed-arm screen is refused, and the tripwire threshold'
  + ' cannot be turned into NaN');
