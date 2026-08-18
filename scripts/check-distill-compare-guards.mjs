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
 *   exit 2  -- an input was rejected (bad env, unreadable ledger, reserved id)
 *   exit 1 AND a module-not-found marker -- the gate LET IT THROUGH and the script
 *           then failed to load the engine
 *
 * The marker is checked, not just the exit code: injecting a `throw` into
 * `requireAnchors` also exits 1, and without the marker a crashing script would count
 * as "the gate passed".
 *
 * Testing the script rather than an exported helper is deliberate: the defect these
 * cases exist for was a gate that was present and passable, and only the real argv ->
 * ledger -> gate path can show that.
 *
 * Verified against the pre-fix file: it fails 18 of these.
 */

import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT = path.join(HERE, 'distill-compare.mjs');
const dir = mkdtempSync(path.join(tmpdir(), 'ww-anchor-'));
const FAILURES = [];

const HEALTHY = { base: 'P', scorePct: 51.2, elo: 8, games: 200, decided: 180, ts: '2026-01-01' };
/** A ledger with 'P' already declared as a parent, which is the normal steady state. */
const declared = (entries = {}) => ({ __parents__: { P: { ts: '2026-01-01' } }, ...entries });

function run({ ledger, raw, args = [], env = {} }) {
  const file = path.join(dir, `ledger-${Math.random().toString(36).slice(2)}.json`);
  writeFileSync(file, raw !== undefined ? raw : JSON.stringify(ledger ?? {}, null, 2));
  const r = spawnSync(process.execPath, [
    SCRIPT, '--a', 's1', '--b', 's2', '--base', 'P', '--ledger', file, ...args,
  ], {
    encoding: 'utf8',
    env: { ...process.env, WW_DISTILL_WASM_DIR: path.join(dir, 'no-such-wasm-dir'), ...env },
  });
  return { code: r.status, out: `${r.stdout || ''}${r.stderr || ''}`, file };
}

function refuses(label, opts) {
  const { code, out } = run(opts);
  if (code === 2 || code === 3) return;
  FAILURES.push(`${label}: exit ${code}, want a refusal (2 or 3)`
    + `${code === 1 ? ' -- THE GATE PASSED IT' : ''}\n      ${out.trim().split('\n').pop() || ''}`);
}

/** Reached loadWasm, which is as far as this harness goes -- and reached it by passing
 *  the gate rather than by crashing inside it. */
function passesGate(label, opts) {
  const { code, out } = run(opts);
  const reachedWasm = /ERR_MODULE_NOT_FOUND|Cannot find module|no-such-wasm-dir/.test(out);
  if (code === 1 && reachedWasm) return;
  FAILURES.push(`${label}: exit ${code}, reachedWasm=${reachedWasm}, want the gate to pass`
    + `\n      ${out.trim().split('\n').pop() || ''}`);
}

// --- the gate does its job ----------------------------------------------------

refuses('empty ledger: student-vs-student with no anchor', { ledger: declared() });

passesGate('both arms properly anchored against a declared base', {
  ledger: declared({ s1: { vsBase: HEALTHY }, s2: { vsBase: { ...HEALTHY, scorePct: 48.9, elo: -8 } } }),
});

passesGate('legitimate anchor run (one arm IS the base)', {
  ledger: declared(), args: ['--b', 'P'],
});

refuses('--a and --b are the same checkpoint', {
  ledger: declared({ s1: { vsBase: HEALTHY } }), args: ['--b', 's1'],
});

// --- parenthood is declared, not inferred -------------------------------------

refuses('base has not been declared as a parent', {
  ledger: { s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } },
});

refuses('--base names a sibling on an EMPTY ledger (the residual hole)', {
  ledger: {}, args: ['--base', 's2'],
});

// A declared parent that is ITSELF a student is the normal lineage case: every net is
// a student of the previous one and the parent of the next. Inferring parenthood from
// ledger membership refused this, which would have broken the second generation of
// every experiment.
passesGate('declared parent that is itself a recorded student (lineage)', {
  ledger: {
    __parents__: { 'g2-030': { ts: '2026-01-01' } },
    'g2-030': { vsBase: { ...HEALTHY, base: 'g2-020' } },
    s1: { vsBase: { ...HEALTHY, base: 'g2-030', elo: 21 } },
    s2: { vsBase: { ...HEALTHY, base: 'g2-030', elo: 10 } },
  },
  args: ['--base', 'g2-030'],
});

// ANCHOR-RUN shape deliberately. In a student-vs-student shape this refuses for a
// different reason (no anchor) even with the collision check deleted, so it would pass
// for the wrong reason and pin nothing. The anchor-run shape is where the check is
// load-bearing: without it the run records `ledger.__parents__.vsBase`, merging a
// student record into the declarations map.
refuses('an id colliding with the reserved ledger key', {
  ledger: declared(), args: ['--a', '__parents__', '--b', 'P'],
});

// --- prototype chain is not a declaration -------------------------------------
//
// `parents[base]` walks the prototype chain, so on an EMPTY ledger every
// Object.prototype member read as declared and passed the gate.

// ANCHOR-RUN shape (`--b X --base X`) for every one of these. In a student-vs-student
// shape the missing anchor refuses first, so the case passes whether or not the
// declaration gate works -- it would pin nothing. Here the declaration gate is the only
// thing that can refuse, so reverting it to truthiness makes these fail.
for (const proto of ['constructor', 'toString', 'valueOf', '__proto__', 'hasOwnProperty']) {
  refuses(`--base ${proto} (Object.prototype member) on an empty ledger`, {
    ledger: {}, args: ['--b', proto, '--base', proto],
  });
}

refuses('__parents__ is a string, and an index into it looks declared', {
  raw: JSON.stringify({ __parents__: 'P' }), args: ['--b', '0', '--base', '0'],
});
refuses('__parents__ is an array', {
  raw: JSON.stringify({ __parents__: [] }), args: ['--b', 'P'],
});
refuses('__parents__ is null', {
  raw: JSON.stringify({ __parents__: null }), args: ['--b', 'P'],
});
refuses('a declared value of null is not a declaration', {
  raw: JSON.stringify({ __parents__: { P: null } }), args: ['--b', 'P'],
});

// A declaration must actually land. `parents['__proto__'] = {...}` triggers the
// inherited setter, reassigns the prototype instead of creating an own property, and
// JSON.stringify drops it -- so the run reported success over a ledger that recorded
// nothing. Fail-closed, but a success message for something that did not happen.
{
  const { code, file } = run({ ledger: {}, args: ['--base', '__proto__', '--declare-parent'] });
  const raw = readFileSync(file, 'utf8');
  const after = JSON.parse(raw);
  if (code !== 0) {
    FAILURES.push(`--declare-parent __proto__: exit ${code}, want 0`);
  } else if (!Object.hasOwn(after.__parents__ ?? {}, '__proto__')) {
    FAILURES.push(`--declare-parent __proto__ reported success but recorded nothing: ${raw.trim()}`);
  }
}

// The `__parents__` type check earns its place HERE rather than on the read path, where
// the value check already covers every reachable shape. Modules are strict mode, so
// assigning a property to a string primitive throws -- without the type check,
// --declare-parent against a corrupt ledger dies with a raw TypeError instead of
// recovering or refusing.
{
  const { code, out, file } = run({
    raw: JSON.stringify({ __parents__: 'P' }), args: ['--declare-parent'],
  });
  if (code !== 0 && code !== 2) {
    FAILURES.push(`--declare-parent over a string __parents__: exit ${code}, want 0 or a`
      + ` clean refusal (2)\n      ${out.trim().split('\n').pop() || ''}`);
  }
  if (/TypeError|Cannot create property/.test(out)) {
    FAILURES.push('--declare-parent over a string __parents__ threw a raw TypeError');
  }
  if (code === 0) {
    const after = JSON.parse(readFileSync(file, 'utf8'));
    if (!after.__parents__?.P?.ts) {
      FAILURES.push('--declare-parent over a string __parents__ did not record a declaration');
    }
  }
}

// --- the anchor must be against THIS base, and be a real record ----------------

refuses('anchors recorded against a DIFFERENT parent', {
  ledger: declared({
    s1: { vsBase: { ...HEALTHY, base: 'OLD-PARENT' } },
    s2: { vsBase: { ...HEALTHY, base: 'OLD-PARENT' } },
  }),
});

refuses('vsBase is `true`', { ledger: declared({ s1: { vsBase: true }, s2: { vsBase: true } }) });
refuses('vsBase is `{}`', { ledger: declared({ s1: { vsBase: {} }, s2: { vsBase: {} } }) });
refuses('vsBase is an array', { ledger: declared({ s1: { vsBase: [] }, s2: { vsBase: HEALTHY } }) });
// `null` specifically, because it is the one shape that makes the object check
// load-bearing rather than redundant: `true` and `[]` are both caught downstream by the
// base mismatch, but `null.base` throws, and a crash is not a refusal.
refuses('vsBase is null', { ledger: declared({ s1: { vsBase: null }, s2: { vsBase: HEALTHY } }) });
refuses('vsBase has no game count', {
  ledger: declared({ s1: { vsBase: { ...HEALTHY, games: undefined } }, s2: { vsBase: HEALTHY } }),
});
refuses('vsBase has a string game count', {
  ledger: declared({ s1: { vsBase: { ...HEALTHY, games: '200' } }, s2: { vsBase: HEALTHY } }),
});
refuses('vsBase has no score', {
  ledger: declared({ s1: { vsBase: { ...HEALTHY, scorePct: undefined } }, s2: { vsBase: HEALTHY } }),
});
refuses('vsBase has a string Elo', {
  ledger: declared({ s1: { vsBase: { ...HEALTHY, elo: '8' } }, s2: { vsBase: HEALTHY } }),
});

// --- an anchor showing collapse is not permission ------------------------------
//
// Two branches: saturated, and magnitude in either direction. Both directions are
// exercised because they share one condition and a sign error would break exactly one
// of them -- the incident's own delta was favourable-looking.

refuses('one arm is 530 Elo below its parent', {
  ledger: declared({
    s1: { vsBase: { ...HEALTHY, scorePct: 4.5, elo: -530 } },
    s2: { vsBase: HEALTHY },
  }),
});

refuses('one arm was annihilated 0-0-200 (saturated, elo null)', {
  ledger: declared({
    s1: { vsBase: { ...HEALTHY, scorePct: 0, elo: null } },
    s2: { vsBase: HEALTHY },
  }),
});

refuses('one arm is beyond the breakage threshold in the FAVOURABLE direction', {
  ledger: declared({
    s1: { vsBase: { ...HEALTHY, scorePct: 96.0, elo: 552 } },
    s2: { vsBase: HEALTHY },
  }),
});

// The floor is -BREAKAGE_ELO, not -POOL_SPREAD_HEALTHY: a legitimately weak early
// generation must still be screenable.
passesGate('an arm 120 Elo below its parent is weak, not broken', {
  ledger: declared({
    s1: { vsBase: { ...HEALTHY, scorePct: 33.4, elo: -120 } },
    s2: { vsBase: HEALTHY },
  }),
});

// --- the threshold itself must be a number ------------------------------------

refuses('WW_BREAKAGE_ELO is a plausible typo ("2x145" -> NaN)', {
  ledger: declared({ s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } }),
  env: { WW_BREAKAGE_ELO: '2x145' },
});

refuses('WW_POOL_SPREAD_HEALTHY is negative', {
  ledger: declared({ s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } }),
  env: { WW_POOL_SPREAD_HEALTHY: '-1' },
});

// Empty is UNSET, not zero. `export WW_BREAKAGE_ELO=` and an unset variable expand
// identically in shell, and the pre-fix `Number('' ?? d)` read that as 0 -- a threshold
// of zero fires SUSPECTED BREAKAGE on every run including a dead-even 50%, which is as
// useless as never firing. Falling back to the default is the safe reading.
passesGate('WW_BREAKAGE_ELO is empty -> falls back to the default, not 0', {
  ledger: declared({ s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY } }),
  env: { WW_BREAKAGE_ELO: '' },
});

// --- a malformed ledger refuses rather than crashing ---------------------------

refuses('ledger file is literal null', { raw: 'null' });
refuses('ledger file is not JSON', { raw: '{ not json' });
refuses('ledger entry is null', { raw: JSON.stringify({ __parents__: { P: {} }, s1: null, s2: null }) });

// --- --declare-parent actually records ----------------------------------------

{
  // Declares, records, and EXITS 0 without playing anything. Declare-then-play forced
  // the refusal message to recommend `--a s1 --b s2 --base s2 --declare-parent`, which
  // is the abuse shape; an operator should not be handed it at the moment they are
  // blocked.
  const { code, out, file } = run({ ledger: {}, args: ['--declare-parent'] });
  const after = JSON.parse(readFileSync(file, 'utf8'));
  if (!after.__parents__?.P) {
    FAILURES.push('--declare-parent did not record the parent in the ledger');
  }
  if (code !== 0) FAILURES.push(`--declare-parent exited ${code}, want 0`);
  if (/ERR_MODULE_NOT_FOUND|Cannot find module/.test(out)) {
    FAILURES.push('--declare-parent proceeded to play instead of exiting after declaring');
  }
  // And a declared base then screens normally.
  const re = run({
    raw: JSON.stringify({
      __parents__: { P: { ts: '2026-01-01' } },
      s1: { vsBase: HEALTHY }, s2: { vsBase: HEALTHY },
    }),
  });
  if (re.code !== 1) FAILURES.push(`after declaring, a normal screen exited ${re.code}, want the gate to pass`);
}

// -------------------------------------------------------------------------------

rmSync(dir, { recursive: true, force: true });
if (FAILURES.length) {
  console.error(`FAIL (${FAILURES.length})`);
  for (const f of FAILURES) console.error(`  ${f}`);
  process.exit(1);
}
console.log('all distill-compare guard checks passed: an undeclared base, an unanchored,'
  + ' stale-anchored, self-anchored or collapsed-arm screen is refused, lineage still'
  + ' works, and the tripwire threshold cannot be turned into NaN');
