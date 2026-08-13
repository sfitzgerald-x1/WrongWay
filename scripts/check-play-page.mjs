#!/usr/bin/env node
/**
 * Execute play.html's module script against stub globals.
 *
 *     node scripts/check-play-page.mjs
 *
 * A parse check cannot catch a ReferenceError -- `auth = d.auth` parses fine and
 * throws only when the line runs, which is how "standings unavailable: auth is not
 * defined" reached the browser. This actually RUNS the boot path with document,
 * fetch and localStorage stubbed, so an undeclared variable or a missing element
 * fails here instead of in front of someone.
 */
import { readFileSync, writeFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = readFileSync(path.join(ROOT, 'play.html'), 'utf8');
const src = html.split('<script type="module">')[1].split('</script>')[0];

const ids = [...html.matchAll(/id="([\w-]+)"/g)].map((m) => m[1]);
// Every write to the DOM is recorded. The page reports its own failures by putting
// a message on screen inside a catch, so an exception-only check sees nothing -- that
// is precisely how "standings unavailable: auth is not defined" got shipped.
const writes = [];
const el = (id) => {
  const node = {
    id, innerHTML: '', value: '', className: '', style: {},
    classList: { add() {}, remove() {}, toggle() {} },
    appendChild() {}, addEventListener() {}, querySelector: () => el('inner'),
    remove() {}, onclick: null, firstChild: { textContent: '' }
  };
  let text = '';
  Object.defineProperty(node, 'textContent', {
    get: () => text,
    set: (v) => { text = String(v); writes.push({ id, text }); }
  });
  return node;
};
const known = new Map(ids.map((i) => [i, el(i)]));
const missing = new Set();

globalThis.document = {
  getElementById(id) {
    if (known.has(id)) return known.get(id);
    missing.add(id);           // records instead of throwing, so all are reported
    return el(id);
  },
  createElement: () => el('created'),
  addEventListener() {},
  body: { appendChild() {} },
  visibilityState: 'visible'
};
globalThis.localStorage = {
  _d: new Map(),
  getItem(k) { return this._d.has(k) ? this._d.get(k) : null; },
  setItem(k, v) { this._d.set(k, String(v)); },
  removeItem(k) { this._d.delete(k); }
};
globalThis.getComputedStyle = () => ({ getPropertyValue: (n) => (n === '--cell' ? '46' : '9') });
const routes = {
  // `id` and `userId` are present because the page matches YOUR row by the opaque
  // handle, not by display name. A mock without them silently exercised the legacy
  // name-matching fallback instead -- i.e. the bug the handle was added to fix.
  // The two rows share a display name on purpose: that collision is the reason
  // name-matching was wrong, so the checker asserts the right row is picked.
  '/api/me': { user: { id: 'h0000000000000001', name: 'Test User',
                       verified: true, provider: 'tailscale' }, source: 'tailscale' },
  '/api/leaderboard': {
    recent: [],
    players: [
      { userId: 'h0000000000000002', name: 'Test User', verified: true, w: 9, l: 1, d: 0, games: 10, last: null },
      { userId: 'h0000000000000001', name: 'Test User', verified: true, w: 2, l: 3, d: 0, games: 5, last: null }
    ],
    totals: { humanWins: 11, botWins: 4, draws: 0, games: 15 } },
  '/api/health': { ok: true, wasm: 'x', network: { version: 'ckpt', device: 'cpu', msPerCall: 8 }, meta: { label: 'ckpt' } },
  '/api/bestmove': { action: { kind: 'pawn', to: { r: 1, c: 4 } }, ms: 100, sims: 512, rootValue: 0.1 },
  '/api/result': { ok: true }
};
globalThis.fetch = async (url) => {
  const key = Object.keys(routes).find((k) => String(url).startsWith(k));
  if (!key) throw new Error(`unstubbed fetch: ${url}`);
  return { ok: true, status: 200, json: async () => routes[key] };
};

// tmpdir, not the repo root: a SIGKILL skips the finally that removes this, and an
// orphaned .check-play-*.mjs in a source tree that does not gitignore it is litter.
const tmp = path.join(tmpdir(), `check-play-${process.pid}.mjs`);
writeFileSync(tmp, src.replace("'/js/normal-duel-engine.mjs'", `'${ROOT}/js/normal-duel-engine.mjs'`));
let failed = false;
try {
  await import(tmp);
  // Boot does async work (identity, standings, first move); let it settle.
  await new Promise((r) => setTimeout(r, 600));
  console.log('  boot path ran with no exception');
} catch (e) {
  failed = true;
  console.log(`  RUNTIME ERROR: ${e.message}`);
} finally { rmSync(tmp, { force: true }); }

// Anything the page put on screen that reads like a failure is a failure, even
// though nothing was thrown out of the module.
const BAD = [/unavailable/i, /is not defined/i, /is not a function/i,
             /undefined/i, /\[object/i, /NaN/];
const complaints = writes.filter((w) => BAD.some((re) => re.test(w.text)));
if (complaints.length) {
  failed = true;
  console.log('  PAGE REPORTED FAILURES:');
  for (const c of complaints.slice(0, 6)) console.log(`    #${c.id}: ${c.text}`);
} else {
  console.log(`  nothing on screen reads as an error (${writes.length} DOM writes checked)`);
}

if (missing.size) {
  failed = true;
  console.log(`  MISSING ELEMENTS: ${[...missing].join(', ')}`);
} else {
  console.log('  every element referenced exists');
}

// NOT nested in the else above: a missing element used to skip this silently, so a
// regression here could hide behind an unrelated failure.
// The two mocked players share a display name; only the handle distinguishes them.
// 'me' is h...01 with 2W 3L, so name-matching would have shown the other row's 9W 1L.
const meRec = writes.filter((w) => w.id === 'meRec').pop();
if (!meRec) {
  console.log('  FAIL: nothing was written to meRec, so the standings path did not run');
  failed = true;
} else if (!/2W\s*3L/.test(meRec.text)) {
  console.log(`  FAIL: meRec says "${meRec.text}"; expected this user's own 2W 3L. `
    + 'Matching by display name would pick the other player with the same name.');
  failed = true;
} else {
  console.log(`  the standings row for "me" is matched by handle (${meRec.text})`);
}

process.exit(failed ? 1 : 0);
