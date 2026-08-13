#!/usr/bin/env node
/**
 * Step 0 of docs/tree-reuse-plan.md: measure what tree reuse WOULD inherit.
 *
 *     node scripts/tree-reuse-probe.mjs --games 4 --sims 512 [--considered 12]
 *
 * The premise behind tree reuse is a 2-4x effective saving. That figure comes from
 * engines with deep trees and millions of nodes, and there is a specific reason to doubt
 * it here: Gumbel sequential halving spends the budget on `considered` root actions in
 * ceil(log2(considered)) rounds, so the visits concentrate on a handful of candidates,
 * and the node a next search would start from is a GRANDCHILD -- our move, then their
 * reply.
 *
 * MEASURED RESULT (self-play, d3-iter-150): median 23.4% of budget at 512 sims / 12
 * considered over 125 transitions, and 20.3% at 128 sims over 126. The specific fear
 * above did NOT materialise -- the reply was never absent from the tree (0% of
 * transitions), because the opponent's reply is a child of the move we searched hardest,
 * not a root candidate we might have discarded. The distribution is wide rather than
 * bimodal: p10 0.2%, p25 8.8%, p50 23.4%, p90 34.8%.
 *
 * So this keeps each move's search alive one move longer than it needs to be, and asks
 * it -- after both moves are known -- how many visits sit under [our move, their reply].
 * That is exactly what a resumed search would inherit.
 *
 * It reports the DISTRIBUTION, not the mean. The quantity is expected to be bimodal
 * (a decent subtree when the reply was a survivor, ~0 when it was not), and a mean over
 * a bimodal quantity is the one summary that would mislead here.
 *
 * IT MEASURES A CEILING. An inherited visit is worth less than a fresh one: the Gumbel
 * schedule still has to be re-run at the new root, so inherited statistics inform the
 * search rather than replacing simulations. A small number here rules reuse out; a large
 * number does not rule it in.
 */

import { readFileSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createInitialState, applyAction, decodeAction } from '../js/normal-duel-engine.mjs';
import { createNetGuard } from './net-guard.mjs';
import { CONFIG_9X9 as CONFIG } from '../tests/support/nn-runtime-fixture.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(`--${n}`); return i === -1 ? d : argv[i + 1]; };
const GAMES = Number(arg('games', 4));
const SIMS = Number(arg('sims', 512));
const CONSIDERED = Number(arg('considered', 12));
const INFER = String(arg('infer', 'http://127.0.0.1:8099')).replace(/\/$/, '');
const MAX_PLIES = Number(arg('maxPlies', 200));

const agent = new http.Agent({ keepAlive: true, maxSockets: 4, noDelay: true });
const inferUrl = new URL(`${INFER}/infer`);
function inferOne(features) {
  const body = Buffer.copyBytesFrom(features);      // copy before any await
  return new Promise((resolve, reject) => {
    const req = http.request({ agent, hostname: inferUrl.hostname, port: inferUrl.port,
      path: inferUrl.pathname, method: 'POST',
      headers: { 'content-type': 'application/octet-stream', 'content-length': body.length } },
      (res) => {
        const c = [];
        res.on('data', (d) => c.push(d));
        res.on('end', () => {
          if (res.statusCode !== 200) return reject(new Error(`infer ${res.statusCode}`));
          const b = Buffer.concat(c);
          resolve(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4));
        });
      });
    req.on('socket', (s) => s.setNoDelay(true));
    req.setTimeout(120000, () => req.destroy(new Error('infer timeout')));
    req.on('error', reject);
    req.end(body);
  });
}

const wasmMod = await import(`${ROOT}/rust/target/wasm-candidate/release/normal-duel-wasm.mjs`);
const wasmBytes = readFileSync(`${ROOT}/rust/target/wasm-candidate/release/normal-duel-wasm_bg.wasm`);
const instance = await wasmMod.default({ module_or_path: wasmBytes });
const memory = instance.memory ?? wasmMod.__wasm?.memory ?? wasmMod.memory;

if (typeof wasmMod.NormalDuelSearch.prototype.subtreeVisitsAfter !== 'function') {
  console.error('this wasm build has no subtreeVisitsAfter; run '
    + 'npm run build:normal-duel-wasm-candidate');
  process.exit(2);
}

/** Run one search to completion and return it UNFREED, plus its chosen action. */
async function search(state, sims, seed, considered) {
  const s = new wasmMod.NormalDuelSearch(
    JSON.stringify(CONFIG), JSON.stringify(state),
    JSON.stringify({ simulations: sims, maxConsidered: considered, cPuct: 1.25, seed }));
  const guard = createNetGuard(s.policyLen());
  while (!s.isDone()) {
    s.nextLeaf();
    if (s.isDone()) break;
    const out = await inferOne(
      new Float32Array(memory.buffer, s.featuresPtr(), s.featuresLen()));
    const len = s.policyLen();
    s.pendingLeafMask();
    const mask = new Float32Array(memory.buffer, s.maskPtr(), len);
    const { probs, value } = guard.classify(out, mask);
    new Float32Array(memory.buffer, s.policyPtr(), len).set(probs);
    s.submit(value);
    if (guard.doomed) break;
  }
  guard.finish();
  return { search: s, code: s.actionCode(), action: decodeAction(CONFIG, s.actionCode()),
           visits: s.visitCounts() };
}

const samples = [];        // { ply, inherited, budget }
let plies = 0;

for (let g = 0; g < GAMES; g += 1) {
  let state = createInitialState(CONFIG);
  let prev = null;         // the search from the previous ply, kept alive on purpose
  let prevCode = null;
  let ply = 0;
  while (state.outcome.kind === 'ongoing' && ply < MAX_PLIES) {
    const cur = await search(state, SIMS, g * 100003 + ply + 1, CONSIDERED);

    // `prev` chose prevCode; the position then advanced by prevCode and, now, by
    // cur.code. So from prev's root, [prevCode, cur.code] is the node a search
    // starting HERE would have inherited.
    if (prev) {
      const inherited = prev.search.subtreeVisitsAfter(
        new Uint16Array([prevCode, cur.code]));
      // Only the inheritance itself is recorded. An earlier version also reported
      // whether the reply was one of PREV's root candidates, which sounds like the
      // mechanism and is not: the inherited node is a child of the move we chose, so
      // whether the reply also happened to be a move WE were considering for ourselves
      // is a coincidence. Reporting it as if it explained the inheritance was wrong.
      samples.push({ ply, inherited, budget: SIMS });
      prev.search.free();
    }
    prev = cur;
    prevCode = cur.code;
    state = applyAction(CONFIG, state, cur.action);
    ply += 1;
    plies += 1;
  }
  if (prev) prev.search.free();
  process.stdout.write(`  game ${g + 1}: ${ply} plies\n`);
}

// ------------------------------------------------------------------ report
const fracs = samples.map((s) => (s.inherited < 0 ? 0 : s.inherited) / s.budget);
const sorted = [...fracs].sort((a, b) => a - b);
const q = (p) => sorted.length ? sorted[Math.min(sorted.length - 1,
  Math.floor(p * sorted.length))] : 0;
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const missing = samples.filter((s) => s.inherited < 0).length;
const zero = samples.filter((s) => s.inherited === 0).length;
const any = samples.filter((s) => s.inherited > 0).length;

console.log(`\ntree-reuse inheritance over ${samples.length} transitions `
  + `(${GAMES} games, ${plies} plies, ${SIMS} sims / ${CONSIDERED} considered)\n`);
console.log(`  reply never expanded in the tree : ${missing} (${pct(missing / samples.length)})`);
console.log(`  reply present but 0 visits       : ${zero} (${pct(zero / samples.length)})`);
console.log(`  some inheritance available       : ${any} (${pct(any / samples.length)})`);
console.log('\n  inherited visits as a fraction of the budget:');
for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
  console.log(`    p${String(Math.round(p * 100)).padStart(2)} : ${pct(q(p))}`);
}
console.log(`    max : ${pct(sorted[sorted.length - 1] ?? 0)}`);
const median = q(0.5);
console.log('\n  DECISION RULE from docs/tree-reuse-plan.md, on the median:');
console.log(median >= 0.25
  ? `    ${pct(median)} >= 25%  -> build it`
  : median >= 0.10
    ? `    ${pct(median)} in 10-25%  -> self-play only, not the play site`
    : `    ${pct(median)} < 10%  -> STOP. Record as settled and spend the effort on depth.`);
console.log('\n  (a ceiling: the Gumbel schedule would still be re-run at the new root,'
  + '\n   so inherited statistics inform the search rather than replacing simulations)');
