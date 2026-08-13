#!/usr/bin/env node
/**
 * Head-to-head between two simulation budgets of the SAME network.
 *
 *     node scripts/sims-match.mjs --a 2048 --b 128 --games 10 [--concurrency 4]
 *
 * Why this and not the `hard` ladder: scoring one net against a fixed opponent
 * saturates. The ladder put 128 sims at 92.0% and 512 at 94.5% vs `hard` with
 * overlapping intervals, so it could not say whether the extra search was worth
 * anything. Playing the budgets against EACH OTHER has no ceiling -- the result is
 * directly "how often does more thinking win".
 *
 * Colours alternate every game, because side A moves first on a 9x9 board and the
 * side advantage here was measured at ~13 points. An odd number of games, or a
 * fixed assignment, would fold that straight into the answer.
 *
 * Both sides share one network process, so this measures search depth only.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import { createInitialState, applyAction, decodeAction } from '../js/normal-duel-engine.mjs';
import { createNetGuard } from './net-guard.mjs';
import { CONFIG_9X9 as CONFIG } from '../tests/support/nn-runtime-fixture.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(`--${n}`); return i === -1 ? d : argv[i + 1]; };
/** A bad number here silently produced 0 games and a NaN% result. */
function positiveInt(name, fallback) {
  const raw = arg(name, String(fallback));
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1) {
    console.error(`--${name} must be a positive integer, got ${JSON.stringify(raw)}`);
    process.exit(2);
  }
  return n;
}
const SIMS_A = positiveInt('a', 2048);
const SIMS_B = positiveInt('b', 128);
const GAMES = positiveInt('games', 10);
const CONC = positiveInt('concurrency', 4);
if (GAMES % 2 !== 0) {
  // Colours alternate every game, and side A moves first with a ~13-point edge, so an
  // odd count folds part of that advantage straight into the result.
  console.error(`--games must be even so the colours balance, got ${GAMES}`);
  process.exit(2);
}
// Per-side candidate breadth. Sequential halving spends the budget on
// `considered` root actions only, so at 12 considered a large budget refines twelve
// options and never examines the other ~56 legal moves. Making this asymmetric is
// what lets "more depth" be tested against "more breadth" at the SAME budget.
const CONS_A = Number(arg('consideredA', arg('considered', 12)));
const CONS_B = Number(arg('consideredB', arg('considered', 12)));
const INFER = String(arg('infer', 'http://127.0.0.1:8099')).replace(/\/$/, '');

const agent = new http.Agent({ keepAlive: true, maxSockets: Math.max(8, CONC * 2) });
const inferUrl = new URL(`${INFER}/infer`);
function inferOne(body) {
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
    // Without a timeout a lost keep-alive request never settles: the worker sits at
    // 0% CPU forever, the game never ends, and Promise.all never resolves -- one
    // 2048v512 game hung this way for 90 min while the other worker finished 19.
    req.setTimeout(60000, () => req.destroy(new Error('infer timeout')));
    req.on('error', reject);
    req.end(body);
  });
}

/**
 * COPY the features out of wasm memory before sending. The body used to be a
 * zero-copy view of `memory.buffer`, but both concurrent games share ONE wasm
 * instance: when the other game allocates its search, wasm memory grows, the old
 * ArrayBuffer is detached, and this request's body is truncated mid-write. The
 * server then blocks forever reading a body that will never arrive, so the worker
 * sits at 0% CPU and Promise.all never resolves -- that is what hung one 2048v512
 * game on its first move, and at 4096 sims it fires on the very first request.
 * The copy is 3.2 KB a call, which is nothing next to a 4 ms forward pass.
 * Inference is stateless, so a dropped request is also safe to replay.
 */
async function infer(features) {
  const body = Buffer.from(Buffer.from(features.buffer, features.byteOffset, features.byteLength));
  for (let attempt = 0; ; attempt += 1) {
    try { return await inferOne(body); }
    catch (err) {
      if (attempt >= 4) throw err;
      console.log(`  [infer retry ${attempt + 1}: ${err.message}]`);
    }
  }
}

const wasmMod = await import(`${ROOT}/rust/target/wasm-candidate/release/normal-duel-wasm.mjs`);
const wasmBytes = readFileSync(`${ROOT}/rust/target/wasm-candidate/release/normal-duel-wasm_bg.wasm`);
const instance = await wasmMod.default({ module_or_path: wasmBytes });
const memory = instance.memory ?? wasmMod.__wasm?.memory ?? wasmMod.memory;
const build = createHash('sha256').update(wasmBytes).digest('hex').slice(0, 16);

/**
 * One move, guarded.
 *
 * This harness is where BOTH historical silent-degradation bugs were found -- the
 * unfilled mask and the coerced-float body -- and until now it was the one driver with
 * no check for them, while its output is the evidence for the search-depth findings. A
 * measurement tool that cannot tell a priorless search from a real one produces numbers
 * about the wrong thing, so it now shares the play server's guard rather than
 * reimplementing a weaker version of it.
 */
async function chooseMove(state, sims, seed, considered) {
  const search = new wasmMod.NormalDuelSearch(
    JSON.stringify(CONFIG), JSON.stringify(state),
    JSON.stringify({ simulations: sims, maxConsidered: considered, cPuct: 1.25, seed }));
  const guard = createNetGuard(search.policyLen());
  try {
    while (!search.isDone()) {
      search.nextLeaf();
      if (search.isDone()) break;
      const out = await infer(
        new Float32Array(memory.buffer, search.featuresPtr(), search.featuresLen()));
      const len = search.policyLen();
      // pendingLeafMask() FILLS the mask buffer; maskPtr() only says where it lives.
      // Without this call the buffer stays zero, the masked softmax finds no legal
      // entry, an ALL-ZERO policy gets submitted (which submit() accepts), and the
      // search runs with no priors at all -- picking near-arbitrary moves while every
      // latency number and the root value still look plausible.
      search.pendingLeafMask();
      const mask = new Float32Array(memory.buffer, search.maskPtr(), len);
      const { probs, value } = guard.classify(out, mask);
      new Float32Array(memory.buffer, search.policyPtr(), len).set(probs);
      search.submit(value);
      if (guard.doomed) break;
    }
    // A match is a measurement, so a degenerate network must abort it rather than
    // quietly contribute games. The caller lets this propagate.
    guard.finish();
    return decodeAction(CONFIG, search.actionCode());
  } finally { search.free(); }
}

/** One game. `aSide` says which colour the A-budget plays. */
async function playGame(gameIndex, aSide) {
  let state = createInitialState(CONFIG);
  const t0 = performance.now();
  let plies = 0;
  while (state.outcome.kind === 'ongoing') {
    const turn = state.position.turn;
    const sims = turn === aSide ? SIMS_A : SIMS_B;
    const cons = turn === aSide ? CONS_A : CONS_B;
    // Seed varies with game and ply so the two budgets do not replay one line.
    const action = await chooseMove(state, sims, gameIndex * 100003 + plies + 1, cons);
    state = applyAction(CONFIG, state, action);
    plies += 1;
  }
  const sec = (performance.now() - t0) / 1000;
  let result;                                   // from the A-budget's viewpoint
  if (state.outcome.kind === 'draw') result = 'draw';
  else result = state.outcome.winner === aSide ? 'win' : 'loss';
  console.log(`  game ${String(gameIndex + 1).padStart(2)}: ${SIMS_A} as ${aSide} -> `
    + `${result.padEnd(4)}  ${plies} plies  ${sec.toFixed(0)}s  (${state.outcome.reason || ''})`);
  return result;
}

const vpc = (sims, cons) => Math.max(1, Math.floor(sims / (Math.max(1, 32 - Math.clz32(cons - 1)) * cons)));
console.log(`sims match: A=${SIMS_A}sims/${CONS_A}cons (~${vpc(SIMS_A,CONS_A)} visits/candidate)`
  + ` vs B=${SIMS_B}sims/${CONS_B}cons (~${vpc(SIMS_B,CONS_B)} visits/candidate),`
  + ` ${GAMES} games, wasm ${build}`);
console.log(`network ${INFER}, concurrency ${CONC}\n`);

const jobs = Array.from({ length: GAMES }, (_, i) => () => playGame(i, i % 2 === 0 ? 'A' : 'B'));
const results = [];
const t0 = performance.now();
let next = 0;
await Promise.all(Array.from({ length: Math.min(CONC, GAMES) }, async () => {
  while (next < jobs.length) {
    const mine = next++;
    results.push(await jobs[mine]());
  }
}));

const w = results.filter((r) => r === 'win').length;
const l = results.filter((r) => r === 'loss').length;
const d = results.filter((r) => r === 'draw').length;
const score = (w + d / 2) / results.length;
// Wilson-free, deliberately: with 10 games the honest statement is the interval,
// and a normal approximation at n=10 is misleading. Report the exact binomial
// two-sided p for "no difference" instead.
function binomP(k, n) {                          // P(X >= k) + P(X <= n-k) under p=0.5
  const choose = (a, b) => { let r = 1; for (let i = 0; i < b; i += 1) r = r * (a - i) / (i + 1); return r; };
  let tail = 0;
  for (let i = k; i <= n; i += 1) tail += choose(n, i);
  return Math.min(1, 2 * tail / 2 ** n);
}
const decided = w + l;
console.log(`\n  ${SIMS_A} sims scored ${w}-${d}-${l} (W-D-L) = ${(score * 100).toFixed(1)}%`);
if (decided > 0) {
  const p = binomP(Math.max(w, l), decided);
  const elo = score > 0 && score < 1
    ? (-400 * Math.log10(1 / score - 1)).toFixed(0) : 'inf';
  console.log(`  implied Elo gap: ${elo}`);
  console.log(`  exact binomial p (no difference, ${decided} decided games): ${p.toFixed(3)}`);
  console.log(p < 0.05
    ? '  -> significant at 5%'
    : `  -> NOT significant; ${GAMES} games cannot resolve a gap this size`);
}
console.log(`  wall clock ${((performance.now() - t0) / 60000).toFixed(1)} min`);
