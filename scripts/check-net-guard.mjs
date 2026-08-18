#!/usr/bin/env node
/**
 * Prove the play server refuses to serve a move a broken network produced.
 *
 *     node scripts/check-net-guard.mjs
 *
 * The hard part is that a broken network still yields a LEGAL move: the search happily
 * runs on uniform or absent priors, returns a plausible action and a plausible root
 * value, and plays like a weak bot. That has shipped twice here. A test that only
 * checks for a 200 and a legal action cannot see it.
 *
 * So this stands up a stub inference server broken in one specific way per case and
 * asserts the server REFUSES. Two cases are controls rather than refusals -- `healthy`,
 * and `oneflat` for the tolerance -- because a suite that never serves a move would
 * pass just as well with everything broken.
 *
 * SCOPE, stated so the summary line is not read as more than it is: every case here is
 * about the RESPONSE. None of them establish that the network read the correct input.
 * Request-side corruption (the Buffer.from bug sent a quarter-length body) yields a
 * well-shaped varying policy for the wrong position, and only the inference server's own
 * length check refuses that. The `healthy` stub deliberately derives its output from the
 * request bytes, because a fixed-vector stub would itself be a network ignoring the
 * board -- which is what an earlier version of this file used as its control.
 *
 * Runs against its own port and its own data dir: pointing it at the live store
 * would write test games onto the real leaderboard.
 */

import { spawn } from 'node:child_process';
import { createNetGuard, NetGuardError } from './net-guard.mjs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const INFER_PORT = 8399;
const PLAY_PORT = 8398;
const POLICY_LEN = 209;               // 9x9 duel action space
const SIMS = 16;                      // enough to reach real leaves, fast enough to loop

/** How the stub network misbehaves. Swapped between cases. */
let mode = 'healthy';

const stub = http.createServer((req, res) => {
  if (req.url === '/healthz') {
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({ ok: true, version: 'stub', device: 'none' }));
  }
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    // The healthy stub READS ITS INPUT. An earlier version returned one fixed vector
    // for every leaf, which is precisely a network ignoring the board -- so the
    // control case was itself a broken network, and a file asserting "served only
    // when the network shaped it" demonstrated the opposite for its own control.
    const features = Buffer.concat(chunks);
    let h = 2166136261;
    for (let i = 0; i < features.length; i += 4) h = ((h ^ features[i]) * 16777619) >>> 0;
    const salt = (h % 1000) / 1000;

    let out;
    if (mode === 'zeros') {
      // The classic silent failure: every legal logit identical, so the masked
      // softmax turns it into a tidy uniform prior and the search flies blind.
      out = new Float32Array(POLICY_LEN + 1);
    } else if (mode === 'short') {
      // Short AND varying, so this isolates the length check. A short vector of zeros
      // would also be degenerate, and would still 500 with the shape check deleted.
      out = new Float32Array(POLICY_LEN);
      for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7 + salt) * 2;
    } else if (mode === 'nanlogits') {
      // Reaches the `blind` branch: exp(NaN) is NaN, so no legal entry gains mass.
      out = new Float32Array(POLICY_LEN + 1).fill(Number.NaN);
      out[POLICY_LEN] = 0.25;
    } else if (mode === 'nanvalue' || mode === 'hugevalue') {
      out = new Float32Array(POLICY_LEN + 1);
      for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7 + salt) * 2;
      out[POLICY_LEN] = mode === 'nanvalue' ? Number.NaN : 42;
    } else if (mode === 'constvec') {
      // Varying across entries, identical across leaves: a network that never reads
      // the position. Every per-leaf check passes; only the cross-leaf test sees it.
      out = new Float32Array(POLICY_LEN + 1);
      for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7) * 2;
      out[POLICY_LEN] = 0.25;
    } else if (mode === 'oneflat') {
      // A single coincidental flat leaf, which must NOT kill the move: retries are
      // seed-deterministic, so refusing here made one float collision permanent.
      oneflatCount += 1;
      out = new Float32Array(POLICY_LEN + 1);
      if (oneflatCount === 5) out[POLICY_LEN] = 0.25;        // all logits 0.0 -> flat
      else {
        for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7 + salt) * 2;
        out[POLICY_LEN] = 0.25;
      }
    } else {
      out = new Float32Array(POLICY_LEN + 1);
      for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7 + salt) * 2;
      out[POLICY_LEN] = 0.25;
    }
    const body = Buffer.copyBytesFrom(out);
    res.writeHead(200, { 'content-type': 'application/octet-stream',
                         'content-length': body.length });
    res.end(body);
  });
});
let oneflatCount = 0;
await new Promise((r) => stub.listen(INFER_PORT, '127.0.0.1', r));

const server = spawn(process.execPath, [
  path.join(ROOT, 'scripts/play-server.mjs'),
  '--port', String(PLAY_PORT),
  '--infer', `http://127.0.0.1:${INFER_PORT}`,
  '--data', '/tmp/ww-net-guard-test'          // NOT the live store
], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
server.stdout.on('data', (d) => { serverLog += d; });
server.stderr.on('data', (d) => { serverLog += d; });

const bestmove = () => new Promise((resolve, reject) => {
  const body = JSON.stringify({
    history: [], sims: SIMS, seed: 1,
    board: { map: 'duel', duelSize: 'standard', initialStock: 10 }
  });
  const req = http.request({ hostname: '127.0.0.1', port: PLAY_PORT, path: '/api/bestmove',
    method: 'POST', headers: { 'content-type': 'application/json',
                               'content-length': Buffer.byteLength(body) } }, (res) => {
    const c = [];
    res.on('data', (d) => c.push(d));
    res.on('end', () => {
      let parsed = null;
      try { parsed = JSON.parse(Buffer.concat(c).toString()); } catch { /* leave null */ }
      resolve({ status: res.statusCode, body: parsed });
    });
  });
  req.setTimeout(120000, () => req.destroy(new Error('bestmove timeout')));
  req.on('error', reject);
  req.end(body);
});

// Wait for the listener rather than sleeping a fixed amount.
for (let i = 0; ; i += 1) {
  try { await bestmove(); break; }
  catch (err) {
    if (i >= 40) { console.error('server never came up\n', serverLog); process.exit(1); }
    await new Promise((r) => setTimeout(r, 250));
  }
}

const CASES = [
  { mode: 'healthy', want: 'move',
    why: 'control: an input-dependent varying policy must produce a served move, else the refusals below mean nothing' },
  { mode: 'zeros', want: 'net_degenerate',
    why: 'flat logits are a broken net laundered into a uniform prior' },
  { mode: 'nanlogits', want: 'net_degenerate',
    why: 'NaN logits leave no legal entry with prior mass -- the priorless branch' },
  { mode: 'short', want: 'net_bad_shape',
    why: 'a wrong-length response is someone else\'s bytes (varying, so only the length can refuse it)' },
  { mode: 'nanvalue', want: 'net_bad_value',
    why: 'a NaN value was clamped to 0 at every leaf, degrading the search while every other signal looked healthy' },
  { mode: 'hugevalue', want: 'net_bad_value',
    why: 'the head is a tanh, so a value of 42 is a broken graph, not a numerical edge' },
  { mode: 'constvec', want: 'net_input_ignored',
    why: 'a network returning one fixed vector for every position passes every per-leaf check' },
  { mode: 'oneflat', want: 'move',
    why: 'a single coincidental flat leaf must survive: retries are seed-deterministic, so refusing makes one float32 collision permanent' }
];

let failures = 0;

// ---------------------------------------------------------------- unit cases
//
// Two refusals cannot be provoked from a stub NETWORK, because they are about the wasm
// MASK rather than the response: an unfilled mask buffer, and a move where no leaf ends
// up network-informed. Those were the two codes with no coverage -- including
// net_mask_empty, which is the original production bug the whole module exists for -- so
// they are driven against the guard directly.
const POLICY = 209;
const varying = () => {
  const out = new Float32Array(POLICY + 1);
  for (let i = 0; i < POLICY; i += 1) out[i] = Math.sin(i * 0.7 + Math.random()) * 2;
  out[POLICY] = 0.25;
  return out;
};
const maskOf = (n) => {
  const m = new Float32Array(POLICY);
  for (let i = 0; i < n; i += 1) m[i] = 1;
  return m;
};

function unitCase(name, want, run) {
  let got = 'no error';
  try { run(); } catch (err) {
    got = err instanceof NetGuardError ? err.code : `${err.constructor.name}: ${err.message}`;
  }
  const ok = got === want;
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name.padEnd(22)} -> ${got}`);
  if (!ok) { console.log(`         expected ${want}`); failures += 1; }
}

unitCase('mask never filled', 'net_mask_empty', () => {
  // Exactly the pendingLeafMask() bug: the buffer stays all zeros.
  const g = createNetGuard(POLICY);
  const empty = new Float32Array(POLICY);
  for (let i = 0; i < 4; i += 1) g.classify(varying(), empty);
  g.finish();
});

unitCase('no informed leaf', 'net_never_consulted', () => {
  // Every leaf flat with only 2 legal moves: tolerated individually (a float32
  // coincidence in a corridor), but nothing is evidence, so the move must still refuse.
  const g = createNetGuard(POLICY);
  const two = maskOf(2);
  for (let i = 0; i < 5; i += 1) {
    const flat = new Float32Array(POLICY + 1);
    flat[POLICY] = 0.25;                       // all logits identical at 0
    g.classify(flat, two);
  }
  g.finish();
});

unitCase('two flat of many', 'no error', () => {
  // The tolerance boundary: 2 flat leaves among plenty of informed ones is survivable.
  const g = createNetGuard(POLICY);
  const many = maskOf(60);
  for (let i = 0; i < 20; i += 1) g.classify(varying(), many);
  for (let i = 0; i < 2; i += 1) {
    const flat = new Float32Array(POLICY + 1);
    flat[POLICY] = 0.25;
    g.classify(flat, many);
  }
  g.finish();
});

unitCase('three flat of many', 'net_degenerate', () => {
  // ...and 3 is not. This is the other side of the same boundary.
  const g = createNetGuard(POLICY);
  const many = maskOf(60);
  for (let i = 0; i < 20; i += 1) g.classify(varying(), many);
  for (let i = 0; i < 3; i += 1) {
    const flat = new Float32Array(POLICY + 1);
    flat[POLICY] = 0.25;
    g.classify(flat, many);
  }
  g.finish();
});

unitCase('narrow flat not evidence', 'no error', () => {
  // A narrow flat leaf is tolerated AND excluded from netLeaves, so it cannot pass
  // itself off as evidence-of-network.
  const g = createNetGuard(POLICY);
  const many = maskOf(60);
  for (let i = 0; i < 5; i += 1) g.classify(varying(), many);
  const flat = new Float32Array(POLICY + 1);
  flat[POLICY] = 0.25;
  g.classify(flat, maskOf(2));
  g.finish();
  if (g.netLeaves !== 5 || g.narrowFlat !== 1) {
    throw new Error(`netLeaves=${g.netLeaves} narrowFlat=${g.narrowFlat}, want 5 and 1`);
  }
});

// A PROPERTY, not an example: `doomed` must never trip without `finish()` throwing.
//
// `doomed` exists to cut a search short as soon as the network is known to be broken. If
// any state could set it while finish() stayed silent, the caller would break out of the
// leaf loop early and then SERVE the move -- a search truncated to a handful of
// simulations, presented as a full one. That is a worse failure than the ones this module
// was written to catch, and it is not something an example-based case would find, so it
// is checked exhaustively over every short sequence of leaf kinds plus long runs of each.
{
  const kinds = ['informed', 'flatWide', 'flatNarrow', 'blind', 'maskEmpty', 'const'];
  const apply = (g, kind, i) => {
    if (kind === 'informed') return g.classify(varying(), maskOf(60));
    if (kind === 'flatWide') { const f = new Float32Array(POLICY + 1); f[POLICY] = 0.25;
                               return g.classify(f, maskOf(60)); }
    if (kind === 'flatNarrow') { const f = new Float32Array(POLICY + 1); f[POLICY] = 0.25;
                                 return g.classify(f, maskOf(2)); }
    if (kind === 'blind') { const n = new Float32Array(POLICY + 1).fill(Number.NaN);
                            n[POLICY] = 0.25; return g.classify(n, maskOf(60)); }
    if (kind === 'maskEmpty') return g.classify(varying(), maskOf(0));
    const c = new Float32Array(POLICY + 1);           // 'const': same vector every leaf
    for (let k = 0; k < POLICY; k += 1) c[k] = Math.sin(k * 0.7) * 2;
    c[POLICY] = 0.25;
    return g.classify(c, maskOf(60));
  };
  const seqs = [];
  const build = (pre) => {
    if (pre.length) seqs.push([...pre]);
    if (pre.length === 4) return;
    for (const k of kinds) build([...pre, k]);
  };
  build([]);
  for (const k of kinds) for (const n of [8, 12, 20]) seqs.push(Array(n).fill(k));

  let violations = 0;
  for (const seq of seqs) {
    const g = createNetGuard(POLICY);
    let threwAtLeaf = false;
    for (let i = 0; i < seq.length; i += 1) {
      try { apply(g, seq[i], i); } catch { threwAtLeaf = true; break; }
      if (g.doomed) break;
    }
    if (threwAtLeaf) continue;
    const wasDoomed = g.doomed;
    let finishThrew = false;
    try { g.finish(); } catch { finishThrew = true; }
    if (wasDoomed && !finishThrew) violations += 1;
  }
  const ok = violations === 0;
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${'doomed => refuses'.padEnd(22)} -> `
    + `${seqs.length} sequences, ${violations} violation(s)`);
  if (!ok) { console.log('         a truncated search could be served as a full one'); failures += 1; }
}

// ------------------------------------------------------- server cases
for (const c of CASES) {
  mode = c.mode;
  const { status, body } = await bestmove();
  const err = body && body.error ? String(body.error) : '';
  let ok;
  let saw;
  if (c.want === 'move') {
    // sims === SIMS matters: a mutation that made `doomed` trip spuriously served a
    // move from a search truncated to 3 of 16 simulations and this case still passed,
    // because it only looked for a legal action and netLeaves > 0.
    ok = status === 200 && body && body.action && body.netLeaves > 0 && body.sims === SIMS;
    saw = ok ? `move served, netLeaves=${body.netLeaves}/${body.leaves}, sims=${body.sims}`
             : `status ${status} sims=${body && body.sims} ${err || JSON.stringify(body)}`;
  } else {
    ok = status === 500 && err.startsWith(c.want);
    saw = status === 200 ? `SERVED A MOVE (${JSON.stringify(body.action)})` : `${status} ${err}`;
  }
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${c.mode.padEnd(8)} -> ${saw}`);
  if (!ok) { console.log(`         expected ${c.want}: ${c.why}`); failures += 1; }
}

server.kill('SIGTERM');
stub.close();
if (failures) {
  console.log(`\n${failures} case(s) failed. Server log:\n${serverLog}`);
  process.exit(1);
}
console.log('\nall cases pass: a live, input-reading network is required for a move to be\nserved -- see the docstring for what these checks do NOT establish');
