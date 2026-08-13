#!/usr/bin/env node
/**
 * Prove the play server refuses to serve a move the network did not shape.
 *
 *     node scripts/check-net-guard.mjs
 *
 * The guarantee under test is "every served move comes from the neural net, or no
 * move is served". The hard part is that a broken network still yields a LEGAL move:
 * the search happily runs on uniform or absent priors, returns a plausible action and
 * a plausible root value, and plays like a weak bot. That has shipped twice here. A
 * test that only checks for a 200 and a legal action cannot see it.
 *
 * So this stands up a stub inference server that is broken in a specific way, and
 * asserts the play server REFUSES. A test that never observes the refusal is not
 * evidence, so the healthy case is included as the control -- if it fails, the
 * refusals below prove nothing except that everything is broken.
 *
 * Runs against its own port and its own data dir: pointing it at the live store
 * would write test games onto the real leaderboard.
 */

import { spawn } from 'node:child_process';
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
    let out;
    if (mode === 'zeros') {
      // The classic silent failure: every legal logit identical, so the masked
      // softmax turns it into a tidy uniform prior and the search flies blind.
      out = new Float32Array(POLICY_LEN + 1);
    } else if (mode === 'short') {
      out = new Float32Array(POLICY_LEN);          // value float missing
    } else {
      out = new Float32Array(POLICY_LEN + 1);
      for (let i = 0; i < POLICY_LEN; i += 1) out[i] = Math.sin(i * 0.7) * 2;
      out[POLICY_LEN] = 0.25;
    }
    const body = Buffer.copyBytesFrom(out);
    res.writeHead(200, { 'content-type': 'application/octet-stream',
                         'content-length': body.length });
    res.end(body);
  });
});
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
    why: 'control: a varying policy must produce a served move, else the refusals below mean nothing' },
  { mode: 'zeros', want: 'net_degenerate',
    why: 'flat logits are a broken net laundered into a uniform prior' },
  { mode: 'short', want: 'net_bad_shape',
    why: 'a wrong-length response is someone else\'s bytes' }
];

let failures = 0;
for (const c of CASES) {
  mode = c.mode;
  const { status, body } = await bestmove();
  const err = body && body.error ? String(body.error) : '';
  let ok;
  let saw;
  if (c.want === 'move') {
    ok = status === 200 && body && body.action && body.netLeaves > 0;
    saw = ok ? `move served, netLeaves=${body.netLeaves}/${body.leaves}`
             : `status ${status} ${err || JSON.stringify(body)}`;
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
console.log('\nall cases pass: a move is served only when the network shaped it');
