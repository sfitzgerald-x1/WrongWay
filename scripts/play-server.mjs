#!/usr/bin/env node
/**
 * Local server for playing against a trained checkpoint.
 *
 *     # 1. the network (needs the training venv, and MPS for sane latency)
 *     cd ../wrongway-training && .venv/bin/python local-infer-mps.py \
 *          --checkpoint /tmp/ww-play/model.pt --port 8099
 *     # 2. this
 *     node scripts/play-server.mjs --port 8177 --infer http://127.0.0.1:8099
 *
 * The split exists because the JS forward runtime on this branch is sized for the
 * 6x64 tower (`unsupported_blocks` on a 15x128 checkpoint), so the network has to
 * run in torch. The SEARCH still runs here, in wasm, driven one leaf at a time
 * through NormalDuelSearch's nextLeaf()/submit() interface -- which is what lets
 * an async HTTP evaluation sit in the middle of a simulation.
 *
 * Latency, measured on an M5 Max: ~3.0 ms per simulation end to end (1.9 ms of
 * that is the forward on MPS, the rest HTTP and tree). So a move costs roughly
 * 3 ms x simulations -- ~0.4 s at 128, ~1.5 s at 512, ~3 s at 1024. That is why
 * the simulation count is a slider in the UI rather than a constant: it is the
 * strength/patience dial, and the sims ladder says each doubling is worth roughly
 * 85 Elo up to 128.
 */

import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import {
  createInitialState, applyAction, legalActions, decodeAction,
  positionKey, validateState, normalizePosition
} from '../js/normal-duel-engine.mjs';
import { CONFIG_9X9 } from '../tests/support/nn-runtime-fixture.mjs';
import { createNetGuard } from './net-guard.mjs';
import { createStore } from './play-store.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(`--${n}`); return i === -1 ? d : argv[i + 1]; };
const PORT = Number(arg('port', 8177));
const INFER = String(arg('infer', 'http://127.0.0.1:8099')).replace(/\/$/, '');
// Loopback by DEFAULT: this serves a game with no authentication, so exposing it on
// a network has to be an explicit choice rather than something that happens because
// a port was free. Pass --host 0.0.0.0 to publish it.
const HOST = String(arg('host', '127.0.0.1'));
const CONFIG = CONFIG_9X9;
/**
 * Where results live. In the pod that is the shared volume, so they outlive the pod.
 *
 * But /shared exists only in the cluster, and hardcoding it as the default made the
 * server UNSTARTABLE anywhere else: mkdir '/shared/wwplay/data' fails under a
 * read-only root, so the process died on boot. That is exactly what happened -- the
 * supervisor could never restart the local server, it only ever reported the
 * long-lived hand-started process as "already up", and the moment that process went
 * away the site went down with no store to come back to.
 *
 * So the default is now conditional on /shared actually being there, and the local
 * fallback is under $HOME rather than /tmp, which macOS prunes.
 */
const defaultDataDir = existsSync('/shared')
  ? '/shared/wwplay/data'
  : path.join(process.env.HOME || '/tmp', 'wwplay/data');
// Google sign-in turns itself on only when a client id is configured AND the site is
// served over HTTPS.
const store = createStore({
  dir: arg('data', process.env.WW_PLAY_DATA || defaultDataDir),
  googleClientId: process.env.WW_GOOGLE_CLIENT_ID || '',
  httpsOrigin: process.env.WW_HTTPS_ORIGIN === '1'
});

// ---------------------------------------------------------------- wasm loading
async function loadWasm() {
  const rel = `${ROOT}/rust/target/wasm-candidate/release`;
  const wasmPath = `${rel}/normal-duel-wasm_bg.wasm`;
  const wasm = await import(`${rel}/normal-duel-wasm.mjs`);
  const bytes = readFileSync(wasmPath);
  const instance = await wasm.default({ module_or_path: bytes });
  const memory = instance.memory ?? wasm.__wasm?.memory ?? wasm.memory;
  if (!memory || !(memory.buffer instanceof ArrayBuffer)) {
    throw new Error('wasm memory not reachable');
  }
  const build = createHash('sha256').update(bytes).digest('hex').slice(0, 16);
  return { wasm, memory, build, bytes: statSync(wasmPath).size };
}

// -------------------------------------------------------------- the evaluator
// One keep-alive agent for the whole process: a 512-simulation move is 512
// requests, and a fresh connection each time would dominate the measurement.
// noDelay: without TCP_NODELAY a 512-simulation move cost 25.6 s on Linux against
// 8.8 ms of actual network time -- ~40 ms of delayed-ACK stall per simulation. Both
// ends have to set it; one is not enough.
const agent = new http.Agent({ keepAlive: true, maxSockets: 4, noDelay: true });
const inferUrl = new URL(`${INFER}/infer`);

/**
 * The features MUST be copied out of wasm memory before any await.
 *
 * `Buffer.from(view.buffer, ...)` is a zero-copy window onto the wasm heap. Any wasm
 * allocation can grow that heap, which DETACHES the old ArrayBuffer -- and every
 * concurrent game shares one wasm instance here. When it detaches mid-request the
 * body write truncates, the server blocks forever waiting for bytes that never come,
 * and the move hangs with no error anywhere. A 3.2 KB copy against a multi-ms forward
 * costs nothing. Found by a search-scaling run where it hung 1 game for 90 minutes at
 * 0% CPU, and it is the likeliest cause of a game freezing mid-turn on the site.
 */
function inferOne(features) {
  // copyBytesFrom, NOT Buffer.from: the latter coerces each float to a single byte
  // and silently sends a body a quarter of the right size.
  const body = Buffer.copyBytesFrom(features);   // copy NOW, before any await
  return new Promise((resolve, reject) => {
    const req = http.request({
      agent, hostname: inferUrl.hostname, port: inferUrl.port,
      path: inferUrl.pathname, method: 'POST',
      headers: { 'content-type': 'application/octet-stream', 'content-length': body.length }
    }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        if (res.statusCode !== 200) return reject(new Error(`infer ${res.statusCode}`));
        const buf = Buffer.concat(chunks);
        resolve(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
      });
    });
    req.on('socket', (sock) => sock.setNoDelay(true));
    // Belt and braces: if a request ever does stall, fail it rather than hanging the
    // turn. The client already retries.
    req.setTimeout(60000, () => req.destroy(new Error('infer_timeout')));
    req.on('error', reject);
    req.end(body);
  });
}

// ------------------------------------------------------------------ AI move
/**
 * EVERY served move comes from the network, or no move is served at all.
 *
 * There is no heuristic path in here to fall back TO, but that alone does not make
 * the guarantee: this search returns a legal move even when the network's
 * contribution is worthless, and it has done exactly that twice. Once because
 * `pendingLeafMask()` was never called, so an all-zero policy was submitted and the
 * search ran with no priors; once because a wrong-sized body made the response
 * garbage. Both times the move was legal, the timings looked right, the root value
 * looked plausible, and the bot was simply weak. Nothing in the response said so.
 *
 * So the net's contribution is checked rather than assumed, and a violation throws
 * instead of returning a move:
 *
 *   net_bad_shape      the response must be exactly policyLen + 1 floats
 *   net_bad_value      the value must be finite and inside [-1, 1]; a NaN used to be
 *                      clamped to 0 at every leaf, which degrades the search to
 *                      priors-plus-visit-counts while every other signal looks fine
 *   net_mask_empty     a leaf handed to us always has a legal move, so an empty mask
 *                      means pendingLeafMask() is not filling the buffer
 *   net_degenerate     a leaf with legal moves must gain prior mass, and the legal
 *                      logits must not be flat (up to two tolerated, refused in bulk)
 *   net_never_consulted no leaf was network-informed
 *   net_input_ignored  every leaf returned bit-identical logits, so the network is
 *                      not reading the position it was given
 *
 * WHAT THIS DOES NOT PROVE. These are all checks on the RESPONSE, so they establish
 * that a live network answered plausibly and read its input -- not that it read the
 * CORRECT input. Request-side corruption (the Buffer.from bug sent a body a quarter of
 * the right size) yields a well-shaped varying policy for the wrong position, and only
 * the inference server's own length check catches that. `netLeaves` is evidence, not a
 * proof of correctness.
 */
async function aiMove({ wasm, memory }, state, sims, seed) {
  const t0 = performance.now();
  const search = new wasm.NormalDuelSearch(
    JSON.stringify(CONFIG), JSON.stringify(state),
    JSON.stringify({ simulations: sims, maxConsidered: 12, cPuct: 1.25, seed })
  );
  const guard = createNetGuard(search.policyLen());
  try {
    while (!search.isDone()) {
      search.nextLeaf();
      if (search.isDone()) break;
      // Views are rebuilt every leaf on purpose: any wasm allocation can grow the
      // heap and detach an old ArrayBuffer, and a stale view then reads freed
      // memory rather than throwing.
      const features = new Float32Array(memory.buffer, search.featuresPtr(), search.featuresLen());
      const out = await inferOne(features);

      // submit() wants NON-NEGATIVE PROBABILITIES over the whole policy vector, not
      // the network's raw logits, and it validates every entry -- including ones behind
      // illegal codes. The guard does that conversion and records what the network
      // actually contributed at this leaf.
      //
      // pendingLeafMask() FILLS the mask buffer; maskPtr() only says where it lives.
      // Without this call the buffer stays zero, the masked softmax finds no legal
      // entry, an ALL-ZERO policy gets submitted (which submit() accepts), and the
      // search runs with no priors at all -- picking near-arbitrary moves while every
      // latency number and the root value still look plausible.
      //
      // Pointers are re-read after the await because any wasm allocation can grow the
      // heap and detach the buffers these views were built on.
      const policyLen = search.policyLen();
      search.pendingLeafMask();
      const mask = new Float32Array(memory.buffer, search.maskPtr(), policyLen);
      const { probs, value } = guard.classify(out, mask);
      new Float32Array(memory.buffer, search.policyPtr(), policyLen).set(probs);
      search.submit(value);

      // Stop as soon as the network is known to be broken. Waiting for the loop to end
      // meant a degenerate network at 4096 sims completed all 4096 round trips before
      // refusing -- and the app retries 12 times, so ~50k pointless forwards per turn.
      if (guard.doomed) break;
    }
    // Refuse rather than return. A caller cannot tell a priorless move from a real one
    // by looking at it, so this is the only place the distinction can be made.
    guard.finish();
    const code = search.actionCode();
    const used = search.simulationsUsed();
    const rootValue = search.rootValue();
    return {
      action: decodeAction(CONFIG, code),
      ms: Math.round(performance.now() - t0),
      sims: used, leaves: guard.leaves, rootValue,
      netLeaves: guard.netLeaves   // evidence-of-network, carried to the caller and log
    };
  } finally {
    search.free();
  }
}

// ------------------------------------------------------------------ the game
const games = new Map();
let nextId = 1;
// Only /dev uses these, and nothing ever removed them: a long-lived server accumulated
// one entry per game it had ever hosted. Bounded by dropping the oldest, which is safe
// because a game is addressed by id and an evicted one simply reads as unknown.
const MAX_GAMES = 200;
function freshGame(humanSide, sims) {
  const id = String(nextId++);
  games.set(id, { id, state: createInitialState(CONFIG), humanSide, sims, history: [] });
  while (games.size > MAX_GAMES) games.delete(games.keys().next().value);
  return games.get(id);
}
function view(g, extra = {}) {
  return {
    id: g.id, humanSide: g.humanSide, sims: g.sims,
    state: g.state, legal: legalActions(CONFIG, g.state),
    outcome: g.state.outcome, history: g.history.slice(-8), ...extra
  };
}

// ------------------------------------------------------------------- HTTP
const wasmBits = await loadWasm();
console.log(`[play] wasm ${wasmBits.build} (${wasmBits.bytes} bytes)`);
console.log(`[play] network at ${INFER}`);
{
  const st = store.stats(); const g = store.googleReady();
  console.log(`[play] store ${st.dir}: ${st.users} users, ${st.games} games`);
  console.log(`[play] google sign-in: ${g.enabled ? 'enabled' : `disabled (missing ${g.missing.join(', ')})`}`);
}

// Read on demand, NOT at boot. A dev-only asset must not be able to stop the server
// from starting: this was a top-level readFileSync, so a missing play-ui.html took the
// whole site down rather than just the /dev route.
const devBoard = () => {
  try { return readFileSync(path.join(HERE, 'play-ui.html'), 'utf8'); }
  catch { return null; }
};

function json(res, code, obj) {
  const body = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': body.length });
  res.end(body);
}
function readBody(req) {
  return new Promise((resolve) => {
    const c = []; req.on('data', (d) => c.push(d));
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(c).toString() || '{}')); } catch { resolve({}); } });
  });
}

// ---------------------------------------- serving the real app, not a mock board
//
// The app in this repo IS the game, with the board, the rules UI and the wall
// interaction already built. Serving it here and answering /api/bestmove means the
// network plugs in as another difficulty rather than needing a second board.
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.webmanifest': 'application/manifest+json',
  '.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.woff2': 'font/woff2'
};
function serveStatic(req, res) {
  const clean = decodeURIComponent(req.url.split('?')[0]);
  // Contain the path: a served file must resolve inside the repo.
  const full = path.resolve(ROOT, `.${clean}`);
  // ROOT + sep, not ROOT: a bare prefix test also matches a SIBLING directory whose
  // name merely extends the repo's (".../ww-play-guard-notes/secret").
  if (full !== ROOT && !full.startsWith(ROOT + path.sep)) {
    res.writeHead(403); return res.end('forbidden');
  }
  let body;
  try { body = readFileSync(full); } catch { return false; }
  res.writeHead(200, { 'content-type': MIME[path.extname(full)] || 'application/octet-stream',
                       'content-length': body.length, 'cache-control': 'no-store' });
  res.end(body);
  return true;
}

/**
 * Build a valid engine state from a bare position.
 *
 * The app tracks pawns/walls/stock but not the engine's positionKey, ply parity or
 * repetition history, so those are reconstructed. Ply is chosen only for its
 * PARITY -- validateState requires the ply to agree with whose turn it is.
 *
 * Repetition history is deliberately NOT reconstructed: the app's own move list
 * would have to be replayed to do it properly. The consequence is that the search
 * cannot see a threefold-repetition draw coming, which affects move choice only in
 * already-drawn-ish lines. Worth knowing rather than hiding.
 */
/**
 * Rebuild the engine state by REPLAYING the app's move history.
 *
 * Reconstructing a state from a bare position does not work: validateState ties
 * `ply`, `historyStartPly` and each side's stock spend together (historyStartPly is
 * the ply after the last wall placement, and must be 0 exactly when no wall is on
 * the board), so a guessed ply is rejected as invalid_state. Replaying gets all of
 * that right for free -- and, unlike a reconstruction, it also carries the true
 * repetition counts, so the search can see a threefold draw coming.
 *
 * The replayed position is then CHECKED against what the client believes it is. A
 * mismatch means the history is incomplete for this mode (hammer/chaos modes
 * record extra action kinds), and the right response is to refuse rather than
 * search a position the player is not looking at.
 */
function stateFromHistory(history, expect) {
  let state = createInitialState(CONFIG);
  for (const entry of history) {
    let action = null;
    if (entry && entry.type === 'move') action = { kind: 'pawn', to: { r: entry.r, c: entry.c } };
    else if (entry && entry.type === 'wall') action = { kind: 'wall', wall: entry.k };
    else continue;                 // emotes, and mode-specific kinds like 'break'
    try {
      state = applyAction(CONFIG, state, action);
    } catch (err) {
      // Say WHICH entry failed and what the board looked like. A bare
      // illegal_action from a 40-move replay is undebuggable, and this path
      // falls back silently in the client, so the log is the only evidence.
      err.detail = {
        failedAt: history.indexOf(entry), entry, action,
        ply: state.ply, turn: state.position.turn,
        pawns: state.position.pawns, walls: state.position.walls,
        stock: state.position.stock,
        historyKinds: history.map((h) => `${h && h.type}/${h && h.p}`).join(' ')
      };
      console.error('[play] replay failed:', JSON.stringify(err.detail));
      throw err;
    }
  }
  if (expect) {
    const p = state.position;
    const same = p.turn === expect.turn
      && p.pawns.A.r === expect.pawns.A.r && p.pawns.A.c === expect.pawns.A.c
      && p.pawns.B.r === expect.pawns.B.r && p.pawns.B.c === expect.pawns.B.c
      && p.stock.A === Number(expect.stock.A) && p.stock.B === Number(expect.stock.B)
      // The wall SET, not just its size. Comparing counts let a same-count,
      // different-keys divergence through, which is the one case where the server
      // would search a board the player is not looking at -- the exact thing this
      // check exists to prevent.
      && [...p.walls].map(String).sort().join('|')
         === Array.from(expect.walls || []).map(String).sort().join('|');
    if (!same) {
      const err = new Error('history_replay_mismatch');
      err.detail = { replayed: p, expected: expect };
      throw err;
    }
  }
  return state;
}

function stateFromPosition(raw) {
  // normalizePosition first: the engine has a canonical wall ordering and a
  // validated pawn/stock shape, and positionKey is only meaningful on that form.
  // Passing the app's own ordering through produced invalid_state.
  const position = normalizePosition(CONFIG, raw);
  const key = positionKey(CONFIG, position);
  const ply = position.turn === CONFIG.firstPlayer ? 0 : 1;
  const state = {
    position, positionKey: key, ply, historyStartPly: ply,
    repetitionCounts: [{ positionKey: key, count: 1 }],
    outcome: { kind: 'ongoing' }
  };
  validateState(CONFIG, state);
  return state;
}

http.createServer(async (req, res) => {
  try {
    const bearer = () => {
      const h = req.headers.authorization || '';
      return h.startsWith('Bearer ') ? h.slice(7) : '';
    };
    /**
     * Who is calling, according to the transport.
     *
     * THE HONEST THREAT MODEL, because an earlier version of this comment claimed the
     * opposite. `tailscale serve` injects Tailscale-User-* after WireGuard has
     * authenticated the device, and its proxied requests arrive on loopback. But the
     * loopback test does NOT establish that a request came from the proxy: when the
     * site is browsed directly at 127.0.0.1:8177 -- the documented local setup -- the
     * browser's own requests are also loopback, and Tailscale-User-* is not a
     * forbidden header name, so same-origin page JS can set it. On a machine where
     * someone can already reach this port, they can assert any identity.
     *
     * What is at stake is leaderboard integrity, not access: there is nothing else
     * behind this. It is left as-is for the tailnet deployment, where the only route in
     * is the proxy, and closed properly by setting WW_PLAY_PROXY_SECRET -- then a
     * matching X-Play-Proxy-Secret is required as well, which page JS cannot obtain.
     */
    const proxySecret = process.env.WW_PLAY_PROXY_SECRET || '';
    const tailnetUser = () => {
      const remote = String(req.socket.remoteAddress || '');
      const local = remote === '127.0.0.1' || remote === '::1' || remote === '::ffff:127.0.0.1';
      const login = req.headers['tailscale-user-login'];
      if (!local || !login) return null;
      if (proxySecret && req.headers['x-play-proxy-secret'] !== proxySecret) return null;
      return store.identify({ login, name: req.headers['tailscale-user-name'] });
    };
    const caller = () => tailnetUser() || store.verifyToken(bearer());
    if (req.method === 'GET' && req.url.startsWith('/api/whoami')) {
      // Diagnostic: what identity, if any, does the transport give us? Tailscale's
      // `serve` proxy injects Tailscale-User-* headers that tailscaled itself
      // verifies, so a browser cannot forge them -- but ONLY on requests that came
      // through the proxy. A request straight to the port has no such headers, which
      // is exactly what this endpoint is for distinguishing.
      const ts = {};
      for (const [k, v] of Object.entries(req.headers)) {
        if (k.toLowerCase().startsWith('tailscale-')) ts[k] = v;
      }
      return json(res, 200, {
        remote: req.socket.remoteAddress,
        tailscaleHeaders: ts,
        identified: Object.keys(ts).length > 0
      });
    }
    if (req.method === 'GET' && req.url.startsWith('/api/leaderboard')) {
      return json(res, 200, { ...store.leaderboard(10), auth: store.googleReady() });
    }
    if (req.method === 'GET' && req.url.startsWith('/api/me')) {
      const u = caller();
      return json(res, 200, {
        user: u ? store.publicUser(u) : null,
        // Tells the page whether to offer a sign-in at all: behind the tailnet proxy
        // there is nothing to sign in to.
        source: u && u.provider === 'tailscale' ? 'tailscale' : (u ? u.provider : null),
        auth: store.googleReady()
      });
    }
    if (req.method === 'POST' && req.url.startsWith('/api/signin/google')) {
      const b = await readBody(req);
      try { return json(res, 200, await store.signInGoogle(b.credential)); }
      catch (e) { return json(res, e.code || 400, { error: e.message }); }
    }
    if (req.method === 'POST' && req.url.startsWith('/api/signin')) {
      const b = await readBody(req);
      try { return json(res, 200, store.signInGuest(b.name)); }
      catch (e) { return json(res, e.code || 400, { error: e.message }); }
    }
    if (req.method === 'POST' && req.url.startsWith('/api/result')) {
      const u = caller();
      if (!u) return json(res, 401, { error: 'not_identified' });
      const b = await readBody(req);
      try { return json(res, 200, { ok: true, game: store.recordGame(u, b) }); }
      catch (e) { return json(res, e.code || 400, { error: e.message }); }
    }
    if (req.method === 'POST' && req.url.startsWith('/api/bestmove')) {
      const b = await readBody(req);
      const sims = Math.max(1, Math.min(4096, Number(b.sims) || 128));
      // The network was trained on ONE board: the 9x9 duel map, stock 10, A first.
      // That IS the app's default (selectedMap starts at 'duel'), so this guard is a
      // safety net rather than a common path. The board it excludes is the RACE mode
      // -- internally map==='classic' -- which is 9x13 with both pawns on the bottom
      // row and a shared goal row: a different game whose moves are not encodable in
      // this 209-move policy. Blitz duel is 7x7 and likewise out.
      const board = b.board || {};
      if (board.map && board.map !== 'duel') {
        return json(res, 400, { error: 'unsupported_board',
          detail: `the network plays the 9x9 duel board; this game is '${board.map}' (race mode)` });
      }
      if (board.duelSize && board.duelSize === 'blitz') {
        return json(res, 400, { error: 'unsupported_board',
          detail: 'blitz is 7x7; the network plays 9x9' });
      }
      if (board.initialStock !== undefined && Number(board.initialStock) !== CONFIG.initialStock.A) {
        return json(res, 400, { error: 'unsupported_board',
          detail: `the network was trained with ${CONFIG.initialStock.A} walls a side, this game has ${board.initialStock}` });
      }
      if (board.chaos || board.hammer || board.drop) {
        return json(res, 400, { error: 'unsupported_board',
          detail: 'chaos/hammer/drop modes add actions the trained rules do not contain' });
      }
      const state = stateFromHistory(b.history || [], b.expect);
      const mv = await aiMove(wasmBits, state, sims, Number(b.seed) || 1);
      // Log every served move. Without this, "the app never asked" and "the app
      // asked and it failed" look identical from here -- and they need completely
      // different fixes.
      console.log(`[play] move ply=${state.ply} sims=${sims} ${mv.ms}ms `
        + `net=${mv.netLeaves}/${mv.leaves} v=${mv.rootValue.toFixed(3)} `
        + `-> ${JSON.stringify(mv.action)}`);
      return json(res, 200, { ...mv, turn: state.position.turn, ply: state.ply });
    }
    if (req.method === 'GET' && req.url.startsWith('/sw.js')) {
      // The app is a PWA and registers a service worker that caches the shell. On
      // a local play server that means an edit to index.html can be invisible
      // behind a stale cache, so this serves a worker that unregisters itself and
      // drops any cache it already made. The repo's real sw.js is untouched.
      const body = Buffer.from(
        "self.addEventListener('install',()=>self.skipWaiting());\n"
        + "self.addEventListener('activate',(e)=>{e.waitUntil((async()=>{\n"
        + "  for (const k of await caches.keys()) await caches.delete(k);\n"
        + "  await self.registration.unregister();\n"
        + "  for (const c of await self.clients.matchAll()) c.navigate(c.url);\n"
        + "})());});\n");
      res.writeHead(200, { 'content-type': 'text/javascript; charset=utf-8',
                           'content-length': body.length, 'cache-control': 'no-store' });
      return res.end(body);
    }
    if (req.method === 'GET' && (req.url === '/' || req.url.startsWith('/?'))) {
      // The home page IS the game. play.html is a standalone site: it borrows the
      // app's visual language (board skin, stone colours, dark palette) and the
      // rules engine, and none of its player-vs-player architecture -- no auth, no
      // lobbies, no websocket, no Firebase. The full app stays reachable at /app
      // for comparison rather than being deleted.
      if (serveStatic({ ...req, url: '/play.html' }, res) !== false) return;
    }
    if (req.method === 'GET' && (req.url === '/app' || req.url.startsWith('/app?'))) {
      if (serveStatic({ ...req, url: '/index.html' }, res) !== false) return;
    }
    if (req.method === 'GET' && req.url.startsWith('/dev')) {
      // The minimal debug board, kept because it exercises the search directly
      // with no app state in the way -- it is how the all-zero-policy bug was found.
      const board = devBoard();
      if (!board) { res.writeHead(404); return res.end('play-ui.html not present'); }
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      return res.end(board);
    }
    if (req.method === 'GET' && req.url.startsWith('/api/health')) {
      // res.ok is checked: a 503 with a JSON body, or a {"ok":false} from the network's
      // own healthz, both used to report this server as healthy.
      const r = await fetch(`${INFER}/healthz`)
        .then((x) => (x.ok ? x.json() : null))
        .then((j) => (j && j.ok === false ? null : j))
        .catch(() => null);
      // Checkpoint metadata, if the deployer left a meta.json beside the weights.
      // Read per request so swapping checkpoints does not need a restart, and kept
      // OUT of the page: hardcoding an Elo in the HTML goes stale the moment a
      // stronger checkpoint is deployed, and a stale strength claim is worse than none.
      let meta = null;
      for (const c of [process.env.WW_PLAY_META, '/shared/wwplay/meta.json',
                       '/tmp/ww-play/meta.json']) {
        if (!c) continue;
        try { meta = JSON.parse(readFileSync(c, 'utf8')); break; } catch { /* next */ }
      }
      // ok tracks the NETWORK, not this process. This server is useless without the
      // net -- it will not substitute another opponent -- so "the HTTP layer is up"
      // is not a health answer worth giving.
      return json(res, 200, { ok: Boolean(r), wasm: wasmBits.build, network: r, meta });
    }
    if (req.method === 'POST' && req.url.startsWith('/api/new')) {
      const b = await readBody(req);
      const g = freshGame(b.humanSide === 'B' ? 'B' : 'A', Number(b.sims) || 128);
      // If the human is B, A moves first and A is the machine.
      if (g.state.position.turn !== g.humanSide) {
        const mv = await aiMove(wasmBits, g.state, g.sims, g.state.ply + 1);
        g.state = applyAction(CONFIG, g.state, mv.action);
        g.history.push({ by: 'ai', ...mv });
        return json(res, 200, view(g, { ai: mv }));
      }
      return json(res, 200, view(g));
    }
    if (req.method === 'POST' && req.url.startsWith('/api/move')) {
      const b = await readBody(req);
      const g = games.get(String(b.id));
      if (!g) return json(res, 404, { error: 'no such game' });
      if (g.state.outcome.kind !== 'ongoing') return json(res, 400, { error: 'game over' });
      if (g.state.position.turn !== g.humanSide) return json(res, 400, { error: 'not your turn' });
      if (Number(b.sims)) g.sims = Number(b.sims);
      // Trust the engine, not the client: the action must be in the legal list.
      const legal = legalActions(CONFIG, g.state);
      const want = JSON.stringify(b.action);
      if (!legal.some((a) => JSON.stringify(a) === want)) {
        return json(res, 400, { error: 'illegal action' });
      }
      g.state = applyAction(CONFIG, g.state, b.action);
      g.history.push({ by: 'human', action: b.action });
      if (g.state.outcome.kind !== 'ongoing') return json(res, 200, view(g));
      const mv = await aiMove(wasmBits, g.state, g.sims, g.state.ply + 1);
      g.state = applyAction(CONFIG, g.state, mv.action);
      g.history.push({ by: 'ai', ...mv });
      return json(res, 200, view(g, { ai: mv }));
    }
    if (req.method === 'GET' && serveStatic(req, res) !== false) return;
    res.writeHead(404); res.end('not found');
  } catch (err) {
    console.error('[play]', err);
    json(res, 500, { error: String(err && err.message || err) });
  }
}).listen(PORT, HOST, () => {
  console.log(`[play] listening on http://${HOST}:${PORT}`
    + (HOST === '127.0.0.1' ? '' : '  (published beyond loopback)'));
});
