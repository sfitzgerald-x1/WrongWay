/**
 * Shared machinery for the deep-target distillation experiment (Bet 2).
 *
 * THE QUESTION. Self-play at 512 simulations produces a policy target. The same
 * net at 4096 simulations plays measurably better than itself at 512, so the deep
 * search holds signal the net has not absorbed. Can that signal be distilled
 * through the POLICY TARGET alone, with the positions, the outcomes and the data
 * volume all held fixed? Arm g1 already tried raising self-play sims and scored
 * worse, but it halved games per iteration at the same time, so depth and volume
 * moved together. This pipeline separates them: one set of positions, one set of
 * `z`, two policy targets.
 *
 * WHY THE SEARCH IS DRIVEN FROM HERE rather than through `/api/bestmove`. That
 * endpoint returns the CHOSEN MOVE and never the visit distribution, and the
 * policy target IS the visit distribution -- so the play server cannot express
 * this experiment's output. It is still the design's model: a position is
 * addressed by its MOVE HISTORY and reconstructed by replay, because
 * `validateState` ties ply, historyStartPly and each side's wall spend together
 * and a hand-assembled board is rejected as invalid_state.
 *
 * WHY LOCK-STEP BATCHING IS THE WHOLE DESIGN. `NormalDuelSearch` holds at most
 * ONE outstanding evaluation, so a single search is a serial chain of batch-1
 * forwards -- latency-bound, and a 4096-simulation search costs 4096 round trips.
 * Measured against `local-infer-mps.py` on an M5 Max:
 *
 *     batch    1 ->    343 positions/s     (2.92 ms/call)
 *     batch   16 ->  6,075 positions/s
 *     batch   32 ->  8,826 positions/s
 *     batch   64 -> 11,686 positions/s     (5.48 ms/call)
 *     batch  128 -> 12,371 positions/s
 *
 * A 34x throughput difference. Concurrency alone does NOT buy it: the server
 * holds one lock around the MPS forward (Metal is not thread safe), so N
 * concurrent batch-1 requests still serialise -- measured 471 positions/s at
 * concurrency 4. The win comes from putting many searches' pending leaves into
 * ONE forward, which is what `runLockStep` below does: every active search is
 * advanced to its next leaf, all their feature vectors go out as a single
 * request, and every one of them is submitted before the next step. So the batch
 * size is the number of concurrent searches, and it stays full because a slot
 * that finishes is refilled inside the same step.
 */

import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';

import { applyAction, createInitialState, decodeAction } from '../js/normal-duel-engine.mjs';
import { encodeState, legalMaskFloat } from '../js/normal-duel-nn-encoding.mjs';
import { CONFIG_9X9 } from '../tests/support/nn-runtime-fixture.mjs';
import { createNetGuard } from './net-guard.mjs';

export const CONFIG = CONFIG_9X9;
export const CONFIG_JSON = JSON.stringify(CONFIG);
export const FEATURE_LEN = 810;
export const POLICY_LEN = 209;
/** features | policyTarget | legalMask | z, the on-disk shard record. */
export const RECORD_FLOATS = FEATURE_LEN + POLICY_LEN + POLICY_LEN + 1;
/** What the collect step stores per ply so the shard step needs no wasm: features | mask. */
export const SIDECAR_FLOATS = FEATURE_LEN + POLICY_LEN;

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(HERE, '..');

// ------------------------------------------------------------------ wasm
/**
 * Load the search engine, from an OVERRIDABLE directory.
 *
 * The default local build is not necessarily the engine production runs, and a
 * search difference changes which move is picked -- so a flip rate measured against
 * the wrong build answers a question about the wrong build.
 *
 * Observed 2026-08-16: this checkout sits on `scott/rust-batched-puct`, which is
 * `puct-az-tree-v1` and records the VISIT DISTRIBUTION as the self-play target.
 * `main` is v2/v3 and records the Gumbel IMPROVED POLICY instead -- merged as
 * "Record the completed-Q improved policy as the self-play target" (#25) and
 * "Adopt the mctx qtransform and v_mix completion in the PUCT root (v3)" (#26).
 * Production h2 reports `engineBuild 4470335d`, matching neither this branch's
 * build nor any local artefact until it was pulled deliberately.
 *
 * The lesson worth keeping: three separate claims were made today from reading
 * THIS branch and asserting them about production, and all three were wrong. The
 * cheap check that settles it is comparing the engine hash against the runlog's
 * `engineBuild`, which is why `build` is returned here and reported by every step.
 * (An earlier version of this comment stated production's boost span as
 * `(50 + maxN) * 0.1`; that came from a design doc, and `git show main` reports
 * `C_SCALE = 1.0`, so the constant was removed rather than restated unverified.)
 *
 * `WW_DISTILL_WASM_DIR` (or `--wasm`) therefore takes a directory holding a
 * `normal-duel-wasm_bg.wasm` plus its `normal-duel-wasm.mjs` bindings. The pair
 * must come from the SAME build: bindings describe the binary's exported shape,
 * and a mismatched pair fails as a missing export rather than as bad numbers.
 *
 * `build` is the binary's own hash and is reported by every step, so a corpus can
 * always be traced to the engine that produced it -- the runlog's `engineBuild`
 * exists for the same reason.
 */
export async function loadWasm(root = REPO_ROOT, dir = process.env.WW_DISTILL_WASM_DIR) {
  const rel = dir || `${root}/rust/target/wasm-candidate/release`;
  const wasmPath = `${rel}/normal-duel-wasm_bg.wasm`;
  const wasm = await import(`${rel}/normal-duel-wasm.mjs`);
  const bytes = readFileSync(wasmPath);
  const instance = await wasm.default({ module_or_path: bytes });
  const memory = instance.memory ?? wasm.__wasm?.memory ?? wasm.memory;
  if (!memory || !(memory.buffer instanceof ArrayBuffer)) throw new Error('wasm memory not reachable');
  return {
    wasm, memory,
    build: createHash('sha256').update(bytes).digest('hex').slice(0, 16),
    bytes: statSync(wasmPath).size
  };
}

// ------------------------------------------------------------- inference
/**
 * A BATCHED inference client. `infer(features, n)` sends n * 810 floats and
 * returns n * (209 + 1) floats: policy logits then value, per position.
 *
 * The body is a copy (`Buffer.copyBytesFrom`), never a view onto wasm memory: a
 * wasm allocation grows the heap and DETACHES the old ArrayBuffer, which
 * truncates the body mid-write and hangs the request forever. And
 * copyBytesFrom, not Buffer.from -- the latter coerces each float to one byte
 * and silently sends a quarter of the data.
 */
export function createInferClient({ url = 'http://127.0.0.1:8301', checkpoint = '', maxSockets = 4 } = {}) {
  const target = new URL(`${String(url).replace(/\/$/, '')}/infer`);
  const agent = new http.Agent({ keepAlive: true, maxSockets, noDelay: true });
  const stats = { calls: 0, positions: 0, ms: 0 };

  async function infer(features, n) {
    const body = Buffer.copyBytesFrom(features.subarray(0, n * FEATURE_LEN));
    const t0 = performance.now();
    const out = await new Promise((resolve, reject) => {
      const req = http.request({
        agent, hostname: target.hostname, port: target.port, path: target.pathname,
        method: 'POST',
        headers: {
          'content-type': 'application/octet-stream', 'content-length': body.length,
          ...(checkpoint ? { 'x-checkpoint': checkpoint } : {})
        }
      }, (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const buf = Buffer.concat(chunks);
          if (res.statusCode !== 200) return reject(new Error(`infer ${res.statusCode}: ${buf}`));
          resolve(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
        });
      });
      req.on('socket', (s) => s.setNoDelay(true));
      req.setTimeout(120000, () => req.destroy(new Error('infer_timeout')));
      req.on('error', reject);
      req.end(body);
    });
    stats.ms += performance.now() - t0;
    stats.calls += 1;
    stats.positions += n;
    if (out.length !== n * (POLICY_LEN + 1)) {
      throw new Error(`infer returned ${out.length} floats, want ${n * (POLICY_LEN + 1)}`);
    }
    return out;
  }
  return { infer, stats };
}

// -------------------------------------------------------- the lock-step runner
/**
 * Drive many searches through one batched evaluator.
 *
 * `sessions` is an iterator of session objects; each session yields a sequence of
 * searches (one per ply for a self-play game, exactly one for a relabel):
 *
 *     session.nextSpec()   -> { state, sims, considered, cPuct, seed } | null
 *     session.onResult(r)  -> void, where r carries features/mask/visits/rootValue
 *
 * `slots` is both the concurrency and the inference batch size.
 */
export async function runLockStep({
  wasm, memory, infer, sessions, slots = 32, cPuct = 1.25, onStep = null
}) {
  const iterator = sessions[Symbol.iterator] ? sessions[Symbol.iterator]() : sessions;
  const batch = new Float32Array(slots * FEATURE_LEN);
  const live = Array.from({ length: slots }, () => ({
    session: null, search: null, guard: null, spec: null, features: null, mask: null,
    scratchMask: new Float32Array(POLICY_LEN), t0: 0, leafIndex: 0
  }));
  const stats = { searches: 0, leaves: 0, steps: 0, maxBatch: 0, wasmBytes: 0 };
  let exhausted = false;

  function fill(slot) {
    for (;;) {
      if (!slot.session) {
        if (exhausted) return false;
        const next = iterator.next();
        if (next.done) { exhausted = true; return false; }
        slot.session = next.value;
      }
      const spec = slot.session.nextSpec();
      if (!spec) { slot.session = null; continue; }
      slot.search = new wasm.NormalDuelSearch(CONFIG_JSON, JSON.stringify(spec.state),
        JSON.stringify({
          simulations: spec.sims,
          maxConsidered: spec.considered,
          cPuct: spec.cPuct ?? cPuct,
          seed: spec.seed
        }));
      slot.guard = createNetGuard(slot.search.policyLen());
      slot.spec = spec;
      slot.features = null;
      slot.mask = null;
      slot.leafIndex = 0;
      slot.t0 = performance.now();
      return true;
    }
  }

  function finish(slot) {
    const search = slot.search;
    let result;
    try {
      // Positions this pipeline must PLAY but must not RECORD.
      //
      // Two shapes, both established by instrumenting the failure rather than reading
      // the error text -- two earlier fixes were written against guesses at it and both
      // were wrong:
      //
      //  1. Nothing to decide: one legal action, so the search short-circuits, hands out
      //     no leaf and spends no budget. Both halves of the signature are required --
      //     `simulationsUsed() === 0` alone could be a search that died before starting,
      //     and a missing leaf alone is the genuine bug the throw below catches.
      //
      //  2. Narrow-flat endgame: a real position (measured at ply 51, both players out
      //     of wall stock, one candidate holding all 512 visits, rootValue 0.9995) whose
      //     1-2 legal moves make the network's logits necessarily flat. The guard counts
      //     that leaf as `narrowFlat` and not as `informed`, so `informed === 0` fires.
      //     Three collection runs died here; it is not rare, just deep in the game where
      //     short runs never reach.
      //
      // Neither is a training example: a target over <= 2 legal moves is the same at
      // every search depth, so recording them would add identical rows to the shallow
      // and deep corpora and dilute every contrast statistic with guaranteed zeros.
      //
      // The shared guard is deliberately NOT loosened. Relaxing its `informed === 0`
      // test was tried and reverted: it broke `check-net-guard.mjs` and did not even
      // prevent the refusal, only downgraded it to `net_input_ignored`. Refusing is
      // correct for the play server, which must not move when it cannot tell "the net
      // answered" from "the net is dead". The asymmetry is the point: play refuses,
      // bulk generation skips.
      let skip = !slot.features && search.simulationsUsed() === 0;
      if (!skip) {
        try {
          slot.guard.finish();
        } catch (err) {
          // Every leaf narrow-flat => shape 2. Requires leaves > 0, so a genuinely
          // unconsulted network (0 leaves, 0 narrowFlat) still throws.
          if (err.code === 'net_never_consulted'
              && slot.guard.leaves > 0
              && slot.guard.leaves === slot.guard.narrowFlat) {
            skip = true;
          } else {
            // Anything else keeps the diagnostic, so the next unknown refusal is read
            // rather than guessed at.
            const vc = Array.from(search.visitCounts());
            const pairs = [];
            for (let i = 0; i < vc.length; i += 2) pairs.push([vc[i], vc[i + 1]]);
            console.log(`[diag] ${err.code || 'error'}: ${err.message}`);
            console.log(`[diag]   guard: leaves=${slot.guard.leaves} netLeaves=${slot.guard.netLeaves}`
              + ` narrowFlat=${slot.guard.narrowFlat} doomed=${slot.guard.doomed}`);
            console.log(`[diag]   search: simsUsed=${search.simulationsUsed()}`
              + ` actionCode=${search.actionCode()} rootValue=${search.rootValue()}`);
            console.log(`[diag]   featuresHandedOut=${slot.features ? 'yes' : 'no'}`
              + ` maskHandedOut=${slot.mask ? 'yes' : 'no'}`
              + ` visitPairs=${JSON.stringify(pairs.slice(0, 12))} totalPairs=${pairs.length}`);
            const totalVisits = pairs.reduce((s, [, n]) => s + n, 0);
            console.log(`[diag]   totalVisits=${totalVisits} spec=${JSON.stringify(slot.spec)}`);
            throw err;
          }
        }
        if (!slot.features && !skip) {
          throw new Error('search completed without handing out a leaf');
        }
      }
      const flat = search.visitCounts();
      const visits = [];
      for (let i = 0; i < flat.length; i += 2) visits.push([flat[i], flat[i + 1]]);
      result = {
        spec: slot.spec,
        forced: skip,
        features: slot.features,
        mask: slot.mask,
        visits,
        actionCode: search.actionCode(),
        rootValue: search.rootValue(),
        simsUsed: search.simulationsUsed(),
        leaves: slot.guard.leaves,
        netLeaves: slot.guard.netLeaves,
        ms: performance.now() - slot.t0
      };
    } finally {
      search.free();
      slot.search = null;
    }
    stats.searches += 1;
    slot.session.onResult(result);
  }

  for (;;) {
    const pending = [];
    for (const slot of live) {
      if (!slot.search && !fill(slot)) continue;
      // Advance to a leaf. A search that finishes here is finalised and the slot
      // refilled inside the same step, so the batch never thins out while there
      // is work left.
      for (;;) {
        slot.search.nextLeaf();
        if (!slot.search.isDone()) break;
        finish(slot);
        if (!fill(slot)) break;
      }
      if (!slot.search) continue;

      // EVERY wasm view is rebuilt here and copied out BEFORE the await below:
      // another slot's nextLeaf() in this same loop can grow the heap and detach
      // an older view, and a stale view reads freed memory rather than throwing.
      const featuresView = new Float32Array(memory.buffer, slot.search.featuresPtr(), slot.search.featuresLen());
      const index = pending.length;
      batch.set(featuresView, index * FEATURE_LEN);
      // pendingLeafMask() FILLS the buffer; maskPtr() only says where it is.
      // Skipping it submits an all-zero policy, which submit() accepts, and the
      // search then runs with no priors at all while every timing looks right.
      slot.search.pendingLeafMask();
      slot.scratchMask.set(new Float32Array(memory.buffer, slot.search.maskPtr(), POLICY_LEN));
      if (slot.leafIndex === 0) {
        // The FIRST leaf of a fresh search is the root (`next_leaf` hands out the
        // unexpanded root's features before it descends), so this is the record's
        // feature vector and legal mask. Asserted against the JS encoder by the
        // callers rather than assumed.
        slot.features = batch.slice(index * FEATURE_LEN, (index + 1) * FEATURE_LEN);
        slot.mask = slot.scratchMask.slice();
      }
      slot.leafIndex += 1;
      pending.push(slot);
    }
    if (pending.length === 0) break;

    const out = await infer(batch, pending.length);
    stats.steps += 1;
    stats.leaves += pending.length;
    stats.maxBatch = Math.max(stats.maxBatch, pending.length);
    for (let i = 0; i < pending.length; i += 1) {
      const slot = pending[i];
      const row = out.subarray(i * (POLICY_LEN + 1), (i + 1) * (POLICY_LEN + 1));
      const { probs, value } = slot.guard.classify(row, slot.scratchMask);
      // Pointers re-read after the await: the heap may have grown since.
      new Float32Array(memory.buffer, slot.search.policyPtr(), POLICY_LEN).set(probs);
      slot.search.submit(value);
      if (slot.guard.doomed) {
        throw new Error('net guard tripped mid-search; refusing to record data from a broken network');
      }
    }
    stats.wasmBytes = memory.buffer.byteLength;
    if (onStep) onStep(stats);
  }
  return stats;
}

// ------------------------------------------------------------------ targets
/**
 * `(code, visits)` pairs -> a 209-float policy target summing to 1.
 *
 * Mirrors `PuctResult::effective_visit_counts` plus the normalisation
 * `SelfPlayBatch` applies: a search with nothing to decide spends no simulations,
 * so the counts sum to zero and the played action carries the whole target. That
 * fallback is correct ONLY when there was one action to choose, so it is counted
 * and reported rather than silently taken.
 */
export function visitTarget(visits, actionCode) {
  const target = new Float32Array(POLICY_LEN);
  let total = 0;
  for (const [, n] of visits) total += n;
  if (total <= 0) {
    target[actionCode] = 1;
    return { target, degenerate: true };
  }
  for (const [code, n] of visits) target[code] = n / total;
  return { target, degenerate: false };
}

/** Shannon entropy in nats. */
export function entropyOf(target) {
  let h = 0;
  for (let i = 0; i < target.length; i += 1) {
    const p = target[i];
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

/** Total-variation distance between two distributions over the same support. */
export function totalVariation(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += Math.abs(a[i] - b[i]);
  return sum / 2;
}

/** KL(a || b) in nats, with b's zeros treated as infinite where a is positive. */
export function klDivergence(a, b) {
  let kl = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] <= 0) continue;
    if (b[i] <= 0) return Infinity;
    kl += a[i] * Math.log(a[i] / b[i]);
  }
  return kl;
}

// -------------------------------------------------------------- exploration
/** mulberry32: a small deterministic PRNG, so a run is reproducible from its seed. */
export function rng32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Sample an action code from `target^(1/T)`. Mirrors
 * `selfplay::sample_visit_temperature`, including the `T == 1` short circuit --
 * sampling from the TARGET rather than the raw counts is what keeps the played
 * move consistent with the label the shard carries.
 */
export function sampleFromTarget(target, temperature, roll) {
  const exponent = 1 / temperature;
  const weight = (p) => (temperature === 1 ? p : p ** exponent);
  let total = 0;
  for (let i = 0; i < target.length; i += 1) if (target[i] > 0) total += weight(target[i]);
  if (!(total > 0)) return null;
  let mark = roll * total;
  for (let code = 0; code < target.length; code += 1) {
    if (target[code] <= 0) continue;
    mark -= weight(target[code]);
    if (mark <= 0) return code;
  }
  for (let code = target.length - 1; code >= 0; code -= 1) if (target[code] > 0) return code;
  return null;
}

// ------------------------------------------------------------------ history
/** The `/api/bestmove` history record for an action taken by `mover`. */
export function historyEntry(mover, action) {
  return action.kind === 'pawn'
    ? { type: 'move', p: mover, r: action.to.r, c: action.to.c }
    : { type: 'wall', p: mover, k: action.wall };
}

/** Replay a `/api/bestmove`-shaped history into an engine state. */
export function stateFromHistory(history) {
  let state = createInitialState(CONFIG);
  for (const entry of history) {
    const action = entry.type === 'move'
      ? { kind: 'pawn', to: { r: entry.r, c: entry.c } }
      : { kind: 'wall', wall: entry.k };
    state = applyAction(CONFIG, state, action);
  }
  return state;
}

/**
 * Independent cross-check of a wasm search's root leaf: the JS encoder must
 * agree with it on the features, and the engine's own legal mask must agree on
 * the mask.
 *
 * This is the check that the deep search actually looked at the SAME position
 * the shallow one did. The features come from Rust and the comparison from JS, so
 * it is two implementations agreeing rather than one asserting about itself.
 */
export function crossCheckRoot(state, features, mask) {
  const jsFeatures = encodeState(CONFIG, state);
  const jsMask = legalMaskFloat(CONFIG, state);
  let featureDiff = 0;
  let featureBitExact = true;
  for (let i = 0; i < FEATURE_LEN; i += 1) {
    const d = Math.abs(features[i] - jsFeatures[i]);
    if (d !== 0) featureBitExact = false;
    if (d > featureDiff) featureDiff = d;
  }
  let maskMismatches = 0;
  let legalCount = 0;
  for (let i = 0; i < POLICY_LEN; i += 1) {
    if (jsMask[i] > 0) legalCount += 1;
    if ((mask[i] > 0) !== (jsMask[i] > 0)) maskMismatches += 1;
  }
  return { featureDiff, featureBitExact, maskMismatches, legalCount };
}

/** sha256 of a float32 array's bytes -- an identity for a feature vector. */
export function floatsHash(floats) {
  return createHash('sha256').update(Buffer.from(floats.buffer, floats.byteOffset, floats.byteLength))
    .digest('hex').slice(0, 24);
}

/** Decode an action code for logging. */
export function describeCode(code) {
  try { return JSON.stringify(decodeAction(CONFIG, code)); } catch { return `code:${code}`; }
}

// ------------------------------------------------------------------- CLI bits
export function parseArgs(argv, defaults) {
  const out = { ...defaults };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) throw new Error(`unexpected argument '${token}'`);
    const name = token.slice(2);
    if (!(name in defaults)) {
      throw new Error(`unknown flag --${name}; known: ${Object.keys(defaults).map((k) => `--${k}`).join(' ')}`);
    }
    const current = defaults[name];
    if (typeof current === 'boolean') { out[name] = true; continue; }
    const value = argv[i + 1];
    if (value === undefined) throw new Error(`--${name} needs a value`);
    i += 1;
    out[name] = typeof current === 'number' ? Number(value) : value;
    if (typeof current === 'number' && !Number.isFinite(out[name])) {
      throw new Error(`--${name}=${value} is not a number`);
    }
  }
  return out;
}

export function fmtDuration(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}min`;
  return `${(ms / 3600000).toFixed(2)}h`;
}
