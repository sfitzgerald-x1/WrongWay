#!/usr/bin/env node
/**
 * Step 2: re-search every collected position at a DEEPER budget.
 *
 * Each position is addressed by its move history and rebuilt by replay -- the
 * same contract `/api/bestmove` trusts, and the reason step 1 recorded histories
 * at all. The seed is the one the shallow search used, so the Gumbel draw and
 * therefore the considered set are the same: the only thing that changes is how
 * many simulations get spent inside that set.
 *
 * THE POSITION IDENTITY IS CHECKED, NOT ASSUMED. A replay that lands somewhere
 * else would produce a perfectly plausible target for the wrong board, and the
 * experiment would then be comparing two different position sets. So the deep
 * search's root features are compared against the hash step 1 recorded, against
 * the JS encoder, and its legal mask against the engine's -- three ways, because
 * this is the assumption the whole comparison rests on.
 *
 * Resumable: results are appended, and an `idx` already present is skipped. A
 * 4096-simulation pass over a full-size collection is hours long, and losing it
 * to a laptop sleep is not acceptable.
 *
 *   node scripts/distill-relabel.mjs --in ~/wwplay/distill/smoke --sims 4096 --slots 32
 */

import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import {
  POLICY_LEN, createInferClient, crossCheckRoot, entropyOf, floatsHash, fmtDuration,
  klDivergence, loadWasm, parseArgs, runLockStep, stateFromHistory, totalVariation, visitTarget
} from './distill-lib.mjs';

const args = parseArgs(process.argv.slice(2), {
  in: path.join(process.env.HOME || '/tmp', 'wwplay/distill/smoke'),
  sims: 4096,
  considered: 0,          // 0 = reuse the shallow pass's value (pure depth contrast)
  cpuct: 1.4,          // matches the collect step and production self-play
  slots: 32,
  seedmode: 'same',       // same | offset -- 'same' keeps the Gumbel draw identical
  limit: 0,               // 0 = every collected position
  tag: '',                // output suffix; defaults to deep-<sims>
  infer: 'http://127.0.0.1:8301',
  checkpoint: 'g2-iter-030',
  fresh: false
});

const tag = args.tag || `deep-${args.sims}${args.considered ? `-c${args.considered}` : ''}`;
const pliesPath = path.join(args.in, 'plies.jsonl');
const outPath = path.join(args.in, `relabel-${tag}.jsonl`);
const metaPath = path.join(args.in, `relabel-${tag}.json`);
if (!existsSync(pliesPath)) throw new Error(`no plies.jsonl in ${args.in}; run distill-collect first`);

const plies = readFileSync(pliesPath, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const wanted = args.limit > 0 ? plies.slice(0, args.limit) : plies;

if (args.fresh && existsSync(outPath)) writeFileSync(outPath, '');
const done = new Set();
if (existsSync(outPath)) {
  for (const line of readFileSync(outPath, 'utf8').split('\n')) {
    if (line) done.add(JSON.parse(line).idx);
  }
}
const todo = wanted.filter((p) => !done.has(p.idx));

const { wasm, memory, build } = await loadWasm();
const { infer, stats: inferStats } = createInferClient({
  url: args.infer, checkpoint: args.checkpoint, maxSockets: 2
});
console.log(`[relabel] wasm ${build}  net ${args.checkpoint} at ${args.infer}`);
console.log(`[relabel] ${todo.length} of ${wanted.length} positions to do `
  + `(${done.size} already present) at sims=${args.sims} `
  + `considered=${args.considered || 'as-collected'} slots=${args.slots} -> ${outPath}`);

const totals = {
  positions: 0, entropySum: 0, nonzeroSum: 0, degenerate: 0, tieTop2: 0,
  tvSum: 0, klSum: 0, klInfinite: 0, agreeTop: 0, searchMs: 0, simsUsedSum: 0
};

class RelabelSession {
  constructor(record) {
    this.record = record;
    this.state = stateFromHistory(record.history);
    this.issued = false;
  }

  nextSpec() {
    if (this.issued) return null;
    this.issued = true;
    if (this.state.outcome.kind !== 'ongoing') {
      throw new Error(`idx ${this.record.idx}: replayed history is already terminal `
        + `(${this.state.outcome.kind}) -- the collected ply cannot be re-searched`);
    }
    if (this.state.ply !== this.record.ply) {
      throw new Error(`idx ${this.record.idx}: replay reached ply ${this.state.ply}, recorded ${this.record.ply}`);
    }
    if (this.state.position.turn !== this.record.mover) {
      throw new Error(`idx ${this.record.idx}: replay has ${this.state.position.turn} to move, recorded ${this.record.mover}`);
    }
    return {
      state: this.state,
      sims: args.sims,
      considered: args.considered || this.record.considered,
      cPuct: args.cpuct,
      seed: args.seedmode === 'same' ? this.record.seed : (this.record.seed + 0x9e3779b9) >>> 0
    };
  }

  onResult(result) {
    const rec = this.record;
    // Same position, three independent ways.
    const hash = floatsHash(result.features);
    if (hash !== rec.featuresHash) {
      throw new Error(`idx ${rec.idx}: deep search root features hash ${hash} != collected ${rec.featuresHash}`);
    }
    const check = crossCheckRoot(this.state, result.features, result.mask);
    if (check.maskMismatches > 0 || check.featureDiff > 1e-6) {
      throw new Error(`idx ${rec.idx}: root cross-check failed `
        + `(maskMismatches=${check.maskMismatches} featureDiff=${check.featureDiff})`);
    }
    if (check.legalCount !== rec.legal) {
      throw new Error(`idx ${rec.idx}: ${check.legalCount} legal moves, collected ${rec.legal}`);
    }

    const { target: deep, degenerate } = visitTarget(result.visits, result.actionCode);
    const { target: shallow } = visitTarget(rec.visitsAll, rec.playedCode);
    if (degenerate) totals.degenerate += 1;
    let nonzero = 0;
    let top1 = 0;
    let top2 = 0;
    let topCode = -1;
    for (let c = 0; c < POLICY_LEN; c += 1) {
      if (deep[c] > 0) nonzero += 1;
      if (deep[c] > top1) { top2 = top1; top1 = deep[c]; topCode = c; }
      else if (deep[c] > top2) top2 = deep[c];
      if (deep[c] > 0 && result.mask[c] <= 0) throw new Error(`idx ${rec.idx}: deep target mass on illegal code ${c}`);
    }
    let shallowTop = -1;
    let shallowBest = -1;
    for (let c = 0; c < POLICY_LEN; c += 1) if (shallow[c] > shallowBest) { shallowBest = shallow[c]; shallowTop = c; }

    const entropy = entropyOf(deep);
    const tv = totalVariation(shallow, deep);
    const kl = klDivergence(shallow, deep);
    totals.positions += 1;
    totals.entropySum += entropy;
    totals.nonzeroSum += nonzero;
    if (top1 === top2 && top1 > 0) totals.tieTop2 += 1;
    totals.tvSum += tv;
    if (Number.isFinite(kl)) totals.klSum += kl; else totals.klInfinite += 1;
    if (topCode === shallowTop) totals.agreeTop += 1;
    totals.searchMs += result.ms;
    totals.simsUsedSum += result.simsUsed;

    appendFileSync(outPath, `${JSON.stringify({
      idx: rec.idx, game: rec.game, ply: rec.ply, sims: args.sims,
      considered: args.considered || rec.considered, seed: result.spec.seed,
      simsUsed: result.simsUsed, leaves: result.leaves,
      rootValue: Number(result.rootValue.toFixed(6)),
      actionCode: result.actionCode,
      entropy: Number(entropy.toFixed(6)), nonzero,
      shallowEntropy: rec.entropy,
      tvFromShallow: Number(tv.toFixed(6)),
      klShallowToDeep: Number.isFinite(kl) ? Number(kl.toFixed(6)) : null,
      topCode, shallowTopCode: shallowTop, topAgrees: topCode === shallowTop,
      visits: result.visits.filter(([, n]) => n > 0),
      visitsAll: result.visits,
      ms: Math.round(result.ms)
    })}\n`);
  }
}

function* sessions() {
  for (const record of todo) yield new RelabelSession(record);
}

const started = performance.now();
let lastLog = started;
const stats = await runLockStep({
  wasm,
  memory,
  infer,
  sessions: sessions(),
  slots: args.slots,
  cPuct: args.cpuct,
  onStep: (s) => {
    const now = performance.now();
    if (now - lastLog < 20000) return;
    lastLog = now;
    const perPosition = (now - started) / Math.max(1, totals.positions);
    const remaining = (todo.length - totals.positions) * perPosition;
    console.log(`[relabel] ${totals.positions}/${todo.length} positions, `
      + `${(s.leaves / ((now - started) / 1000)).toFixed(0)} leaves/s, `
      + `${(perPosition / 1000).toFixed(2)} s/position, `
      + `eta ${fmtDuration(remaining)}, wasm ${(s.wasmBytes / 1e6).toFixed(0)}MB`);
  }
});
const wall = performance.now() - started;

const n = Math.max(1, totals.positions);
const meta = {
  step: 'relabel',
  tag,
  wasmBuild: build,
  config: { ...args },
  counts: {
    positions: totals.positions,
    skippedAlreadyDone: done.size,
    degenerateTargets: totals.degenerate,
    meanSimsUsed: Number((totals.simsUsedSum / n).toFixed(1))
  },
  target: {
    meanEntropyNats: Number((totals.entropySum / n).toFixed(4)),
    meanNonzeroEntries: Number((totals.nonzeroSum / n).toFixed(2)),
    tieTop2Fraction: Number((totals.tieTop2 / n).toFixed(4))
  },
  contrast: {
    meanTotalVariationFromShallow: Number((totals.tvSum / n).toFixed(4)),
    meanKlShallowToDeep: Number((totals.klSum / n).toFixed(4)),
    positionsWithInfiniteKl: totals.klInfinite,
    topMoveAgreementFraction: Number((totals.agreeTop / n).toFixed(4))
  },
  cost: {
    wallMs: Math.round(wall),
    secondsPerPosition: Number((wall / 1000 / n).toFixed(3)),
    leavesPerSecond: Number((stats.leaves / (wall / 1000)).toFixed(0)),
    inferCalls: inferStats.calls,
    inferMsPerCall: Number((inferStats.ms / Math.max(1, inferStats.calls)).toFixed(2)),
    meanBatch: Number((inferStats.positions / Math.max(1, inferStats.calls)).toFixed(1))
  }
};
writeFileSync(metaPath, `${JSON.stringify(meta, null, 2)}\n`);

console.log('');
console.log(`[relabel] ${totals.positions} positions at ${args.sims} sims in ${fmtDuration(wall)} `
  + `(${meta.cost.secondsPerPosition} s/position, ${meta.cost.leavesPerSecond} leaves/s, `
  + `batch ${meta.cost.meanBatch} at ${meta.cost.inferMsPerCall} ms/call)`);
console.log(`[relabel] deep target: entropy ${meta.target.meanEntropyNats} nats over `
  + `${meta.target.meanNonzeroEntries} nonzero entries`);
console.log(`[relabel] contrast vs shallow: TV ${meta.contrast.meanTotalVariationFromShallow}, `
  + `KL ${meta.contrast.meanKlShallowToDeep} nats, `
  + `top move agrees ${(meta.contrast.topMoveAgreementFraction * 100).toFixed(1)}%`);
console.log(`[relabel] wrote ${outPath} and ${metaPath}`);
