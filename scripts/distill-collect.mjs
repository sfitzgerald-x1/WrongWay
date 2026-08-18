#!/usr/bin/env node
/**
 * Step 1 of the deep-target experiment: play games at the SHALLOW budget and
 * record everything needed to reproduce every position later.
 *
 * WHY THE HISTORIES ARE THE POINT. Existing self-play shards carry the 810-float
 * feature encoding and nothing else -- no board, no move list -- so there is
 * nothing in them to replay and no way to run a second search on a stored
 * position. Relabelling an existing shard is therefore impossible, which is why
 * this step exists: the positions have to be generated fresh WITH their move
 * histories, and the deep pass replays those histories.
 *
 * What lands in the work directory:
 *
 *   games.jsonl      one line per game: full move history, outcome, plies
 *   plies.jsonl      one line per position: game, ply, history prefix, the
 *                    shallow visit counts, z, and the feature hash
 *   features.f32     810 features + 209 mask per position, in `idx` order --
 *                    so the shard step needs neither wasm nor the network
 *   collect.json     the run's configuration and measured cost
 *
 * The exploration recipe mirrors `cluster/batched-selfplay.mjs`: sample the played
 * move from the search's own target for the first T_MOVES absolute plies, then
 * play argmax. That is what makes these positions the same DISTRIBUTION the real
 * pipeline trains on, which the experiment needs -- an off-distribution position
 * set would confound the comparison as badly as the volume change arm g1 made.
 *
 *   node scripts/distill-collect.mjs --games 8 --slots 32 --out ~/wwplay/distill/smoke
 */

import { closeSync, existsSync, mkdirSync, openSync, writeFileSync, writeSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

import { applyAction, createInitialState, decodeAction } from '../js/normal-duel-engine.mjs';
import {
  CONFIG, FEATURE_LEN, POLICY_LEN, SIDECAR_FLOATS, createInferClient, crossCheckRoot,
  entropyOf, floatsHash, fmtDuration, historyEntry, loadWasm, parseArgs,
  rng32, runLockStep, sampleFromTarget, visitTarget
} from './distill-lib.mjs';

const args = parseArgs(process.argv.slice(2), {
  games: 8,
  sims: 512,
  considered: 12,
  // 1.4, not the play server's 1.25: `cluster/batched-selfplay.mjs` defaults
  // C_PUCT to 1.4 and the cluster orchestrator never overrides it, so 1.4 is what
  // produced the training distribution these positions have to belong to.
  cpuct: 1.4,
  slots: 32,
  temperature: 1.0,
  tmoves: 24,
  plycap: 200,
  seed: 1,
  infer: 'http://127.0.0.1:8301',
  checkpoint: 'g2-iter-030',
  out: path.join(process.env.HOME || '/tmp', 'wwplay/distill/smoke')
});

mkdirSync(args.out, { recursive: true });
const paths = {
  games: path.join(args.out, 'games.jsonl'),
  plies: path.join(args.out, 'plies.jsonl'),
  features: path.join(args.out, 'features.f32'),
  meta: path.join(args.out, 'collect.json')
};
for (const p of [paths.games, paths.plies, paths.features]) writeFileSync(p, '');

const { wasm, memory, build } = await loadWasm();
const { infer, stats: inferStats } = createInferClient({
  url: args.infer, checkpoint: args.checkpoint, maxSockets: 2
});
console.log(`[collect] wasm ${build}  net ${args.checkpoint} at ${args.infer}`);
console.log(`[collect] ${args.games} games at sims=${args.sims} considered=${args.considered} `
  + `slots=${args.slots} T=${args.temperature} tmoves=${args.tmoves}`);

const fdGames = openSync(paths.games, 'a');
const fdPlies = openSync(paths.plies, 'a');
const fdFeatures = openSync(paths.features, 'a');

const totals = {
  games: 0, plies: 0, wins: 0, draws: 0, degenerate: 0,
  entropySum: 0, legalSum: 0, nonzeroSum: 0, tieTop2: 0,
  featureBitExact: 0, maxFeatureDiff: 0, maskMismatches: 0,
  searchMs: 0, sampledPlies: 0
};
let nextIdx = 0;

class GameSession {
  constructor(index) {
    this.index = index;
    this.state = createInitialState(CONFIG);
    this.history = [];
    this.plies = [];
    this.roll = rng32(args.seed * 1_000_003 + index * 7919 + 13);
    this.finalized = false;
  }

  get over() {
    return this.state.outcome.kind !== 'ongoing' || this.state.ply >= args.plycap;
  }

  nextSpec() {
    if (this.over) { this.finalize(); return null; }
    return {
      state: this.state,
      sims: args.sims,
      considered: args.considered,
      cPuct: args.cpuct,
      // The seed is a pure function of (run, game, ply) so the deep pass can
      // reuse the IDENTICAL seed at the same position: same Gumbel draw, same
      // considered set, and therefore a contrast that is purely about DEPTH.
      seed: (args.seed * 1_000_003 + this.index * 7919 + this.state.ply + 1) >>> 0
    };
  }

  onResult(result) {
    const state = this.state;
    const mover = state.position.turn;
    // A forced position is PLAYED but not RECORDED. There is no training signal in a
    // one-hot over a single legal move, and no teacher at any depth can differ from it,
    // so it would enter both the shallow and the deep corpus as an identical row and
    // dilute every contrast statistic with guaranteed-zero-contrast records.
    //
    // Dropping it keeps the two corpora aligned: `relabel` replays these same plies, so
    // a ply absent here is absent there too. It also has no features or mask to
    // cross-check -- Rust handed out no leaf -- and synthesising them from the JS
    // encoder would report agreement between the encoder and itself.
    if (result.forced) {
      totals.forcedSkipped = (totals.forcedSkipped || 0) + 1;
      const forcedAction = decodeAction(CONFIG, result.actionCode);
      this.history.push(historyEntry(mover, forcedAction));
      this.state = applyAction(CONFIG, this.state, forcedAction);
      return;
    }
    // Two implementations must agree that this is the same position: the features
    // came out of Rust, the comparison comes from the JS encoder and the JS
    // engine's own legal mask.
    const check = crossCheckRoot(state, result.features, result.mask);
    if (check.maskMismatches > 0) {
      throw new Error(`game ${this.index} ply ${state.ply}: legal mask disagrees with the engine `
        + `in ${check.maskMismatches} entries`);
    }
    if (check.featureDiff > 1e-6) {
      throw new Error(`game ${this.index} ply ${state.ply}: root features differ from the JS `
        + `encoder by ${check.featureDiff}`);
    }
    if (check.featureBitExact) totals.featureBitExact += 1;
    totals.maxFeatureDiff = Math.max(totals.maxFeatureDiff, check.featureDiff);

    const { target, degenerate } = visitTarget(result.visits, result.actionCode);
    if (degenerate) totals.degenerate += 1;
    let nonzero = 0;
    let top1 = 0;
    let top2 = 0;
    for (let c = 0; c < POLICY_LEN; c += 1) {
      if (target[c] > 0) nonzero += 1;
      if (target[c] > top1) { top2 = top1; top1 = target[c]; }
      else if (target[c] > top2) top2 = target[c];
      if (target[c] > 0 && result.mask[c] <= 0) {
        throw new Error(`game ${this.index} ply ${state.ply}: target mass on illegal code ${c}`);
      }
    }
    totals.entropySum += entropyOf(target);
    totals.legalSum += check.legalCount;
    totals.nonzeroSum += nonzero;
    if (top1 === top2 && top1 > 0) totals.tieTop2 += 1;
    totals.searchMs += result.ms;

    // Choose the played move. Temperature phase first, argmax after -- the
    // recorded target is unaffected either way.
    let playedCode = result.actionCode;
    let sampled = false;
    if (state.ply < args.tmoves) {
      const drawn = sampleFromTarget(target, args.temperature, this.roll());
      if (drawn !== null) { playedCode = drawn; sampled = true; totals.sampledPlies += 1; }
    }
    const action = decodeAction(CONFIG, playedCode);

    this.plies.push({
      ply: state.ply,
      mover,
      history: this.history.slice(),
      visits: result.visits.filter(([, n]) => n > 0).map(([c, n]) => [c, n]),
      visitsAll: result.visits.map(([c, n]) => [c, n]),
      legal: check.legalCount,
      nonzero,
      rootValue: Number(result.rootValue.toFixed(6)),
      entropy: Number(entropyOf(target).toFixed(6)),
      simsUsed: result.simsUsed,
      leaves: result.leaves,
      seed: result.spec.seed,
      playedCode,
      sampled,
      featuresHash: floatsHash(result.features),
      ms: Math.round(result.ms),
      features: result.features,
      mask: result.mask
    });

    this.history.push(historyEntry(mover, action));
    this.state = applyAction(CONFIG, this.state, action);
  }

  /** Stamp z on every ply and flush the game to disk. */
  finalize() {
    if (this.finalized) return;
    this.finalized = true;
    const outcome = this.state.outcome;
    const sidecar = new Float32Array(this.plies.length * SIDECAR_FLOATS);
    const lines = [];
    this.plies.forEach((ply, i) => {
      // z is from THAT PLY'S MOVER's perspective, exactly as the trainer wants
      // and as `SelfPlayBatch::flush` stamps it. It is a property of the game,
      // so the deep pass must never re-derive it.
      const z = outcome.kind === 'win' ? (outcome.winner === ply.mover ? 1 : -1) : 0;
      const idx = nextIdx + i;
      sidecar.set(ply.features, i * SIDECAR_FLOATS);
      sidecar.set(ply.mask, i * SIDECAR_FLOATS + FEATURE_LEN);
      lines.push(JSON.stringify({
        idx, game: this.index, ply: ply.ply, mover: ply.mover, z,
        legal: ply.legal, nonzero: ply.nonzero, entropy: ply.entropy,
        rootValue: ply.rootValue, sims: args.sims, considered: args.considered,
        simsUsed: ply.simsUsed, leaves: ply.leaves, seed: ply.seed,
        playedCode: ply.playedCode, sampled: ply.sampled,
        featuresHash: ply.featuresHash, ms: ply.ms,
        visits: ply.visits, visitsAll: ply.visitsAll,
        history: ply.history
      }));
    });
    nextIdx += this.plies.length;
    writeSync(fdFeatures, Buffer.from(sidecar.buffer, 0, sidecar.length * 4));
    if (lines.length) writeSync(fdPlies, `${lines.join('\n')}\n`);
    writeSync(fdGames, `${JSON.stringify({
      game: this.index, plies: this.plies.length, outcome,
      firstIdx: nextIdx - this.plies.length, history: this.history
    })}\n`);
    totals.games += 1;
    totals.plies += this.plies.length;
    if (outcome.kind === 'win') totals.wins += 1; else totals.draws += 1;
    const done = totals.games;
    console.log(`[collect] game ${this.index} ${outcome.kind}`
      + `${outcome.kind === 'win' ? ` ${outcome.winner}` : ` (${outcome.reason})`}`
      + ` in ${this.plies.length} plies  [${done}/${args.games} games, ${totals.plies} positions]`);
    this.plies = [];        // release the feature buffers
  }
}

function* sessions() {
  for (let i = 0; i < args.games; i += 1) yield new GameSession(i);
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
    if (now - lastLog < 15000) return;
    lastLog = now;
    const rate = s.leaves / ((now - started) / 1000);
    console.log(`[collect] ${s.steps} steps, ${s.searches} searches, ${s.leaves} leaves, `
      + `${rate.toFixed(0)} leaves/s, batch<=${s.maxBatch}, wasm ${(s.wasmBytes / 1e6).toFixed(0)}MB`);
  }
});
const wall = performance.now() - started;

closeSync(fdGames); closeSync(fdPlies); closeSync(fdFeatures);

const meta = {
  step: 'collect',
  wasmBuild: build,
  config: { ...args },
  counts: {
    games: totals.games, positions: totals.plies,
    wins: totals.wins, draws: totals.draws,
    drawRate: Number((totals.draws / Math.max(1, totals.games)).toFixed(4)),
    sampledPlies: totals.sampledPlies,
    degenerateTargets: totals.degenerate
  },
  target: {
    meanEntropyNats: Number((totals.entropySum / Math.max(1, totals.plies)).toFixed(4)),
    meanLegalMoves: Number((totals.legalSum / Math.max(1, totals.plies)).toFixed(2)),
    meanNonzeroEntries: Number((totals.nonzeroSum / Math.max(1, totals.plies)).toFixed(2)),
    tieTop2Fraction: Number((totals.tieTop2 / Math.max(1, totals.plies)).toFixed(4))
  },
  parity: {
    featureBitExactFraction: Number((totals.featureBitExact / Math.max(1, totals.plies)).toFixed(4)),
    maxFeatureAbsDiff: totals.maxFeatureDiff,
    maskMismatches: totals.maskMismatches
  },
  cost: {
    wallMs: Math.round(wall),
    searchMsSum: Math.round(totals.searchMs),
    msPerPosition: Number((wall / Math.max(1, totals.plies)).toFixed(1)),
    msPerGame: Number((wall / Math.max(1, totals.games)).toFixed(0)),
    inferCalls: inferStats.calls,
    inferPositions: inferStats.positions,
    inferMsPerCall: Number((inferStats.ms / Math.max(1, inferStats.calls)).toFixed(2)),
    meanBatch: Number((inferStats.positions / Math.max(1, inferStats.calls)).toFixed(1)),
    leavesPerSecond: Number((stats.leaves / (wall / 1000)).toFixed(0))
  }
};
writeFileSync(paths.meta, `${JSON.stringify(meta, null, 2)}\n`);

console.log('');
console.log(`[collect] ${totals.plies} positions from ${totals.games} games in ${fmtDuration(wall)}`);
console.log(`[collect] draws ${totals.draws}/${totals.games}, `
  + `mean target entropy ${meta.target.meanEntropyNats} nats over `
  + `${meta.target.meanNonzeroEntries} nonzero of ${meta.target.meanLegalMoves} legal`);
console.log(`[collect] cost ${meta.cost.msPerPosition} ms/position, ${meta.cost.msPerGame} ms/game, `
  + `mean batch ${meta.cost.meanBatch} at ${meta.cost.inferMsPerCall} ms/call`);
console.log(`[collect] wrote ${paths.plies}, ${paths.features}, ${paths.games}, ${paths.meta}`);
