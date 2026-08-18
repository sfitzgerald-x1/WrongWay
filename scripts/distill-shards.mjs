#!/usr/bin/env node
/**
 * Step 3: write the two shard sets.
 *
 * IDENTICAL positions, IDENTICAL `z`, DIFFERENT policy target. Both records for a
 * position are cut from the SAME 810-float feature vector and the SAME legal mask
 * -- the ones step 1 stored in `features.f32` -- and carry the SAME `z`, which
 * comes from the game's final outcome and is never re-derived from the deep
 * search. Only the 209 policy floats differ.
 *
 * `cluster/shard-format.mjs` is IMPORTED, never copied: the format's own writer
 * is the only thing that should ever produce a `.bin` the trainer reads, and a
 * second implementation of a 64-byte header is a truncation bug waiting to
 * happen. Verification is a separate step (`distill-verify.mjs`) that reads the
 * files back from disk.
 *
 *   node scripts/distill-shards.mjs --in ~/wwplay/distill/smoke --deep deep-4096
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

import {
  FEATURE_LEN, POLICY_LEN, RECORD_FLOATS, SIDECAR_FLOATS, REPO_ROOT,
  entropyOf, parseArgs, totalVariation, visitTarget
} from './distill-lib.mjs';

const args = parseArgs(process.argv.slice(2), {
  in: path.join(process.env.HOME || '/tmp', 'wwplay/distill/smoke'),
  deep: '',                     // relabel tag; defaults to the only one present
  out: '',                      // defaults to <in>/shards
  perShard: 4096,
  shardFormat: process.env.WW_SHARD_FORMAT
    || path.resolve(REPO_ROOT, '../wrongway-training/cluster/shard-format.mjs')
});

const { writeShard } = await import(args.shardFormat);

const pliesPath = path.join(args.in, 'plies.jsonl');
const featuresPath = path.join(args.in, 'features.f32');
for (const p of [pliesPath, featuresPath]) {
  if (!existsSync(p)) throw new Error(`missing ${p}; run distill-collect first`);
}
let deepTag = args.deep;
if (!deepTag) {
  const candidates = [];
  const { readdirSync } = await import('node:fs');
  for (const f of readdirSync(args.in)) {
    const m = /^relabel-(.+)\.jsonl$/.exec(f);
    if (m) candidates.push(m[1]);
  }
  if (candidates.length !== 1) {
    throw new Error(`--deep is required: found ${candidates.length} relabel sets (${candidates.join(', ')})`);
  }
  deepTag = candidates[0];
}
const deepPath = path.join(args.in, `relabel-${deepTag}.jsonl`);
if (!existsSync(deepPath)) throw new Error(`missing ${deepPath}`);

const plies = readFileSync(pliesPath, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const deepByIdx = new Map();
for (const line of readFileSync(deepPath, 'utf8').split('\n')) {
  if (!line) continue;
  const record = JSON.parse(line);
  deepByIdx.set(record.idx, record);
}
const sidecar = new Float32Array(readFileSync(featuresPath).buffer);
if (sidecar.length !== plies.length * SIDECAR_FLOATS) {
  throw new Error(`features.f32 holds ${sidecar.length} floats, expected `
    + `${plies.length * SIDECAR_FLOATS} for ${plies.length} plies`);
}

// Only positions the deep pass actually reached. A partial relabel is usable --
// both sets simply cover the same subset -- but it must be the SAME subset, so
// the filter is applied once, here, and both sets are written from it.
const usable = plies.filter((p) => deepByIdx.has(p.idx));
if (usable.length === 0) throw new Error('no position has both a shallow and a deep target');
const shallowSims = usable[0].sims;
const deepSims = deepByIdx.get(usable[0].idx).sims;

const outRoot = args.out || path.join(args.in, 'shards');
// The deep set is named from the relabel TAG, not from its sim count: two deep
// arms can share a budget and differ in `maxConsidered`, and naming both
// `deep-4096` would have the second silently overwrite the first.
const sets = [
  { name: `shallow-${shallowSims}`, deep: false },
  { name: deepTag.startsWith('deep') ? deepTag : `deep-${deepTag}`, deep: true }
];
// Refuse to half-overwrite a pairing that is already here: the shallow set has
// the same name for every pairing, so writing a second deep arm into the same
// directory would leave one manifest describing two different experiments.
const existingManifest = path.join(outRoot, 'shards.json');
if (existsSync(existingManifest)) {
  const previous = JSON.parse(readFileSync(existingManifest, 'utf8'));
  if (previous.deepTag !== deepTag) {
    throw new Error(`${outRoot} already holds the ${previous.deepTag} pairing; `
      + `pass --out for the ${deepTag} pairing rather than mixing them`);
  }
}
for (const set of sets) {
  set.dir = path.join(outRoot, set.name);
  mkdirSync(set.dir, { recursive: true });
  set.buffer = new Float32Array(Math.min(args.perShard, usable.length) * RECORD_FLOATS);
  set.inShard = 0;
  set.shardIndex = 0;
  set.files = [];
  set.stats = { records: 0, entropySum: 0, nonzeroSum: 0, tieTop2: 0, zSum: 0, zNonZero: 0 };
}
let tvSum = 0;
let differing = 0;

function flush(set, final = false) {
  if (set.inShard === 0) return;
  if (!final && set.inShard < args.perShard) return;
  const file = path.join(set.dir, `shard-${String(set.shardIndex).padStart(3, '0')}.bin`);
  const bytes = writeShard(file, set.buffer, set.inShard, {
    featureLen: FEATURE_LEN, policyLen: POLICY_LEN, maskLen: POLICY_LEN, zLen: 1
  });
  set.files.push({ file, records: set.inShard, bytes });
  set.shardIndex += 1;
  set.inShard = 0;
}

for (const ply of usable) {
  const base = ply.idx * SIDECAR_FLOATS;
  const features = sidecar.subarray(base, base + FEATURE_LEN);
  const mask = sidecar.subarray(base + FEATURE_LEN, base + FEATURE_LEN + POLICY_LEN);
  const deepRecord = deepByIdx.get(ply.idx);
  const shallowTarget = visitTarget(ply.visitsAll, ply.playedCode).target;
  const deepTarget = visitTarget(deepRecord.visitsAll, deepRecord.actionCode).target;
  tvSum += totalVariation(shallowTarget, deepTarget);
  let same = true;
  for (let c = 0; c < POLICY_LEN && same; c += 1) if (shallowTarget[c] !== deepTarget[c]) same = false;
  if (!same) differing += 1;

  for (const set of sets) {
    const target = set.deep ? deepTarget : shallowTarget;
    const at = set.inShard * RECORD_FLOATS;
    set.buffer.set(features, at);
    set.buffer.set(target, at + FEATURE_LEN);
    set.buffer.set(mask, at + FEATURE_LEN + POLICY_LEN);
    set.buffer[at + FEATURE_LEN + POLICY_LEN + POLICY_LEN] = ply.z;
    set.inShard += 1;
    set.stats.records += 1;
    set.stats.entropySum += entropyOf(target);
    let nonzero = 0;
    let top1 = 0;
    let top2 = 0;
    for (let c = 0; c < POLICY_LEN; c += 1) {
      if (target[c] > 0) nonzero += 1;
      if (target[c] > top1) { top2 = top1; top1 = target[c]; }
      else if (target[c] > top2) top2 = target[c];
    }
    set.stats.nonzeroSum += nonzero;
    if (top1 === top2 && top1 > 0) set.stats.tieTop2 += 1;
    set.stats.zSum += ply.z;
    if (ply.z !== 0) set.stats.zNonZero += 1;
    flush(set);
  }
}
for (const set of sets) flush(set, true);

const manifest = {
  step: 'shards',
  source: args.in,
  deepTag,
  shallowSims,
  deepSims,
  records: usable.length,
  skippedWithoutDeepTarget: plies.length - usable.length,
  contrast: {
    meanTotalVariation: Number((tvSum / usable.length).toFixed(4)),
    recordsWithDifferentTarget: differing,
    fractionDifferent: Number((differing / usable.length).toFixed(4))
  },
  sets: sets.map((set) => ({
    name: set.name,
    dir: set.dir,
    files: set.files,
    records: set.stats.records,
    meanTargetEntropyNats: Number((set.stats.entropySum / set.stats.records).toFixed(4)),
    meanNonzeroTargetEntries: Number((set.stats.nonzeroSum / set.stats.records).toFixed(2)),
    tieTop2Fraction: Number((set.stats.tieTop2 / set.stats.records).toFixed(4)),
    meanZ: Number((set.stats.zSum / set.stats.records).toFixed(4)),
    decisiveFraction: Number((set.stats.zNonZero / set.stats.records).toFixed(4))
  }))
};
const manifestPath = path.join(outRoot, 'shards.json');
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`[shards] ${usable.length} records per set from ${args.in}`
  + (manifest.skippedWithoutDeepTarget
    ? ` (${manifest.skippedWithoutDeepTarget} collected positions had no deep target and were skipped in BOTH sets)`
    : ''));
for (const set of manifest.sets) {
  console.log(`[shards] ${set.name.padEnd(14)} ${set.files.length} file(s), ${set.records} records, `
    + `entropy ${set.meanTargetEntropyNats} nats, ${set.meanNonzeroTargetEntries} nonzero, `
    + `ties ${set.tieTop2Fraction}, meanZ ${set.meanZ}`);
}
console.log(`[shards] target contrast: mean TV ${manifest.contrast.meanTotalVariation}, `
  + `${manifest.contrast.recordsWithDifferentTarget}/${usable.length} records differ`);
console.log(`[shards] manifest ${manifestPath}`);
