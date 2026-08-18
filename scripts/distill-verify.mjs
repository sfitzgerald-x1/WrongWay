#!/usr/bin/env node
/**
 * Step 4: read the two shard sets back OFF DISK and prove they are what the
 * experiment claims.
 *
 * Nothing here trusts the writer. The legal mask is re-derived from the ENGINE by
 * replaying each position's move history, the features are re-derived from the JS
 * encoder, and the header arithmetic is checked against the actual file length.
 * The assertion that matters most is the last one:
 *
 *   A  header valid; file length is exactly 64 + records * 1229 * 4
 *   B  every record's legalMask equals the engine's legal actions for that position
 *   C  every policyTarget sums to 1 and is zero wherever the mask is zero
 *   D  the two sets have identical features, identical legalMask, identical z
 *   E  the two sets have DIFFERENT policyTarget  <- this is what makes it an experiment
 *
 * E is reported as a fraction and a mean total-variation, not just a boolean: if
 * the deep target were merely a rounding away from the shallow one, D-and-E would
 * both pass while the experiment tested nothing. The entropy of both sets is
 * printed side by side for the same reason.
 *
 *   node scripts/distill-verify.mjs --in ~/wwplay/distill/smoke
 */

import { readFileSync, readdirSync, existsSync } from 'node:fs';
import path from 'node:path';

import {
  FEATURE_LEN, POLICY_LEN, RECORD_FLOATS, REPO_ROOT,
  crossCheckRoot, entropyOf, klDivergence, parseArgs, stateFromHistory, totalVariation
} from './distill-lib.mjs';

const args = parseArgs(process.argv.slice(2), {
  in: path.join(process.env.HOME || '/tmp', 'wwplay/distill/smoke'),
  shards: '',            // defaults to <in>/shards
  sample: 0,             // 0 = re-derive the mask for EVERY record (the honest default)
  shardFormat: process.env.WW_SHARD_FORMAT
    || path.resolve(REPO_ROOT, '../wrongway-training/cluster/shard-format.mjs')
});

const { readShardHeader, SHARD_HEADER_BYTES } = await import(args.shardFormat);

const shardRoot = args.shards || path.join(args.in, 'shards');
const manifestPath = path.join(shardRoot, 'shards.json');
if (!existsSync(manifestPath)) throw new Error(`no shards.json in ${shardRoot}; run distill-shards first`);
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
const plies = readFileSync(path.join(args.in, 'plies.jsonl'), 'utf8')
  .split('\n').filter(Boolean).map((l) => JSON.parse(l));
const pliesByIdx = new Map(plies.map((p) => [p.idx, p]));

const failures = [];
const notes = [];
function check(ok, label, detail = '') {
  if (ok) { console.log(`  PASS  ${label}${detail ? `  (${detail})` : ''}`); return true; }
  console.log(`  FAIL  ${label}${detail ? `  (${detail})` : ''}`);
  failures.push(`${label}${detail ? `: ${detail}` : ''}`);
  return false;
}

/** Load every record of a set as one flat Float32Array, validating each file. */
function loadSet(set) {
  const files = readdirSync(set.dir).filter((f) => f.startsWith('shard-') && f.endsWith('.bin')).sort();
  if (files.length === 0) throw new Error(`${set.dir}: no shard-*.bin`);
  const chunks = [];
  let records = 0;
  const headers = [];
  for (const f of files) {
    const full = path.join(set.dir, f);
    const buf = readFileSync(full);
    // totalBytes is passed, which is the check that catches a truncated shard.
    const header = readShardHeader(buf, buf.length, f);
    if (header.featureLen !== FEATURE_LEN || header.policyLen !== POLICY_LEN
        || header.maskLen !== POLICY_LEN || header.zLen !== 1
        || header.recordFloats !== RECORD_FLOATS) {
      throw new Error(`${f}: unexpected field sizes ${JSON.stringify(header)}`);
    }
    const expect = SHARD_HEADER_BYTES + header.recordCount * RECORD_FLOATS * 4;
    if (buf.length !== expect) throw new Error(`${f}: ${buf.length} bytes, expected ${expect}`);
    const body = new Float32Array(header.recordCount * RECORD_FLOATS);
    // Copy rather than view: the Buffer's byteOffset is not guaranteed 4-aligned.
    Buffer.from(buf.buffer, buf.byteOffset + SHARD_HEADER_BYTES, header.recordCount * RECORD_FLOATS * 4)
      .copy(Buffer.from(body.buffer));
    chunks.push(body);
    records += header.recordCount;
    headers.push({ file: f, bytes: buf.length, records: header.recordCount });
  }
  const all = new Float32Array(records * RECORD_FLOATS);
  let at = 0;
  for (const c of chunks) { all.set(c, at); at += c.length; }
  return { records, floats: all, headers, files };
}

console.log(`[verify] ${shardRoot}`);
console.log(`[verify] manifest: ${manifest.records} records per set, deep tag ${manifest.deepTag}`);

const loaded = manifest.sets.map((set) => ({ set, ...loadSet(set) }));

console.log('');
console.log('A. header and file length');
for (const { set, records, headers } of loaded) {
  check(records === manifest.records,
    `${set.name}: ${records} records read back`,
    headers.map((h) => `${h.file}=${h.records}r/${h.bytes}B`).join(' '));
}

console.log('');
console.log('B. legalMask re-derived from the engine, per record');
console.log('C. policyTarget sums to 1 and is zero off the mask');
const sampleEvery = args.sample > 0 ? Math.max(1, Math.floor(manifest.records / args.sample)) : 1;
// The shard order is the plies.jsonl order filtered to positions that got a deep
// target -- reconstruct that ordering to know which position each record is.
const deepPath = path.join(args.in, `relabel-${manifest.deepTag}.jsonl`);
const deepIdx = new Set(readFileSync(deepPath, 'utf8').split('\n').filter(Boolean)
  .map((l) => JSON.parse(l).idx));
const order = plies.filter((p) => deepIdx.has(p.idx)).map((p) => p.idx);
check(order.length === manifest.records,
  'record order reconstructed from plies.jsonl + relabel set', `${order.length} positions`);

const perSet = new Map();
for (const { set } of loaded) {
  perSet.set(set.name, {
    maskMismatch: 0, featureMismatch: 0, sumBad: 0, offMask: 0, zBad: 0,
    entropySum: 0, nonzeroSum: 0, tieTop2: 0, checked: 0,
    maxSumErr: 0, maxFeatureDiff: 0
  });
}
for (let r = 0; r < manifest.records; r += 1) {
  if (r % sampleEvery !== 0) continue;
  const ply = pliesByIdx.get(order[r]);
  const state = stateFromHistory(ply.history);
  for (const { set, floats } of loaded) {
    const acc = perSet.get(set.name);
    const at = r * RECORD_FLOATS;
    const features = floats.subarray(at, at + FEATURE_LEN);
    const target = floats.subarray(at + FEATURE_LEN, at + FEATURE_LEN + POLICY_LEN);
    const mask = floats.subarray(at + FEATURE_LEN + POLICY_LEN, at + FEATURE_LEN + 2 * POLICY_LEN);
    const z = floats[at + FEATURE_LEN + 2 * POLICY_LEN];
    const cross = crossCheckRoot(state, features, mask);
    if (cross.maskMismatches > 0) acc.maskMismatch += 1;
    if (cross.featureDiff > 0) acc.featureMismatch += 1;
    acc.maxFeatureDiff = Math.max(acc.maxFeatureDiff, cross.featureDiff);
    let sum = 0;
    let nonzero = 0;
    let top1 = 0;
    let top2 = 0;
    for (let c = 0; c < POLICY_LEN; c += 1) {
      sum += target[c];
      if (target[c] > 0) {
        nonzero += 1;
        if (mask[c] <= 0) acc.offMask += 1;
      }
      if (target[c] > top1) { top2 = top1; top1 = target[c]; }
      else if (target[c] > top2) top2 = target[c];
    }
    const sumErr = Math.abs(sum - 1);
    acc.maxSumErr = Math.max(acc.maxSumErr, sumErr);
    if (sumErr > 1e-5) acc.sumBad += 1;
    if (z !== ply.z) acc.zBad += 1;
    acc.entropySum += entropyOf(target);
    acc.nonzeroSum += nonzero;
    if (top1 === top2 && top1 > 0) acc.tieTop2 += 1;
    acc.checked += 1;
  }
}
for (const { set } of loaded) {
  const acc = perSet.get(set.name);
  check(acc.maskMismatch === 0, `${set.name}: legalMask matches the engine on all ${acc.checked} records`,
    acc.maskMismatch ? `${acc.maskMismatch} mismatched` : '');
  check(acc.featureMismatch === 0, `${set.name}: features match the JS encoder bit for bit`,
    `maxAbsDiff ${acc.maxFeatureDiff}`);
  check(acc.sumBad === 0, `${set.name}: every policyTarget sums to 1`, `maxErr ${acc.maxSumErr.toExponential(2)}`);
  check(acc.offMask === 0, `${set.name}: no target mass off the legal mask`,
    acc.offMask ? `${acc.offMask} entries` : '');
  check(acc.zBad === 0, `${set.name}: z matches the collected game outcome`,
    acc.zBad ? `${acc.zBad} records` : '');
}

console.log('');
console.log('D. the two sets agree on features, legalMask and z');
console.log('E. the two sets DISAGREE on policyTarget');
const [a, b] = loaded;
let featuresSame = 0;
let maskSame = 0;
let zSame = 0;
let targetDiffer = 0;
let tvSum = 0;
let klSum = 0;
let klInf = 0;
let maxTv = 0;
let topAgree = 0;
// Records where the deep target is BIT-IDENTICAL to the shallow one carry no
// contrast at all, so they are counted and characterised rather than averaged
// away: if they were the bulk of the set the experiment would be mostly a
// no-op, and their shape says why they happen.
const identical = { count: 0, nonzeroSum: 0, legalSum: 0, top1Sum: 0 };
const differing = { count: 0, nonzeroSum: 0, legalSum: 0, top1Sum: 0 };
for (let r = 0; r < manifest.records; r += 1) {
  const at = r * RECORD_FLOATS;
  const fa = a.floats.subarray(at, at + FEATURE_LEN);
  const fb = b.floats.subarray(at, at + FEATURE_LEN);
  let same = true;
  for (let i = 0; i < FEATURE_LEN; i += 1) if (fa[i] !== fb[i]) { same = false; break; }
  if (same) featuresSame += 1;

  const ma = a.floats.subarray(at + FEATURE_LEN + POLICY_LEN, at + FEATURE_LEN + 2 * POLICY_LEN);
  const mb = b.floats.subarray(at + FEATURE_LEN + POLICY_LEN, at + FEATURE_LEN + 2 * POLICY_LEN);
  same = true;
  for (let i = 0; i < POLICY_LEN; i += 1) if (ma[i] !== mb[i]) { same = false; break; }
  if (same) maskSame += 1;

  if (a.floats[at + FEATURE_LEN + 2 * POLICY_LEN] === b.floats[at + FEATURE_LEN + 2 * POLICY_LEN]) zSame += 1;

  const ta = a.floats.subarray(at + FEATURE_LEN, at + FEATURE_LEN + POLICY_LEN);
  const tb = b.floats.subarray(at + FEATURE_LEN, at + FEATURE_LEN + POLICY_LEN);
  same = true;
  for (let i = 0; i < POLICY_LEN; i += 1) if (ta[i] !== tb[i]) { same = false; break; }
  if (!same) targetDiffer += 1;
  {
    const bucket = same ? identical : differing;
    let nonzero = 0;
    let legal = 0;
    let top1 = 0;
    for (let c = 0; c < POLICY_LEN; c += 1) {
      if (ta[c] > 0) nonzero += 1;
      if (ma[c] > 0) legal += 1;
      if (ta[c] > top1) top1 = ta[c];
    }
    bucket.count += 1;
    bucket.nonzeroSum += nonzero;
    bucket.legalSum += legal;
    bucket.top1Sum += top1;
  }
  const tv = totalVariation(ta, tb);
  tvSum += tv;
  maxTv = Math.max(maxTv, tv);
  const kl = klDivergence(ta, tb);
  if (Number.isFinite(kl)) klSum += kl; else klInf += 1;
  let bestA = -1;
  let bestB = -1;
  let codeA = -1;
  let codeB = -1;
  for (let c = 0; c < POLICY_LEN; c += 1) {
    if (ta[c] > bestA) { bestA = ta[c]; codeA = c; }
    if (tb[c] > bestB) { bestB = tb[c]; codeB = c; }
  }
  if (codeA === codeB) topAgree += 1;
}
const n = manifest.records;
check(featuresSame === n, 'features identical across the two sets', `${featuresSame}/${n}`);
check(maskSame === n, 'legalMask identical across the two sets', `${maskSame}/${n}`);
check(zSame === n, 'z identical across the two sets', `${zSame}/${n}`);
// NOT "every record": a position whose search has nothing left to decide -- a
// forced line, or a considered set the halving schedule splits the same way at
// both budgets -- yields a bit-identical target however deep the search goes.
// Those records are real data, they simply carry no contrast, so the assertion is
// on the fraction and the magnitude and the exact count is printed.
check(targetDiffer / n > 0.5, 'policyTarget differs on the large majority of records',
  `${targetDiffer}/${n} differ (${(targetDiffer / n * 100).toFixed(1)}%), `
  + `${n - targetDiffer} bit-identical`);
check(tvSum / n > 0.02, 'target contrast is materially larger than float noise',
  `mean TV ${(tvSum / n).toFixed(4)}, max ${maxTv.toFixed(4)}`);
if (identical.count) {
  console.log(`  ..    the ${identical.count} zero-contrast records average `
    + `${(identical.nonzeroSum / identical.count).toFixed(2)} nonzero target entries and `
    + `${(identical.legalSum / identical.count).toFixed(1)} legal moves, top mass `
    + `${(identical.top1Sum / identical.count).toFixed(3)}; the ${differing.count} others average `
    + `${(differing.nonzeroSum / differing.count).toFixed(2)} nonzero, `
    + `${(differing.legalSum / differing.count).toFixed(1)} legal, top mass `
    + `${(differing.top1Sum / differing.count).toFixed(3)}`);
}

console.log('');
console.log('Target shape, side by side');
console.log('  set              records   entropy(nats)   nonzero   ties(top1==top2)');
for (const { set } of loaded) {
  const acc = perSet.get(set.name);
  console.log(`  ${set.name.padEnd(15)}  ${String(acc.checked).padStart(7)}   `
    + `${(acc.entropySum / acc.checked).toFixed(4).padStart(13)}   `
    + `${(acc.nonzeroSum / acc.checked).toFixed(2).padStart(7)}   `
    + `${(acc.tieTop2 / acc.checked).toFixed(4).padStart(16)}`);
}
const accA = perSet.get(loaded[0].set.name);
const accB = perSet.get(loaded[1].set.name);
const dEntropy = (accB.entropySum / accB.checked) - (accA.entropySum / accA.checked);
console.log('');
console.log(`  entropy delta (deep - shallow): ${dEntropy >= 0 ? '+' : ''}${dEntropy.toFixed(4)} nats`);
console.log(`  mean TV(shallow, deep):         ${(tvSum / n).toFixed(4)}`);
console.log(`  mean KL(shallow || deep):       ${(klSum / n).toFixed(4)} nats`
  + (klInf ? ` (+${klInf} records with support the deep target does not cover)` : ''));
console.log(`  top-move agreement:             ${(topAgree / n * 100).toFixed(1)}%`);

console.log('');
if (failures.length) {
  console.log(`[verify] ${failures.length} FAILED assertion(s):`);
  for (const f of failures) console.log(`  - ${f}`);
  process.exitCode = 1;
} else {
  console.log('[verify] all assertions passed');
}
for (const note of notes) console.log(`[verify] note: ${note}`);
