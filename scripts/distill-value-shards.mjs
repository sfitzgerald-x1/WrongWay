/**
 * Bet 3: value distillation. Build a shard set whose VALUE target is the deep
 * search's root value instead of the game outcome `z`.
 *
 * WHY THIS AND NOT MORE POLICY DEPTH. Bet 2 is settled and negative: a 4096-sim
 * POLICY target produced a student ~100 Elo WEAKER than a 512-sim one, replicated
 * over 600 games (64.1% and 65.9% to the shallow student, both significant). The
 * likely mechanism is that the games were PLAYED at 512, so a deep label recommends
 * moves the play-time search will not find, degrading the prior that search leans
 * on. A VALUE target cannot fail that way: it does not ask the net to play anything,
 * it tells the net what a position is worth. Every decay episode in this project has
 * also run through the value head, which is the component this touches.
 *
 * WHAT CHANGES, EXACTLY ONE COLUMN. Features, policy target and legal mask are taken
 * from the SHALLOW set -- the arm that won -- so this is not confounded with the
 * policy-depth change that just lost. Only the final float, the value target, is
 * replaced with `rootValue` from the 4096-sim relabel pass.
 *
 * ORDERING IS THE FAILURE MODE. `distill-shards.mjs` walks `usable` plies in order
 * and keys the deep record by `ply.idx`; if this script paired root values to
 * records in any other order, every label would be silently wrong and the shards
 * would still look perfectly well-formed. So it does NOT trust the ordering: it
 * rebuilds `z` independently and asserts it matches the `z` already on disk, record
 * for record, before writing anything. If the ordering is wrong, that check fails
 * instead of producing a plausible corpus.
 *
 *   node scripts/distill-value-shards.mjs --in /Users/scott/wwplay/distill/g4 --deep deep-4096
 */

import { readFileSync, mkdirSync, readdirSync } from 'node:fs';
import path from 'node:path';

import { FEATURE_LEN, POLICY_LEN, RECORD_FLOATS, parseArgs } from './distill-lib.mjs';
import {
  SHARD_HEADER_BYTES, readShardHeader, writeShard
} from '../../wrongway-training/cluster/shard-format.mjs';

/**
 * Read a shard back into a flat Float32Array.
 *
 * `shard-format.mjs` exports a writer and a HEADER parser but no whole-file reader,
 * so the body is sliced here. `readShardHeader` still does the validating -- magic,
 * version, field widths and the exact file length -- and the layout assertion below
 * catches a shard written with different dimensions, which would otherwise reindex
 * every record silently.
 */
function readShardFile(file) {
  const buf = readFileSync(file);
  const header = readShardHeader(buf, buf.length, path.basename(file));
  if (header.recordFloats !== RECORD_FLOATS) {
    throw new Error(`${file}: recordFloats ${header.recordFloats}, expected ${RECORD_FLOATS}`);
  }
  const body = buf.subarray(SHARD_HEADER_BYTES);
  // Copy rather than view: the returned array is mutated and written back out.
  return {
    count: header.recordCount,
    records: new Float32Array(
      body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength)
    )
  };
}

const args = parseArgs(process.argv.slice(2), {
  in: path.join(process.env.HOME || '/tmp', 'wwplay/distill/g4'),
  deep: 'deep-4096',
  from: 'shallow-512',
  out: 'value-deep'
});

const root = args.in;
const srcDir = path.join(root, 'shards', args.from);
const outDir = path.join(root, 'shards', args.out);
mkdirSync(outDir, { recursive: true });

// Deep root values, keyed the same way distill-shards keys its deep records.
const rootValueByIdx = new Map();
for (const line of readFileSync(path.join(root, `relabel-${args.deep}.jsonl`), 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const r = JSON.parse(line);
  rootValueByIdx.set(r.idx, r.rootValue);
}

// `z` rebuilt from plies.jsonl, used ONLY as an ordering proof.
const zByIdx = new Map();
for (const line of readFileSync(path.join(root, 'plies.jsonl'), 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const p = JSON.parse(line);
  zByIdx.set(p.idx, p.z);
}
const order = [...zByIdx.keys()].sort((a, b) => a - b);

const files = readdirSync(srcDir).filter((f) => f.endsWith('.bin')).sort();
let cursor = 0;
let written = 0;
let vSum = 0;
let zSum = 0;
let agree = 0;
const outFiles = [];

for (const f of files) {
  const { records, count } = readShardFile(path.join(srcDir, f));
  const buf = new Float32Array(records);
  for (let r = 0; r < count; r += 1) {
    const at = r * RECORD_FLOATS;
    const zAt = at + FEATURE_LEN + POLICY_LEN + POLICY_LEN;
    const idx = order[cursor];
    if (idx === undefined) throw new Error(`ran out of plies at record ${cursor}`);
    const expectZ = zByIdx.get(idx);
    const onDisk = buf[zAt];
    // THE ORDERING PROOF. Exact equality: both sides are the same float32 written by
    // the same writer, so any mismatch means the pairing is wrong, not that a
    // tolerance was too tight.
    if (onDisk !== expectZ) {
      throw new Error(`ordering mismatch at record ${cursor} (ply idx ${idx}): `
        + `shard z=${onDisk} but plies.jsonl z=${expectZ}. Pairing is wrong -- refusing to write.`);
    }
    const rv = rootValueByIdx.get(idx);
    if (rv === undefined || !Number.isFinite(rv)) {
      throw new Error(`no finite rootValue for ply idx ${idx}`);
    }
    buf[zAt] = rv;
    if (Math.sign(rv) === Math.sign(expectZ) || expectZ === 0) agree += 1;
    vSum += Math.abs(rv);
    zSum += Math.abs(expectZ);
    cursor += 1;
    written += 1;
  }
  const outFile = path.join(outDir, f);
  const bytes = writeShard(outFile, buf, count, {
    featureLen: FEATURE_LEN, policyLen: POLICY_LEN, maskLen: POLICY_LEN, zLen: 1
  });
  outFiles.push({ file: outFile, records: count, bytes });
  console.log(`[value-shards] ${f}: ${count} records, value target swapped`);
}

if (cursor !== order.length) {
  throw new Error(`wrote ${cursor} records but plies.jsonl has ${order.length}`);
}

console.log(`\n[value-shards] ${written} records -> ${outDir}`);
console.log(`  ordering proof: PASSED (every record's on-disk z matched plies.jsonl)`);
console.log(`  mean |z| (outcome)      ${(zSum / written).toFixed(4)}`);
console.log(`  mean |rootValue| (deep) ${(vSum / written).toFixed(4)}`);
console.log(`  sign agreement z vs rootValue: ${(100 * agree / written).toFixed(1)}%`);
console.log(`  policy target unchanged (taken from ${args.from}), only the value column differs`);
