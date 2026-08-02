#!/usr/bin/env node
/**
 * Reference side of the `js_math` parity harness.
 *
 * `crate::js_math::js_log` claims to be `Math.log` bit-for-bit. That claim is
 * worth exactly as much as the evidence behind it, so this dumps real V8
 * `Math.log` output over the ranges the PUCT search actually feeds it — the
 * `(raw + 0.5) / 2^32` grid the Gumbel draw uses, the `-log(u)` values that grid
 * produces, renormalised priors, a wide exponent sweep, and the near-1 region
 * where the algorithm switches to its short polynomial.
 *
 *   <logs.bin>    interleaved (x, Math.log(x)) as little-endian f64
 *   <stream.bin>  the first N outputs of `createLcg32(seed)` as u32 LE
 *
 * Raw bytes, not decimal: the comparison is on bit patterns.
 */

import { writeFileSync } from 'node:fs';

import { createLcg32 } from '../js/lcg32.mjs';

const [logsPath, streamPath] = process.argv.slice(2);
if (!logsPath || !streamPath) {
  console.error('usage: dump-normal-duel-js-math.mjs <logs.bin> <stream.bin>');
  process.exit(2);
}

const inputs = [];
const random = createLcg32(99991);

// The exact grid `gumbel()` draws from, and the `-log(u)` values it then logs.
for (let index = 0; index < 120000; index += 1) {
  const u = (random() + 0.5) / 4294967296;
  inputs.push(u, -Math.log(u));
}
// Renormalised priors: a policy entry over a mass of up to ~200 entries.
for (let index = 0; index < 60000; index += 1) {
  inputs.push((random() / 4294967296) / (1 + (random() / 4294967296) * 200));
}
// Wide exponent sweep, including subnormal-adjacent scales.
for (let index = 0; index < 60000; index += 1) {
  inputs.push((1 + random() / 4294967296) * 2 ** ((random() % 80) - 40));
}
// The |f| < 2^-20 branch, where the algorithm drops to two polynomial terms.
for (let index = 0; index < 20000; index += 1) {
  inputs.push(1 + (random() / 4294967296 - 0.5) * 1e-6);
}
// The floor the search clamps a zero prior to, and the exact powers of two.
inputs.push(1e-9, 1, 0.5, 2, 0.25, 4, Number.MIN_VALUE, Number.EPSILON);

const usable = inputs.filter((x) => Number.isFinite(x) && x > 0);
const logs = new Float64Array(usable.length * 2);
for (let index = 0; index < usable.length; index += 1) {
  logs[index * 2] = usable[index];
  logs[index * 2 + 1] = Math.log(usable[index]);
}
writeFileSync(logsPath, Buffer.from(logs.buffer, logs.byteOffset, logs.byteLength));

const stream = new Uint32Array(4096);
const streamRandom = createLcg32(1234);
for (let index = 0; index < stream.length; index += 1) stream[index] = streamRandom();
writeFileSync(streamPath, Buffer.from(stream.buffer, stream.byteOffset, stream.byteLength));

console.error(`dumped ${usable.length} log samples and ${stream.length} lcg32 draws`);
