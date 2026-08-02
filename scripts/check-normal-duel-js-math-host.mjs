#!/usr/bin/env node
/**
 * Diagnostic side of the `js_math` parity harness.
 *
 * Compares the *host's live* `Math.log` against the committed fixture and
 * reports how far apart they are. This is deliberately not a pass/fail check:
 * V8's `Math.log` is C++ compiled by the host toolchain, and whether `a*b+c`
 * contracts to an FMA is target-dependent, so the honest answer on x86-64 is
 * "the local engine differs from the fixture" — a fact about the engine, not a
 * defect in the Rust port.
 *
 * Usage: node scripts/check-normal-duel-js-math-host.mjs <js_log_reference.txt>
 * Prints a one-line-per-field report on stdout; exit status is 0 unless the
 * fixture could not be read.
 */

import { readFileSync } from 'node:fs';

const [fixturePath] = process.argv.slice(2);
if (!fixturePath) {
  console.error('usage: check-normal-duel-js-math-host.mjs <js_log_reference.txt>');
  process.exit(2);
}

const scratch = new DataView(new ArrayBuffer(8));
function bits(value) {
  scratch.setFloat64(0, value);
  return scratch.getBigUint64(0).toString(16).padStart(16, '0');
}
function fromBits(hex) {
  scratch.setBigUint64(0, BigInt(`0x${hex}`));
  return scratch.getFloat64(0);
}

let fixtureArch = 'unknown';
let fixtureNode = 'unknown';
let total = 0;
let mismatches = 0;
let first = null;

for (const line of readFileSync(fixturePath, 'utf8').split('\n')) {
  if (line.startsWith('# arch:')) fixtureArch = line.slice(7).trim();
  if (line.startsWith('# node:')) fixtureNode = line.slice(7).trim();
  if (line.startsWith('#') || line.trim() === '') continue;
  const [xHex, expectedHex] = line.split(' ');
  const x = fromBits(xHex);
  const actualHex = bits(Math.log(x));
  total += 1;
  if (actualHex !== expectedHex) {
    mismatches += 1;
    if (first === null) first = { x, expectedHex, actualHex };
  }
}

console.log(`host-node ${process.version}`);
console.log(`host-arch ${process.arch}`);
console.log(`host-platform ${process.platform}`);
console.log(`fixture-node ${fixtureNode}`);
console.log(`fixture-arch ${fixtureArch}`);
console.log(`total ${total}`);
console.log(`mismatches ${mismatches}`);
if (first !== null) {
  console.log(`first-x ${first.x}`);
  console.log(`first-fixture ${first.expectedHex}`);
  console.log(`first-host ${first.actualHex}`);
}
