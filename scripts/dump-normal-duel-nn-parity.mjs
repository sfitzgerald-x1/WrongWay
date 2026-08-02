#!/usr/bin/env node
/**
 * Reference side of the Rust hot-path parity harness.
 *
 * Reads `{ config, states: [...] }`, runs the real `js/` modules over every
 * state, and writes what the Rust test compares against:
 *
 *   <codes.json>  { "codes": [[actionCode, ...], ...] }
 *   <planes.bin>  every `encodeState` result concatenated, raw little-endian
 *                 f32, so the comparison is on bit patterns rather than on a
 *                 decimal round trip.
 *
 * Nothing here reimplements a rule or an encoding; both answers come straight
 * out of the shipped modules.
 */

import { writeFileSync, readFileSync } from 'node:fs';

import { legalActionCodes } from '../js/normal-duel-engine.mjs';
import { encodeState, NN_INPUT_PLANES } from '../js/normal-duel-nn-encoding.mjs';

const [inputPath, codesPath, planesPath] = process.argv.slice(2);
if (!inputPath || !codesPath || !planesPath) {
  console.error('usage: dump-normal-duel-nn-parity.mjs <input.json> <codes.json> <planes.bin>');
  process.exit(2);
}

const { config, states } = JSON.parse(readFileSync(inputPath, 'utf8'));
const cells = config.rows * config.columns;
const planeFloats = NN_INPUT_PLANES * cells;

const codes = [];
const planes = new Float32Array(states.length * planeFloats);
for (let index = 0; index < states.length; index += 1) {
  const state = states[index];
  codes.push([...legalActionCodes(config, state)]);
  planes.set(encodeState(config, state), index * planeFloats);
}

writeFileSync(codesPath, JSON.stringify({ codes }));
writeFileSync(planesPath, Buffer.from(planes.buffer, planes.byteOffset, planes.byteLength));
