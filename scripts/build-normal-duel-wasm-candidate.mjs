#!/usr/bin/env node

/** Build the sealed browser-WASM candidate release used by strength evaluation. */
import { createHash } from 'node:crypto';
import {
  copyFileSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync
} from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

import { CANDIDATE_ARTIFACT_MANIFEST_FORMAT } from './evaluation/candidate-artifact-manifest.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WEB_OUTPUT = resolve(ROOT, 'rust/target/wasm-bindgen/web');
const RELEASE = resolve(ROOT, 'rust/target/wasm-candidate/release');
const ADAPTER_TEMPLATE = resolve(ROOT, 'scripts/evaluation/normal-duel-wasm-candidate-adapter.mjs');
const RELEASE_FILES = Object.freeze([
  Object.freeze({ source: ADAPTER_TEMPLATE, target: 'adapter.mjs' }),
  Object.freeze({ source: resolve(WEB_OUTPUT, 'wrongway_normal_duel.js'), target: 'normal-duel-wasm.mjs' }),
  Object.freeze({ source: resolve(WEB_OUTPUT, 'wrongway_normal_duel_bg.wasm'), target: 'normal-duel-wasm_bg.wasm' })
]);

function fail(message) {
  throw new Error(`build-normal-duel-wasm-candidate: ${message}`);
}

function sha256(filename) {
  return createHash('sha256').update(readFileSync(filename)).digest('hex');
}

function runBrowserBuild() {
  const result = spawnSync(process.execPath, [resolve(ROOT, 'scripts/build-normal-duel-wasm.mjs')], {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: 'inherit'
  });
  if (result.error) fail(`browser WASM build could not start: ${result.error.message}`);
  if (result.status !== 0) fail(`browser WASM build failed with exit ${result.status}`);
}

function renderManifest() {
  const files = RELEASE_FILES
    .map(({ target }) => ({ path: target, sha256: sha256(resolve(RELEASE, target)) }))
    .sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0));
  return `${JSON.stringify({
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    entry: 'adapter.mjs',
    files
  }, null, 2)}\n`;
}

runBrowserBuild();
rmSync(RELEASE, { recursive: true, force: true });
mkdirSync(RELEASE, { recursive: true });
for (const { source, target } of RELEASE_FILES) {
  copyFileSync(source, resolve(RELEASE, target));
}
writeFileSync(resolve(RELEASE, 'manifest.json'), renderManifest());

const actual = readdirSync(RELEASE).sort();
const expected = ['adapter.mjs', 'manifest.json', 'normal-duel-wasm.mjs', 'normal-duel-wasm_bg.wasm'];
if (JSON.stringify(actual) !== JSON.stringify(expected)) fail('release directory shape is not hermetic');
process.stdout.write(`${RELEASE}\n`);
