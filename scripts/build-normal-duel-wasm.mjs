#!/usr/bin/env node

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WASM_BINDGEN_VERSION = '0.2.126';
const EXPECTED_CLI_VERSION = `wasm-bindgen ${WASM_BINDGEN_VERSION}`;
const TARGET = resolve(ROOT, 'rust/target');
const INPUT = resolve(TARGET, 'wasm32-unknown-unknown/release/wrongway_normal_duel_wasm.wasm');
const OUTPUT = resolve(TARGET, 'wasm-bindgen/web');
const WASM_BINDGEN_EXECUTABLE = process.platform === 'win32' ? 'wasm-bindgen.exe' : 'wasm-bindgen';
const CARGO_WASM_BINDGEN = process.env.CARGO_HOME
  ? resolve(process.env.CARGO_HOME, 'bin', WASM_BINDGEN_EXECUTABLE)
  : resolve(homedir(), '.cargo/bin', WASM_BINDGEN_EXECUTABLE);
const WASM_BINDGEN_OVERRIDE = process.env.WASM_BINDGEN_BIN?.trim();
const WASM_BINDGEN_COMMAND = WASM_BINDGEN_OVERRIDE
  || (existsSync(CARGO_WASM_BINDGEN) ? CARGO_WASM_BINDGEN : WASM_BINDGEN_EXECUTABLE);

function fail(message) {
  throw new Error(`build-normal-duel-wasm: ${message}`);
}

function run(command, args, capture = false) {
  const result = spawnSync(command, args, {
    cwd: ROOT,
    encoding: 'utf8',
    stdio: capture ? ['ignore', 'pipe', 'pipe'] : 'inherit'
  });
  if (result.error) fail(`${command} could not start: ${result.error.message}`);
  if (result.status !== 0) {
    const detail = capture ? result.stderr.trim() : `exit ${result.status}`;
    fail(`${command} ${args.join(' ')} failed: ${detail}`);
  }
  return result.stdout?.trim() ?? '';
}

const installedVersion = run(WASM_BINDGEN_COMMAND, ['--version'], true);
if (installedVersion !== EXPECTED_CLI_VERSION) {
  fail(
    `requires ${EXPECTED_CLI_VERSION}, found ${installedVersion || 'no version'}; `
    + `install with "cargo install wasm-bindgen-cli --version ${WASM_BINDGEN_VERSION} --locked"`
  );
}

run('cargo', [
  'build',
  '--manifest-path', 'rust/Cargo.toml',
  '--package', 'wrongway-normal-duel-wasm',
  '--release',
  '--target', 'wasm32-unknown-unknown',
  '--locked'
]);

mkdirSync(OUTPUT, { recursive: true });
run(WASM_BINDGEN_COMMAND, [
  '--target', 'web',
  '--out-dir', OUTPUT,
  '--out-name', 'wrongway_normal_duel',
  INPUT
]);

// Node uses this browser-target package only for boundary verification. The
// production browser can load the same standards-based ESM glue directly.
writeFileSync(resolve(OUTPUT, 'package.json'), '{"private":true,"type":"module"}\n');
process.stdout.write(`${OUTPUT}\n`);
