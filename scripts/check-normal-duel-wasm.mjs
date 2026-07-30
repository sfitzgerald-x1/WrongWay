#!/usr/bin/env node

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';

import {
  applyAction,
  createInitialState,
  decodeAction,
  legalActionCodes,
  positionKey
} from '../js/normal-duel-engine.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = resolve(ROOT, 'rust/target/wasm-bindgen/web');
const fixture = JSON.parse(
  readFileSync(resolve(ROOT, 'tests/fixtures/normal-duel-perft-v1.json'), 'utf8')
);
const wasm = await import(pathToFileURL(resolve(OUTPUT, 'wrongway_normal_duel.js')));
const bytes = readFileSync(resolve(OUTPUT, 'wrongway_normal_duel_bg.wasm'));

await wasm.default({ module_or_path: bytes });
assert.equal(wasm.normalDuelRuleset(), 'normal-duel-v1');
assert.equal(wasm.normalDuelVersion(), '0.1.0');

for (const configId of ['standardA', 'standardB', 'blitzA']) {
  const config = fixture.configs[configId];
  const configJson = JSON.stringify(config);
  const expectedInitial = createInitialState(config);
  const actualInitialJson = wasm.normalDuelInitialState(configJson);
  const actualInitial = JSON.parse(actualInitialJson);
  assert.deepEqual(actualInitial, expectedInitial, `${configId}: initial state`);

  const expectedCodes = legalActionCodes(config, expectedInitial);
  assert.deepEqual(
    JSON.parse(wasm.normalDuelLegalActionCodes(configJson, actualInitialJson)),
    expectedCodes,
    `${configId}: legal action codes`
  );

  for (const code of [expectedCodes[0], expectedCodes.at(-1)]) {
    const action = decodeAction(config, code);
    const expectedChild = applyAction(config, expectedInitial, action);
    const actualChild = JSON.parse(
      wasm.normalDuelApplyAction(configJson, actualInitialJson, JSON.stringify(action))
    );
    assert.deepEqual(actualChild, expectedChild, `${configId}: apply action ${code}`);
    assert.equal(
      wasm.normalDuelPositionKey(configJson, JSON.stringify(actualChild.position)),
      positionKey(config, expectedChild.position),
      `${configId}: position key ${code}`
    );
  }
}

const exponentConfig = JSON.stringify(fixture.configs.standardA).replace(
  '"plyCap":200',
  '"plyCap":2e2'
);
assert.deepEqual(
  JSON.parse(wasm.normalDuelInitialState(exponentConfig)),
  createInitialState(fixture.configs.standardA),
  'JavaScript-safe exponent integers'
);

function expectError(code, operation) {
  assert.throws(operation, (error) => error === code, code);
}

expectError('invalid_config', () => wasm.normalDuelInitialState('{'));
expectError('unsupported_feature', () => wasm.normalDuelInitialState(JSON.stringify({
  ...fixture.configs.standardA,
  features: { hammer: true }
})));
const standardConfigJson = JSON.stringify(fixture.configs.standardA);
const standardInitialJson = wasm.normalDuelInitialState(standardConfigJson);
expectError('invalid_action', () => wasm.normalDuelApplyAction(
  standardConfigJson,
  standardInitialJson,
  '{"kind":"wall","wall":"H-9-0"}'
));
const terminal = fixture.cases.find((entry) => entry.id === 'terminal-goal-win-7x7-a');
assert.ok(terminal, 'terminal fixture exists');
expectError('terminal_state', () => wasm.normalDuelApplyAction(
  JSON.stringify(fixture.configs[terminal.configId]),
  JSON.stringify(terminal.state),
  '{}'
));

process.stdout.write('browser-target wasm instantiated and matched the JavaScript boundary\n');
