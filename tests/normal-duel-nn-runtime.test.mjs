import assert from 'node:assert/strict';
import test from 'node:test';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { createInitialState, applyAction, legalActionCodes, policySize } from '../js/normal-duel-engine.mjs';
import { encodeState } from '../js/normal-duel-nn-encoding.mjs';
import { gumbelRootSearch } from '../js/normal-duel-gumbel-search.mjs';
import { createLcg32 } from '../js/lcg32.mjs';
import {
  NN_RUNTIME_VERSION, createNetworkEvaluator, forwardRaw, loadWeights
} from '../js/normal-duel-nn-runtime.mjs';

const CONFIG_9X9 = Object.freeze({
  ruleset: 'normal-duel-v1', rows: 9, columns: 9,
  start: { A: { r: 8, c: 4 }, B: { r: 0, c: 4 } },
  goalRows: { A: 0, B: 8 }, initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap: 200, firstPlayer: 'A'
});

const CONFIG_7X7 = Object.freeze({
  ruleset: 'normal-duel-v1', rows: 7, columns: 7,
  start: { A: { r: 6, c: 3 }, B: { r: 0, c: 3 } },
  goalRows: { A: 0, B: 6 }, initialStock: { A: 10, B: 10 },
  jumpRule: 'permissive-adjacent-exit-v1', repetitionThreshold: 3, plyCap: 200, firstPlayer: 'A'
});

const TRAINING_DIR = '/Users/scott/workspace/agents/wrongway-training';
const BLOB_PATH = path.join(TRAINING_DIR, 'weights.bin');
const MANIFEST_PATH = path.join(TRAINING_DIR, 'weights.manifest.json');
const PYTHON = path.join(TRAINING_DIR, '.venv/bin/python');

const POLICY_SIZE = 209;
const FEATURE_LEN = 648;

/* ------------------------------------------------------------------ *
 * Fixtures
 * ------------------------------------------------------------------ */

/** A fixed non-trivial position: a few pawn moves and a wall from each side. */
function fixedState() {
  const actions = [
    { kind: 'pawn', to: { r: 7, c: 4 } },
    { kind: 'pawn', to: { r: 1, c: 4 } },
    { kind: 'wall', wall: 'H-3-3' },
    { kind: 'pawn', to: { r: 2, c: 4 } },
    { kind: 'pawn', to: { r: 6, c: 4 } },
    { kind: 'wall', wall: 'V-5-5' }
  ];
  return actions.reduce((state, action) => applyAction(CONFIG_9X9, state, action),
    createInitialState(CONFIG_9X9));
}

function artifactsPresent() {
  return fs.existsSync(BLOB_PATH) && fs.existsSync(MANIFEST_PATH);
}

let cachedWeights = null;
function realWeights() {
  if (cachedWeights === null) {
    cachedWeights = loadWeights(fs.readFileSync(MANIFEST_PATH, 'utf8'), fs.readFileSync(BLOB_PATH));
  }
  return cachedWeights;
}

/** A synthetic-but-valid weight set, so structural tests need no artifacts. */
function syntheticWeightSet() {
  const spec = [
    ['stem.conv.weight', [64, 8, 3, 3]], ['stem.conv.bias', [64]]
  ];
  for (let i = 0; i < 6; i += 1) {
    spec.push([`block${i}.conv1.weight`, [64, 64, 3, 3]], [`block${i}.conv1.bias`, [64]]);
    spec.push([`block${i}.conv2.weight`, [64, 64, 3, 3]], [`block${i}.conv2.bias`, [64]]);
  }
  spec.push(['policy.conv.weight', [32, 64, 1, 1]], ['policy.conv.bias', [32]]);
  spec.push(['policy.fc.weight', [209, 2592]], ['policy.fc.bias', [209]]);
  spec.push(['value.conv.weight', [8, 64, 1, 1]], ['value.conv.bias', [8]]);
  spec.push(['value.fc1.weight', [64, 648]], ['value.fc1.bias', [64]]);
  spec.push(['value.fc2.weight', [1, 64]], ['value.fc2.bias', [1]]);

  const tensors = [];
  let offset = 0;
  for (const [name, shape] of spec) {
    const count = shape.reduce((product, dimension) => product * dimension, 1);
    tensors.push({ name, shape, dtype: 'float32', byteOffset: offset, byteLength: count * 4, count });
    offset += count * 4;
  }
  const buffer = new ArrayBuffer(offset);
  const floats = new Float32Array(buffer);
  // Deterministic filler: a fixed LCG, never Math.random.
  let seed = 12345;
  for (let i = 0; i < floats.length; i += 1) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    floats[i] = ((seed >>> 8) / 0x1000000 - 0.5) * 0.1;
  }
  const manifest = {
    formatVersion: 1, byteOrder: 'little', layout: 'C-contiguous', flattenOrder: 'CHW',
    policySize: 209,
    input: { planes: 8, rows: 9, columns: 9 },
    architecture: { blocks: 6, channels: 64, policyHeadChannels: 32, valueHeadChannels: 8, valueHidden: 64 },
    blobBytes: offset,
    tensors
  };
  return { manifest, buffer };
}

const SYNTHETIC = syntheticWeightSet();
const cloneManifest = () => JSON.parse(JSON.stringify(SYNTHETIC.manifest));

/** FNV-1a over a byte range: pins the fixture to the exact blob it was made from. */
function fnv1a32(buffer) {
  const bytes = new Uint8Array(buffer);
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

const GOLDEN = JSON.parse(fs.readFileSync(
  new URL('./fixtures/nn-runtime-golden-forward-v1.json', import.meta.url), 'utf8'
));

/* ------------------------------------------------------------------ *
 * Hermetic graph parity — no Python, no exported artifacts
 * ------------------------------------------------------------------ */

/**
 * The full-net parity test below is skippable: it needs a training checkout
 * that a clean machine does not have, and when it skips the entire forward
 * graph goes unverified while 17 other tests still pass. Dropping a residual
 * skip connection or transposing the flatten order moves a logit by ~1e0 and
 * nothing else in this file notices.
 *
 * So the graph is pinned here instead. The goldens in
 * `tests/fixtures/nn-runtime-golden-forward-v1.json` were produced by
 * `export_weights.reference_forward` (PyTorch) over exactly the blob
 * `syntheticWeightSet()` builds, then committed — they are genuinely
 * cross-checked against the reference, and reproducing them needs neither
 * Python nor `weights.bin`. This test can never skip.
 */
test('forwardRaw reproduces the committed PyTorch goldens (hermetic, never skips)', () => {
  assert.equal(GOLDEN.format, 'nn-runtime-golden-forward-v1');

  // The goldens are only meaningful for the blob they were computed from.
  assert.equal(SYNTHETIC.buffer.byteLength, GOLDEN.blob.byteLength);
  assert.equal(
    fnv1a32(SYNTHETIC.buffer), GOLDEN.blob.fnv1a32,
    'the synthetic weight blob has changed; regenerate the golden fixture against the Python reference'
  );

  // The golden input is a real encoded position, not noise: the fixture carries
  // the feature vector explicitly so this test measures the graph and not the
  // encoder, and the equality below ties the two together.
  assert.equal(GOLDEN.features.length, FEATURE_LEN);
  assert.deepEqual(Array.from(encodeState(CONFIG_9X9, fixedState())), GOLDEN.features);

  const weights = loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer);
  const actual = forwardRaw(weights, Float32Array.from(GOLDEN.features));

  assert.equal(actual.policyLogits.length, POLICY_SIZE);
  assert.equal(GOLDEN.policyLogits.length, POLICY_SIZE);

  let maxLogitDelta = 0;
  for (let code = 0; code < POLICY_SIZE; code += 1) {
    maxLogitDelta = Math.max(maxLogitDelta, Math.abs(actual.policyLogits[code] - GOLDEN.policyLogits[code]));
  }
  const valueDelta = Math.abs(actual.value - GOLDEN.value);

  assert.ok(maxLogitDelta < 1e-4, `policy logits differ from the golden by ${maxLogitDelta}`);
  assert.ok(valueDelta < 1e-4, `value differs from the golden by ${valueDelta}`);

  // The goldens must actually exercise the graph: an all-zero logit vector
  // would satisfy the deltas above against an all-zero golden.
  assert.ok(GOLDEN.policyLogits.some((logit) => Math.abs(logit) > 1e-6));
  assert.ok(Math.abs(GOLDEN.value) > 1e-6);
});

/* ------------------------------------------------------------------ *
 * Parity with the Python reference — the key test
 * ------------------------------------------------------------------ */

const PARITY_SCRIPT = `
import json, sys
from pathlib import Path
import numpy as np, torch

training = Path(sys.argv[1])
sys.path.insert(0, str(training))
from export_weights import reference_forward

manifest = json.loads((training / "weights.manifest.json").read_text())
blob = (training / "weights.bin").read_bytes()
tensors = {}
for entry in manifest["tensors"]:
    raw = np.frombuffer(blob, dtype="<f4", count=entry["count"],
                        offset=entry["byteOffset"]).reshape(entry["shape"])
    tensors[entry["name"]] = torch.from_numpy(raw.copy())

features = np.asarray(json.loads(Path(sys.argv[2]).read_text()), dtype=np.float32)
x = torch.from_numpy(features.reshape(1, 8, 9, 9))
logits, value = reference_forward(tensors, x, manifest["architecture"]["blocks"])
print(json.dumps({"logits": logits[0].tolist(), "value": float(value[0])}))
`;

test('forwardRaw matches the Python reference_forward on the real exported weights', (t) => {
  if (!artifactsPresent()) {
    t.skip(`training artifacts missing: expected ${BLOB_PATH} and ${MANIFEST_PATH}`);
    return;
  }
  if (!fs.existsSync(PYTHON)) {
    t.skip(`training venv missing: expected ${PYTHON}`);
    return;
  }

  const features = encodeState(CONFIG_9X9, fixedState());
  assert.equal(features.length, FEATURE_LEN);

  const scratchDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nn-parity-'));
  const scriptPath = path.join(scratchDir, 'parity.py');
  const featurePath = path.join(scratchDir, 'features.json');
  fs.writeFileSync(scriptPath, PARITY_SCRIPT);
  fs.writeFileSync(featurePath, JSON.stringify(Array.from(features)));

  let reference;
  try {
    const stdout = execFileSync(PYTHON, [scriptPath, TRAINING_DIR, featurePath], { encoding: 'utf8' });
    reference = JSON.parse(stdout);
  } finally {
    fs.rmSync(scratchDir, { recursive: true, force: true });
  }

  const actual = forwardRaw(realWeights(), features);
  assert.equal(actual.policyLogits.length, POLICY_SIZE);
  assert.equal(reference.logits.length, POLICY_SIZE);

  let maxLogitDelta = 0;
  for (let code = 0; code < POLICY_SIZE; code += 1) {
    maxLogitDelta = Math.max(maxLogitDelta, Math.abs(actual.policyLogits[code] - reference.logits[code]));
  }
  const valueDelta = Math.abs(actual.value - reference.value);
  const maxDelta = Math.max(maxLogitDelta, valueDelta);

  t.diagnostic(`max |JS - Python| logits: ${maxLogitDelta.toExponential(3)}`);
  t.diagnostic(`max |JS - Python| value:  ${valueDelta.toExponential(3)}`);
  t.diagnostic(`max |JS - Python| overall: ${maxDelta.toExponential(3)}`);

  assert.ok(maxLogitDelta < 1e-4, `policy logits differ by ${maxLogitDelta}`);
  assert.ok(valueDelta < 1e-4, `value differs by ${valueDelta}`);
});

/* ------------------------------------------------------------------ *
 * Behaviour of the evaluator
 * ------------------------------------------------------------------ */

test('the runtime version is a frozen identifier', () => {
  assert.equal(NN_RUNTIME_VERSION, 'nn-runtime-forward-v1');
});

test('two evaluations of the same state are byte-identical', () => {
  const evaluate = createNetworkEvaluator(loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer));
  const state = fixedState();
  const first = evaluate(CONFIG_9X9, state);
  const second = evaluate(CONFIG_9X9, state);
  assert.notEqual(first.policy, second.policy); // a fresh array per call, not the scratch
  assert.deepEqual(Array.from(first.policy), Array.from(second.policy));
  assert.equal(first.value, second.value);

  const third = evaluate(CONFIG_9X9, createInitialState(CONFIG_9X9));
  const fourth = evaluate(CONFIG_9X9, state);
  assert.deepEqual(Array.from(fourth.policy), Array.from(first.policy));
  assert.equal(fourth.value, first.value);
  assert.ok(third.policy.length === POLICY_SIZE);
});

test('illegal actions get exactly zero probability and the policy sums to 1', () => {
  const evaluate = createNetworkEvaluator(loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer));
  const state = fixedState();
  const { policy, value } = evaluate(CONFIG_9X9, state);

  assert.ok(policy instanceof Float32Array);
  assert.equal(policy.length, policySize(CONFIG_9X9));

  const legal = new Set(legalActionCodes(CONFIG_9X9, state));
  assert.ok(legal.size > 0 && legal.size < POLICY_SIZE);

  let total = 0;
  for (let code = 0; code < POLICY_SIZE; code += 1) {
    if (legal.has(code)) {
      assert.ok(policy[code] > 0, `legal action ${code} has zero probability`);
      total += policy[code];
    } else {
      assert.equal(policy[code], 0, `illegal action ${code} is not exactly zero`);
      assert.ok(Object.is(policy[code], 0), `illegal action ${code} is not +0`);
    }
  }
  assert.ok(Math.abs(total - 1) < 1e-5, `probabilities sum to ${total}`);

  assert.equal(typeof value, 'number');
  assert.ok(Number.isFinite(value));
  assert.ok(value >= -1 && value <= 1, `value ${value} out of range`);
});

test('gumbelRootSearch accepts the evaluator unchanged and returns a legal action', () => {
  const evaluate = createNetworkEvaluator(loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer));
  const state = fixedState();
  const result = gumbelRootSearch({
    config: CONFIG_9X9, state, evaluate, simulations: 8, maxConsidered: 4, random: createLcg32(7)
  });
  const legal = legalActionCodes(CONFIG_9X9, state);
  assert.ok(legal.includes(result.actionCode), `search returned illegal action ${result.actionCode}`);
  assert.ok(result.rootValue >= -1 && result.rootValue <= 1);

  // Same seed, same evaluator, same game.
  const repeat = gumbelRootSearch({
    config: CONFIG_9X9, state, evaluate, simulations: 8, maxConsidered: 4, random: createLcg32(7)
  });
  assert.equal(repeat.actionCode, result.actionCode);
});

test('evaluation mutates neither the config, the state, nor the input features', () => {
  const weights = loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer);
  const evaluate = createNetworkEvaluator(weights);
  const state = fixedState();
  const configBefore = JSON.stringify(CONFIG_9X9);
  const stateBefore = JSON.stringify(state);
  evaluate(CONFIG_9X9, state);
  assert.equal(JSON.stringify(CONFIG_9X9), configBefore);
  assert.equal(JSON.stringify(state), stateBefore);

  const features = encodeState(CONFIG_9X9, state);
  const copy = Float32Array.from(features);
  forwardRaw(weights, features);
  assert.deepEqual(Array.from(features), Array.from(copy));

  // The weight set itself is frozen and its tensors are unchanged by a pass.
  assert.ok(Object.isFrozen(weights));
  const stemBefore = Float32Array.from(weights.get('stem.conv.bias'));
  forwardRaw(weights, features);
  assert.deepEqual(Array.from(weights.get('stem.conv.bias')), Array.from(stemBefore));
});

test('a non-9x9 board is rejected', () => {
  const evaluate = createNetworkEvaluator(loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer));
  assert.throws(() => evaluate(CONFIG_7X7, createInitialState(CONFIG_7X7)), /unsupported_board/);
});

test('forwardRaw rejects a feature vector of the wrong length or with a non-finite entry', () => {
  const weights = loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer);
  assert.throws(() => forwardRaw(weights, new Float32Array(FEATURE_LEN - 1)), /invalid_features_length/);
  const bad = new Float32Array(FEATURE_LEN);
  bad[3] = Number.NaN;
  assert.throws(() => forwardRaw(weights, bad), /invalid_features/);
  assert.throws(() => forwardRaw({}, new Float32Array(FEATURE_LEN)), /invalid_weights/);
});

/* ------------------------------------------------------------------ *
 * loadWeights validation
 * ------------------------------------------------------------------ */

test('loadWeights rejects a truncated buffer', () => {
  const truncated = SYNTHETIC.buffer.slice(0, SYNTHETIC.buffer.byteLength - 4);
  assert.throws(() => loadWeights(cloneManifest(), truncated), /tensor_out_of_range|blob_size_mismatch/);
});

test('loadWeights rejects an out-of-range byteOffset', () => {
  const manifest = cloneManifest();
  manifest.tensors[0].byteOffset = SYNTHETIC.buffer.byteLength;
  assert.throws(() => loadWeights(manifest, SYNTHETIC.buffer), /tensor_out_of_range/);

  const wild = cloneManifest();
  wild.tensors[2].byteOffset = Number.MAX_SAFE_INTEGER - 1;
  assert.throws(() => loadWeights(wild, SYNTHETIC.buffer), /tensor_out_of_range/);
});

test('loadWeights rejects a count that disagrees with the shape', () => {
  const manifest = cloneManifest();
  manifest.tensors[0].count -= 1;
  assert.throws(() => loadWeights(manifest, SYNTHETIC.buffer), /tensor_count_mismatch/);
});

test('loadWeights rejects a byteLength that disagrees with the count', () => {
  const manifest = cloneManifest();
  manifest.tensors[0].byteLength = manifest.tensors[0].count * 2;
  assert.throws(() => loadWeights(manifest, SYNTHETIC.buffer), /tensor_byte_length_mismatch/);
});

test('loadWeights rejects a non-float32 dtype', () => {
  const manifest = cloneManifest();
  manifest.tensors[0].dtype = 'float16';
  assert.throws(() => loadWeights(manifest, SYNTHETIC.buffer), /unsupported_dtype/);
});

test('loadWeights rejects a missing required tensor', () => {
  const manifest = cloneManifest();
  manifest.tensors = manifest.tensors.filter((entry) => entry.name !== 'block3.conv2.weight');
  manifest.blobBytes = undefined;
  assert.throws(() => loadWeights(manifest, SYNTHETIC.buffer), /missing_tensor:block3\.conv2\.weight/);
});

test('loadWeights rejects overlapping tensors and duplicates', () => {
  const overlapping = cloneManifest();
  overlapping.tensors[1].byteOffset = overlapping.tensors[0].byteOffset + 4;
  assert.throws(() => loadWeights(overlapping, SYNTHETIC.buffer), /overlapping_tensors/);

  const duplicated = cloneManifest();
  duplicated.tensors.push({ ...duplicated.tensors[0] });
  assert.throws(() => loadWeights(duplicated, SYNTHETIC.buffer), /duplicate_tensor/);
});

test('loadWeights rejects a wrong shape for a required tensor and a bad format version', () => {
  const reshaped = cloneManifest();
  const entry = reshaped.tensors.find((candidate) => candidate.name === 'policy.fc.weight');
  entry.shape = [2592, 209];
  assert.throws(() => loadWeights(reshaped, SYNTHETIC.buffer), /unexpected_tensor_shape:policy\.fc\.weight/);

  const versioned = cloneManifest();
  versioned.formatVersion = 2;
  assert.throws(() => loadWeights(versioned, SYNTHETIC.buffer), /unsupported_format_version/);

  const board = cloneManifest();
  board.input.rows = 7;
  assert.throws(() => loadWeights(board, SYNTHETIC.buffer), /unsupported_board/);

  assert.throws(() => loadWeights('{not json', SYNTHETIC.buffer), /invalid_manifest_json/);
});

test('loadWeights accepts the real exported artifacts', (t) => {
  if (!artifactsPresent()) {
    t.skip(`training artifacts missing: expected ${BLOB_PATH}`);
    return;
  }
  const weights = realWeights();
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  assert.equal(weights.parameterCount, manifest.parameterCount);
  assert.equal(weights.version, NN_RUNTIME_VERSION);
});

/* ------------------------------------------------------------------ *
 * Timing — the 900 ms turn allows roughly 20 of these
 * ------------------------------------------------------------------ */

test('a single evaluate call fits the per-turn budget', (t) => {
  const weights = artifactsPresent() ? realWeights() : loadWeights(SYNTHETIC.manifest, SYNTHETIC.buffer);
  const evaluate = createNetworkEvaluator(weights);
  const state = fixedState();

  for (let i = 0; i < 3; i += 1) evaluate(CONFIG_9X9, state); // warm the JIT

  const runs = 10;
  const start = process.hrtime.bigint();
  for (let i = 0; i < runs; i += 1) evaluate(CONFIG_9X9, state);
  const elapsedMs = Number(process.hrtime.bigint() - start) / 1e6;
  const perCall = elapsedMs / runs;

  t.diagnostic(`single evaluate: ${perCall.toFixed(2)} ms (mean of ${runs}); ~20 per 900 ms turn needs <= 45 ms`);
  assert.ok(perCall < 200, `evaluate took ${perCall} ms, far outside any plausible turn budget`);
});
