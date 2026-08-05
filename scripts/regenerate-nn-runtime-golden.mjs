#!/usr/bin/env node
/**
 * Regenerate `tests/fixtures/nn-runtime-golden-forward-v1.json` from the PyTorch
 * reference implementation.
 *
 *     node scripts/regenerate-nn-runtime-golden.mjs [--training DIR]
 *                                                   [--update-exporter-shapes]
 *
 * `--update-exporter-shapes` additionally re-captures
 * `tests/fixtures/exporter-tensor-shapes.json` from the training checkout's real
 * `weights.manifest.json`, which is what the exporter-drift test compares the
 * runtime against.
 *
 * The golden pins the JS forward graph against `export_weights.reference_forward`
 * so that `tests/normal-duel-nn-runtime.test.mjs` can verify the whole graph
 * without Python and without any exported artifact. Its value comes entirely
 * from having been produced by the reference: regenerating it from `forwardRaw`
 * would turn the test into a tautology that agrees with whatever the JS does.
 * So this script refuses to run without a torch venv rather than falling back.
 *
 * Run it when the encoder's plane count, the synthetic weight spec, or the
 * reference graph changes -- the test's fnv1a32 assertion fails loudly and names
 * this script when the committed fixture and the blob have drifted apart.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { encodeState } from '../js/normal-duel-nn-encoding.mjs';
import {
  CONFIG_9X9, POLICY_SIZE, fixedState, fnv1a32, syntheticWeightSet
} from '../tests/support/nn-runtime-fixture.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const OUT = path.join(REPO, 'tests/fixtures/nn-runtime-golden-forward-v1.json');
const SHAPES_OUT = path.join(REPO, 'tests/fixtures/exporter-tensor-shapes.json');

const argv = process.argv.slice(2);
for (const flag of argv.filter((a) => a.startsWith('--'))) {
  if (!['--training', '--update-exporter-shapes'].includes(flag)) {
    console.error(`unknown flag ${flag}`);
    process.exit(2);
  }
}
const UPDATE_SHAPES = argv.includes('--update-exporter-shapes');
const trainingFlag = argv.indexOf('--training');
const TRAINING = trainingFlag === -1
  ? '/Users/scott/workspace/agents/wrongway-training'
  : argv[trainingFlag + 1];
const PYTHON = path.join(TRAINING, '.venv/bin/python');

for (const [label, target] of [
  ['training checkout', path.join(TRAINING, 'export_weights.py')],
  ['torch venv', PYTHON]
]) {
  if (!fs.existsSync(target)) {
    console.error(`missing ${label}: ${target}`);
    console.error('the golden must come from the PyTorch reference, so this script will not guess.');
    process.exit(1);
  }
}

const REFERENCE = `
import json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

training = Path(sys.argv[1])
sys.path.insert(0, str(training))
from export_weights import reference_forward

payload = json.loads(Path(sys.argv[2]).read_text())
manifest = payload["manifest"]
blob = Path(sys.argv[3]).read_bytes()

tensors = {}
for entry in manifest["tensors"]:
    raw = np.frombuffer(blob, dtype="<f4", count=entry["count"],
                        offset=entry["byteOffset"]).reshape(entry["shape"])
    tensors[entry["name"]] = torch.from_numpy(raw.copy())

planes = manifest["input"]["planes"]
rows, columns = manifest["input"]["rows"], manifest["input"]["columns"]
features = np.asarray(payload["features"], dtype=np.float32)
x = torch.from_numpy(features.reshape(1, planes, rows, columns))
logits, value = reference_forward(tensors, x, manifest["architecture"]["blocks"])

# The pooled value-head vector, recomputed here from the same tensors. The value
# is a single scalar and cannot pin down how 96 numbers were pooled, so the
# fixture carries the vector too and the test asserts it elementwise.
h = F.relu(F.conv2d(x, tensors["stem.conv.weight"], tensors["stem.conv.bias"], padding=1))
for i in range(manifest["architecture"]["blocks"]):
    y = F.relu(F.conv2d(h, tensors[f"block{i}.conv1.weight"],
                        tensors[f"block{i}.conv1.bias"], padding=1))
    y = F.conv2d(y, tensors[f"block{i}.conv2.weight"],
                 tensors[f"block{i}.conv2.bias"], padding=1)
    h = F.relu(h + y)
v = F.relu(F.conv2d(h, tensors["value.conv.weight"], tensors["value.conv.bias"]))
flat = v.flatten(2)
pooled = torch.cat((flat.mean(dim=2), flat.amax(dim=2),
                    flat.std(dim=2, unbiased=False)), dim=1)

# The trunk above is a second implementation of the same graph, so cross-check it
# against reference_forward rather than trusting it: recompute the policy logits
# from this h and require them to match. Without this the pooled vector would have
# a different provenance from the rest of the fixture, unverified.
p = F.conv2d(h, tensors["policy.conv.weight"], tensors["policy.conv.bias"])
WR = WC = 8
check = torch.cat((p[:, 0].flatten(1),
                   p[:, 1, :WR, :WC].flatten(1),
                   p[:, 2, :WR, :WC].flatten(1)), dim=1)
drift = float((check - logits).abs().max())
if drift > 1e-6:
    raise SystemExit(f"the local trunk disagrees with reference_forward by {drift}")

print(json.dumps({"logits": logits[0].tolist(), "value": float(value[0]),
                  "valuePooled": pooled[0].tolist(), "trunkCrossCheck": drift}))
`;

const synthetic = syntheticWeightSet();
const features = Array.from(encodeState(CONFIG_9X9, fixedState()));

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'ww-golden-'));
try {
  const payloadPath = path.join(scratch, 'payload.json');
  const blobPath = path.join(scratch, 'weights.bin');
  const scriptPath = path.join(scratch, 'reference.py');
  fs.writeFileSync(payloadPath, JSON.stringify({ manifest: synthetic.manifest, features }));
  fs.writeFileSync(blobPath, Buffer.from(synthetic.buffer));
  fs.writeFileSync(scriptPath, REFERENCE);

  const raw = execFileSync(PYTHON, [scriptPath, TRAINING, payloadPath, blobPath], { encoding: 'utf8' });
  const reference = JSON.parse(raw);

  if (reference.logits.length !== POLICY_SIZE) {
    throw new Error(`reference returned ${reference.logits.length} logits, expected ${POLICY_SIZE}`);
  }

  const fixture = {
    format: 'nn-runtime-golden-forward-v1',
    note: 'Generated by scripts/regenerate-nn-runtime-golden.mjs from export_weights.reference_forward. Do not hand-edit.',
    blob: { byteLength: synthetic.buffer.byteLength, fnv1a32: fnv1a32(synthetic.buffer) },
    input: synthetic.manifest.input,
    features,
    policyLogits: reference.logits,
    valuePooled: reference.valuePooled,
    value: reference.value
  };
  fs.writeFileSync(OUT, `${JSON.stringify(fixture, null, 2)}\n`);
  console.log(`wrote ${path.relative(REPO, OUT)}`);
  console.log(`  planes ${fixture.input.planes}, features ${features.length}, blob ${fixture.blob.byteLength} B (${fixture.blob.fnv1a32})`);
  console.log(`  value ${reference.value}`);
  console.log(`  local trunk vs reference_forward: ${reference.trunkCrossCheck}`);
  if (UPDATE_SHAPES) {
    // Re-capture the exporter shape table from the REAL export, not from the
    // synthetic spec -- the fixture's whole value is that it comes from the other
    // side of the boundary.
    const manifestPath = path.join(TRAINING, 'weights.manifest.json');
    if (!fs.existsSync(manifestPath)) {
      console.error(`no real export to capture: ${manifestPath}`);
      console.error('run export_weights.py in the training checkout first.');
      process.exit(1);
    }
    const real = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const shapes = {
      note: 'Shape table only, no weights. Captured from a real export_weights.py run so'
        + " the runtime's REQUIRED_TENSORS cannot drift from the exporter without a red"
        + ' build. Regenerate with'
        + ' scripts/regenerate-nn-runtime-golden.mjs --update-exporter-shapes.',
      input: real.input,
      architecture: real.architecture,
      policySize: real.policySize,
      tensors: real.tensors.map((tensor) => ({ name: tensor.name, shape: tensor.shape }))
    };
    fs.writeFileSync(SHAPES_OUT, `${JSON.stringify(shapes, null, 2)}\n`);
    console.log(`wrote ${path.relative(REPO, SHAPES_OUT)}`);
    console.log(`  ${shapes.tensors.length} tensors, ${shapes.input.planes} planes,`
      + ` ${shapes.architecture.channels}ch/${shapes.architecture.blocks} blocks`);
  }
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}
