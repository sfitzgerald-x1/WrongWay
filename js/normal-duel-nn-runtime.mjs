/**
 * Stage 4 neural-network forward pass for the canonical 9x9 normal duel.
 *
 * This is the inference half of the exporter in `wrongway-training`: it loads
 * the float32 blob described by `weights.manifest.json` and runs the exact
 * BN-free graph that `export_weights.reference_forward` specifies. BatchNorm
 * was folded into the preceding convolution at export time, so nothing here
 * runs a BN — every convolution carries a bias and is followed by ReLU (or, in
 * the residual second conv, by the skip add and then ReLU).
 *
 *     input  (10, 9, 9) planes in `NN_PLANE_LAYOUT` order, absolute engine (r, c)
 *     stem   conv3x3 -> 64ch, ReLU
 *     6x     [conv3x3 + ReLU, conv3x3, + skip, ReLU]
 *     policy conv1x1 -> 3ch, no activation, read as [pawn 9x9 | H 8x8 | V 8x8]
 *     value  conv1x1 -> 32ch, ReLU, mean/max/std pool (96), FC -> 64, ReLU, FC -> 1, tanh
 *
 * The heads are the fully-convolutional policy and globally-pooled value of the
 * current `model.py`; the dense `policy.fc` and the flattened value head that
 * preceded them are gone, and weight sets carrying those tensors are rejected.
 *
 * Nothing mirrors, rotates or otherwise transforms coordinates: the planes and
 * the action codes are both engine-absolute, and they stay that way.
 *
 * Numerics: activations are stored as float32 (matching the reference), but
 * every dot product accumulates in a float64 accumulator before the single
 * rounding store. That is strictly more accurate than a float32 running sum,
 * so the gap to the PyTorch reference is bounded by the reference's own
 * accumulation error rather than compounding on top of it.
 *
 * Purity: no `Math.random`, no `Date`, no filesystem and no network. The
 * caller supplies the manifest text and the blob bytes. Identical input gives
 * byte-identical output.
 *
 * Allocation: `createNetworkEvaluator` allocates *this module's own* scratch
 * buffers exactly once and reuses them across calls. Roughly 20 evaluations
 * happen inside the canonical 900 ms turn, and the deployed candidate runs in
 * WASM linear memory that never shrinks, so per-call churn is a containment
 * problem (PRs #16/#17) and not merely a speed one.
 *
 * The once-only guarantee covers this module's scratch and nothing else. It is
 * NOT a statement about the whole evaluate path: `encodeState`,
 * `legalMaskFloat` and the `validateState` inside them each allocate per call,
 * on the order of 36 KB in total, which dwarfs the 836 B policy array this
 * module returns. That policy array is deliberately fresh per call because the
 * caller owns it; the encode-path churn is `normal-duel-nn-encoding.mjs`'s to
 * fix, not something the scratch reuse here removes.
 */

import { policySize, validateConfig } from './normal-duel-engine.mjs';
import { encodeState, legalMaskFloat, NN_INPUT_PLANES } from './normal-duel-nn-encoding.mjs';

/** Frozen identifier for this runtime and the weight format it accepts. */
export const NN_RUNTIME_VERSION = 'nn-runtime-forward-v1';

export class NormalDuelRuntimeError extends Error {
  constructor(reason) { super(reason); this.name = 'NormalDuelRuntimeError'; this.reason = reason; }
}

function fail(reason) { throw new NormalDuelRuntimeError(reason); }

/* ------------------------------------------------------------------ *
 * Fixed shape contract — mirrors `model.py`, not tunable from outside.
 * ------------------------------------------------------------------ */

const ROWS = 9;
const COLUMNS = 9;
const CELLS = ROWS * COLUMNS;          // 81
// Taken from the encoder rather than restated. This was a literal 8 while the
// encoder grew to 10 planes, and the two only disagreed inside a golden fixture
// -- the stem convolution would have read a feature vector of the wrong length
// with no signal beyond that one test.
const INPUT_PLANES = NN_INPUT_PLANES;
const FEATURE_LEN = INPUT_PLANES * CELLS;
const CHANNELS = 64;
const BLOCKS = 6;
// The policy head is fully convolutional: 3 planes out of a 1x1 conv, no
// activation, read out as [pawn 9x9 | wall H 8x8 | wall V 8x8] = 209. The wall
// planes are cropped to their top-left 8x8, which is where the anchors live.
const POLICY_CHANNELS = 3;
const WALL_ROWS = 8;
const WALL_COLUMNS = 8;
const VALUE_CHANNELS = 32;
// The value head pools each channel to mean/max/std over the 81 cells rather
// than flattening them, so fc1 reads 3 numbers per channel and the head no
// longer scales with board area.
const VALUE_POOLS = 3;
const VALUE_HIDDEN = 64;
const POLICY_SIZE = 209;
const FORMAT_VERSION = 1;
const BYTES_PER_FLOAT = 4;

/** Same finite fill as `model.MASK_FILL`; exp() of it underflows to exactly 0. */
const MASK_FILL = -1e9;

/** `name -> expected shape`, in the exporter's order. */
const REQUIRED_TENSORS = (() => {
  const spec = [
    ['stem.conv.weight', [CHANNELS, INPUT_PLANES, 3, 3]],
    ['stem.conv.bias', [CHANNELS]]
  ];
  for (let i = 0; i < BLOCKS; i += 1) {
    spec.push([`block${i}.conv1.weight`, [CHANNELS, CHANNELS, 3, 3]]);
    spec.push([`block${i}.conv1.bias`, [CHANNELS]]);
    spec.push([`block${i}.conv2.weight`, [CHANNELS, CHANNELS, 3, 3]]);
    spec.push([`block${i}.conv2.bias`, [CHANNELS]]);
  }
  spec.push(['policy.conv.weight', [POLICY_CHANNELS, CHANNELS, 1, 1]]);
  spec.push(['policy.conv.bias', [POLICY_CHANNELS]]);
  spec.push(['value.conv.weight', [VALUE_CHANNELS, CHANNELS, 1, 1]]);
  spec.push(['value.conv.bias', [VALUE_CHANNELS]]);
  spec.push(['value.fc1.weight', [VALUE_HIDDEN, VALUE_POOLS * VALUE_CHANNELS]]);
  spec.push(['value.fc1.bias', [VALUE_HIDDEN]]);
  spec.push(['value.fc2.weight', [1, VALUE_HIDDEN]]);
  spec.push(['value.fc2.bias', [1]]);
  return Object.freeze(new Map(spec));
})();

/**
 * The tensor table this runtime demands, as `name -> shape`.
 *
 * Exported so a test can compare it against a shape table captured from a real
 * `export_weights.py` run. Everything else that guards this module is internal:
 * the golden is generated from a synthetic spec that lives in this repo, so if
 * the exporter grows a plane or moves a head width, nothing here changes, the
 * blob hash is unchanged, the golden still matches, and CI stays green while
 * `loadWeights` rejects every real export. That is exactly how this module came
 * to implement a dead architecture for three commits.
 */
export function requiredTensorShapes() {
  const out = {};
  for (const [name, shape] of REQUIRED_TENSORS) out[name] = [...shape];
  return out;
}

/** Brand for weight sets this module produced; nothing else is accepted. */
const WEIGHTS_BRAND = Symbol.for('wrongway.nnRuntime.weights.v1');

/** The blob is little-endian; `Float32Array` is host-endian. Swap if they differ. */
const HOST_IS_LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/* ------------------------------------------------------------------ *
 * Weight loading and validation
 * ------------------------------------------------------------------ */

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonNegativeInteger(value) {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/** Normalise ArrayBuffer / TypedArray / Node Buffer to a byte view. */
function asBytes(buffer) {
  if (buffer instanceof ArrayBuffer) return new Uint8Array(buffer);
  if (ArrayBuffer.isView(buffer)) return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return fail('invalid_buffer');
}

function parseManifest(manifestJson) {
  if (typeof manifestJson === 'string') {
    try {
      return JSON.parse(manifestJson);
    } catch {
      return fail('invalid_manifest_json');
    }
  }
  if (isPlainObject(manifestJson)) return manifestJson;
  return fail('invalid_manifest');
}

function checkArchitecture(manifest) {
  if (manifest.formatVersion !== FORMAT_VERSION) fail('unsupported_format_version');
  if (manifest.byteOrder !== undefined && manifest.byteOrder !== 'little') fail('unsupported_byte_order');
  if (manifest.layout !== undefined && manifest.layout !== 'C-contiguous') fail('unsupported_layout');
  if (manifest.flattenOrder !== undefined && manifest.flattenOrder !== 'CHW') fail('unsupported_flatten_order');
  if (manifest.policySize !== undefined && manifest.policySize !== POLICY_SIZE) fail('unsupported_policy_size');

  const input = manifest.input;
  if (isPlainObject(input)) {
    // Canonical 9x9 only: the tower was sized and measured for this board.
    if (input.rows !== ROWS || input.columns !== COLUMNS) fail('unsupported_board');
    if (input.planes !== INPUT_PLANES) fail('unsupported_input_planes');
  }

  const arch = manifest.architecture;
  if (isPlainObject(arch)) {
    if (arch.blocks !== BLOCKS) fail('unsupported_blocks');
    if (arch.channels !== CHANNELS) fail('unsupported_channels');
    if (arch.policyHeadChannels !== undefined && arch.policyHeadChannels !== POLICY_CHANNELS) fail('unsupported_policy_channels');
    if (arch.valueHeadChannels !== undefined && arch.valueHeadChannels !== VALUE_CHANNELS) fail('unsupported_value_channels');
    if (arch.valueHidden !== undefined && arch.valueHidden !== VALUE_HIDDEN) fail('unsupported_value_hidden');
  }
}

/** Copy `[byteOffset, byteOffset + byteLength)` out of the blob as float32. */
function readTensor(bytes, entry) {
  // Copy rather than view: the caller's buffer may be reused or mutated after
  // the call, the source offset need not be 4-byte aligned, and a copy is what
  // lets the weight set be treated as frozen.
  const copy = bytes.slice(entry.byteOffset, entry.byteOffset + entry.byteLength);
  if (!HOST_IS_LITTLE_ENDIAN) {
    for (let i = 0; i < copy.length; i += BYTES_PER_FLOAT) {
      const b0 = copy[i]; const b1 = copy[i + 1]; const b2 = copy[i + 2]; const b3 = copy[i + 3];
      copy[i] = b3; copy[i + 1] = b2; copy[i + 2] = b1; copy[i + 3] = b0;
    }
  }
  return new Float32Array(copy.buffer, copy.byteOffset, entry.count);
}

/**
 * Validate a manifest against a blob and return a frozen weight set.
 *
 * Strict by construction: every declared tensor must be float32, must have
 * `count` equal to the product of `shape` and `byteLength` equal to
 * `count * 4`, must lie wholly inside the buffer, and must not overlap another
 * tensor. Every tensor the graph needs must be present with the exact expected
 * shape. Any violation throws — the runtime never reads past the end of the
 * blob and never silently zero-fills a missing tensor.
 */
export function loadWeights(manifestJson, buffer) {
  const manifest = parseManifest(manifestJson);
  const bytes = asBytes(buffer);
  checkArchitecture(manifest);

  const entries = manifest.tensors;
  if (!Array.isArray(entries) || entries.length === 0) fail('invalid_tensor_table');

  const seen = new Map();
  for (const entry of entries) {
    if (!isPlainObject(entry)) fail('invalid_tensor_entry');
    const { name, shape, dtype, byteOffset, byteLength, count } = entry;
    if (typeof name !== 'string' || name.length === 0) fail('invalid_tensor_name');
    if (seen.has(name)) fail('duplicate_tensor');
    if (dtype !== 'float32') fail('unsupported_dtype');
    if (!Array.isArray(shape) || shape.length === 0) fail('invalid_tensor_shape');

    let product = 1;
    for (const dimension of shape) {
      if (!isNonNegativeInteger(dimension) || dimension === 0) fail('invalid_tensor_shape');
      product *= dimension;
    }
    if (!isNonNegativeInteger(count) || count !== product) fail('tensor_count_mismatch');
    if (!isNonNegativeInteger(byteLength) || byteLength !== count * BYTES_PER_FLOAT) fail('tensor_byte_length_mismatch');
    if (!isNonNegativeInteger(byteOffset)) fail('invalid_tensor_offset');
    // Written as a subtraction so a huge offset cannot overflow past the check.
    if (byteOffset > bytes.byteLength || byteLength > bytes.byteLength - byteOffset) fail('tensor_out_of_range');

    seen.set(name, entry);
  }

  // Overlap check: sort by offset once, then compare each start to the
  // previous end. Zero-length tensors are already rejected above.
  const ordered = [...seen.values()].sort((left, right) => left.byteOffset - right.byteOffset);
  for (let i = 1; i < ordered.length; i += 1) {
    const previous = ordered[i - 1];
    if (ordered[i].byteOffset < previous.byteOffset + previous.byteLength) fail('overlapping_tensors');
  }

  if (manifest.blobBytes !== undefined && manifest.blobBytes !== bytes.byteLength) fail('blob_size_mismatch');

  const tensors = new Map();
  for (const [name, expected] of REQUIRED_TENSORS) {
    const entry = seen.get(name);
    if (entry === undefined) fail(`missing_tensor:${name}`);
    if (entry.shape.length !== expected.length
      || entry.shape.some((dimension, index) => dimension !== expected[index])) {
      fail(`unexpected_tensor_shape:${name}`);
    }
    tensors.set(name, readTensor(bytes, entry));
  }

  const weights = {
    [WEIGHTS_BRAND]: true,
    version: NN_RUNTIME_VERSION,
    parameterCount: [...tensors.values()].reduce((total, tensor) => total + tensor.length, 0),
    names: Object.freeze([...tensors.keys()]),
    /** The float32 view for `name`. Private copies — treat them as read-only. */
    get(name) {
      const tensor = tensors.get(name);
      if (tensor === undefined) fail(`missing_tensor:${name}`);
      return tensor;
    }
  };
  return Object.freeze(weights);
}

function checkWeights(weights) {
  if (weights === null || typeof weights !== 'object' || weights[WEIGHTS_BRAND] !== true) fail('invalid_weights');
  return weights;
}

/* ------------------------------------------------------------------ *
 * Kernels
 * ------------------------------------------------------------------ */

/**
 * 3x3 convolution, stride 1, zero padding 1, over a 9x9 plane stack.
 * `weight` is (outC, inC, 3, 3) C-contiguous, `acc` a 81-long float64 scratch.
 */
function conv3x3(input, inChannels, weight, bias, output, outChannels, acc) {
  for (let oc = 0; oc < outChannels; oc += 1) {
    acc.fill(bias[oc]);
    for (let ic = 0; ic < inChannels; ic += 1) {
      const inputBase = ic * CELLS;
      const weightBase = (oc * inChannels + ic) * 9;
      for (let kr = 0; kr < 3; kr += 1) {
        // Output rows whose input row `r + kr - 1` stays on the board.
        const rowLow = kr === 0 ? 1 : 0;
        const rowHigh = kr === 2 ? ROWS - 1 : ROWS;
        for (let kc = 0; kc < 3; kc += 1) {
          const w = weight[weightBase + kr * 3 + kc];
          const columnLow = kc === 0 ? 1 : 0;
          const columnHigh = kc === 2 ? COLUMNS - 1 : COLUMNS;
          for (let r = rowLow; r < rowHigh; r += 1) {
            const outputRow = r * COLUMNS;
            const inputRow = inputBase + (r + kr - 1) * COLUMNS + kc - 1;
            for (let c = columnLow; c < columnHigh; c += 1) {
              acc[outputRow + c] += w * input[inputRow + c];
            }
          }
        }
      }
    }
    const outputBase = oc * CELLS;
    for (let i = 0; i < CELLS; i += 1) output[outputBase + i] = acc[i];
  }
}

/** 1x1 convolution, no activation — what the policy head takes. */
function conv1x1(input, inChannels, weight, bias, output, outChannels, acc) {
  for (let oc = 0; oc < outChannels; oc += 1) {
    acc.fill(bias[oc]);
    const weightBase = oc * inChannels;
    for (let ic = 0; ic < inChannels; ic += 1) {
      const w = weight[weightBase + ic];
      const inputBase = ic * CELLS;
      for (let i = 0; i < CELLS; i += 1) acc[i] += w * input[inputBase + i];
    }
    const outputBase = oc * CELLS;
    for (let i = 0; i < CELLS; i += 1) output[outputBase + i] = acc[i];
  }
}

/**
 * Flatten the three policy planes into the 209 codes: the pawn plane entire,
 * then each wall plane cropped to its top-left 8x8. The crop is a row-stride
 * read, not a contiguous slice — the plane is 9 wide, so row r of the crop
 * starts at r * 9 and the ninth column of every row is skipped.
 */
function readPolicyPlanes(planes, logits) {
  for (let i = 0; i < CELLS; i += 1) logits[i] = planes[i];
  let out = CELLS;
  for (let plane = 1; plane <= 2; plane += 1) {
    const base = plane * CELLS;
    for (let r = 0; r < WALL_ROWS; r += 1) {
      const rowBase = base + r * COLUMNS;
      for (let c = 0; c < WALL_COLUMNS; c += 1) logits[out++] = planes[rowBase + c];
    }
  }
}

/**
 * Global mean/max/std per channel, laid out as the reference concatenates them:
 * all the means, then all the maxima, then all the standard deviations. The std
 * is the population form (`unbiased=False`), so it divides by 81 and not 80.
 */
function poolValuePlanes(planes, pooled) {
  for (let ch = 0; ch < VALUE_CHANNELS; ch += 1) {
    const base = ch * CELLS;
    let sum = 0;
    let max = -Infinity;
    for (let i = 0; i < CELLS; i += 1) {
      const value = planes[base + i];
      sum += value;
      if (value > max) max = value;
    }
    const mean = sum / CELLS;
    let variance = 0;
    for (let i = 0; i < CELLS; i += 1) {
      const delta = planes[base + i] - mean;
      variance += delta * delta;
    }
    pooled[ch] = mean;
    pooled[VALUE_CHANNELS + ch] = max;
    pooled[2 * VALUE_CHANNELS + ch] = Math.sqrt(variance / CELLS);
  }
}

/** 1x1 convolution followed by ReLU — what the value head takes. */
function conv1x1Relu(input, inChannels, weight, bias, output, outChannels, acc) {
  for (let oc = 0; oc < outChannels; oc += 1) {
    acc.fill(bias[oc]);
    const weightBase = oc * inChannels;
    for (let ic = 0; ic < inChannels; ic += 1) {
      const w = weight[weightBase + ic];
      const inputBase = ic * CELLS;
      for (let i = 0; i < CELLS; i += 1) acc[i] += w * input[inputBase + i];
    }
    const outputBase = oc * CELLS;
    for (let i = 0; i < CELLS; i += 1) {
      const value = acc[i];
      output[outputBase + i] = value > 0 ? value : 0;
    }
  }
}

/** `output = weight * input + bias`, weight (outLen, inLen) C-contiguous. */
function linear(input, inLength, weight, bias, output, outLength) {
  for (let o = 0; o < outLength; o += 1) {
    const base = o * inLength;
    let sum = bias[o];
    for (let i = 0; i < inLength; i += 1) sum += weight[base + i] * input[i];
    output[o] = sum;
  }
}

function reluInPlace(buffer) {
  for (let i = 0; i < buffer.length; i += 1) if (buffer[i] < 0) buffer[i] = 0;
}

/* ------------------------------------------------------------------ *
 * Forward pass
 * ------------------------------------------------------------------ */

/** Every buffer the forward pass writes to, allocated once. */
function createScratch() {
  return {
    acc: new Float64Array(CELLS),
    h: new Float32Array(CHANNELS * CELLS),
    t1: new Float32Array(CHANNELS * CELLS),
    t2: new Float32Array(CHANNELS * CELLS),
    policyPlanes: new Float32Array(POLICY_CHANNELS * CELLS),
    valuePlanes: new Float32Array(VALUE_CHANNELS * CELLS),
    pooled: new Float32Array(VALUE_POOLS * VALUE_CHANNELS),
    valueHidden: new Float32Array(VALUE_HIDDEN),
    valueOut: new Float32Array(1),
    logits: new Float32Array(POLICY_SIZE)
  };
}

function checkFeatures(features) {
  if (!(features instanceof Float32Array) && !(features instanceof Float64Array) && !Array.isArray(features)) {
    fail('invalid_features');
  }
  if (features.length !== FEATURE_LEN) fail('invalid_features_length');
  for (let i = 0; i < FEATURE_LEN; i += 1) {
    const value = features[i];
    if (typeof value !== 'number' || !Number.isFinite(value)) fail('invalid_features');
  }
  return features;
}

/**
 * Run the graph. Writes `scratch.logits` and returns the tanh value. Mirrors
 * `export_weights.reference_forward` operation for operation, including the
 * order of the residual add and the ReLU that follows it. `features` is only
 * ever read.
 */
function forwardInto(weights, features, scratch) {
  const { acc, h, t1, t2 } = scratch;

  conv3x3(features, INPUT_PLANES, weights.get('stem.conv.weight'), weights.get('stem.conv.bias'),
    h, CHANNELS, acc);
  reluInPlace(h);

  for (let i = 0; i < BLOCKS; i += 1) {
    conv3x3(h, CHANNELS, weights.get(`block${i}.conv1.weight`), weights.get(`block${i}.conv1.bias`),
      t1, CHANNELS, acc);
    reluInPlace(t1);
    conv3x3(t1, CHANNELS, weights.get(`block${i}.conv2.weight`), weights.get(`block${i}.conv2.bias`),
      t2, CHANNELS, acc);
    for (let j = 0; j < h.length; j += 1) {
      const sum = h[j] + t2[j];
      h[j] = sum > 0 ? sum : 0;
    }
  }

  // Policy: no activation on the conv, then the 209 codes are read straight out
  // of the three planes -- the pawn plane whole, the wall planes cropped.
  conv1x1(h, CHANNELS, weights.get('policy.conv.weight'), weights.get('policy.conv.bias'),
    scratch.policyPlanes, POLICY_CHANNELS, acc);
  readPolicyPlanes(scratch.policyPlanes, scratch.logits);

  conv1x1Relu(h, CHANNELS, weights.get('value.conv.weight'), weights.get('value.conv.bias'),
    scratch.valuePlanes, VALUE_CHANNELS, acc);
  poolValuePlanes(scratch.valuePlanes, scratch.pooled);
  linear(scratch.pooled, VALUE_POOLS * VALUE_CHANNELS,
    weights.get('value.fc1.weight'), weights.get('value.fc1.bias'), scratch.valueHidden, VALUE_HIDDEN);
  reluInPlace(scratch.valueHidden);
  linear(scratch.valueHidden, VALUE_HIDDEN,
    weights.get('value.fc2.weight'), weights.get('value.fc2.bias'), scratch.valueOut, 1);

  const value = Math.tanh(scratch.valueOut[0]);
  // tanh is already in [-1, 1]; clamp only so a float edge case cannot hand the
  // search a value it would reject as out of range.
  return value > 1 ? 1 : (value < -1 ? -1 : value);
}

/**
 * Unmasked forward pass over a raw feature vector — the parity surface against
 * the Python reference. Allocates its own scratch, so prefer
 * `createNetworkEvaluator` on any hot path.
 */
export function forwardRaw(weights, features) {
  const checked = checkWeights(weights);
  checkFeatures(features);
  const scratch = createScratch();
  const source = features instanceof Float32Array ? features : Float32Array.from(features);
  const value = forwardInto(checked, source, scratch);
  // `valuePooled` is exposed for the golden to assert against. The value is one
  // tanh scalar summarising 96 pooled numbers, so it cannot discriminate an
  // error in how they are pooled: reordering the mean/max/std blocks moves it by
  // 2e-5 and switching to the sample std by 2e-6, both inside any tolerance the
  // scalar can carry. Comparing the pooled vector elementwise does catch them.
  return { policyLogits: scratch.logits, value, valuePooled: scratch.pooled };
}

/* ------------------------------------------------------------------ *
 * Evaluator
 * ------------------------------------------------------------------ */

/**
 * Build an evaluator with the same signature and return shape as
 * `uniformStubEvaluator` in `normal-duel-gumbel-search.mjs`, so
 * `gumbelRootSearch` and `selfPlayGame` take it unchanged:
 *
 *     evaluate(config, state) -> { policy: Float32Array(209), value: number }
 *
 * The legal mask is applied to the logits *before* the softmax, so illegal
 * actions come out at exactly zero rather than merely small. Scratch is
 * allocated here, once; each call returns a fresh policy array because that
 * array belongs to the caller.
 */
/**
 * The same forward-plus-mask-plus-softmax as `createNetworkEvaluator`, but
 * entered from a feature vector and a legal mask the caller already holds
 * rather than from a state.
 *
 *     evaluate(features, mask, policyOut) -> value
 *
 * This is what the native PUCT tree needs: it hands out the encoded leaf and
 * its mask directly, so re-deriving them from a state object would be pure
 * waste. `policyOut` is filled in place — it may be a view into wasm memory —
 * and scratch is allocated once here, so nothing allocates per leaf.
 */
export function createFeatureEvaluator(weights) {
  const checked = checkWeights(weights);
  const scratch = createScratch();
  const masked = new Float64Array(POLICY_SIZE);

  return function evaluateFeatures(features, mask, policyOut) {
    checkFeatures(features);
    if (mask.length !== POLICY_SIZE || policyOut.length !== POLICY_SIZE) {
      fail('unsupported_policy_size');
    }
    const value = forwardInto(checked, features, scratch);

    const logits = scratch.logits;
    let max = -Infinity;
    for (let code = 0; code < POLICY_SIZE; code += 1) {
      const logit = mask[code] > 0 ? logits[code] : MASK_FILL;
      masked[code] = logit;
      if (logit > max) max = logit;
    }

    let total = 0;
    for (let code = 0; code < POLICY_SIZE; code += 1) {
      if (mask[code] <= 0) { policyOut[code] = 0; continue; }
      const weight = Math.exp(masked[code] - max);
      policyOut[code] = weight;
      total += weight;
    }
    if (total > 0) {
      for (let code = 0; code < POLICY_SIZE; code += 1) {
        if (policyOut[code] !== 0) policyOut[code] /= total;
      }
    }
    return value;
  };
}

export function createNetworkEvaluator(weights) {
  const checked = checkWeights(weights);
  const scratch = createScratch();
  const masked = new Float64Array(POLICY_SIZE);

  return function evaluate(config, state) {
    const config9x9 = validateConfig(config);
    if (config9x9.rows !== ROWS || config9x9.columns !== COLUMNS) fail('unsupported_board');
    const size = policySize(config9x9);
    if (size !== POLICY_SIZE) fail('unsupported_policy_size');

    // `encodeState` and `legalMaskFloat` both re-validate the state and copy
    // out of it; neither the config nor the state is mutated here.
    const features = encodeState(config9x9, state);
    const mask = legalMaskFloat(config9x9, state);
    const value = forwardInto(checked, features, scratch);

    const logits = scratch.logits;
    let max = -Infinity;
    for (let code = 0; code < POLICY_SIZE; code += 1) {
      const logit = mask[code] > 0 ? logits[code] : MASK_FILL;
      masked[code] = logit;
      if (logit > max) max = logit;
    }

    const policy = new Float32Array(POLICY_SIZE);
    let total = 0;
    for (let code = 0; code < POLICY_SIZE; code += 1) {
      if (mask[code] <= 0) continue; // exactly zero, never a rounded-down epsilon
      const weight = Math.exp(masked[code] - max);
      policy[code] = weight;
      total += weight;
    }
    // A terminal state has no legal actions; return the all-zero policy rather
    // than dividing by zero. The search never evaluates one, but a caller can.
    if (total > 0) {
      for (let code = 0; code < POLICY_SIZE; code += 1) {
        if (policy[code] !== 0) policy[code] /= total;
      }
    }

    return { policy, value };
  };
}
