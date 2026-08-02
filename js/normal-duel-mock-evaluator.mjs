/**
 * The parity harness's stand-in for the network.
 *
 * This must stay byte-identical to `rust/normal-duel-core/src/mock_evaluator.rs`;
 * it exists so the cross-engine PUCT parity test compares two *searches* and
 * not two ONNX runtimes. Every value it produces is a 24-bit fraction times a
 * power of two, so it is exact in f32 and in the f64 JavaScript computes in —
 * there is no rounding here for the two implementations to disagree about.
 *
 * The hash runs over the feature vector's raw f32 bit patterns, so an encoder
 * that is merely close to the Rust encoder produces a completely different
 * evaluation and the parity test fails loudly instead of quietly passing.
 */

import { encodeState } from './normal-duel-nn-encoding.mjs';
import { policySize } from './normal-duel-engine.mjs';

/** FNV-1a over the feature vector's f32 bit patterns. */
export function hashFeatures(features) {
  const words = new Uint32Array(features.buffer, features.byteOffset, features.length);
  let hash = 2166136261;
  for (let index = 0; index < words.length; index += 1) {
    hash = (hash ^ words[index]) >>> 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash;
}

/** 32-bit avalanche, so neighbouring codes get unrelated priors. */
export function mix32(seed) {
  let x = seed >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

/** A 24-bit fraction in [0, 1), exact in both f32 and f64. */
function unit(word) {
  return (word >>> 8) / 16777216;
}

/**
 * `evaluate(config, state)` in the shape `puctSearch` expects. The policy is a
 * Float32Array so the priors the search divides are f32 values, exactly as they
 * are on the Rust side.
 */
export function mockEvaluator(config, state) {
  const features = encodeState(config, state);
  const size = policySize(config);
  const hash = hashFeatures(features);
  const policy = new Float32Array(size);
  for (let code = 0; code < size; code += 1) {
    policy[code] = unit(mix32((hash ^ Math.imul(code, 0x9e3779b1)) >>> 0));
  }
  return { policy, value: unit(mix32((hash ^ 0xdeadbeef) >>> 0)) * 2 - 1 };
}
