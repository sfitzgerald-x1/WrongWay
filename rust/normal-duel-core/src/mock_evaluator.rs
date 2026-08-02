//! A deterministic stand-in for the network, used by the cross-engine parity
//! harness and by the throughput benchmark.
//!
//! Why a mock and not the real network
//! -----------------------------------
//! The parity claim is about the *search*, so the evaluator has to be removed
//! from the experiment. If both engines ran ONNX, a disagreement could be an
//! ONNX build difference, a BLAS difference, or a genuine search bug, and there
//! would be no way to tell from the outside. A pure function of the feature
//! vector cannot differ between the two engines unless the feature vector does,
//! and the feature encoder is already pinned bit-for-bit against JS by
//! `tests/js_hot_path_parity.rs`.
//!
//! For the benchmark the same property matters for a different reason: with a
//! mock on both sides, the measured difference is tree plus engine cost and
//! nothing else.
//!
//! Exactness
//! ---------
//! Every value this produces is a 24-bit fraction scaled by a power of two, so
//! it is exactly representable in `f32` *and* in the `f64` a JavaScript
//! implementation naturally computes in. There is no rounding anywhere in the
//! mock, so `Math.fround` is unnecessary on the JS side and the two
//! implementations cannot drift. `js/normal-duel-mock-evaluator.mjs` is the
//! matching implementation; `tests/js_puct_parity.rs` drives both.

use crate::MAX_POLICY_CODES;

/// FNV-1a over the feature vector's raw `f32` bit patterns.
///
/// Hashing bit patterns rather than values means a feature encoding that
/// differs by one ULP produces a completely different evaluation, so the parity
/// test cannot pass on an encoder that is merely close.
#[must_use]
pub fn hash_features(features: &[f32]) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for value in features {
        hash ^= value.to_bits();
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}

/// A 32-bit avalanche (the `splitmix32`-style mixer) so neighbouring codes give
/// unrelated priors.
#[must_use]
pub fn mix32(seed: u32) -> u32 {
    let mut x = seed;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// A 24-bit fraction in `[0, 1)`, exact in both `f32` and `f64`.
fn unit(word: u32) -> f64 {
    f64::from(word >> 8) / 16_777_216.0
}

/// Policy logits in `[0, 1)` for every policy code, and a value in `[-1, 1)`.
///
/// The "logits" are used directly as unnormalised priors, which is what the
/// search does with a real policy head's output after masking.
pub fn evaluate(features: &[f32], policy: &mut [f32]) -> f64 {
    assert_eq!(policy.len(), MAX_POLICY_CODES, "policy buffer is policy_size");
    let hash = hash_features(features);
    for (code, slot) in policy.iter_mut().enumerate() {
        *slot = unit(mix32(hash ^ (code as u32).wrapping_mul(0x9e37_79b1))) as f32;
    }
    unit(mix32(hash ^ 0xdead_beef)) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluations_are_exactly_representable_in_f32() {
        let features: Vec<f32> = (0..648).map(|index| index as f32 / 648.0).collect();
        let mut policy = vec![0.0_f32; MAX_POLICY_CODES];
        let value = evaluate(&features, &mut policy);
        assert!((-1.0..1.0).contains(&value));
        assert_eq!(f64::from(value as f32), value, "value survives an f32 round trip");
        for probability in &policy {
            assert!((0.0..1.0).contains(probability));
        }
    }

    #[test]
    fn a_one_ulp_feature_change_changes_the_evaluation() {
        let mut features = vec![0.5_f32; 648];
        let mut left = vec![0.0_f32; MAX_POLICY_CODES];
        let mut right = vec![0.0_f32; MAX_POLICY_CODES];
        let value_left = evaluate(&features, &mut left);
        features[17] = f32::from_bits(features[17].to_bits() + 1);
        let value_right = evaluate(&features, &mut right);
        assert_ne!(value_left, value_right);
        assert_ne!(left, right);
    }
}
