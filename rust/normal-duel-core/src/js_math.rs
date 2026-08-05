//! Bit-exact reimplementations of the JavaScript primitives the PUCT search
//! depends on.
//!
//! Why this module exists
//! ----------------------
//! The acceptance criterion for the Rust search is *exact* parity with
//! `js/normal-duel-puct-search.mjs`: identical visit counts, identical chosen
//! action, identical root value. Every arithmetic operation in that file is
//! IEEE-754 double precision and therefore reproducible in Rust — except
//! `Math.log`, which is a library function, not an IEEE operation. Measured on
//! this machine over 22,001 search-representative inputs:
//!
//!   * Rust `f64::ln` (Apple libm) disagrees with V8's `Math.log` on 1,450 of
//!     them, always by 1 ULP.
//!   * A straight transcription of FDLIBM `__ieee754_log` (the algorithm V8
//!     ships in `src/base/ieee754.cc`) still disagrees on 315.
//!   * The same transcription with the two tail expressions evaluated as fused
//!     multiply-adds — which is what clang emits for that source under its
//!     default `-ffp-contract=on` — agrees on all 22,001, and on a wider
//!     380,004-input sweep as well.
//!
//! So [`js_log`] is FDLIBM's `log` with the tail contracted, and
//! `tests/js_math_parity.rs` pins that against real `Math.log` output rather
//! than trusting the reasoning above. Using [`f64::mul_add`] (a genuine fused
//! multiply-add, correctly rounded once) rather than relying on the compiler to
//! contract also makes the result identical on native and on wasm32, where the
//! backend has no fused instruction and calls `fma` instead.
//!
//! A 1-ULP disagreement would not obviously matter — the logarithm only ever
//! reaches a comparison, never a visit count or a value — but "obviously" is
//! how sign errors survive. Matching bit patterns removes the argument.
//!
//! What is portable and what is not
//! --------------------------------
//! These two are easy to conflate, and the distinction decides how the parity
//! test has to be written:
//!
//!   * **[`js_log`] is bit-portable.** Every step is an IEEE-754 operation with
//!     one correctly rounded result, including the tail, because [`f64::mul_add`]
//!     is a *specified* fused multiply-add — one rounding, by definition — not a
//!     request that the optimiser might decline. x86-64 without FMA3 lowers it to
//!     a libm `fma` call and wasm32 to a runtime call; both return the same bits
//!     an arm64 `fmadd` does. Same input, same output, every target.
//!   * **V8's `Math.log` is not.** It is C++ — `src/base/ieee754.cc`, the same
//!     FDLIBM algorithm — compiled by whatever toolchain built that node binary.
//!     Under clang's default `-ffp-contract=on`, `a*b+c` in the tail contracts to
//!     an FMA when the target has the instruction and stays as two rounded
//!     operations when it does not. arm64 always has it; baseline x86-64 does
//!     not. So `Math.log` returns different bits on the two architectures, 1 ULP
//!     apart, on roughly 1% of search-representative inputs — 0.6819084220333025
//!     is the smallest one the sweep finds: `0xbfd880c6d8b3c8d6` on arm64,
//!     `...d5` on x86-64.
//!
//! The consequence is not academic. `js_log` feeds the Gumbel draw and the PUCT
//! exploration term, so a 1-ULP difference in a logit can reorder two nearly
//! tied actions and select a different move — from there the games diverge
//! completely. **A mixed JS/Rust deployment is only bit-identical when the
//! JavaScript side runs on arm64.** The same JS on x86-64 is a different
//! searcher. Production self-play runs on arm64 and emits byte-identical
//! shards there, which is why the shipped code is correct and why the fixture in
//! `tests/fixtures/js_log_reference.txt` records arm64 values.
//!
//! Hence the shape of the test. The exact assertion is against that committed
//! fixture, so it holds on every target with no engine in the loop; a separate
//! check runs the host's live `Math.log` against the same fixture and *reports*
//! any divergence instead of failing, so the architecture dependence stays
//! visible rather than latent.
// The fdlibm constants below are transcribed verbatim from `e_log.c`, including
// digits beyond what f64 can represent. Trimming them to fit would change no
// value -- the compiler rounds either way -- but it would break the property
// that makes this file auditable: every literal here can be diffed character by
// character against the published source. The bit patterns are pinned by
// `constants_match_their_fdlibm_bit_patterns`, so a mistyped digit fails loudly.
#![allow(clippy::excessive_precision)]

/// `Math.LN2` split high/low, and the FDLIBM `log` polynomial coefficients.
///
/// These are the decimal literals from `e_log.c`. Every one of them is also
/// pinned to its hex encoding — the value `e_log.c` quotes alongside the
/// decimal — by `constants_match_their_fdlibm_bit_patterns`, because a
/// mistyped digit here produces a plausible-looking logarithm that is wrong in
/// the fourth decimal place, and nothing else in the crate would notice.
const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01; // 0x3fe62e42fee00000
const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10; // 0x3dea39ef35793c76
const TWO54: f64 = 1.801_439_850_948_198_400_00e+16; // 0x4350000000000000
const LG1: f64 = 6.666_666_666_666_735_130e-01; // 0x3fe5555555555593
const LG2: f64 = 3.999_999_999_940_941_908e-01; // 0x3fd999999997fa04
const LG3: f64 = 2.857_142_874_366_239_149e-01; // 0x3fd2492494229359
const LG4: f64 = 2.222_219_843_214_978_396e-01; // 0x3fcc71c51d8e78af
const LG5: f64 = 1.818_357_216_161_805_012e-01; // 0x3fc7466496cb03de
const LG6: f64 = 1.531_383_769_920_937_332e-01; // 0x3fc39a09d078c69f
const LG7: f64 = 1.479_819_860_511_658_591e-01; // 0x3fc2f112df3e5244
const ONE_THIRD: f64 = 0.333_333_333_333_333_33;

fn high_word(value: f64) -> i32 {
    (value.to_bits() >> 32) as u32 as i32
}

fn with_high_word(value: f64, high: i32) -> f64 {
    f64::from_bits((u64::from(high as u32) << 32) | (value.to_bits() & 0xffff_ffff))
}

/// `Math.log(x)`, bit-for-bit.
///
/// The structure is FDLIBM's: reduce `x` to `2^k * (1 + f)` with
/// `sqrt(2)/2 < 1 + f < sqrt(2)`, evaluate `log(1 + f)` from the odd
/// polynomial in `s = f / (2 + f)`, and reassemble with `k * ln2`. Only the
/// two tail expressions are written as `mul_add`; see the module comment.
#[must_use]
pub fn js_log(x: f64) -> f64 {
    let mut x = x;
    let mut hx = high_word(x);
    let lx = x.to_bits() as u32;
    let mut k: i32 = 0;

    if hx < 0x0010_0000 {
        // Zero, negative, or subnormal.
        if (hx & 0x7fff_ffff) as u32 | lx == 0 {
            return f64::NEG_INFINITY; // log(+-0)
        }
        if hx < 0 {
            return f64::NAN; // log(negative)
        }
        k -= 54;
        x *= TWO54; // scale a subnormal into the normal range
        hx = high_word(x);
    }
    if hx >= 0x7ff0_0000 {
        return x + x; // infinity or NaN
    }

    k += (hx >> 20) - 1023;
    hx &= 0x000f_ffff;
    // Choose the binade so the reduced argument straddles 1 rather than 1.5.
    let selector = (hx + 0x9_5f64) & 0x10_0000;
    x = with_high_word(x, hx | (selector ^ 0x3ff0_0000));
    k += selector >> 20;
    let f = x - 1.0;
    let dk = f64::from(k);

    if (0x000f_ffff & (2 + hx)) < 3 {
        // |f| < 2^-20: the polynomial degenerates to its first two terms.
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (-ONE_THIRD).mul_add(f, 0.5);
        if k == 0 {
            return f - r;
        }
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }

    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let mut selector = hx - 0x6_147a;
    let complement = 0x6_b851 - hx;
    let t1 = w * w.mul_add(w.mul_add(LG6, LG4), LG2);
    let t2 = z * w.mul_add(w.mul_add(w.mul_add(LG7, LG5), LG3), LG1);
    selector |= complement;
    let r = t2 + t1;

    if selector > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - (-s).mul_add(hfsq + r, hfsq)
        } else {
            dk * LN2_HI - ((hfsq - s.mul_add(hfsq + r, dk * LN2_LO)) - f)
        }
    } else if k == 0 {
        (-s).mul_add(f - r, f)
    } else {
        dk * LN2_HI - (s.mul_add(f - r, -(dk * LN2_LO)) - f)
    }
}

/// `js/lcg32.mjs`: `state = (state * 1664525 + 1013904223) mod 2^32`, advanced
/// before every sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lcg32 {
    state: u32,
}

impl Lcg32 {
    #[must_use]
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// One standard Gumbel draw, consuming exactly one `next_u32`.
    ///
    /// `u` is placed strictly inside `(0, 1)` so `-log(-log(u))` is finite. The
    /// two guards are unreachable for a `u32` draw and exist only because the
    /// JavaScript has them: keeping the shapes identical is cheaper than
    /// re-deriving that they cannot fire.
    // `!(u > 0.0)` is not `u <= 0.0`: they differ on NaN, and the negated form is
    // both the NaN-correct guard and the shape the JavaScript uses. Rewriting it
    // via `partial_cmp` would obscure exactly the correspondence this file exists
    // to preserve.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn gumbel(&mut self) -> f64 {
        let raw = self.next_u32();
        let mut u = (f64::from(raw) + 0.5) / 4_294_967_296.0;
        if !(u > 0.0) {
            u = f64::from_bits(1); // Number.MIN_VALUE
        }
        if !(u < 1.0) {
            u = 1.0 - f64::EPSILON / 2.0;
        }
        -js_log(-js_log(u))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg32_matches_the_reference_stream() {
        // Generated by actually running `createLcg32(1234)` from js/lcg32.mjs:
        //   node -e 'import("./js/lcg32.mjs").then(({createLcg32})=>{
        //     const r=createLcg32(1234); for(let i=0;i<5;i++) console.log(r()); })'
        // The previous values here were not produced that way and did not match
        // the engine; the Rust side was correct and the fixture was wrong.
        let mut rng = Lcg32::new(1234);
        assert_eq!(rng.next_u32(), 3_067_928_073);
        assert_eq!(rng.next_u32(), 889_114_580);
        assert_eq!(rng.next_u32(), 3_219_257_635);
        assert_eq!(rng.next_u32(), 1_486_326_822);
        assert_eq!(rng.next_u32(), 3_450_746_189);
    }

    #[test]
    fn constants_match_their_fdlibm_bit_patterns() {
        for (value, bits, name) in [
            (LN2_HI, 0x3fe6_2e42_fee0_0000_u64, "ln2_hi"),
            (LN2_LO, 0x3dea_39ef_3579_3c76, "ln2_lo"),
            (TWO54, 0x4350_0000_0000_0000, "two54"),
            (LG1, 0x3fe5_5555_5555_5593, "Lg1"),
            (LG2, 0x3fd9_9999_9997_fa04, "Lg2"),
            (LG3, 0x3fd2_4924_9422_9359, "Lg3"),
            (LG4, 0x3fcc_71c5_1d8e_78af, "Lg4"),
            (LG5, 0x3fc7_4664_96cb_03de, "Lg5"),
            (LG6, 0x3fc3_9a09_d078_c69f, "Lg6"),
            (LG7, 0x3fc2_f112_df3e_5244, "Lg7"),
        ] {
            assert_eq!(value.to_bits(), bits, "{name} is mistyped: {value:e}");
        }
    }

    #[test]
    fn js_log_handles_the_domain_edges() {
        assert_eq!(js_log(1.0), 0.0);
        assert!(js_log(0.0).is_infinite() && js_log(0.0) < 0.0);
        assert!(js_log(-1.0).is_nan());
        assert!(js_log(f64::INFINITY).is_infinite());
        assert!(js_log(f64::NAN).is_nan());
        // Subnormal input takes the scale-up path.
        assert!((js_log(f64::from_bits(1)) - (-744.440_071_921_381_3)).abs() < 1e-9);
    }

    #[test]
    fn gumbel_is_finite_across_the_whole_draw_range() {
        for seed in [0_u32, 1, 7, 0xffff_ffff, 0x8000_0000] {
            let mut rng = Lcg32::new(seed);
            for _ in 0..1000 {
                assert!(rng.gumbel().is_finite());
            }
        }
    }
}
