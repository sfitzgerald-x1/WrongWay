/**
 * Shared deterministic PRNG used by checked-in fixture generators.
 *
 * lcg32-v1 initializes from a uint32 seed and advances before every sample:
 * state = (state * 1664525 + 1013904223) mod 2^32.
 */
export const LCG32_ALGORITHM = 'lcg32-v1';

export function createLcg32(seed) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
    throw new TypeError(`${LCG32_ALGORITHM} requires a uint32 seed; received ${String(seed)}`);
  }
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}
