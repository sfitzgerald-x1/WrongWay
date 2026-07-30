import { validateConfig } from '../../js/normal-duel-engine.mjs';

export const CANONICAL_STRENGTH_CONFIGURATION_SHA256 =
  '1869dc2a71b4043aebee681e64283f42b93d13665f7ead9f1eca9ccb1a0b1da0';
export const CANONICAL_STRENGTH_SEED = 0x6d2b79f5;
export const CANONICAL_STRENGTH_DEADLINE_MS = 900;
export const CANONICAL_STRENGTH_INITIALIZATION_CAP_MS = 900;
export const CANONICAL_STRENGTH_OBSERVER_CAP_MS = 900;
export const MINIMUM_STRENGTH_OPENING_PAIRS = 200;
export const CANONICAL_STRENGTH_GENERATOR_VERSION =
  'normal-duel-balanced-opening-generator-1.0.0';
export const CANONICAL_STRENGTH_GENERATOR_ALGORITHM = 'lcg32-v1';

export function normalDuelConfig({ size = 9, firstPlayer = 'A', plyCap = 200 } = {}) {
  if (size !== 7 && size !== 9) {
    throw new TypeError('normal-duel strength config size must be 7 or 9');
  }
  const center = Math.floor(size / 2);
  return validateConfig({
    ruleset: 'normal-duel-v1',
    rows: size,
    columns: size,
    start: { A: { r: size - 1, c: center }, B: { r: 0, c: center } },
    goalRows: { A: 0, B: size - 1 },
    initialStock: { A: 10, B: 10 },
    jumpRule: 'permissive-adjacent-exit-v1',
    repetitionThreshold: 3,
    plyCap,
    firstPlayer
  });
}

export const CANONICAL_STRENGTH_CONFIG = normalDuelConfig({
  size: 9,
  firstPlayer: 'A',
  plyCap: 200
});
