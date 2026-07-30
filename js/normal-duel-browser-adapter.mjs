/** Browser-facing, rule-free publication layer for the normal Duel consumer. */
import {
  classifyNormalDuelConsumer,
  normalDuelLegalMoves,
  normalDuelMoveTowardGoal,
  normalDuelTryWall
} from './normal-duel-consumer.mjs';

function requireEligible(scope) {
  const result = classifyNormalDuelConsumer(scope);
  if (!result.eligible) throw new TypeError(`normal-duel-v1 consumer is unavailable: ${result.reason}`);
  return scope.duelSize;
}

export const NormalDuelConsumer = Object.freeze({
  classify: classifyNormalDuelConsumer,
  legalMoves(scope, snapshot) {
    return normalDuelLegalMoves(requireEligible(scope), snapshot);
  },
  tryWall(scope, snapshot, wall) {
    return normalDuelTryWall(requireEligible(scope), snapshot, wall);
  },
  moveTowardGoal(scope, snapshot, path) {
    return normalDuelMoveTowardGoal(requireEligible(scope), snapshot, path);
  }
});

if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'NormalDuelConsumer', {
    configurable: true,
    enumerable: false,
    value: NormalDuelConsumer,
    writable: false
  });
}
