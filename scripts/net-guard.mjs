/**
 * The network-contribution guard, shared by everything that drives the wasm search.
 *
 * WHY THIS IS A MODULE. The search returns a LEGAL move even when the network
 * contributed nothing, which has shipped twice in this repo. The play server grew
 * checks for that; `sims-match.mjs` -- the harness both of those bugs were actually
 * FOUND in, and the one that produces the strength numbers those bugs invalidated --
 * did not. A guard that lives in one caller protects one caller. So it lives here and
 * both use it, which also means the tests for it cover both.
 *
 * WHAT IT ESTABLISHES: a live network answered, its answer was the right shape, it was
 * not degenerate, and it varied with the position it was given.
 *
 * WHAT IT DOES NOT: that the network read the CORRECT position. These are checks on the
 * response, so request-side corruption -- the `Buffer.from(typedArray)` bug sent a body
 * a quarter of the right size -- produces a well-shaped varying policy for the wrong
 * board, and only the inference server's own length check catches it.
 */

/** Refused because the network's contribution could not be established. */
export class NetGuardError extends Error {
  constructor(code, detail) {
    super(`${code}: ${detail}`);
    this.code = code;
  }
}

export function createNetGuard(policyLen) {
  let leaves = 0;
  let informed = 0;      // leaves given real, varying priors
  let blind = 0;         // leaves with legal moves but no prior mass at all
  let flat = 0;          // leaves whose legal logits were bit-identical
  let maskEmpty = 0;     // leaves handed out with NO legal move in the mask
  let narrowFlat = 0;    // flat, but with < 3 legal moves: tolerated, and not evidence
  let firstSignature = null;
  let allIdentical = true;

  return {
    get leaves() { return leaves; },
    get narrowFlat() { return narrowFlat; },
    get netLeaves() { return informed; },
    /** True once the network is known to be broken, so the caller can stop early. */
    get doomed() {
      // Includes the flat and constant-response trips, not just the two per-leaf ones.
      // Omitting them meant the canonical broken networks -- all-zero logits, and a
      // fixed vector for every position -- consumed the ENTIRE budget before refusing,
      // times the client's twelve retries. That is what this exists to prevent.
      return maskEmpty > 0 || blind > 0 || flat >= 3
        || (leaves >= 8 && allIdentical);
    },

    /**
     * Turn one raw response into the probabilities `submit()` wants, and record what
     * the network's contribution at this leaf actually was.
     *
     * `out` is the raw response (policyLen logits then one value); `mask` is the filled
     * legal mask. Returns { probs, value }. Throws NetGuardError on a per-leaf
     * violation; aggregate violations surface from finish().
     */
    classify(out, mask) {
      if (out.length !== policyLen + 1) {
        throw new NetGuardError('net_bad_shape',
          `got ${out.length} floats, want ${policyLen + 1}`);
      }
      let max = -Infinity;
      let min = Infinity;
      let legal = 0;
      for (let i = 0; i < policyLen; i += 1) {
        if (mask[i] > 0) {
          legal += 1;
          if (out[i] > max) max = out[i];
          if (out[i] < min) min = out[i];
        }
      }
      const probs = new Float32Array(policyLen);
      let sum = 0;
      if (max !== -Infinity) {
        for (let i = 0; i < policyLen; i += 1) {
          if (mask[i] > 0) { const e = Math.exp(out[i] - max); probs[i] = e; sum += e; }
        }
      }
      if (sum > 0) for (let i = 0; i < policyLen; i += 1) probs[i] /= sum;
      // NOT normalisable means NOT submittable. With only SOME legal logits NaN, sum is
      // NaN, the NaN entries survived unnormalised, and the caller still submitted them
      // -- so Rust rejected them with an opaque invalid_evaluation instead of this
      // module reporting net_degenerate. Zeroing keeps submit()'s contract (non-negative
      // probabilities); the leaf is already counted as blind, so the move still refuses.
      else probs.fill(0);

      // legal === 0 is NOT a terminal leaf: next_leaf() scores terminal leaves
      // internally and never hands them out, so a leaf we were given always has a
      // legal move. An empty mask means pendingLeafMask() did not fill the buffer --
      // the original bug this guard exists for.
      if (legal === 0) maskEmpty += 1;
      else if (!(sum > 0)) blind += 1;
      // `legal >= 3`, not `> 1`: with two legal moves -- a late-game pawn in a corridor
      // with both sides out of stock -- two float32 logits coinciding is a coincidence,
      // not evidence. Requiring three keeps the test out of the endgame.
      else if (max === min) {
        // Flat legal logits. Only counted as a violation when there were >= 3 legal
        // moves: with two, a float32 coincidence is ordinary. But a narrow flat leaf is
        // not EVIDENCE either, so it goes in its own bucket rather than inflating
        // netLeaves, which is reported to the caller as evidence-of-network.
        if (legal >= 3) flat += 1;
        else narrowFlat += 1;
      } else informed += 1;

      // Cross-leaf signature over the WHOLE response, quantised to ~1e-6 -- so this
      // detects "identical to within a millionth", NOT bit equality. That is deliberate
      // (a network is not going to differ by 1e-7 between positions and mean it), but the
      // error message says "identical" rather than "bit-identical" for that reason.
      // Masking the signature would be wrong: the
      // legal SET differs per leaf, so a network returning one constant vector would
      // still produce a different signature each time and slip through. FNV-1a rather
      // than string building, since a 4096-sim move would build 4096 long strings.
      let signature = 2166136261;
      for (let i = 0; i < out.length; i += 1) {
        signature = (signature ^ (Math.fround(out[i]) * 1e6 | 0)) >>> 0;
        signature = (signature * 16777619) >>> 0;
      }
      if (firstSignature === null) firstSignature = signature;
      else if (signature !== firstSignature) allIdentical = false;

      // The value head gets the same scrutiny as the policy. It had none: a NaN was
      // silently clamped to 0 at every leaf, so the search fell back to
      // priors-plus-visit-counts while the move stayed legal and every other signal
      // looked healthy -- exactly the failure mode this guard was written to stop.
      const raw = out[out.length - 1];
      if (!Number.isFinite(raw)) {
        throw new NetGuardError('net_bad_value',
          `value head returned ${raw} at leaf ${leaves}`);
      }
      // The head is a tanh, so beyond float slack this is a broken graph, and clamping
      // it would hide that.
      if (Math.abs(raw) > 1 + 1e-4) {
        throw new NetGuardError('net_bad_value',
          `value ${raw} is outside [-1, 1] at leaf ${leaves}`);
      }
      leaves += 1;
      return { probs, value: Math.max(-1, Math.min(1, raw)) };
    },

    /** Throws if the move as a whole was not network-shaped. Call once, at the end. */
    finish() {
      if (maskEmpty > 0) {
        throw new NetGuardError('net_mask_empty',
          `${maskEmpty} of ${leaves} leaves had no legal move in the mask; `
          + 'the mask buffer is not being filled');
      }
      if (blind > 0) {
        throw new NetGuardError('net_degenerate',
          `${blind} priorless leaf/leaves of ${leaves}`);
      }
      // Flat leaves are tolerated in ones and twos, and refused in bulk. Zero tolerance could not
      // recover: both clients derive the seed from the move number, so a retry rebuilds
      // the identical tree, reaches the identical leaf and fails identically -- one
      // float32 collision (~1e-7 per leaf) killed the turn for good, and a reload
      // replayed it. A broken network is flat on essentially every leaf, so a
      // count-plus-all-decided test separates the two while leaving one or two
      // coincidences survivable. (It is a count, not a fraction.)
      // `flat === decided` is kept for the message it produces, not for reach: with
      // flat > 0 it is equivalent to informed === 0, which the next check refuses anyway.
      const decided = informed + flat;
      if (flat > 0 && (flat >= 3 || flat === decided)) {
        throw new NetGuardError('net_degenerate',
          `${flat} of ${decided} decided leaves had flat legal logits`);
      }
      // Applies at every budget: from_state rejects terminal roots and the root is
      // always evaluated, so there is always at least one non-terminal leaf. An earlier
      // `sims > 1` exemption could only ever take effect once something was broken.
      // Fires even when every leaf was `narrowFlat`. That is deliberate and pinned by
      // `check-net-guard.mjs` ("narrow flat not evidence"): tolerated means "not proof
      // the network is broken", NOT "proof it answered".
      //
      // It does refuse some legitimate positions. Observed live 2026-08-16 at ply 51,
      // both players out of wall stock, a single candidate holding all 512 visits,
      // rootValue 0.9995: features and mask went out, the network answered, and flat
      // logits over a 1-2 move legal set left `informed` at 0. Refusing is still right
      // HERE -- a play server that cannot distinguish "answered" from "dead" must not
      // move -- so callers that meet such positions in bulk (corpus generation) absorb
      // the refusal on their side instead of loosening this.
      //
      // Relaxing the condition to `leaves > narrowFlat` was tried on 2026-08-16 and
      // reverted: it does not stop the refusal, it just downgrades it to the less
      // accurate `net_input_ignored` from the check below, and it broke that test.
      if (informed === 0) {
        throw new NetGuardError('net_never_consulted',
          `${leaves} leaves (${narrowFlat} narrow-flat), none network-informed`);
      }
      // A network ignoring its input returns one fixed vector for every position; each
      // response passes every per-leaf check. Tests ALL identical rather than ANY
      // repeat, because transpositions make a genuine repeat possible.
      if (leaves >= 2 && allIdentical) {
        throw new NetGuardError('net_input_ignored',
          `all ${leaves} leaves returned an identical response (to ~1e-6)`);
      }
    }
  };
}
