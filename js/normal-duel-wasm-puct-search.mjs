/**
 * JS driver for the native PUCT tree exposed as `NormalDuelSearch`.
 *
 * Why this exists
 * ---------------
 * `puctSearch` in `js/normal-duel-puct-search.mjs` spends most of its time in
 * move generation, not in the network: at 128 simulations a ply the JS tree
 * costs milliseconds per simulation. The Rust core does the same search; this
 * module is the pump that lets JS keep owning the network. The tree calls out
 * once per leaf, the caller writes probabilities into wasm memory, and the
 * chosen action comes back as a code in the same absolute frame `puctSearch`
 * uses.
 *
 * Equivalence
 * -----------
 * `rust/normal-duel-core/tests/js_puct_parity.rs` proves the two trees agree
 * (visit counts, action code, bit-identical root value). What THIS file has to
 * get right is only the passthrough, which
 * `tests/normal-duel-wasm-search-wrapper-parity.test.mjs` pins against
 * `puctSearch` on the shared mock evaluator.
 *
 * Memory-growth hazard
 * --------------------
 * The tree arenas grow as a search deepens, and a wasm memory growth detaches
 * every existing `Float32Array` view. Every view here is therefore taken AFTER
 * the call that can grow memory (`nextLeaf`) and dropped before the next one.
 * Do not hoist them out of the loop.
 */
import { readFileSync } from 'node:fs';

/** Frozen identifier for this driver's protocol. */
export const WASM_PUCT_DRIVER_VERSION = 'wasm-puct-driver-v1';

const loaded = new Map();

/**
 * Instantiate the candidate wasm package once per release directory and hand
 * back the module plus its `WebAssembly.Memory`.
 */
export async function loadWasmPuct(releaseDir) {
  const cached = loaded.get(releaseDir);
  if (cached) return cached;
  const mod = await import(`${releaseDir}/normal-duel-wasm.mjs`);
  const instance = await mod.default({
    module_or_path: readFileSync(`${releaseDir}/normal-duel-wasm_bg.wasm`)
  });
  const memory = instance.memory ?? mod.__wasm?.memory ?? mod.memory;
  if (!memory || !(memory.buffer instanceof ArrayBuffer)) {
    throw new Error('wasm_memory_unavailable');
  }
  if (typeof mod.NormalDuelSearch !== 'function') {
    throw new Error('wasm_package_missing_NormalDuelSearch');
  }
  const handle = Object.freeze({ mod, memory });
  loaded.set(releaseDir, handle);
  return handle;
}

/**
 * Run one search-to-move over the native tree.
 *
 * `evaluateFeatures(features, mask, policyOut)` must write the leaf's
 * probabilities into `policyOut` and return the value in the leaf's own frame —
 * the same sign convention `puctSearch`'s `evaluate` uses. All three arguments
 * are live views into wasm memory and are invalid after the call returns.
 *
 * Returns the same four fields the parity surface compares.
 */
export function wasmPuctSearch({
  handle, config, state, evaluateFeatures, simulations, maxConsidered, cPuct, seed
}) {
  const { mod, memory } = handle;
  if (typeof evaluateFeatures !== 'function') throw new Error('invalid_evaluator');
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff) {
    throw new Error(`invalid_seed: ${String(seed)}`);
  }
  const options = { simulations, maxConsidered, seed };
  if (cPuct !== undefined) options.cPuct = cPuct;
  const search = new mod.NormalDuelSearch(
    JSON.stringify(config), JSON.stringify(state), JSON.stringify(options)
  );
  // Views are rebuilt every iteration: `nextLeaf` grows the arenas.
  const f32 = (ptr, len) => new Float32Array(memory.buffer, ptr, len);
  while (search.nextLeaf()) {
    search.pendingLeafMask();
    const policyLen = search.policyLen();
    const value = evaluateFeatures(
      f32(search.featuresPtr(), search.featuresLen()),
      f32(search.maskPtr(), policyLen),
      f32(search.policyPtr(), policyLen)
    );
    if (!Number.isFinite(value)) throw new Error('invalid_leaf_value');
    search.submit(value);
  }
  const pairs = search.visitCounts();
  const visitCounts = new Map();
  for (let index = 0; index < pairs.length; index += 2) {
    visitCounts.set(pairs[index], pairs[index + 1]);
  }
  return {
    actionCode: search.actionCode(),
    rootValue: search.rootValue(),
    simulationsUsed: search.simulationsUsed(),
    visitCounts
  };
}
