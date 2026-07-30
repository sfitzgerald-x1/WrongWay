#!/usr/bin/env node
/**
 * JavaScript half of the normal-duel throughput benchmark.
 *
 * The companion Rust example implements this exact workload.  Parsing the
 * fixture and rebuilding its states are deliberately outside the timed region:
 * every timed pass starts from the same already-validated fixture states and
 * performs legal generation, a checked apply for every root action, a child
 * legal-code query, and an exact-depth perft for every frozen root.
 */
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  applyLegalAction,
  createInitialState,
  encodeAction,
  legalActionCodes,
  legalActions,
  validateState
} from '../js/normal-duel-engine.mjs';
import { perft, seededWallState, stateFromActionCodes } from '../js/normal-duel-perft.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const FIXTURE_PATH = resolve(ROOT, 'tests/fixtures/normal-duel-perft-v1.json');
const WORKLOAD_VERSION = 'normal-duel-throughput-v1';
const DEFAULT_WARMUP = 2;
const DEFAULT_SAMPLES = 9;

function fail(message) {
  throw new Error(`normal-duel-throughput-benchmark: ${message}`);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

/** A compact, portable FNV-1a-64 digest with unambiguous UTF-8 framing. */
class Fnv1a64 {
  #value = 0xcbf29ce484222325n;

  add(value) {
    const text = String(value);
    const bytes = Buffer.from(text, 'utf8');
    this.#update(Buffer.from(`${bytes.length}:`, 'ascii'));
    this.#update(bytes);
    this.#update(Buffer.from('|', 'ascii'));
  }

  #update(bytes) {
    for (const byte of bytes) {
      this.#value ^= BigInt(byte);
      this.#value = (this.#value * 0x100000001b3n) & 0xffffffffffffffffn;
    }
  }

  hex() {
    return this.#value.toString(16).padStart(16, '0');
  }
}

function numberArgument(raw, name, minimum) {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) fail(`${name} must be an integer >= ${minimum}`);
  return value;
}

function parseArguments(argv) {
  const options = { mode: 'measure', warmup: DEFAULT_WARMUP, samples: DEFAULT_SAMPLES };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--verify') options.mode = 'verify';
    else if (argument === '--measure') options.mode = 'measure';
    else if (argument === '--profile') options.mode = 'profile';
    else if (argument === '--smoke') {
      options.mode = 'measure';
      options.warmup = 0;
      options.samples = 1;
    } else if (argument === '--warmup') options.warmup = numberArgument(argv[++index], 'warmup', 0);
    else if (argument === '--samples') options.samples = numberArgument(argv[++index], 'samples', 1);
    else fail(`unknown argument ${argument}`);
  }
  return options;
}

function sameJson(left, right) {
  return canonicalJson(left) === canonicalJson(right);
}

function replayCase(fixture, entry) {
  const config = fixture.configs[entry.configId];
  if (!config) fail(`${entry.id} refers to unknown config ${entry.configId}`);
  let state;
  if (entry.kind === 'initial') state = createInitialState(config);
  else if (entry.kind === 'action-codes') state = stateFromActionCodes(config, entry.actionCodes);
  else if (entry.kind === 'seeded-walls') {
    state = seededWallState(config, { seed: entry.generator?.seed, plies: entry.generator?.plies });
  } else {
    fail(`${entry.id} has unsupported fixture kind ${entry.kind}`);
  }
  if (!sameJson(state, entry.state)) fail(`${entry.id} replay differs from its frozen state`);
  return Object.freeze({
    id: entry.id,
    config: Object.freeze(config),
    state: Object.freeze(validateState(config, state)),
    depth: entry.expect.depth,
    expectedLeaves: entry.expect.leavesByDepth.at(-1),
    expectedRootCodes: Object.freeze([...entry.expect.rootActionCodes]),
    perftOptions: entry.perftOptions ?? undefined
  });
}

function makePlan() {
  const source = readFileSync(FIXTURE_PATH);
  const fixture = JSON.parse(source);
  if (!Array.isArray(fixture.cases) || fixture.cases.length === 0) fail('fixture has no cases');
  return Object.freeze({
    fixtureSha256: createHash('sha256').update(source).digest('hex'),
    cases: Object.freeze(fixture.cases.map((entry) => replayCase(fixture, entry)))
  });
}

/**
 * Validate the output of every operation once before timing and create a
 * language-neutral checksum. This uses the same action flow as `runPass`, but
 * deliberately does not contribute to its timings.
 */
function verifyPlan(plan) {
  const actionsDigest = new Fnv1a64();
  const perftDigest = new Fnv1a64();
  actionsDigest.add(WORKLOAD_VERSION);
  perftDigest.add(WORKLOAD_VERSION);
  let rootActionCount = 0;
  let childActionCount = 0;
  let expectedLeafTotal = 0;

  for (const entry of plan.cases) {
    const actions = legalActions(entry.config, entry.state);
    const codes = actions.map((action) => encodeAction(entry.config, action));
    if (!sameJson(codes, entry.expectedRootCodes)) fail(`${entry.id} root legal action codes differ from fixture`);

    actionsDigest.add(entry.id);
    actionsDigest.add(entry.state.positionKey);
    actionsDigest.add(entry.depth);
    actionsDigest.add(codes.length);
    rootActionCount += codes.length;
    for (let index = 0; index < actions.length; index += 1) {
      const code = codes[index];
      const child = applyLegalAction(entry.config, entry.state, actions[index]);
      const childCodes = legalActionCodes(entry.config, child);
      actionsDigest.add(code);
      actionsDigest.add(child.positionKey);
      actionsDigest.add(childCodes.length);
      for (const childCode of childCodes) actionsDigest.add(childCode);
      childActionCount += childCodes.length;
    }

    const leaves = perft(entry.config, entry.state, entry.depth, entry.perftOptions);
    if (leaves !== entry.expectedLeaves) fail(`${entry.id} perft differs from fixture`);
    perftDigest.add(entry.id);
    perftDigest.add(entry.depth);
    perftDigest.add(leaves);
    expectedLeafTotal += leaves;
  }

  return Object.freeze({
    actionChecksum: actionsDigest.hex(),
    perftChecksum: perftDigest.hex(),
    rootActionCount,
    childActionCount,
    expectedLeafTotal
  });
}

/** The sole timed unit: one fixed pass of legal/apply/child-legal/perft work. */
function runPass(plan) {
  let result = 0;
  for (const entry of plan.cases) {
    const actions = legalActions(entry.config, entry.state);
    result += actions.length;
    for (const action of actions) {
      result += encodeAction(entry.config, action);
      const child = applyLegalAction(entry.config, entry.state, action);
      result += legalActionCodes(entry.config, child).length;
    }
    result += perft(entry.config, entry.state, entry.depth, entry.perftOptions);
  }
  if (!Number.isSafeInteger(result)) fail('work result exceeds Number.MAX_SAFE_INTEGER');
  return result;
}

function measure(plan, warmup, samples) {
  for (let index = 0; index < warmup; index += 1) runPass(plan);
  const sampleMs = [];
  let workResult = null;
  for (let index = 0; index < samples; index += 1) {
    const start = performance.now();
    const result = runPass(plan);
    const elapsed = performance.now() - start;
    // Preserve the pass result as an observable value: neither engine gets to
    // replace legal/apply/perft with a precomputed fixture constant.
    if (workResult !== null && result !== workResult) fail('timed workload is not deterministic');
    workResult = result;
    sampleMs.push(elapsed);
  }
  return Object.freeze({ workResult, sampleMs });
}

function median(samples) {
  const sorted = [...samples].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function profileOperation(work, warmup, samples) {
  for (let index = 0; index < warmup; index += 1) work();
  const sampleMs = [];
  let result = null;
  for (let index = 0; index < samples; index += 1) {
    const start = performance.now();
    const value = work();
    const elapsed = performance.now() - start;
    if (result !== null && value !== result) fail('profiled operation is not deterministic');
    result = value;
    sampleMs.push(elapsed);
  }
  return Object.freeze({ result, sampleMilliseconds: sampleMs, medianMilliseconds: median(sampleMs) });
}

/**
 * A diagnostic decomposition of the fixed pass.  The setup arrays are made
 * outside timings so each row attributes just one public-core operation.
 * `runPass`, not this decomposition, remains the benchmark's source of truth.
 */
function profile(plan, warmup, samples) {
  const rootActions = plan.cases.map((entry) => legalActions(entry.config, entry.state));
  const children = rootActions.map((actions, caseIndex) => actions.map((action) =>
    applyLegalAction(plan.cases[caseIndex].config, plan.cases[caseIndex].state, action)));
  const rootLegal = profileOperation(() => plan.cases.reduce(
    (total, entry) => total + legalActions(entry.config, entry.state).length, 0), warmup, samples);
  const checkedApply = profileOperation(() => rootActions.reduce((total, actions, caseIndex) => total + actions.reduce(
    (caseTotal, action) => caseTotal + applyLegalAction(plan.cases[caseIndex].config, plan.cases[caseIndex].state, action).positionKey.length, 0), 0), warmup, samples);
  const childLegal = profileOperation(() => children.reduce((total, caseChildren, caseIndex) => total + caseChildren.reduce(
    (caseTotal, child) => caseTotal + legalActionCodes(plan.cases[caseIndex].config, child).length, 0), 0), warmup, samples);
  const scalarPerft = profileOperation(() => plan.cases.reduce(
    (total, entry) => total + perft(entry.config, entry.state, entry.depth, entry.perftOptions), 0), warmup, samples);
  return Object.freeze({ rootLegal, checkedApply, childLegal, scalarPerft });
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const plan = makePlan();
  const integrity = verifyPlan(plan);
  const baselineWorkResult = runPass(plan);
  const result = {
    benchmarkFormat: WORKLOAD_VERSION,
    engine: 'javascript-reference-engine',
    fixture: {
      path: 'tests/fixtures/normal-duel-perft-v1.json',
      sha256: plan.fixtureSha256,
      caseCount: plan.cases.length
    },
    integrity: { ...integrity, workResult: String(baselineWorkResult) },
    environment: {
      node: process.version,
      os: process.platform,
      arch: process.arch
    },
    verifiedBeforeTiming: true
  };
  if (options.mode === 'measure') {
    const timed = measure(plan, options.warmup, options.samples);
    if (timed.workResult !== baselineWorkResult) fail('timed work result differs from verified work result');
    result.measurement = {
      warmupPasses: options.warmup,
      sampleCount: options.samples,
      sampleMilliseconds: timed.sampleMs,
      workResult: String(timed.workResult)
    };
  } else if (options.mode === 'profile') {
    result.profile = profile(plan, options.warmup, options.samples);
  }
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
