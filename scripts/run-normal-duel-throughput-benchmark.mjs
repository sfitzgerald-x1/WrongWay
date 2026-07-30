#!/usr/bin/env node
/**
 * Cross-engine driver for the normal-duel throughput benchmark.
 *
 * It performs a JS/Rust checksum handshake before either engine is asked to
 * measure. The ordinary command reports the observed ratio only; a threshold
 * is opt-in for a pinned, scheduled performance runner.
 */
import { execFileSync } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const JS_BENCHMARK = 'scripts/normal-duel-throughput-benchmark.mjs';
const RUST_MANIFEST = 'rust/Cargo.toml';
const RUST_EXAMPLE = 'throughput_benchmark';
const DEFAULT_WARMUP = 2;
const DEFAULT_SAMPLES = 9;

function fail(message) {
  throw new Error(`run-normal-duel-throughput-benchmark: ${message}`);
}

function integer(value, name, minimum) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < minimum) fail(`${name} must be an integer >= ${minimum}`);
  return number;
}

function parseArguments(argv) {
  const options = { smoke: false, warmup: DEFAULT_WARMUP, samples: DEFAULT_SAMPLES, minimumSpeedup: null };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--smoke') options.smoke = true;
    else if (argument === '--warmup') options.warmup = integer(argv[++index], 'warmup', 0);
    else if (argument === '--samples') options.samples = integer(argv[++index], 'samples', 1);
    else if (argument === '--min-speedup') {
      const value = Number(argv[++index]);
      if (!Number.isFinite(value) || value <= 0) fail('min-speedup must be a positive number');
      options.minimumSpeedup = value;
    } else fail(`unknown argument ${argument}`);
  }
  if (options.smoke) {
    if (options.minimumSpeedup !== null) fail('--min-speedup cannot be used with --smoke');
    options.warmup = 0;
    options.samples = 1;
  } else {
    if (options.warmup < 1) fail('a benchmark run needs at least one warmup pass');
    if (options.samples < 7) fail('a benchmark run needs at least seven samples');
  }
  return options;
}

function execute(command, commandArgs) {
  try {
    return execFileSync(command, commandArgs, {
      cwd: ROOT,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    });
  } catch (error) {
    const detail = error.stderr?.trim() || error.message;
    fail(`${command} ${commandArgs.join(' ')} failed: ${detail}`);
  }
}

function executeJson(command, commandArgs) {
  const output = execute(command, commandArgs).trim();
  try {
    return JSON.parse(output);
  } catch (error) {
    fail(`${command} did not emit one JSON object: ${error.message}`);
  }
}

function jsCommand(mode, options) {
  return ['node', [JS_BENCHMARK, mode, '--warmup', String(options.warmup), '--samples', String(options.samples)]];
}

function rustCommand(mode, options) {
  return [
    'cargo',
    [
      'run', '--quiet', '--release', '--locked', '--manifest-path', RUST_MANIFEST,
      '--example', RUST_EXAMPLE, '--', mode,
      '--warmup', String(options.warmup), '--samples', String(options.samples)
    ]
  ];
}

function runEngine(command, commandArgs) {
  return executeJson(command, commandArgs);
}

function assertField(actual, expected, path) {
  if (actual !== expected) fail(`${path} differs (expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)})`);
}

function assertIntegrity(left, right, leftName, rightName) {
  for (const field of ['benchmarkFormat', 'verifiedBeforeTiming']) {
    assertField(left[field], right[field], `${leftName}.${field}/${rightName}.${field}`);
  }
  for (const field of ['path', 'sha256', 'caseCount']) {
    assertField(left.fixture?.[field], right.fixture?.[field], `${leftName}.fixture.${field}/${rightName}.fixture.${field}`);
  }
  for (const field of ['actionChecksum', 'perftChecksum', 'rootActionCount', 'childActionCount', 'expectedLeafTotal', 'workResult']) {
    assertField(left.integrity?.[field], right.integrity?.[field], `${leftName}.integrity.${field}/${rightName}.integrity.${field}`);
  }
}

function assertMeasurement(output, verified, label, expectedSamples, expectedWarmup) {
  assertIntegrity(output, verified, label, `${label}-verification`);
  const measurement = output.measurement;
  if (!measurement) fail(`${label} did not emit a measurement`);
  assertField(measurement.sampleCount, expectedSamples, `${label}.measurement.sampleCount`);
  assertField(measurement.warmupPasses, expectedWarmup, `${label}.measurement.warmupPasses`);
  assertField(measurement.workResult, verified.integrity.workResult, `${label}.measurement.workResult`);
  if (!Array.isArray(measurement.sampleMilliseconds) || measurement.sampleMilliseconds.length !== expectedSamples
    || measurement.sampleMilliseconds.some((sample) => !Number.isFinite(sample) || sample < 0)) {
    fail(`${label}.measurement.sampleMilliseconds must have ${expectedSamples} non-negative finite values`);
  }
}

function median(samples) {
  const sorted = [...samples].sort((left, right) => left - right);
  const center = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[center] : (sorted[center - 1] + sorted[center]) / 2;
}

function rustcVersion() {
  return execute('rustc', ['--version']).trim();
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  // This is the pre-timing, cross-language handshake. A mismatch exits before
  // either --measure process is started.
  const [jsVerifyCommand, jsVerifyArguments] = jsCommand('--verify', options);
  const [rustVerifyCommand, rustVerifyArguments] = rustCommand('--verify', options);
  const jsVerification = runEngine(jsVerifyCommand, jsVerifyArguments);
  const rustVerification = runEngine(rustVerifyCommand, rustVerifyArguments);
  assertIntegrity(jsVerification, rustVerification, 'javascript', 'rust');
  assertField(
    rustVerification.integrity?.compactTraceVerified,
    true,
    'rust.integrity.compactTraceVerified'
  );

  const [jsMeasureCommand, jsMeasureArguments] = jsCommand('--measure', options);
  const [rustMeasureCommand, rustMeasureArguments] = rustCommand('--measure', options);
  const js = runEngine(jsMeasureCommand, jsMeasureArguments);
  const rust = runEngine(rustMeasureCommand, rustMeasureArguments);
  assertMeasurement(js, jsVerification, 'javascript', options.samples, options.warmup);
  assertMeasurement(rust, rustVerification, 'rust', options.samples, options.warmup);
  assertField(rust.integrity?.compactTraceVerified, true, 'rust measurement.integrity.compactTraceVerified');
  assertIntegrity(js, rust, 'javascript measurement', 'rust measurement');

  const jsMedianMilliseconds = median(js.measurement.sampleMilliseconds);
  const rustMedianMilliseconds = median(rust.measurement.sampleMilliseconds);
  if (rustMedianMilliseconds === 0) fail('Rust median is zero; increase workload or timer resolution');
  const speedup = jsMedianMilliseconds / rustMedianMilliseconds;
  const result = {
    benchmarkFormat: js.benchmarkFormat,
    mode: options.smoke ? 'smoke' : 'benchmark',
    workload: {
      fixture: js.fixture,
      verifiedBeforeTiming: true,
      description: 'one pass over frozen roots: legal actions, checked apply for each root action, child legal-action codes, and frozen-depth scalar perft'
    },
    correctness: js.integrity,
    environment: {
      os: process.platform,
      arch: process.arch,
      node: js.environment.node,
      rustc: rustcVersion(),
      javascriptEngine: js.engine,
      rustEngine: rust.engine,
      javascriptRuntime: js.environment,
      rustRuntime: rust.environment
    },
    measurements: {
      warmupPasses: options.warmup,
      sampleCount: options.samples,
      javascript: {
        sampleMilliseconds: js.measurement.sampleMilliseconds,
        medianMilliseconds: jsMedianMilliseconds,
        passesPerSecond: 1_000 / jsMedianMilliseconds
      },
      rust: {
        sampleMilliseconds: rust.measurement.sampleMilliseconds,
        medianMilliseconds: rustMedianMilliseconds,
        passesPerSecond: 1_000 / rustMedianMilliseconds
      },
      rustSpeedupOverJavascript: speedup
    },
    threshold: options.minimumSpeedup === null
      ? { enforced: false, note: 'No machine-specific performance threshold is enforced by default.' }
      : { enforced: true, minimumRustSpeedupOverJavascript: options.minimumSpeedup, passed: speedup >= options.minimumSpeedup }
  };
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (options.minimumSpeedup !== null && speedup < options.minimumSpeedup) {
    process.exitCode = 1;
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
