#!/usr/bin/env node
/**
 * CLI for the Stage 2 paired strength protocol.
 *
 * Candidate modules export either an engine descriptor as `default` or a
 * `createEngineAdapter()` function. See docs/normal-duel-strength-evaluation.md.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { createPinnedHardBaseline } from './evaluation/hard-baseline-46a871c7.mjs';
import {
  createPinnedHardWorkerAdapter,
  createWorkerEngineAdapter
} from './evaluation/worker-engine-proxy.mjs';
import {
  CANONICAL_STRENGTH_DEADLINE_MS,
  CANONICAL_STRENGTH_SEED,
  MINIMUM_STRENGTH_OPENING_PAIRS,
  REGRESSION_MODE,
  runEvaluation,
  STRENGTH_MODE
} from './evaluation/normal-duel-strength.mjs';
import { verifyOpeningArtifacts } from './generate-normal-duel-balanced-openings.mjs';

const DEFAULT_BOOK = 'tests/fixtures/normal-duel-balanced-openings-v1.json';

function fail(message) {
  throw new TypeError(`run-normal-duel-strength: ${message}`);
}

function integer(raw, name, minimum = 0) {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) fail(`${name} must be an integer >= ${minimum}`);
  return value;
}

function parseArguments(argv) {
  const options = {
    candidate: null,
    candidateManifest: null,
    book: DEFAULT_BOOK,
    manifest: null,
    mode: STRENGTH_MODE,
    seed: null,
    deadlineMs: CANONICAL_STRENGTH_DEADLINE_MS,
    nodeBudget: null,
    openingLimit: null,
    minimumOpeningPairs: MINIMUM_STRENGTH_OPENING_PAIRS,
    enforceGate: false
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--candidate') options.candidate = argv[++index];
    else if (argument === '--candidate-manifest') options.candidateManifest = argv[++index];
    else if (argument === '--book') options.book = argv[++index];
    else if (argument === '--manifest') options.manifest = argv[++index];
    else if (argument === '--mode') {
      const mode = argv[++index];
      options.mode = mode === 'strength' ? STRENGTH_MODE
        : mode === 'regression' ? REGRESSION_MODE
          : fail('mode must be strength or regression');
    } else if (argument === '--seed') options.seed = integer(argv[++index], 'seed');
    else if (argument === '--deadline-ms') options.deadlineMs = integer(argv[++index], 'deadline-ms', 1);
    else if (argument === '--node-budget') options.nodeBudget = integer(argv[++index], 'node-budget', 1);
    else if (argument === '--opening-limit') options.openingLimit = integer(argv[++index], 'opening-limit', 1);
    else if (argument === '--minimum-opening-pairs') {
      options.minimumOpeningPairs = integer(argv[++index], 'minimum-opening-pairs', 1);
    } else if (argument === '--enforce-gate') options.enforceGate = true;
    else fail(`unknown argument ${argument}`);
  }
  if (!options.candidate) fail('--candidate is required');
  if (options.mode === REGRESSION_MODE && options.nodeBudget === null) {
    fail('--node-budget is required in regression mode');
  }
  if (options.enforceGate && options.mode !== STRENGTH_MODE) {
    fail('--enforce-gate requires strength mode');
  }
  if (options.enforceGate && options.minimumOpeningPairs < MINIMUM_STRENGTH_OPENING_PAIRS) {
    fail(`--enforce-gate requires --minimum-opening-pairs >= ${MINIMUM_STRENGTH_OPENING_PAIRS}`);
  }
  if (options.enforceGate && options.openingLimit !== null) {
    fail('--opening-limit is not allowed with --enforce-gate');
  }
  if (options.enforceGate && options.deadlineMs !== CANONICAL_STRENGTH_DEADLINE_MS) {
    fail(`--enforce-gate requires --deadline-ms ${CANONICAL_STRENGTH_DEADLINE_MS}`);
  }
  if (options.enforceGate && options.seed !== null
    && options.seed !== CANONICAL_STRENGTH_SEED) {
    fail(`--enforce-gate requires --seed ${CANONICAL_STRENGTH_SEED}`);
  }
  if (options.enforceGate && !options.candidateManifest) {
    fail('--enforce-gate requires --candidate-manifest');
  }
  return options;
}

async function loadCandidate(filename, isolated, candidateManifest, requireCanonicalMemoryIsolation) {
  const moduleUrl = pathToFileURL(resolve(filename)).href;
  if (isolated) {
    return createWorkerEngineAdapter({
      moduleUrl,
      candidateManifestPath: candidateManifest ? resolve(candidateManifest) : null,
      requireCanonicalMemoryIsolation
    });
  }
  const module = await import(moduleUrl);
  const candidate = typeof module.createEngineAdapter === 'function'
    ? await module.createEngineAdapter()
    : module.default;
  if (!candidate) fail('candidate module must export default or createEngineAdapter()');
  return candidate;
}

function sha256(source) {
  return createHash('sha256').update(source).digest('hex');
}

function derivedManifestPath(bookPath) {
  return bookPath.endsWith('.json')
    ? `${bookPath.slice(0, -'.json'.length)}.manifest.json`
    : `${bookPath}.manifest.json`;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const bookPath = resolve(options.book);
  const bookText = readFileSync(bookPath, 'utf8');
  const manifestPath = resolve(options.manifest ?? derivedManifestPath(bookPath));
  let book;
  let corpusProvenance;
  if (existsSync(manifestPath)) {
    const verified = verifyOpeningArtifacts(bookText, readFileSync(manifestPath, 'utf8'));
    book = verified.book;
    corpusProvenance = verified.provenance;
  } else {
    if (options.manifest !== null) fail(`opening manifest does not exist: ${manifestPath}`);
    if (options.enforceGate) fail('--enforce-gate requires a matching opening manifest');
    book = JSON.parse(bookText);
    corpusProvenance = Object.freeze({
      verified: false,
      verification: 'none',
      bookSha256: sha256(bookText),
      manifestSha256: null
    });
  }
  if (options.openingLimit !== null) {
    book = {
      ...book,
      openings: book.openings.slice(0, options.openingLimit)
    };
    corpusProvenance = Object.freeze({
      ...corpusProvenance,
      verified: false,
      verification: 'verified-source-subset-non-enforced-v1',
      evaluatedOpeningCount: book.openings.length
    });
  }
  if (options.enforceGate && book.openings.length < options.minimumOpeningPairs) {
    fail(`--enforce-gate requires at least ${options.minimumOpeningPairs} evaluated openings`);
  }
  const useSubprocesses = options.mode === STRENGTH_MODE;
  const candidate = await loadCandidate(
    options.candidate,
    useSubprocesses,
    options.candidateManifest,
    options.enforceGate
  );
  const baseline = useSubprocesses
    ? await createPinnedHardWorkerAdapter({
      requireCanonicalMemoryIsolation: options.enforceGate
    })
    : createPinnedHardBaseline();
  const report = await runEvaluation({
    contender: candidate,
    baseline,
    book,
    mode: options.mode,
    seed: options.seed ?? book.generator.seed,
    perMoveDeadlineMs: options.deadlineMs,
    nodeBudget: options.nodeBudget,
    minimumOpeningPairs: options.minimumOpeningPairs,
    enforceGate: options.enforceGate,
    corpusProvenance
  });
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  if (options.enforceGate && !report.summary.gate.passed) process.exitCode = 1;
}

try {
  await main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exitCode = 1;
}
