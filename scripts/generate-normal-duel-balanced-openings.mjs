#!/usr/bin/env node
/**
 * Seeded, balanced 4–6-ply opening-book generator for the strength protocol.
 *
 * `lcg32-v1` advances before each selection. At each ply it chooses a seeded
 * start in canonical legal-action order, prefers a seeded pawn/wall action
 * class, then scans cyclically for the first ongoing position whose two
 * wall-only shortest-path distances differ by at most one. This is a balance
 * filter, not an engine evaluation.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  applyLegalAction,
  createInitialState,
  encodeAction,
  legalActions
} from '../js/normal-duel-engine.mjs';
import { createLcg32, LCG32_ALGORITHM } from '../js/lcg32.mjs';
import {
  CANONICAL_STRENGTH_GENERATOR_VERSION,
  CANONICAL_STRENGTH_SEED,
  normalDuelConfig
} from './evaluation/normal-duel-strength-constants.mjs';

export const OPENING_BOOK_FORMAT = 'normal-duel-balanced-opening-book-v1';
export const OPENING_GENERATOR_VERSION = CANONICAL_STRENGTH_GENERATOR_VERSION;
const DEFAULT_SEED = CANONICAL_STRENGTH_SEED;
const DEFAULT_COUNT = 12;
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_BOOK_PATH = resolve(ROOT, 'tests/fixtures/normal-duel-balanced-openings-v1.json');
const DEFAULT_MANIFEST_PATH = resolve(ROOT, 'tests/fixtures/normal-duel-balanced-openings-v1.manifest.json');
const VERIFIED_BOOKS = new WeakMap();
const VERIFIED_PROVENANCE = new WeakMap();

function fail(message) {
  throw new TypeError(`${OPENING_GENERATOR_VERSION}: ${message}`);
}

function uint32(value, name) {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) fail(`${name} must be a uint32`);
  return value;
}

function positiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${name} must be a positive integer`);
  return value;
}

function sha256(source) {
  return createHash('sha256').update(source).digest('hex');
}

function deepFreeze(value, seen = new Set()) {
  if (value === null || typeof value !== 'object' || seen.has(value)) return value;
  seen.add(value);
  for (const child of Object.values(value)) deepFreeze(child, seen);
  return Object.freeze(value);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function edgeBlocked(from, to, walls) {
  const dr = to.r - from.r;
  const dc = to.c - from.c;
  if (dr === 1) return walls.has(`H-${from.r}-${from.c}`) || (from.c > 0 && walls.has(`H-${from.r}-${from.c - 1}`));
  if (dr === -1) return walls.has(`H-${to.r}-${from.c}`) || (from.c > 0 && walls.has(`H-${to.r}-${from.c - 1}`));
  if (dc === 1) return walls.has(`V-${from.r}-${from.c}`) || (from.r > 0 && walls.has(`V-${from.r - 1}-${from.c}`));
  if (dc === -1) return walls.has(`V-${from.r}-${to.c}`) || (from.r > 0 && walls.has(`V-${from.r - 1}-${to.c}`));
  return false;
}

function shortestWallOnlyDistance(config, start, goalRow, wallList) {
  const walls = new Set(wallList);
  const queue = [{ position: start, distance: 0 }];
  const visited = new Set([`${start.r},${start.c}`]);
  for (let index = 0; index < queue.length; index += 1) {
    const { position, distance } = queue[index];
    if (position.r === goalRow) return distance;
    for (const [dr, dc] of [[-1, 0], [0, -1], [0, 1], [1, 0]]) {
      const next = { r: position.r + dr, c: position.c + dc };
      const key = `${next.r},${next.c}`;
      if (next.r < 0 || next.r >= config.rows || next.c < 0 || next.c >= config.columns
        || visited.has(key) || edgeBlocked(position, next, walls)) continue;
      visited.add(key);
      queue.push({ position: next, distance: distance + 1 });
    }
  }
  return Infinity;
}

function balance(config, state) {
  const distanceA = shortestWallOnlyDistance(
    config, state.position.pawns.A, config.goalRows.A, state.position.walls
  );
  const distanceB = shortestWallOnlyDistance(
    config, state.position.pawns.B, config.goalRows.B, state.position.walls
  );
  return Object.freeze({
    distanceA,
    distanceB,
    distanceDelta: distanceA - distanceB,
    balanced: Number.isFinite(distanceA) && Number.isFinite(distanceB)
      && Math.abs(distanceA - distanceB) <= 1
  });
}

function selectBalancedAction(config, state, random) {
  const actions = legalActions(config, state);
  if (actions.length === 0) fail(`ongoing state at ply ${state.ply} has no actions`);
  const sample = random();
  const preferredKind = ((sample >>> 24) % 4 === 0) ? 'wall' : 'pawn';
  const start = sample % actions.length;
  const ordered = [];
  for (const restrictKind of [true, false]) {
    for (let offset = 0; offset < actions.length; offset += 1) {
      const action = actions[(start + offset) % actions.length];
      if (restrictKind && action.kind !== preferredKind) continue;
      ordered.push(action);
    }
  }
  const seen = new Set();
  for (const action of ordered) {
    const code = encodeAction(config, action);
    if (seen.has(code)) continue;
    seen.add(code);
    const next = applyLegalAction(config, state, action);
    const diagnostics = balance(config, next);
    if (next.outcome.kind === 'ongoing' && diagnostics.balanced) {
      return Object.freeze({ action, code, next, diagnostics });
    }
  }
  fail(`no balanced action at ply ${state.ply}`);
}

function generateOne(config, openingSeed, index) {
  const random = createLcg32(openingSeed);
  const targetPlies = 4 + (index % 3);
  let state = createInitialState(config);
  const actionCodes = [];
  const actionMix = { pawn: 0, wall: 0 };
  for (let ply = 0; ply < targetPlies; ply += 1) {
    const selected = selectBalancedAction(config, state, random);
    actionCodes.push(selected.code);
    actionMix[selected.action.kind] += 1;
    state = selected.next;
  }
  const diagnostics = balance(config, state);
  return Object.freeze({
    id: `balanced-${config.rows}x${config.columns}-${String(index + 1).padStart(3, '0')}`,
    seed: openingSeed,
    targetPlies,
    actionCodes: Object.freeze(actionCodes),
    positionKey: state.positionKey,
    diagnostics: Object.freeze({
      ...diagnostics,
      wallCount: state.position.walls.length,
      stock: Object.freeze({ A: state.position.stock.A, B: state.position.stock.B }),
      actionMix: Object.freeze(actionMix),
      sideToMove: state.position.turn
    })
  });
}

export function generateBalancedOpeningBook({
  seed = DEFAULT_SEED,
  count = DEFAULT_COUNT,
  size = 9,
  firstPlayer = 'A'
} = {}) {
  uint32(seed, 'seed');
  positiveInteger(count, 'count');
  const config = normalDuelConfig({ size, firstPlayer });
  const nextSeed = createLcg32(seed);
  const openings = [];
  const positions = new Set();
  let candidateIndex = 0;
  while (openings.length < count) {
    if (candidateIndex >= count * 100) fail('could not generate enough unique balanced openings');
    const candidate = generateOne(config, nextSeed(), candidateIndex);
    candidateIndex += 1;
    if (positions.has(candidate.positionKey)) continue;
    positions.add(candidate.positionKey);
    openings.push(Object.freeze({
      ...candidate,
      id: `balanced-${config.rows}x${config.columns}-${String(openings.length + 1).padStart(3, '0')}`
    }));
  }
  return Object.freeze({
    bookFormat: OPENING_BOOK_FORMAT,
    ruleset: config.ruleset,
    config,
    configurationSha256: sha256(canonicalJson(config)),
    generator: Object.freeze({
      version: OPENING_GENERATOR_VERSION,
      algorithm: LCG32_ALGORITHM,
      seed,
      openingCount: count,
      targetPlies: Object.freeze([4, 5, 6]),
      selection: 'advance lcg32; seeded canonical start; prefer wall iff high-byte modulo 4 is zero, otherwise pawn; cyclic first balanced ongoing candidate',
      balance: 'absolute wall-only shortest-path distance delta <= 1 after every selected ply',
      moduloBias: 'intentional'
    }),
    openings: Object.freeze(openings)
  });
}

export function openingArtifacts(options) {
  const book = generateBalancedOpeningBook(options);
  const bookText = `${JSON.stringify(book, null, 2)}\n`;
  const manifest = {
    manifestFormat: 'normal-duel-balanced-opening-manifest-v1',
    bookFormat: book.bookFormat,
    generatorVersion: book.generator.version,
    prng: book.generator.algorithm,
    seed: book.generator.seed,
    openingCount: book.openings.length,
    targetPlies: book.generator.targetPlies,
    configurationSha256: book.configurationSha256,
    sha256: sha256(bookText),
    openings: book.openings.map((opening) => ({
      id: opening.id,
      seed: opening.seed,
      targetPlies: opening.targetPlies,
      actionCodes: opening.actionCodes,
      positionKey: opening.positionKey
    }))
  };
  return Object.freeze({
    book,
    bookText,
    manifest: Object.freeze(manifest),
    manifestText: `${JSON.stringify(manifest, null, 2)}\n`
  });
}

export function verifyOpeningArtifacts(bookText, manifestText) {
  if (typeof bookText !== 'string' || typeof manifestText !== 'string') {
    fail('bookText and manifestText must be strings');
  }
  let book;
  let manifest;
  try {
    book = JSON.parse(bookText);
    manifest = JSON.parse(manifestText);
  } catch (error) {
    fail(`opening corpus is not valid JSON: ${error.message}`);
  }
  if (book?.bookFormat !== OPENING_BOOK_FORMAT
    || book?.generator?.version !== OPENING_GENERATOR_VERSION
    || book?.generator?.algorithm !== LCG32_ALGORITHM
    || !Number.isSafeInteger(book?.generator?.seed)
    || !Number.isSafeInteger(book?.generator?.openingCount)
    || book?.generator?.openingCount !== book?.openings?.length
    || book?.config?.rows !== book?.config?.columns) {
    fail('opening book metadata does not identify this generator');
  }
  const regenerated = openingArtifacts({
    seed: book.generator.seed,
    count: book.generator.openingCount,
    size: book.config.rows,
    firstPlayer: book.config.firstPlayer
  });
  if (bookText !== regenerated.bookText) {
    fail('opening book differs from exact seeded generator output');
  }
  if (manifestText !== regenerated.manifestText) {
    fail('opening manifest differs from exact seeded generator output');
  }
  if (manifest.manifestFormat !== 'normal-duel-balanced-opening-manifest-v1'
    || manifest.sha256 !== sha256(bookText)) {
    fail('opening manifest does not authenticate the exact book bytes');
  }
  const frozenBook = deepFreeze(book);
  const frozenManifest = deepFreeze(manifest);
  const provenance = deepFreeze({
      verification: 'exact-seeded-regeneration-v1',
      verified: true,
      bookSha256: sha256(bookText),
      manifestSha256: sha256(manifestText),
      manifestFormat: manifest.manifestFormat,
      bookFormat: book.bookFormat,
      generatorVersion: book.generator.version,
      generatorAlgorithm: book.generator.algorithm,
      generatorSeed: book.generator.seed,
      generatorOpeningCount: book.generator.openingCount,
      configurationSha256: book.configurationSha256
  });
  const verified = Object.freeze({
    book: frozenBook,
    manifest: frozenManifest,
    provenance
  });
  VERIFIED_BOOKS.set(frozenBook, verified);
  VERIFIED_PROVENANCE.set(provenance, verified);
  return verified;
}

export function isVerifiedOpeningCorpus(book, provenance) {
  const byBook = book && typeof book === 'object' ? VERIFIED_BOOKS.get(book) : null;
  const byProvenance = provenance && typeof provenance === 'object'
    ? VERIFIED_PROVENANCE.get(provenance)
    : null;
  return Boolean(byBook && byBook === byProvenance
    && byBook.book === book && byBook.provenance === provenance);
}

export function assertVerifiedOpeningCorpus(book, provenance) {
  if (!isVerifiedOpeningCorpus(book, provenance)) {
    fail('enforced evaluation requires the exact privately verified book and provenance objects');
  }
  return VERIFIED_BOOKS.get(book);
}

function parseArguments(argv) {
  const options = {
    seed: DEFAULT_SEED,
    count: DEFAULT_COUNT,
    size: 9,
    mode: 'stdout',
    bookPath: null,
    manifestPath: null
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--seed') options.seed = Number(argv[++index]);
    else if (argument === '--count') options.count = Number(argv[++index]);
    else if (argument === '--size') options.size = Number(argv[++index]);
    else if (argument === '--book-path') options.bookPath = argv[++index];
    else if (argument === '--manifest-path') options.manifestPath = argv[++index];
    else if (argument === '--write') options.mode = 'write';
    else if (argument === '--check') options.mode = 'check';
    else if (argument === '--stdout') options.mode = 'stdout';
    else fail(`unknown argument ${argument}`);
  }
  return options;
}

function main() {
  const options = parseArguments(process.argv.slice(2));
  const artifacts = openingArtifacts(options);
  const bookPath = options.bookPath ? resolve(options.bookPath) : DEFAULT_BOOK_PATH;
  const manifestPath = options.manifestPath ? resolve(options.manifestPath) : DEFAULT_MANIFEST_PATH;
  if (options.mode === 'stdout') {
    process.stdout.write(artifacts.bookText);
    return;
  }
  if ((options.bookPath === null) !== (options.manifestPath === null)) {
    fail('--book-path and --manifest-path must be provided together');
  }
  if (options.mode === 'write' && options.bookPath === null
    && (options.seed !== DEFAULT_SEED || options.count !== DEFAULT_COUNT || options.size !== 9)) {
    fail('custom generation requires explicit --book-path and --manifest-path');
  }
  if (options.mode === 'write') {
    writeFileSync(bookPath, artifacts.bookText);
    writeFileSync(manifestPath, artifacts.manifestText);
    return;
  }
  const actualBook = readFileSync(bookPath, 'utf8');
  const actualManifest = readFileSync(manifestPath, 'utf8');
  if (actualBook !== artifacts.bookText || actualManifest !== artifacts.manifestText) {
    fail('checked-in opening book or manifest is stale');
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
