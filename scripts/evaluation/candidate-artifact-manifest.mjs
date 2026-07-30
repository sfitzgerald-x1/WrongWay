/**
 * Canonical content-addressed candidate artifact manifests.
 *
 * An enforced candidate is a set of exact regular-file bytes rooted beside a
 * canonical manifest. Paths are lexical, relative POSIX paths and symlinks are
 * rejected. Runtime read permissions are derived from the verified file list.
 */
import { createHash } from 'node:crypto';
import {
  closeSync,
  fstatSync,
  lstatSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  realpathSync
} from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

export const CANDIDATE_ARTIFACT_MANIFEST_FORMAT =
  'wrongway-candidate-artifact-manifest-v1';

const MAX_MANIFEST_BYTES = 1_048_576;
const MAX_FILES = 4_096;
const MAX_RELATIVE_PATH_LENGTH = 512;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const READ_BUFFER_BYTES = 1_048_576;

function fail(message) {
  throw new TypeError(`candidate artifact manifest: ${message}`);
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || Object.getPrototypeOf(value) !== Object.prototype) {
    fail(`${label} must be a plain object`);
  }
  const keys = Object.keys(value);
  if (keys.length !== expected.length
    || keys.some((key, index) => key !== expected[index])) {
    fail(`${label} keys must be exactly ${expected.join(', ')} in canonical order`);
  }
}

function canonicalRelativePath(value, label) {
  if (typeof value !== 'string' || value.length === 0
    || value.length > MAX_RELATIVE_PATH_LENGTH
    || value.includes('\\')
    || value.startsWith('/')
    || value.split('/').some((part) => part === '' || part === '.' || part === '..')) {
    fail(`${label} must be a canonical relative POSIX path`);
  }
  return value;
}

function sha256Buffer(source) {
  return createHash('sha256').update(source).digest('hex');
}

export function sha256ArtifactFile(filename) {
  const hash = createHash('sha256');
  const buffer = Buffer.allocUnsafe(READ_BUFFER_BYTES);
  const descriptor = openSync(filename, 'r');
  try {
    const stat = fstatSync(descriptor);
    if (!stat.isFile()) fail(`${filename} is not a regular file`);
    let offset = 0;
    while (offset < stat.size) {
      const count = readSync(
        descriptor,
        buffer,
        0,
        Math.min(buffer.length, stat.size - offset),
        offset
      );
      if (count <= 0) fail(`short read while hashing ${filename}`);
      hash.update(buffer.subarray(0, count));
      offset += count;
    }
    return hash.digest('hex');
  } finally {
    closeSync(descriptor);
  }
}

function resolveRegularArtifact(root, relativePath) {
  const filename = resolve(root, ...relativePath.split('/'));
  const lexicalRelative = relative(root, filename);
  if (lexicalRelative.startsWith(`..${sep}`) || lexicalRelative === '..'
    || lexicalRelative.startsWith(sep)) {
    fail(`artifact path escapes manifest root: ${relativePath}`);
  }
  const stat = lstatSync(filename);
  if (!stat.isFile() || stat.isSymbolicLink()) {
    fail(`artifact must be a non-symlink regular file: ${relativePath}`);
  }
  const realRoot = realpathSync(root);
  const realFile = realpathSync(filename);
  const realRelative = relative(realRoot, realFile);
  if (realRelative.startsWith(`..${sep}`) || realRelative === '..'
    || realRelative.startsWith(sep)) {
    fail(`artifact resolves outside manifest root: ${relativePath}`);
  }
  return realFile;
}

function verifyHermeticArtifactRoot(root, manifestFilename, files) {
  const expectedFiles = new Set([
    manifestFilename,
    ...files.map(({ filename }) => filename)
  ]);
  const expectedDirectories = new Set([root]);
  for (const filename of expectedFiles) {
    let directory = dirname(filename);
    while (directory !== root) {
      const relativeDirectory = relative(root, directory);
      if (relativeDirectory.startsWith(`..${sep}`) || relativeDirectory === '..'
        || relativeDirectory.startsWith(sep)) {
        fail('hermetic release contains a path outside its root');
      }
      expectedDirectories.add(directory);
      directory = dirname(directory);
    }
  }
  const foundFiles = new Set();
  const visit = (directory) => {
    for (const name of readdirSync(directory)) {
      const filename = resolve(directory, name);
      const stat = lstatSync(filename);
      if (stat.isSymbolicLink()) {
        fail(`hermetic release may not contain symlinks: ${relative(root, filename)}`);
      }
      if (stat.isDirectory()) {
        if (!expectedDirectories.has(filename)) {
          fail(`hermetic release contains an unlisted directory: ${relative(root, filename)}`);
        }
        visit(filename);
        continue;
      }
      if (!stat.isFile()) {
        fail(`hermetic release contains a special file: ${relative(root, filename)}`);
      }
      const realFile = realpathSync(filename);
      if (!expectedFiles.has(realFile)) {
        fail(`hermetic release contains an unlisted file: ${relative(root, filename)}`);
      }
      foundFiles.add(realFile);
    }
  };
  visit(root);
  if (foundFiles.size !== expectedFiles.size
    || [...expectedFiles].some((filename) => !foundFiles.has(filename))) {
    fail('hermetic release is missing an expected file');
  }
}

function parseCanonicalManifest(source) {
  if (source.length > MAX_MANIFEST_BYTES) fail('manifest exceeds 1 MiB');
  let parsed;
  try {
    parsed = JSON.parse(source.toString('utf8'));
  } catch (error) {
    fail(`invalid JSON: ${error.message}`);
  }
  exactKeys(parsed, ['format', 'entry', 'files'], 'root');
  if (parsed.format !== CANDIDATE_ARTIFACT_MANIFEST_FORMAT) {
    fail(`format must be ${CANDIDATE_ARTIFACT_MANIFEST_FORMAT}`);
  }
  const entry = canonicalRelativePath(parsed.entry, 'entry');
  if (!Array.isArray(parsed.files) || parsed.files.length === 0
    || parsed.files.length > MAX_FILES) {
    fail(`files must contain 1..${MAX_FILES} records`);
  }
  const files = [];
  let previousPath = null;
  for (let index = 0; index < parsed.files.length; index += 1) {
    const record = parsed.files[index];
    exactKeys(record, ['path', 'sha256'], `files[${index}]`);
    const path = canonicalRelativePath(record.path, `files[${index}].path`);
    if (previousPath !== null && path <= previousPath) {
      fail('files must be strictly sorted by path with no duplicates');
    }
    if (typeof record.sha256 !== 'string' || !SHA256_PATTERN.test(record.sha256)) {
      fail(`files[${index}].sha256 must be lowercase SHA-256`);
    }
    files.push(Object.freeze({ path, sha256: record.sha256 }));
    previousPath = path;
  }
  if (!files.some((record) => record.path === entry)) {
    fail('files must include the entry module');
  }
  const normalized = Object.freeze({
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    entry,
    files: Object.freeze(files)
  });
  const canonicalSource = `${JSON.stringify(normalized, null, 2)}\n`;
  if (!source.equals(Buffer.from(canonicalSource))) {
    fail('manifest bytes are not canonical');
  }
  return Object.freeze({ normalized, canonicalSource });
}

export function loadCandidateArtifactManifest({ manifestPath, moduleUrl }) {
  if (typeof manifestPath !== 'string' || manifestPath.length === 0) {
    fail('manifestPath must be a nonempty path');
  }
  let entryFilename;
  try {
    const parsedUrl = new URL(moduleUrl);
    if (parsedUrl.protocol !== 'file:') fail('enforced candidate entry must use file:');
    entryFilename = realpathSync(fileURLToPath(parsedUrl));
  } catch (error) {
    if (error instanceof TypeError && error.message.startsWith('candidate artifact manifest:')) {
      throw error;
    }
    fail(`invalid candidate module URL: ${error.message}`);
  }
  const absoluteManifest = resolve(manifestPath);
  const manifestStat = lstatSync(absoluteManifest);
  if (!manifestStat.isFile() || manifestStat.isSymbolicLink()) {
    fail('manifest must be a non-symlink regular file');
  }
  const manifestFilename = realpathSync(absoluteManifest);
  const root = dirname(manifestFilename);
  const source = readFileSync(manifestFilename);
  const { normalized } = parseCanonicalManifest(source);
  const files = normalized.files.map((record) => {
    const filename = resolveRegularArtifact(root, record.path);
    if (sha256ArtifactFile(filename) !== record.sha256) {
      fail(`SHA-256 mismatch for ${record.path}`);
    }
    return Object.freeze({ ...record, filename });
  });
  const entryRecord = files.find((record) => record.path === normalized.entry);
  if (entryRecord.filename !== entryFilename) {
    fail('entry module path does not match manifest entry');
  }
  verifyHermeticArtifactRoot(root, manifestFilename, files);
  const manifestSha256 = sha256Buffer(source);
  const provenance = Object.freeze({
    verification: 'content-addressed-manifest-v1',
    rootPolicy: 'hermetic-release-directory-v1',
    format: CANDIDATE_ARTIFACT_MANIFEST_FORMAT,
    manifestSha256,
    entry: normalized.entry,
    files: Object.freeze(files.map(({ path, sha256 }) =>
      Object.freeze({ path, sha256 })))
  });
  return Object.freeze({
    root,
    manifestFilename,
    manifestSha256,
    canonicalSource: source.toString('utf8'),
    files,
    entryFilename,
    provenance
  });
}

export function verifyCandidateArtifactManifest(binding) {
  if (!binding || typeof binding !== 'object') fail('missing private artifact binding');
  const source = readFileSync(binding.manifestFilename);
  if (sha256Buffer(source) !== binding.manifestSha256
    || source.toString('utf8') !== binding.canonicalSource) {
    fail('manifest bytes changed after adapter creation');
  }
  for (const record of binding.files) {
    const stat = lstatSync(record.filename);
    if (!stat.isFile() || stat.isSymbolicLink()
      || sha256ArtifactFile(record.filename) !== record.sha256) {
      fail(`artifact changed after adapter creation: ${record.path}`);
    }
  }
  verifyHermeticArtifactRoot(binding.root, binding.manifestFilename, binding.files);
  return true;
}

export function candidateArtifactRuntimePolicy(binding) {
  verifyCandidateArtifactManifest(binding);
  return Object.freeze({
    root: binding.root,
    manifest: Object.freeze({
      filename: binding.manifestFilename,
      sha256: binding.manifestSha256
    }),
    files: Object.freeze(binding.files.map(({ filename, sha256 }) =>
      Object.freeze({ filename, sha256 })))
  });
}
