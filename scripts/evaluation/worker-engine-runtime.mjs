/**
 * Private subprocess runtime for hard-deadline engine isolation.
 *
 * Candidate values never cross IPC directly. Descriptors, decisions, and
 * errors are reduced to bounded primitive schemas before the private channel
 * is used.
 */
import { Buffer } from 'node:buffer';
import { createHash } from 'node:crypto';
import dgram from 'node:dgram';
import dns from 'node:dns';
import fs, { closeSync, fstatSync, openSync, readSync } from 'node:fs';
import fsPromises from 'node:fs/promises';
import http from 'node:http';
import http2 from 'node:http2';
import https from 'node:https';
import inspector from 'node:inspector';
import moduleBuiltin, { registerHooks, syncBuiltinESMExports } from 'node:module';
import net from 'node:net';
import { resolve as resolvePath } from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import tls from 'node:tls';
import { fileURLToPath as runtimeFileURLToPath, pathToFileURL } from 'node:url';

// Capture every intrinsic used at the candidate boundary before importing any
// candidate code. Candidate monkeypatches then affect only the candidate.
const objectCreate = Object.create;
const objectDefineProperty = Object.defineProperty;
const objectFreeze = Object.freeze;
const objectGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
const objectGetOwnPropertyDescriptors = Object.getOwnPropertyDescriptors;
const objectGetPrototypeOf = Object.getPrototypeOf;
const objectHasOwn = Object.hasOwn;
const objectKeys = Object.keys;
const objectValues = Object.values;
const plainObjectPrototype = Object.prototype;
const reflectApply = Reflect.apply;
const reflectOwnKeys = Reflect.ownKeys;
const numberIsFinite = Number.isFinite;
const numberIsSafeInteger = Number.isSafeInteger;
const jsonStringify = JSON.stringify;
const bufferByteLength = Buffer.byteLength.bind(Buffer);
const stringSlice = Function.prototype.call.bind(String.prototype.slice);
const stringStartsWith = Function.prototype.call.bind(String.prototype.startsWith);
const regexpTest = Function.prototype.call.bind(RegExp.prototype.test);
const promiseThen = Function.prototype.call.bind(Promise.prototype.then);
const setAdd = Function.prototype.call.bind(Set.prototype.add);
const setHas = Function.prototype.call.bind(Set.prototype.has);
const runtimeNow = performance.now.bind(performance);
const runtimeExit = process.exit.bind(process);
const runtimeSend = typeof process.send === 'function' ? process.send.bind(process) : null;
const RuntimeError = Error;
const RuntimeTypeError = TypeError;
const NativeSet = Set;
const runtimeRegisterHooks = registerHooks;
const runtimePathToFileURL = pathToFileURL;
const runtimeFileUrlToPath = runtimeFileURLToPath;
const runtimeResolvePath = resolvePath;
const runtimeCwd = process.cwd();
const runtimeFsRealpathSync = fs.realpathSync;
const runtimeFsReadFile = fs.readFile;
const runtimeFsReadFileSync = fs.readFileSync;
const runtimeFsCreateReadStream = fs.createReadStream;
const runtimeFsOpen = fs.open;
const runtimeFsOpenSync = fs.openSync;
const runtimeFsPromisesReadFile = fsPromises.readFile;
const runtimeFsPromisesOpen = fsPromises.open;
const runtimeCreateHash = createHash;
const runtimeOpenSync = openSync;
const runtimeReadSync = readSync;
const runtimeFstatSync = fstatSync;
const runtimeCloseSync = closeSync;

const MAX_DESCRIPTOR_STRING_CODE_UNITS = 256;
const MAX_ERROR_NAME_CODE_UNITS = 128;
const MAX_ERROR_MESSAGE_CODE_UNITS = 1_024;
const MAX_ERROR_CODE_CODE_UNITS = 128;
const MAX_ERROR_STACK_CODE_UNITS = 2_048;
const MAX_OUTBOUND_FRAME_BYTES = 16_384;
const WALL_PATTERN = /^[HV]-\d+-\d+$/;
const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]/;

function keySet(keys) {
  const result = new NativeSet();
  for (let index = 0; index < keys.length; index += 1) {
    setAdd(result, keys[index]);
  }
  return result;
}

const DESCRIPTOR_KEY_NAMES = objectFreeze([
  'id',
  'version',
  'sourceCommit',
  'baselineTrustRoot',
  'capabilities',
  'createSession'
]);
const DESCRIPTOR_REQUIRED_KEYS = objectFreeze(['id', 'version', 'createSession']);
const TRUST_ROOT_KEYS = [
  'baselineId',
  'baselineVersion',
  'sourceCommit',
  'manifestSha256',
  'gameLogicSha256',
  'aiSha256',
  'orchestrationSha256',
  'originalIndexSha256'
];
const TRUST_ROOT_KEY_SET = keySet(TRUST_ROOT_KEYS);
const CAPABILITY_KEY_NAMES = objectFreeze(['nodeBudget', 'deadline', 'deterministicClock']);
const ENVELOPE_KEYS = keySet(['action', 'stats']);
const ENVELOPE_REQUIRED_KEYS = objectFreeze(['action']);
const ACTION_PAWN_KEYS = keySet(['kind', 'to']);
const ACTION_PAWN_REQUIRED_KEYS = objectFreeze(['kind', 'to']);
const ACTION_WALL_KEYS = keySet(['kind', 'wall']);
const ACTION_WALL_REQUIRED_KEYS = objectFreeze(['kind', 'wall']);
const COORDINATE_KEYS = keySet(['r', 'c']);
const COORDINATE_REQUIRED_KEYS = objectFreeze(['r', 'c']);
const STATS_KEY_NAMES = objectFreeze(['elapsedMs', 'nodes', 'depth', 'antiStallReplaced']);
const ACTION_UNION_KEYS = keySet(['kind', 'to', 'wall']);
const DECISION_UNION_KEYS = keySet(['action', 'stats', 'kind', 'to', 'wall']);
const ARTIFACT_READ_BUFFER_BYTES = 1_048_576;
const NETWORK_BUILTINS = keySet([
  'dgram',
  'dns',
  'dns/promises',
  'http',
  'http2',
  'https',
  'inspector',
  'net',
  'tls',
  'node:dgram',
  'node:dns',
  'node:dns/promises',
  'node:http',
  'node:http2',
  'node:https',
  'node:inspector',
  'node:net',
  'node:tls'
]);
const SAFE_CANDIDATE_BUILTIN_URLS = keySet([
  'node:assert',
  'node:assert/strict',
  'node:buffer',
  'node:console',
  'node:crypto',
  'node:events',
  'node:fs',
  'node:fs/promises',
  'node:path',
  'node:path/posix',
  'node:path/win32',
  'node:perf_hooks',
  'node:process',
  'node:stream',
  'node:stream/consumers',
  'node:stream/promises',
  'node:stream/web',
  'node:string_decoder',
  'node:timers',
  'node:timers/promises',
  'node:url',
  'node:util',
  'node:util/types',
  'node:zlib'
]);

let channelId = null;
let descriptor = null;
let session = null;
let initialized = false;
let operationQueue = Promise.resolve();

function runtimeRecord() {
  return objectCreate(null);
}

function accessDenied(permission) {
  const error = new RuntimeError(`Access to ${permission} is denied inside an engine subprocess`);
  error.name = 'EngineSubprocessAccessError';
  error.code = 'ERR_ACCESS_DENIED';
  error.permission = permission;
  throw error;
}

for (const [name, permission] of [
  ['kill', 'process.kill'],
  ['_kill', 'process._kill'],
  ['abort', 'process.abort'],
  ['exit', 'process.exit'],
  ['reallyExit', 'process.reallyExit'],
  ['send', 'engine IPC'],
  ['disconnect', 'engine IPC']
]) {
  if (!(name in process)) continue;
  objectDefineProperty(process, name, {
    value: () => accessDenied(permission),
    configurable: false,
    enumerable: false,
    writable: false
  });
}
syncBuiltinESMExports();

function deepFreeze(value, seen = new NativeSet()) {
  if (value === null || typeof value !== 'object' || setHas(seen, value)) return value;
  setAdd(seen, value);
  const children = objectValues(value);
  for (let index = 0; index < children.length; index += 1) {
    deepFreeze(children[index], seen);
  }
  return objectFreeze(value);
}

function runtimeSha256File(filename) {
  const hash = runtimeCreateHash('sha256');
  const buffer = Buffer.allocUnsafe(ARTIFACT_READ_BUFFER_BYTES);
  const descriptor = runtimeOpenSync(filename, 'r');
  try {
    const stat = runtimeFstatSync(descriptor);
    if (!stat.isFile()) throw new RuntimeTypeError('candidate artifact is not a regular file');
    let offset = 0;
    while (offset < stat.size) {
      const count = runtimeReadSync(
        descriptor,
        buffer,
        0,
        Math.min(buffer.length, stat.size - offset),
        offset
      );
      if (count <= 0) throw new RuntimeTypeError('candidate artifact produced a short read');
      hash.update(buffer.subarray(0, count));
      offset += count;
    }
    return hash.digest('hex');
  } finally {
    runtimeCloseSync(descriptor);
  }
}

function verifyRuntimeArtifactPolicy(policy) {
  if (policy === null || policy === undefined) return;
  if (!policy || typeof policy !== 'object'
    || !policy.manifest || typeof policy.manifest.filename !== 'string'
    || typeof policy.manifest.sha256 !== 'string'
    || !Array.isArray(policy.files) || policy.files.length === 0) {
    throw new RuntimeTypeError('invalid candidate artifact runtime policy');
  }
  if (runtimeSha256File(policy.manifest.filename) !== policy.manifest.sha256) {
    throw new RuntimeTypeError('candidate artifact manifest changed before import');
  }
  for (let index = 0; index < policy.files.length; index += 1) {
    const record = policy.files[index];
    if (!record || typeof record.filename !== 'string' || typeof record.sha256 !== 'string'
      || runtimeSha256File(record.filename) !== record.sha256) {
      throw new RuntimeTypeError('candidate artifact changed before import');
    }
  }
}

function denyBuiltinMethods(moduleValue, names) {
  for (let index = 0; index < names.length; index += 1) {
    const name = names[index];
    try {
      objectDefineProperty(moduleValue, name, {
        value: () => accessDenied(`network API ${name}`),
        configurable: false,
        enumerable: false,
        writable: false
      });
    } catch {
      // The synchronous resolution hook and global network denial remain the
      // primary boundary if a platform builtin exposes a fixed property.
    }
  }
}

function lockDownAuthenticatedCandidate(policy) {
  if (policy === null || policy === undefined) return;
  const allowedUrls = new NativeSet();
  const allowedFiles = new NativeSet();
  for (let index = 0; index < policy.files.length; index += 1) {
    setAdd(allowedFiles, policy.files[index].filename);
    setAdd(allowedUrls, runtimePathToFileURL(policy.files[index].filename).href);
  }
  runtimeRegisterHooks({
    resolve(specifier, context, nextResolve) {
      if (typeof specifier !== 'string') {
        throw new RuntimeTypeError('candidate module specifier must be a string');
      }
      if (setHas(NETWORK_BUILTINS, specifier)) {
        accessDenied(`network module ${specifier}`);
      }
      const resolved = nextResolve(specifier, context);
      if (stringStartsWith(resolved.url, 'node:')) {
        if (!setHas(SAFE_CANDIDATE_BUILTIN_URLS, resolved.url)) {
          accessDenied(`unapproved Node builtin ${resolved.url}`);
        }
        return resolved;
      }
      if (!setHas(allowedUrls, resolved.url)) {
        throw new RuntimeTypeError('candidate imported an unlisted artifact dependency');
      }
      return resolved;
    }
  });

  const authorizedFilename = (pathLike) => {
    let filename;
    try {
      filename = typeof pathLike === 'string'
        ? runtimeResolvePath(runtimeCwd, pathLike)
        : runtimeFileUrlToPath(pathLike);
      filename = runtimeFsRealpathSync(filename);
    } catch {
      accessDenied('unlisted filesystem content');
    }
    if (!setHas(allowedFiles, filename)) {
      accessDenied('unlisted filesystem content');
    }
    return filename;
  };
  const readOnlyFlags = (flags) => {
    if (flags === undefined || flags === 'r' || flags === 'rs' || flags === 'sr'
      || flags === 0) {
      return flags;
    }
    accessDenied('filesystem write-capable open flags');
  };
  objectDefineProperty(fs, 'readFileSync', {
    value: (pathLike, ...argumentsList) =>
      reflectApply(runtimeFsReadFileSync, fs, [
        authorizedFilename(pathLike),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fs, 'readFile', {
    value: (pathLike, ...argumentsList) =>
      reflectApply(runtimeFsReadFile, fs, [
        authorizedFilename(pathLike),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fs, 'openSync', {
    value: (pathLike, flags, ...argumentsList) =>
      reflectApply(runtimeFsOpenSync, fs, [
        authorizedFilename(pathLike),
        readOnlyFlags(flags),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fs, 'open', {
    value: (pathLike, flags, ...argumentsList) =>
      reflectApply(runtimeFsOpen, fs, [
        authorizedFilename(pathLike),
        readOnlyFlags(flags),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fs, 'createReadStream', {
    value: (pathLike, options = undefined) => {
      if (options && typeof options === 'object' && options.fd !== undefined) {
        accessDenied('caller-supplied filesystem descriptor');
      }
      if (options && typeof options === 'object') readOnlyFlags(options.flags);
      return reflectApply(runtimeFsCreateReadStream, fs, [
        authorizedFilename(pathLike),
        options
      ]);
    },
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fsPromises, 'readFile', {
    value: (pathLike, ...argumentsList) =>
      reflectApply(runtimeFsPromisesReadFile, fsPromises, [
        authorizedFilename(pathLike),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  objectDefineProperty(fsPromises, 'open', {
    value: (pathLike, flags, ...argumentsList) =>
      reflectApply(runtimeFsPromisesOpen, fsPromises, [
        authorizedFilename(pathLike),
        readOnlyFlags(flags),
        ...argumentsList
      ]),
    configurable: false,
    enumerable: true,
    writable: false
  });
  for (const [target, names] of [
    [fs, [
      'write',
      'writeSync',
      'writev',
      'writevSync',
      'ReadStream',
      'FileReadStream'
    ]],
    [fsPromises, ['openAsBlob']]
  ]) {
    for (let index = 0; index < names.length; index += 1) {
      const name = names[index];
      if (!(name in target)) continue;
      objectDefineProperty(target, name, {
        value: () => accessDenied(`raw filesystem API ${name}`),
        configurable: false,
        enumerable: true,
        writable: false
      });
    }
  }
  if ('openAsBlob' in fs) {
    objectDefineProperty(fs, 'openAsBlob', {
      value: () => accessDenied('raw filesystem API openAsBlob'),
      configurable: false,
      enumerable: true,
      writable: false
    });
  }

  denyBuiltinMethods(net, ['connect', 'createConnection', 'createServer', 'Socket', 'Server']);
  denyBuiltinMethods(http, ['get', 'request', 'createServer']);
  denyBuiltinMethods(https, ['get', 'request', 'createServer']);
  denyBuiltinMethods(http2, ['connect', 'createServer', 'createSecureServer']);
  denyBuiltinMethods(tls, ['connect', 'createServer', 'TLSSocket']);
  denyBuiltinMethods(dgram, ['createSocket']);
  denyBuiltinMethods(dns, [
    'lookup',
    'lookupService',
    'resolve',
    'resolve4',
    'resolve6',
    'resolveAny',
    'resolveCaa',
    'resolveCname',
    'resolveMx',
    'resolveNaptr',
    'resolveNs',
    'resolvePtr',
    'resolveSoa',
    'resolveSrv',
    'resolveTxt',
    'reverse',
    'Resolver'
  ]);
  denyBuiltinMethods(inspector, ['open']);
  denyBuiltinMethods(moduleBuiltin, ['createRequire', 'register', 'registerHooks']);
  for (const name of ['binding', '_linkedBinding', 'getBuiltinModule']) {
    if (!(name in process)) continue;
    try {
      objectDefineProperty(process, name, {
        value: () => accessDenied(`process.${name}`),
        configurable: false,
        enumerable: false,
        writable: false
      });
    } catch {
      throw new RuntimeTypeError(`could not disable process.${name}`);
    }
  }
  objectDefineProperty(process, 'chdir', {
    value: () => accessDenied('process.chdir'),
    configurable: false,
    enumerable: false,
    writable: false
  });

  for (const name of ['fetch', 'WebSocket', 'EventSource']) {
    if (!(name in globalThis)) continue;
    try {
      objectDefineProperty(globalThis, name, {
        value: () => accessDenied(`global network API ${name}`),
        configurable: false,
        enumerable: false,
        writable: false
      });
    } catch {
      // Fail below if the platform prevents a required network global lock.
      throw new RuntimeTypeError(`could not disable global network API ${name}`);
    }
  }

  const environmentKeys = objectKeys(process.env);
  for (let index = 0; index < environmentKeys.length; index += 1) {
    delete process.env[environmentKeys[index]];
  }
  process.env.LANG = 'C';
  process.env.LC_ALL = 'C';
  process.env.TZ = 'UTC';
  syncBuiltinESMExports();
}

function ownDataRecord(value, label, allowedKeys, requiredKeys = null) {
  if (value === null || typeof value !== 'object') {
    throw new RuntimeTypeError(`${label} must be a plain data object`);
  }
  let prototype;
  let descriptors;
  let keys;
  try {
    prototype = objectGetPrototypeOf(value);
    descriptors = objectGetOwnPropertyDescriptors(value);
    keys = reflectOwnKeys(descriptors);
  } catch {
    throw new RuntimeTypeError(`${label} could not be inspected safely`);
  }
  if (prototype !== plainObjectPrototype && prototype !== null) {
    throw new RuntimeTypeError(`${label} must be a plain data object`);
  }
  const record = runtimeRecord();
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index];
    if (typeof key !== 'string' || !setHas(allowedKeys, key)) {
      throw new RuntimeTypeError(`${label} contains an unsupported property`);
    }
    const descriptorRecord = descriptors[key];
    if (!descriptorRecord || objectHasOwn(descriptorRecord, 'get')
      || objectHasOwn(descriptorRecord, 'set')
      || !objectHasOwn(descriptorRecord, 'value')) {
      throw new RuntimeTypeError(`${label}.${key} must be an own data property`);
    }
    record[key] = descriptorRecord.value;
  }
  if (requiredKeys) {
    for (let index = 0; index < requiredKeys.length; index += 1) {
      if (!objectHasOwn(record, requiredKeys[index])) {
        throw new RuntimeTypeError(`${label} is missing ${requiredKeys[index]}`);
      }
    }
  }
  return record;
}

function selectedOwnDataRecord(value, label, selectedKeys, requiredKeys = null) {
  if (value === null || typeof value !== 'object') {
    throw new RuntimeTypeError(`${label} must be a plain data object`);
  }
  let prototype;
  try {
    prototype = objectGetPrototypeOf(value);
  } catch {
    throw new RuntimeTypeError(`${label} could not be inspected safely`);
  }
  if (prototype !== plainObjectPrototype && prototype !== null) {
    throw new RuntimeTypeError(`${label} must be a plain data object`);
  }
  const record = runtimeRecord();
  for (let index = 0; index < selectedKeys.length; index += 1) {
    const key = selectedKeys[index];
    let property;
    try {
      property = objectGetOwnPropertyDescriptor(value, key);
    } catch {
      throw new RuntimeTypeError(`${label}.${key} could not be inspected safely`);
    }
    if (!property) continue;
    if (objectHasOwn(property, 'get') || objectHasOwn(property, 'set')
      || !objectHasOwn(property, 'value')) {
      throw new RuntimeTypeError(`${label}.${key} must be an own data property`);
    }
    record[key] = property.value;
  }
  if (requiredKeys) {
    for (let index = 0; index < requiredKeys.length; index += 1) {
      if (!objectHasOwn(record, requiredKeys[index])) {
        throw new RuntimeTypeError(`${label} is missing ${requiredKeys[index]}`);
      }
    }
  }
  return record;
}

function boundedDescriptorString(value, label, { optional = false } = {}) {
  if (optional && (value === null || value === undefined)) return null;
  if (typeof value !== 'string' || value.length === 0
    || value.length > MAX_DESCRIPTOR_STRING_CODE_UNITS
    || regexpTest(CONTROL_CHARACTER_PATTERN, value)) {
    throw new RuntimeTypeError(
      `${label} must be a nonempty bounded string without control characters`
    );
  }
  return value;
}

function normalizeTrustRoot(value) {
  if (value === null || value === undefined) return null;
  const raw = ownDataRecord(
    value,
    'descriptor.baselineTrustRoot',
    TRUST_ROOT_KEY_SET,
    TRUST_ROOT_KEYS
  );
  const normalized = runtimeRecord();
  for (let index = 0; index < TRUST_ROOT_KEYS.length; index += 1) {
    const key = TRUST_ROOT_KEYS[index];
    normalized[key] = boundedDescriptorString(
      raw[key],
      `descriptor.baselineTrustRoot.${key}`
    );
  }
  return objectFreeze(normalized);
}

function normalizeCapabilities(value) {
  if (value === null || value === undefined) return objectFreeze(runtimeRecord());
  const raw = selectedOwnDataRecord(
    value,
    'descriptor.capabilities',
    CAPABILITY_KEY_NAMES
  );
  const normalized = runtimeRecord();
  const keys = reflectOwnKeys(raw);
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index];
    if (typeof raw[key] !== 'boolean') {
      throw new RuntimeTypeError(`descriptor.capabilities.${key} must be boolean`);
    }
    normalized[key] = raw[key];
  }
  return objectFreeze(normalized);
}

function normalizeDescriptor(value) {
  const raw = selectedOwnDataRecord(
    value,
    'engine descriptor',
    DESCRIPTOR_KEY_NAMES,
    DESCRIPTOR_REQUIRED_KEYS
  );
  if (typeof raw.createSession !== 'function') {
    throw new RuntimeTypeError('engine descriptor createSession must be a function');
  }
  const metadata = runtimeRecord();
  metadata.id = boundedDescriptorString(raw.id, 'descriptor.id');
  metadata.version = boundedDescriptorString(raw.version, 'descriptor.version');
  metadata.sourceCommit = boundedDescriptorString(
    raw.sourceCommit,
    'descriptor.sourceCommit',
    { optional: true }
  );
  metadata.baselineTrustRoot = normalizeTrustRoot(raw.baselineTrustRoot);
  metadata.capabilities = normalizeCapabilities(raw.capabilities);
  const normalized = runtimeRecord();
  normalized.target = value;
  normalized.createSession = raw.createSession;
  normalized.metadata = objectFreeze(metadata);
  return objectFreeze(normalized);
}

function ownDataExport(moduleNamespace, key) {
  let property;
  try {
    property = objectGetOwnPropertyDescriptor(moduleNamespace, key);
  } catch {
    throw new RuntimeTypeError(`engine module export ${key} could not be inspected safely`);
  }
  return property && objectHasOwn(property, 'value') ? property.value : undefined;
}

async function loadDescriptor(moduleUrl, loadMode) {
  const imported = await import(moduleUrl);
  let loaded;
  if (loadMode?.kind === 'named-factory') {
    const factory = ownDataExport(imported, loadMode.exportName);
    if (typeof factory !== 'function') {
      throw new RuntimeTypeError(`missing engine factory ${loadMode.exportName}`);
    }
    loaded = await reflectApply(factory, undefined, []);
  } else {
    const factory = ownDataExport(imported, 'createEngineAdapter');
    loaded = typeof factory === 'function'
      ? await reflectApply(factory, undefined, [])
      : ownDataExport(imported, 'default');
  }
  return normalizeDescriptor(loaded);
}

function getSessionMethod(target, name, engineId) {
  let property;
  try {
    const prototype = objectGetPrototypeOf(target);
    if (prototype !== plainObjectPrototype && prototype !== null) {
      throw new RuntimeTypeError(`${engineId} session must be a plain object`);
    }
    property = objectGetOwnPropertyDescriptor(target, name);
  } catch {
    throw new RuntimeTypeError(`${engineId} session.${name} could not be inspected safely`);
  }
  if (!property || !objectHasOwn(property, 'value') || typeof property.value !== 'function') {
    throw new RuntimeTypeError(`${engineId} session.${name} must be an own data function`);
  }
  return property.value;
}

function normalizeSession(value, engineId) {
  if (value === null || typeof value !== 'object') {
    throw new RuntimeTypeError(`${engineId} createSession() returned an invalid session`);
  }
  const normalized = runtimeRecord();
  normalized.target = value;
  normalized.selectAction = getSessionMethod(value, 'selectAction', engineId);
  normalized.observe = getSessionMethod(value, 'observe', engineId);
  normalized.close = getSessionMethod(value, 'close', engineId);
  return objectFreeze(normalized);
}

function normalizeCoordinate(value, label) {
  const raw = ownDataRecord(value, label, COORDINATE_KEYS, COORDINATE_REQUIRED_KEYS);
  if (!numberIsSafeInteger(raw.r) || !numberIsSafeInteger(raw.c)) {
    throw new RuntimeTypeError(`${label} must contain safe-integer r/c fields`);
  }
  const normalized = runtimeRecord();
  normalized.r = raw.r;
  normalized.c = raw.c;
  return objectFreeze(normalized);
}

function normalizeAction(value) {
  if (value === null) return null;
  const initial = ownDataRecord(
    value,
    'decision action',
    ACTION_UNION_KEYS
  );
  if (initial.kind === 'pawn') {
    const raw = ownDataRecord(
      value,
      'pawn action',
      ACTION_PAWN_KEYS,
      ACTION_PAWN_REQUIRED_KEYS
    );
    const normalized = runtimeRecord();
    normalized.kind = 'pawn';
    normalized.to = normalizeCoordinate(raw.to, 'pawn action.to');
    return objectFreeze(normalized);
  }
  if (initial.kind === 'wall') {
    const raw = ownDataRecord(
      value,
      'wall action',
      ACTION_WALL_KEYS,
      ACTION_WALL_REQUIRED_KEYS
    );
    if (typeof raw.wall !== 'string' || raw.wall.length > 32
      || !regexpTest(WALL_PATTERN, raw.wall)) {
      throw new RuntimeTypeError('wall action.wall must be a bounded canonical wall string');
    }
    const normalized = runtimeRecord();
    normalized.kind = 'wall';
    normalized.wall = raw.wall;
    return objectFreeze(normalized);
  }
  throw new RuntimeTypeError('decision action.kind must be pawn or wall');
}

function normalizeStats(value) {
  if (value === null || value === undefined) return objectFreeze(runtimeRecord());
  const raw = selectedOwnDataRecord(value, 'decision stats', STATS_KEY_NAMES);
  const normalized = runtimeRecord();
  const keys = reflectOwnKeys(raw);
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index];
    const field = raw[key];
    if (field === null || field === undefined) continue;
    if (key === 'antiStallReplaced') {
      if (typeof field !== 'boolean') {
        throw new RuntimeTypeError('decision stats.antiStallReplaced must be boolean');
      }
    } else if (!numberIsFinite(field) || field < 0
      || (key === 'nodes' && !numberIsSafeInteger(field))) {
      throw new RuntimeTypeError(`decision stats.${key} must be a finite nonnegative scalar`);
    }
    normalized[key] = field;
  }
  return objectFreeze(normalized);
}

function normalizeDecision(value) {
  if (value === null) return null;
  if (value === undefined || typeof value !== 'object') {
    throw new RuntimeTypeError('decision must be null, an action, or an action envelope');
  }
  const inspected = ownDataRecord(
    value,
    'decision',
    DECISION_UNION_KEYS
  );
  if (!objectHasOwn(inspected, 'action')) return normalizeAction(value);
  const envelope = ownDataRecord(
    value,
    'decision envelope',
    ENVELOPE_KEYS,
    ENVELOPE_REQUIRED_KEYS
  );
  const normalized = runtimeRecord();
  normalized.action = normalizeAction(envelope.action);
  normalized.stats = normalizeStats(envelope.stats);
  return objectFreeze(normalized);
}

function safeErrorString(value, fallback, maximumCodeUnits) {
  return typeof value === 'string'
    ? stringSlice(value, 0, maximumCodeUnits)
    : fallback;
}

function ownErrorField(error, key) {
  try {
    const property = objectGetOwnPropertyDescriptor(error, key);
    return property && objectHasOwn(property, 'value') ? property.value : undefined;
  } catch {
    return undefined;
  }
}

function serializeError(error) {
  try {
    const record = runtimeRecord();
    if (typeof error === 'string') {
      record.name = 'Error';
      record.message = safeErrorString(error, 'engine subprocess failed',
        MAX_ERROR_MESSAGE_CODE_UNITS);
      record.code = null;
      record.stack = null;
      return objectFreeze(record);
    }
    if (error === null || (typeof error !== 'object' && typeof error !== 'function')) {
      record.name = 'Error';
      record.message = 'engine subprocess threw a non-error value';
      record.code = null;
      record.stack = null;
      return objectFreeze(record);
    }
    record.name = safeErrorString(
      ownErrorField(error, 'name'),
      'Error',
      MAX_ERROR_NAME_CODE_UNITS
    );
    record.message = safeErrorString(
      ownErrorField(error, 'message'),
      'engine subprocess failed',
      MAX_ERROR_MESSAGE_CODE_UNITS
    );
    record.code = safeErrorString(
      ownErrorField(error, 'code'),
      null,
      MAX_ERROR_CODE_CODE_UNITS
    );
    record.stack = safeErrorString(
      ownErrorField(error, 'stack'),
      null,
      MAX_ERROR_STACK_CODE_UNITS
    );
    return objectFreeze(record);
  } catch {
    const fallback = runtimeRecord();
    fallback.name = 'Error';
    fallback.message = 'engine subprocess error could not be inspected safely';
    fallback.code = null;
    fallback.stack = null;
    return objectFreeze(fallback);
  }
}

function sendFrame(frame) {
  if (!runtimeSend) runtimeExit(1);
  try {
    const encoded = jsonStringify(frame);
    if (typeof encoded !== 'string'
      || bufferByteLength(encoded, 'utf8') > MAX_OUTBOUND_FRAME_BYTES) {
      runtimeExit(1);
      return false;
    }
    runtimeSend(frame, undefined, undefined, (error) => {
      if (error) runtimeExit(1);
    });
    return true;
  } catch {
    runtimeExit(1);
    return false;
  }
}

function baseFrame(type) {
  const frame = runtimeRecord();
  frame.type = type;
  frame.channelId = channelId;
  return frame;
}

function sendReady(metadata) {
  const frame = baseFrame('ready');
  frame.descriptor = metadata;
  return sendFrame(objectFreeze(frame));
}

function sendResponse(id, value) {
  const frame = baseFrame('response');
  frame.id = id;
  frame.value = value;
  return sendFrame(objectFreeze(frame));
}

function sendFailure(type, id, error) {
  const frame = baseFrame(type);
  if (type === 'response') frame.id = id;
  frame.error = serializeError(error);
  return sendFrame(objectFreeze(frame));
}

function decisionRequest(payload) {
  const received = deepFreeze(payload);
  const budgetMs = received.limits?.wallClockBudgetMs;
  const startedAt = runtimeNow();
  const deadlineAtMs = numberIsFinite(budgetMs) ? startedAt + budgetMs : Infinity;
  return objectFreeze({
    ...received,
    clock: objectFreeze({ now: runtimeNow }),
    deadlineAtMs,
    limits: objectFreeze({
      ...received.limits,
      deadlineAtMs: numberIsFinite(deadlineAtMs) ? deadlineAtMs : null
    })
  });
}

async function initialize(message) {
  if (initialized || typeof message.channelId !== 'string' || message.channelId.length < 16) {
    throw new RuntimeTypeError('invalid or repeated engine subprocess initialization');
  }
  initialized = true;
  channelId = message.channelId;
  try {
    verifyRuntimeArtifactPolicy(message.artifactPolicy);
  } catch (error) {
    throw new RuntimeError(`candidate artifact verification failed: ${
      safeErrorString(ownErrorField(error, 'message'), 'unknown error',
        MAX_ERROR_MESSAGE_CODE_UNITS)
    }`);
  }
  try {
    lockDownAuthenticatedCandidate(message.artifactPolicy);
  } catch (error) {
    throw new RuntimeError(`candidate boundary setup failed: ${
      safeErrorString(ownErrorField(error, 'message'), 'unknown error',
        MAX_ERROR_MESSAGE_CODE_UNITS)
    }`);
  }
  try {
    descriptor = await loadDescriptor(message.moduleUrl, message.loadMode);
  } catch (error) {
    throw new RuntimeError(`candidate module import failed: ${
      safeErrorString(ownErrorField(error, 'message'), 'unknown error',
        MAX_ERROR_MESSAGE_CODE_UNITS)
    }`);
  }
  if (message.mode === 'probe') {
    sendReady(descriptor.metadata);
    return;
  }
  if (message.mode !== 'session') {
    throw new RuntimeTypeError('unsupported engine subprocess mode');
  }
  const rawSession = await reflectApply(
    descriptor.createSession,
    descriptor.target,
    [deepFreeze(message.context)]
  );
  session = normalizeSession(rawSession, descriptor.metadata.id);
  sendReady(descriptor.metadata);
}

async function rpc(message) {
  if (!initialized || message.channelId !== channelId || !session
    || !numberIsSafeInteger(message.id)) {
    throw new RuntimeTypeError('invalid engine subprocess RPC channel');
  }
  if (message.method === 'selectAction') {
    const value = await reflectApply(
      session.selectAction,
      session.target,
      [decisionRequest(message.payload)]
    );
    sendResponse(message.id, normalizeDecision(value));
    return;
  }
  if (message.method === 'observe') {
    await reflectApply(session.observe, session.target, [deepFreeze(message.payload)]);
    // Observer return values are intentionally discarded rather than crossing
    // IPC as candidate-controlled data.
    sendResponse(message.id, null);
    return;
  }
  throw new RuntimeTypeError('unsupported subprocess request');
}

function ownMessageField(message, key) {
  if (message === null || typeof message !== 'object') return undefined;
  try {
    const property = objectGetOwnPropertyDescriptor(message, key);
    return property && objectHasOwn(property, 'value') ? property.value : undefined;
  } catch {
    return undefined;
  }
}

process.on('message', (message) => {
  operationQueue = promiseThen(operationQueue, async () => {
    let type;
    let id;
    try {
      type = ownMessageField(message, 'type');
      id = ownMessageField(message, 'id');
      if (type === 'initialize') {
        const normalized = runtimeRecord();
        normalized.type = type;
        normalized.channelId = ownMessageField(message, 'channelId');
        normalized.mode = ownMessageField(message, 'mode');
        normalized.moduleUrl = ownMessageField(message, 'moduleUrl');
        normalized.loadMode = ownMessageField(message, 'loadMode');
        normalized.artifactPolicy = ownMessageField(message, 'artifactPolicy');
        normalized.context = ownMessageField(message, 'context');
        await initialize(normalized);
      } else if (type === 'rpc') {
        const normalized = runtimeRecord();
        normalized.type = type;
        normalized.channelId = ownMessageField(message, 'channelId');
        normalized.id = id;
        normalized.method = ownMessageField(message, 'method');
        normalized.payload = ownMessageField(message, 'payload');
        await rpc(normalized);
      }
      else throw new RuntimeTypeError('unsupported IPC message');
    } catch (error) {
      if (type === 'rpc' && numberIsSafeInteger(id)) {
        sendFailure('response', id, error);
      } else {
        sendFailure('fatal', null, error);
      }
    }
  });
});

process.on('disconnect', () => runtimeExit(0));
