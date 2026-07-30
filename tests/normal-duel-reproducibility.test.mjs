import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { appendFileSync, cpSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = fileURLToPath(new URL('../', import.meta.url));
const checks = Object.freeze([
  Object.freeze({
    script: 'scripts/generate-normal-duel-trajectories.mjs',
    artifact: 'tests/fixtures/normal-duel-v1-trajectories.jsonl'
  }),
  Object.freeze({
    script: 'scripts/generate-normal-duel-perft.mjs',
    artifact: 'tests/fixtures/normal-duel-perft-v1.json'
  })
]);

function runCheck(root, script) {
  return spawnSync(process.execPath, [script, '--check'], {
    cwd: root,
    encoding: 'utf8'
  });
}

for (const { script, artifact } of checks) {
  test(`${script} --check reproduces its committed artifacts`, () => {
    const result = runCheck(repositoryRoot, script);

    assert.equal(
      result.status,
      0,
      `${script} --check failed (signal: ${result.signal ?? 'none'}):\n${result.stderr}${result.stdout}`
    );
  });

  test(`${script} --check rejects a stale artifact without changing the repository`, (context) => {
    const isolatedRoot = mkdtempSync(join(tmpdir(), 'wrongway-corpus-check-'));
    context.after(() => rmSync(isolatedRoot, { recursive: true, force: true }));
    mkdirSync(join(isolatedRoot, 'tests'), { recursive: true });
    cpSync(join(repositoryRoot, 'scripts'), join(isolatedRoot, 'scripts'), { recursive: true });
    cpSync(join(repositoryRoot, 'js'), join(isolatedRoot, 'js'), { recursive: true });
    cpSync(join(repositoryRoot, 'tests', 'fixtures'), join(isolatedRoot, 'tests', 'fixtures'), { recursive: true });
    appendFileSync(join(isolatedRoot, artifact), ' ', 'utf8');

    const result = runCheck(isolatedRoot, script);
    assert.notEqual(result.status, 0, `${script} --check unexpectedly accepted a stale ${artifact}`);
    assert.match(result.stderr, /stale; regenerate from this script\./);
  });
}
