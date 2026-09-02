/**
 * The roster is display data for a picker AND the routing table for which network
 * answers a move. Both halves have a way of going quietly wrong, and each of those
 * is pinned here.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { RosterError, parseRoster, pick, publicView } from '../scripts/play-roster.mjs';

const ok = {
  default: 'b',
  opponents: [
    { id: 'a', label: 'A', url: 'http://127.0.0.1:8100/', elo: 100, measuredAt: '256 sims' },
    { id: 'b', label: 'B', url: 'http://127.0.0.1:8101', note: 'hi' }
  ]
};

test('parses, trims the url, and honours the default', () => {
  const r = parseRoster(ok);
  assert.equal(r.defaultId, 'b');
  assert.equal(r.opponents[0].url, 'http://127.0.0.1:8100');  // trailing slash gone
  assert.equal(pick(r, 'a').id, 'a');
});

test('an unknown id falls back to the default rather than erroring', () => {
  // A stale id in a client's localStorage must not break the game; it should just
  // play the default network.
  assert.equal(pick(parseRoster(ok), 'nope').id, 'b');
});

test('an elo without the search it was measured under is refused', () => {
  // The same checkpoints reorder under different search settings, so an unlabelled
  // rating shown beside a bot playing some other configuration is a claim nobody
  // checked. This is the whole reason measuredAt exists.
  const bad = { opponents: [{ id: 'a', label: 'A', url: 'http://x', elo: 100 }] };
  assert.throws(() => parseRoster(bad), RosterError);
  assert.throws(() => parseRoster(bad), /measuredAt/);
});

test('rejects duplicate ids', () => {
  const dup = { opponents: [
    { id: 'a', label: 'A', url: 'http://x' },
    { id: 'a', label: 'A2', url: 'http://y' }
  ] };
  assert.throws(() => parseRoster(dup), /duplicate/);
});

test('rejects a default that names nothing', () => {
  assert.throws(() => parseRoster({ default: 'zzz', opponents: ok.opponents }), /default/);
});

test('rejects an empty roster and malformed json', () => {
  assert.throws(() => parseRoster({ opponents: [] }), RosterError);
  assert.throws(() => parseRoster('{not json'), /valid JSON/);
});

test('rejects a non-finite elo', () => {
  assert.throws(() => parseRoster({ opponents: [
    { id: 'a', label: 'A', url: 'http://x', elo: 'lots', measuredAt: '256 sims' }] }), /finite/);
});

test('the public view never leaks an internal url', () => {
  // These are loopback addresses of the inference servers. They are not secret,
  // but they are not the client's business and shipping them invites a client
  // that talks to them directly.
  const v = publicView(parseRoster(ok));
  assert.equal(JSON.stringify(v).includes('127.0.0.1'), false);
  assert.deepEqual(v.opponents.map((o) => o.id), ['a', 'b']);
  assert.equal(v.opponents[0].measuredAt, '256 sims');
});

test('no roster is a normal state, not an error', () => {
  assert.equal(publicView(null), null);
  assert.equal(pick(null, 'a'), null);
});
