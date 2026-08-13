/**
 * Persistence and identity for the Quoridor Zero site.
 *
 * Storage lives on the shared NFS volume, not in the container, so it survives pod
 * restarts and rescheduling. Two append-only JSONL files plus a secret:
 *
 *   <dir>/games.jsonl   one line per finished game, append-only
 *   <dir>/users.jsonl   one line per user upsert, last line for an id wins
 *   <dir>/secret        HMAC key for session tokens, generated once, mode 0600
 *
 * Append-only rather than a database: the whole dataset is small, a single server
 * process writes it, and a truncated final line can only ever lose the last record
 * instead of corrupting the file. Reads rebuild an in-memory index at boot and stay
 * in sync on write, so a request never re-reads the disk.
 *
 * IDENTITY, STATED PLAINLY. Two tiers, and the difference is recorded per user:
 *
 *   tailscale  the tainet's own identity, stamped onto the request by tailscaled
 *            after WireGuard has already authenticated the device. Unforgeable by
 *            the browser, and requires no sign-in at all. This is the default.
 *   guest    a name you typed, bound to a signed token. Retained only for a
 *            deployment reached outside the tailnet; it proves you are the same
 *            VISITOR, not who you are.
 *   google   a Google account, verified server-side against Google's public keys.
 *            This one is a real identity claim.
 *
 * Google sign-in stays dormant until a client id is configured AND the site is
 * served over HTTPS -- Google Identity Services refuses non-HTTPS origins other
 * than localhost. `googleReady()` reports which of those is missing rather than
 * failing at the browser with an opaque error.
 */

import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const RESULTS = new Set(['win', 'loss', 'draw', 'forfeit']);

export function createStore({ dir, googleClientId = '', httpsOrigin = false }) {
  mkdirSync(dir, { recursive: true });
  const gamesPath = path.join(dir, 'games.jsonl');
  const usersPath = path.join(dir, 'users.jsonl');
  const secretPath = path.join(dir, 'secret');

  if (!existsSync(secretPath)) {
    // Persisted, so tokens issued before a restart keep working.
    writeFileSync(secretPath, randomBytes(32).toString('hex'), { mode: 0o600 });
  }
  const secret = readFileSync(secretPath, 'utf8').trim();

  const users = new Map();   // id -> {id, name, verified, provider, created, lastSeen}
  const games = [];          // chronological

  const readLines = (p) => {
    if (!existsSync(p)) return [];
    return readFileSync(p, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean)
      .map((l) => { try { return JSON.parse(l); } catch { return null; } })
      .filter(Boolean);   // a half-written final line is dropped, not fatal
  };
  for (const u of readLines(usersPath)) users.set(u.id, u);
  for (const g of readLines(gamesPath)) games.push(g);

  const sign = (id) => createHmac('sha256', secret).update(id).digest('base64url');
  const tokenFor = (id) => `${Buffer.from(id).toString('base64url')}.${sign(id)}`;

  function verifyToken(token) {
    if (typeof token !== 'string' || !token.includes('.')) return null;
    const [b64, mac] = token.split('.');
    let id;
    try { id = Buffer.from(b64, 'base64url').toString('utf8'); } catch { return null; }
    const want = Buffer.from(sign(id));
    const got = Buffer.from(String(mac));
    // Constant-time compare, and length-checked first because timingSafeEqual
    // throws on a length mismatch rather than returning false.
    if (want.length !== got.length || !timingSafeEqual(want, got)) return null;
    return users.get(id) || null;
  }

  // The directory is checked before every append, not just at boot. A long-lived
  // server outlives its filesystem assumptions: if the dir is moved or cleared while
  // it runs, appendFileSync throws ENOENT, the caller treats a failed record as a
  // non-fatal blip, and results vanish while the in-memory leaderboard still shows
  // them -- so the site looks like it is persisting and is not.
  const ensureDir = () => { if (!existsSync(dir)) mkdirSync(dir, { recursive: true }); };

  function upsert(user) {
    users.set(user.id, user);
    ensureDir();
    appendFileSync(usersPath, `${JSON.stringify(user)}\n`);
    return user;
  }

  function nameTaken(name, exceptId) {
    const key = name.trim().toLowerCase();
    for (const u of users.values()) {
      if (u.id !== exceptId && u.name.trim().toLowerCase() === key) return u;
    }
    return null;
  }

  /**
   * Identity from the tailnet, which is the strongest claim available here.
   *
   * The caller has already been authenticated by WireGuard before HTTP existed, and
   * tailscaled itself stamps the login onto the proxied request, so the browser
   * cannot forge it. There is no token and nothing to sign in to -- the user simply
   * IS whoever the tailnet says they are.
   */
  function identify({ login, name }) {
    const id = `ts:${String(login).toLowerCase()}`;
    const now = new Date().toISOString();
    const prev = users.get(id);
    // The display name follows the tailnet profile, so a rename there follows here
    // rather than freezing whatever it was on first sight.
    const nice = String(name || login).trim().slice(0, 40) || login;
    if (prev && prev.name === nice) { prev.lastSeen = now; return prev; }
    return upsert({ id, name: nice, verified: true, provider: 'tailscale',
                    created: prev ? prev.created : now, lastSeen: now });
  }

  function signInGuest(rawName) {
    const name = String(rawName || '').trim().slice(0, 24);
    if (name.length < 2) throw Object.assign(new Error('name_too_short'), { code: 400 });
    if (!/^[\w .\-']+$/u.test(name)) throw Object.assign(new Error('name_invalid'), { code: 400 });
    // A guest name is first-come. Without this, two visitors picking the same name
    // would share one row on the leaderboard, which would be a silent lie about
    // whose record it is.
    const clash = nameTaken(name);
    if (clash && clash.provider === 'google') {
      throw Object.assign(new Error('name_reserved'), { code: 409 });
    }
    if (clash) throw Object.assign(new Error('name_taken'), { code: 409 });
    const now = new Date().toISOString();
    const user = upsert({ id: `guest:${randomBytes(9).toString('base64url')}`,
                          name, verified: false, provider: 'guest',
                          created: now, lastSeen: now });
    return { token: tokenFor(user.id), user: publicUser(user) };
  }

  /**
   * Verify a Google ID token against Google's published keys.
   *
   * Implemented now so that turning Google sign-in on is configuration rather than
   * code, but it stays unreachable until googleReady() passes.
   */
  async function signInGoogle(credential) {
    if (!googleClientId) throw Object.assign(new Error('google_not_configured'), { code: 501 });
    const parts = String(credential || '').split('.');
    if (parts.length !== 3) throw Object.assign(new Error('bad_credential'), { code: 400 });
    const header = JSON.parse(Buffer.from(parts[0], 'base64url').toString());
    const certs = await fetch('https://www.googleapis.com/oauth2/v3/certs').then((r) => r.json());
    const jwk = (certs.keys || []).find((k) => k.kid === header.kid);
    if (!jwk) throw Object.assign(new Error('unknown_key'), { code: 401 });
    const { createPublicKey, createVerify } = await import('node:crypto');
    const key = createPublicKey({ key: jwk, format: 'jwk' });
    const ok = createVerify('RSA-SHA256')
      .update(`${parts[0]}.${parts[1]}`)
      .verify(key, Buffer.from(parts[2], 'base64url'));
    if (!ok) throw Object.assign(new Error('bad_signature'), { code: 401 });
    const claims = JSON.parse(Buffer.from(parts[1], 'base64url').toString());
    if (claims.aud !== googleClientId) throw Object.assign(new Error('wrong_audience'), { code: 401 });
    if (!['accounts.google.com', 'https://accounts.google.com'].includes(claims.iss)) {
      throw Object.assign(new Error('wrong_issuer'), { code: 401 });
    }
    if (claims.exp * 1000 < Date.now()) throw Object.assign(new Error('expired'), { code: 401 });
    const id = `google:${claims.sub}`;
    const now = new Date().toISOString();
    const prev = users.get(id);
    const user = upsert({ id, name: (claims.name || claims.email || 'player').slice(0, 24),
                          verified: true, provider: 'google',
                          created: prev ? prev.created : now, lastSeen: now });
    return { token: tokenFor(user.id), user: publicUser(user) };
  }

  function googleReady() {
    return { enabled: Boolean(googleClientId) && httpsOrigin,
             clientId: googleClientId || null,
             missing: [!googleClientId && 'client_id', !httpsOrigin && 'https_origin']
               .filter(Boolean) };
  }

  const publicUser = (u) => u && ({ name: u.name, verified: u.verified, provider: u.provider });

  function recordGame(user, g) {
    if (!RESULTS.has(g.result)) throw Object.assign(new Error('bad_result'), { code: 400 });
    const row = {
      ts: new Date().toISOString(),
      userId: user.id, name: user.name, verified: user.verified,
      result: g.result,                       // from the PLAYER's point of view
      side: g.side === 'B' ? 'B' : 'A',
      plies: Number.isFinite(g.plies) ? Math.max(0, Math.round(g.plies)) : null,
      checkpoint: String(g.checkpoint || '').slice(0, 40) || null,
      sims: Number.isFinite(g.sims) ? g.sims : null
    };
    games.push(row);
    ensureDir();
    appendFileSync(gamesPath, `${JSON.stringify(row)}\n`);
    const u = users.get(user.id);
    if (u) upsert({ ...u, lastSeen: row.ts });
    return row;
  }

  function leaderboard(recentLimit = 10) {
    const per = new Map();
    for (const g of games) {
      const e = per.get(g.userId)
        || { name: g.name, verified: g.verified, w: 0, l: 0, d: 0, games: 0, last: null };
      // A forfeit is a loss. Counting it separately would let a player farm a
      // clean-looking record by resigning everything they were losing.
      if (g.result === 'win') e.w += 1;
      else if (g.result === 'draw') e.d += 1;
      else e.l += 1;
      e.games += 1;
      e.name = g.name;               // last name used wins, so renames follow through
      e.verified = g.verified;
      e.last = g.ts;
      per.set(g.userId, e);
    }
    const players = [...per.values()].sort((a, b) =>
      b.w - a.w || a.l - b.l || b.games - a.games || String(a.name).localeCompare(b.name));
    const totals = games.reduce((t, g) => {
      if (g.result === 'win') t.humanWins += 1;
      else if (g.result === 'draw') t.draws += 1;
      else t.botWins += 1;
      return t;
    }, { humanWins: 0, botWins: 0, draws: 0, games: games.length });
    return {
      recent: games.slice(-recentLimit).reverse(),
      players, totals,
      players_count: per.size
    };
  }

  return { identify, signInGuest, signInGoogle, googleReady, verifyToken, recordGame,
           leaderboard, publicUser,
           stats: () => ({ users: users.size, games: games.length, dir }) };
}
