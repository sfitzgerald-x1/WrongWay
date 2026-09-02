/**
 * The set of checkpoints a player can choose between, and where each one is served.
 *
 * The site used to serve exactly one network, named by --infer. A roster turns that
 * into a choice: several checkpoints, each with its own inference server, one of
 * them the default.
 *
 * The Elo carried here is DISPLAY DATA, and it is only meaningful alongside the
 * search it was measured under -- the same checkpoints reorder under different
 * search settings, so a rating measured at one configuration and shown next to a
 * bot playing another is a claim nobody checked. `measuredAt` is therefore
 * required on every entry that carries an elo, and is shown to the player.
 */
import { readFileSync } from 'node:fs';

export class RosterError extends Error {}

function requireString(v, what) {
  if (typeof v !== 'string' || !v.trim()) throw new RosterError(`${what} must be a non-empty string`);
  return v;
}

export function parseRoster(raw) {
  let doc;
  try { doc = typeof raw === 'string' ? JSON.parse(raw) : raw; }
  catch (err) { throw new RosterError(`roster is not valid JSON: ${err.message}`); }
  if (!doc || !Array.isArray(doc.opponents) || doc.opponents.length === 0) {
    throw new RosterError('roster needs a non-empty "opponents" array');
  }
  const seen = new Set();
  const opponents = doc.opponents.map((o, i) => {
    const id = requireString(o && o.id, `opponents[${i}].id`);
    if (seen.has(id)) throw new RosterError(`duplicate opponent id "${id}"`);
    seen.add(id);
    const url = requireString(o.url, `opponents[${i}].url`).replace(/\/$/, '');
    const entry = {
      id,
      label: requireString(o.label, `opponents[${i}].label`),
      url,
      note: typeof o.note === 'string' ? o.note : '',
      // require is the checkpoint the server at `url` must report. Without it a
      // restart pointed at the wrong weights serves a different opponent under
      // the same name and rating, silently.
      require: typeof o.require === 'string' ? o.require : ''
    };
    if (o.elo !== undefined && o.elo !== null) {
      if (typeof o.elo !== 'number' || !Number.isFinite(o.elo)) {
        throw new RosterError(`opponents[${i}].elo must be a finite number`);
      }
      if (!o.measuredAt || typeof o.measuredAt !== 'string') {
        throw new RosterError(
          `opponents[${i}] carries an elo but no "measuredAt": a rating without the `
          + 'search it was measured under is not interpretable');
      }
      entry.elo = o.elo;
      entry.measuredAt = o.measuredAt;
    }
    return entry;
  });
  const defaultId = doc.default || opponents[0].id;
  if (!opponents.some((o) => o.id === defaultId)) {
    throw new RosterError(`default "${defaultId}" is not one of the opponents`);
  }
  return { defaultId, opponents };
}

export function loadRoster(path) {
  if (!path) return null;
  return parseRoster(readFileSync(path, 'utf8'));
}

/** The entry to play against. Falls back to the default for an unknown id. */
export function pick(roster, id) {
  if (!roster) return null;
  return roster.opponents.find((o) => o.id === id)
      || roster.opponents.find((o) => o.id === roster.defaultId);
}

/** What the client is allowed to see: no internal URLs. */
export function publicView(roster) {
  if (!roster) return null;
  return {
    default: roster.defaultId,
    opponents: roster.opponents.map(({ id, label, note, elo, measuredAt }) => ({
      id, label, note, ...(elo === undefined ? {} : { elo, measuredAt })
    }))
  };
}
