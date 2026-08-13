#!/usr/bin/env node
/**
 * Browser-verify the /app neural opponent, including the no-fallback behaviour.
 *
 *     node scripts/check-app-neural.mjs            # needs the play server on :8177
 *
 * WHY A BROWSER. The claim under test is a behavioural one about a React effect:
 * choosing the neural difficulty must never let the built-in heuristic bot play, and
 * when the network is unreachable the bot must refuse to move rather than substitute.
 * Every cheaper check I ran -- parsing the script block with the vendored babel,
 * `node --check` on the extracted block, counting delimiters -- can only say the code
 * PARSES. None of them execute the effect, and the bug they missed for two rounds was
 * behavioural: the heuristic ran before the neural call and assigned the move.
 *
 * Driven over the DevTools Protocol using Node's built-in WebSocket (Node >= 22), so
 * there is no puppeteer/playwright dependency to install.
 *
 * The second phase deliberately BREAKS the network by pointing the app at a dead port,
 * rather than stopping the real inference server -- a check that takes the live server
 * down is a check nobody will run twice.
 */

import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
// A RANDOM port, not 9222. On the fixed port, if a browser is already listening the
// spawned instance fails to bind, /json/list still answers -- from the USER'S browser --
// and this script would navigate one of their real tabs and click in it. Two concurrent
// runs collided the same way.
const PORT = 9500 + Math.floor(Math.random() * 400);
const APP = process.env.WW_APP_URL || 'http://127.0.0.1:8177/app';

const profile = mkdtempSync(path.join(tmpdir(), 'ww-chrome-'));
let cleanedUp = false;
function cleanup() {
  // Ran only on the happy path before, so any throw from target() or evaluate() left a
  // headless Chrome running AND ~57 MB of profile behind -- $TMPDIR had accumulated
  // several hundred MB of ww-chrome-* dirs. check-play-page.mjs already takes care to
  // avoid exactly this litter.
  if (cleanedUp) return;
  cleanedUp = true;
  try { chrome.kill('SIGKILL'); } catch { /* already gone */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* best effort */ }
}
process.on('exit', cleanup);
process.on('SIGINT', () => { cleanup(); process.exit(130); });
const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--no-first-run', '--no-default-browser-check', '--disable-gpu',
  'about:blank'
], { stdio: ['ignore', 'pipe', 'pipe'] });
let chromeLog = '';
chrome.stderr.on('data', (d) => { chromeLog += d; });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function target() {
  for (let i = 0; i < 60; i += 1) {
    try {
      const list = await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json());
      const page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page.webSocketDebuggerUrl;
    } catch { /* not up yet */ }
    await sleep(250);
  }
  throw new Error(`chrome never exposed a page target\n${chromeLog}`);
}

const ws = new WebSocket(await target());
await new Promise((r, j) => { ws.onopen = r; ws.onerror = () => j(new Error('ws failed')); });
let nextId = 1;
const pending = new Map();
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
function send(method, params = {}) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve) => pending.set(id, resolve));
}
/** Evaluate in the page and return the value, throwing on a page-side exception. */
async function evaluate(expression) {
  const r = await send('Runtime.evaluate',
    { expression, awaitPromise: true, returnByValue: true });
  if (r.error) throw new Error(JSON.stringify(r.error));
  const res = r.result;
  if (res.exceptionDetails) {
    throw new Error(`page exception: ${res.exceptionDetails.exception?.description
      || res.exceptionDetails.text}`);
  }
  return res.result.value;
}

await send('Runtime.enable');
await send('Log.enable');
const consoleErrors = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
    consoleErrors.push(m.params.entry.text);
  }
});

let failures = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name}${detail ? ` -> ${detail}` : ''}`);
  if (!ok) failures += 1;
};

// ---------------------------------------------------------------- phase 1: it plays
await send('Page.enable');
await send('Page.navigate', { url: APP });
await sleep(4000);

const booted = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return { hasRoot: !!document.querySelector('#root, #app, div'), len: t.length };
})()`);
check('app boots', booted.len > 0, `${booted.len} chars of text rendered`);

// Walk the menu by visible text: vs Computer, then the Neural difficulty.
const clicked = await evaluate(`(() => {
  const hit = (re) => [...document.querySelectorAll('button,div,span,a')]
    .filter(el => re.test(el.textContent || '') && el.offsetParent !== null)
    .sort((a,b) => (a.textContent||'').length - (b.textContent||'').length)[0];
  // The menu entry reads "Play vs Bots / Against the computer", not "vs Computer" --
  // matching the string I assumed instead of the string the app renders is why the first
  // run of this check reported a failure that was mine, not the app's.
  const vs = hit(/Play vs Bots|Against the computer|vs Computer|Gegen Computer/i);
  if (!vs) return { step: 'vsAI', found: false,
                    text: (document.body.innerText||'').slice(0, 400) };
  vs.click();
  return { step: 'vsAI', found: true };
})()`);
check('the vs-computer menu is reachable', clicked.found === true, clicked.step);
await sleep(900);

// Select the difficulty ROW by its own class, not by "shortest element containing
// /Neural/i". That heuristic picked the sims slider's `NEURAL SEARCH` label (13 chars,
// versus `Neural (local)` at 14), which is an inert span with no .mi-btn ancestor -- so
// the click did nothing, the game never started, and every later phase of this file
// asserted against the difficulty menu while reporting success. The row is a MenuItem
// button carrying .mi-btn; that is the thing with the handler.
const picked = await evaluate(`(() => {
  const rows = [...document.querySelectorAll('.mi-btn')]
    .filter(el => /Neural/i.test(el.textContent || '') && el.offsetParent !== null);
  if (!rows.length) {
    return { found: false, text: (document.body.innerText||'').slice(0, 300) };
  }
  rows[0].click();
  return { found: true, label: (rows[0].textContent || '').slice(0, 40) };
})()`);
check('the Neural difficulty row is clickable', picked.found === true,
  picked.found ? `clicked "${picked.label}"` : `not on screen: ${picked.text}`);

// POSITIVE EVIDENCE, not the absence of a banner. The previous version looked for a
// "neural panel" by matching /ms\b/ and /sims?/i in the page text -- which the
// difficulty MENU itself satisfies ("128 sims - ~704 ms/move"), so it was true while no
// game had started. And "no unreachable banner" passes trivially when no request was
// ever attempted. The only thing that proves the network played is a request to
// /api/bestmove, which the Resource Timing API records whether or not we instrumented it.
await sleep(2000);
const started = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return { leftMenu: !/Choose difficulty/i.test(t), text: t.slice(0, 200) };
})()`);
check('selecting Neural starts a game', started.leftMenu === true,
  started.leftMenu ? 'left the difficulty menu'
    : `still on the menu: ${started.text.replace(/\n/g, ' ').slice(0, 120)}`);

// PLAY A MOVE FIRST. initGame forces turn 'A', which is the human, so the bot has no
// turn and correctly issues zero requests until we move. Asserting "requests > 0" before
// moving measured my own ordering mistake and reported it as an app failure.
const opened = await evaluate(`(() => {
  const cell = document.querySelector('[data-r="7"][data-c="4"]');
  if (!cell) return { found: false };
  cell.click();
  return { found: true };
})()`);
check('a human opening move is playable', opened.found === true,
  opened.found ? 'clicked (7,4)' : 'no board cell found');

await sleep(14000);
const played = await evaluate(`(() => {
  const calls = performance.getEntriesByType('resource')
    .filter(e => e.name.includes('/api/bestmove')).length;
  const t = document.body.innerText || '';
  return { calls, sawUnavailable: /unreachable|unavailable/i.test(t),
           text: t.slice(0, 300) };
})()`);
check('the app actually requests moves from the network', played.calls > 0,
  `${played.calls} /api/bestmove request(s)`);
check('no "unreachable" banner while the network is up', played.sawUnavailable === false,
  played.sawUnavailable ? played.text.replace(/\n/g, ' ').slice(0, 160) : 'clean');

// ------------------------------------------------- phase 2: it REFUSES, not substitutes
//
// Point the app's own fetch at a dead port for /api/bestmove only. This is the exact
// condition that used to make the built-in bot play: the neural call rejects.
await evaluate(`(() => {
  window.__wwBestmoveCalls = 0;
  const real = window.fetch;
  window.fetch = (u, o) => {
    if (String(u).includes('/api/bestmove')) {
      window.__wwBestmoveCalls += 1;
      return Promise.reject(new TypeError('forced network failure'));
    }
    return real(u, o);
  };
  return true;
})()`);

// Snapshot the move count, then force the bot to have a turn by making a legal human
// move: click any highlighted target square the app is offering.
const before = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return { calls: window.__wwBestmoveCalls, text: t.slice(0, 200) };
})()`);

// Play a real opening move by addressing the CELL, not by guessing at highlight classes.
// Board squares render with data-r/data-c and an onClick handler (index.html), so this is
// unambiguous: the human is A, starts at row 8 col 4, and moves toward row 0, so (7,4) is
// a legal first step. Class-name sniffing was the wrong approach -- /move/ also matched
// move-history containers, and a wrong hit would have blamed the app for the selector.
// (7,4) is where phase 1 moved to, so step again toward the goal for this turn.
const moved = await evaluate(`(() => {
  const cell = document.querySelector('[data-r="6"][data-c="4"]');
  if (!cell) return { found: false, why: 'no [data-r][data-c] cell on screen' };
  cell.click();
  return { found: true };
})()`);

// Wait out the WHOLE retry budget (12 attempts, 500ms*2^n capped at 8s, so ~65 s), not
// just a few attempts. The decisive evidence is the FINAL state: the app's own words
// saying it will not move, with no move played. Sampling mid-retry only shows it is
// still trying, which the old substituting code would also have got past.
await sleep(72000);
const after = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return {
    calls: window.__wwBestmoveCalls,
    sawRetry: /retrying|unreachable|unavailable/i.test(t),
    // The app's refusal, in its own words. This is the exact no-substitution claim:
    // the built-in bot would have moved instead of saying this.
    sawRefusal: /will not move/i.test(t) && /built-in bot will not play/i.test(t),
    text: t.slice(0, 400)
  };
})()`);

let refusalVerified = false;
if (!moved.found) {
  console.log('  note  no human target square was clickable; the bot-turn phase could '
    + 'not be forced, so the refusal path is UNVERIFIED in this run');
} else {
  refusalVerified = true;
  check('the app retries /api/bestmove rather than giving up silently',
    after.calls > before.calls, `${before.calls} -> ${after.calls} calls`);
  check('the app SAYS the neural opponent is unreachable',
    after.sawRetry === true,
    after.sawRetry ? 'banner shown' : `no banner: ${after.text.replace(/\n/g, ' ').slice(0, 200)}`);
  check('the app REFUSES to move rather than substituting the built-in bot',
    after.sawRefusal === true,
    after.sawRefusal ? 'states it will not move, and that the built-in bot will not play'
      : `no refusal wording: ${after.text.replace(/\n/g, ' ').slice(0, 200)}`);
}

const fatal = consoleErrors.filter((t) => /is not defined|is not a function|SyntaxError/.test(t));
check('no fatal console errors', fatal.length === 0, fatal.slice(0, 2).join(' | ') || 'none');

ws.close();
cleanup();
// The summary must not claim the phase that did not run. An earlier version printed
// "reports unreachability instead of substituting" whenever failures === 0, including
// runs where the refusal phase was skipped entirely -- announcing a verification it had
// not performed, which is the exact habit this check exists to break.
if (failures) console.log(`\n${failures} check(s) failed`);
else if (refusalVerified) {
  console.log('\nthe app plays via the network AND refuses rather than substituting '
    + 'when it is unreachable');
} else {
  console.log('\nthe app boots and plays via the network. The REFUSAL path was not '
    + 'exercised in this run (no human move was available to force a bot turn), so it '
    + 'remains unverified -- run again from a position where it is your move.');
}
process.exit(failures ? 1 : 0);
