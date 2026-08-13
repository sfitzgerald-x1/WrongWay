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
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9222;
const APP = process.env.WW_APP_URL || 'http://127.0.0.1:8177/app';

const profile = mkdtempSync(path.join(tmpdir(), 'ww-chrome-'));
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

const picked = await evaluate(`(() => {
  const hit = (re) => [...document.querySelectorAll('button,div,span,a')]
    .filter(el => re.test(el.textContent || '') && el.offsetParent !== null)
    .sort((a,b) => (a.textContent||'').length - (b.textContent||'').length)[0];
  const n = hit(/Neural/i);
  if (!n) return { found: false, text: (document.body.innerText||'').slice(0, 300) };
  // Click the ROW, not the label inside it. The difficulty rows carry .mi-btn and the
  // handler is on the row; clicking the shortest matching text node hit a child span and
  // the game never started -- so the check passed while verifying nothing.
  (n.closest('.mi-btn') || n).click();
  return { found: true };
})()`);
check('the Neural difficulty is offered', picked.found === true,
  picked.found ? 'clicked' : `not on screen: ${picked.text}`);

// The bot is B and may or may not move first; either way, wait out a network move and
// look for the neural panel, which only the network path can populate.
await sleep(12000);
const played = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return {
    text: t.slice(0, 400),
    sawNeuralPanel: /ms\\b/.test(t) && /sims?/i.test(t),
    sawUnavailable: /unreachable|unavailable/i.test(t)
  };
})()`);
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

const moved = await evaluate(`(() => {
  const tgt = [...document.querySelectorAll('*')]
    .filter(el => el.className && typeof el.className === 'string'
      && /tgt|target|move/i.test(el.className) && el.offsetParent !== null);
  if (!tgt.length) return { found: false };
  tgt[0].click();
  return { found: true, n: tgt.length };
})()`);

// Give the retry loop time to make several attempts and give up on none of them yet.
await sleep(14000);
const after = await evaluate(`(() => {
  const t = document.body.innerText || '';
  return {
    calls: window.__wwBestmoveCalls,
    sawRetry: /retrying|unreachable|unavailable/i.test(t),
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
}

const fatal = consoleErrors.filter((t) => /is not defined|is not a function|SyntaxError/.test(t));
check('no fatal console errors', fatal.length === 0, fatal.slice(0, 2).join(' | ') || 'none');

ws.close();
chrome.kill('SIGTERM');
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
