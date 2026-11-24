import assert from 'node:assert/strict';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
import { startVersionCheck } from './app-update.js';

const body = installDOM();
body.dataset.appVersion = '2.0.2';
body.dataset.appVersionUrl = '/api/version/';

const overlay = new Element();
overlay.id = 'modal-container';
const template = new Element('template');
template.id = 'tmpl-app-update';
template.content = new Element('#fragment');
const button = new Element('button', 'Reload');
button.dataset.action = 'reload';
template.content.appendChild(button);
body.append(overlay, template);

let tick;
globalThis.setInterval = (callback, delay) => {
  assert.equal(delay, 5 * 60 * 1000);
  tick = callback;
};
let remoteVersion = '2.0.2';
globalThis.fetch = async url => {
  assert.equal(url, body.dataset.appVersionUrl);
  if (remoteVersion === null) throw new Error('Offline');
  return new Response(JSON.stringify({ version: remoteVersion }));
};
let reloads = 0;
globalThis.window = { location: { reload: () => reloads++ } };

startVersionCheck();
await tick();
assert.equal(overlay.classList.contains('open'), false);
remoteVersion = null;
await tick();
assert.equal(overlay.classList.contains('open'), false);

remoteVersion = '2.0.3';
const existingModal = new Element('form');
overlay.appendChild(existingModal);
overlay.classList.add('open');
await tick();
assert.equal(overlay.firstChild, existingModal);
overlay.replaceChildren();
overlay.classList.remove('open');

await tick();
assert.equal(overlay.classList.contains('open'), true);
assert.equal(reloads, 0);
await overlay.querySelector('[data-action="reload"]').dispatch();
assert.equal(reloads, 1);
