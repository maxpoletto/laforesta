import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-import-review-'));
process.on('exit', () => fs.rmSync(temporary, { recursive: true, force: true }));
const staticRoot = path.join(temporary, 'static');
fs.mkdirSync(path.join(staticRoot, 'ipso'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'ipso/js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'), path.join(staticRoot, 'base/js'), { recursive: true });
const body = installDOM();
document.documentElement = { lang: 'it' };
const modal = new Element();
modal.id = 'modal-container';
body.appendChild(modal);
const template = new Element('template');
template.id = 'tmpl-ipso-import-review-modal';
template.content = new Element('#fragment');
for (const field of ['summary', 'message', 'errors', 'warnings', 'export-required']) {
  const node = new Element('p');
  node.dataset.field = field;
  template.content.appendChild(node);
}
for (const action of ['cancel', 'export', 'confirm']) {
  const node = new Element('button');
  node.dataset.action = action;
  template.content.appendChild(node);
}
body.appendChild(template);
let downloads = 0;
Element.prototype.click = function() { if (this.tagName === 'a') downloads++; };
let exportAttempt = 0;
const requests = [];
let changed = false;
const review = { status: 'import_review', review_token: 'review-1', valid_count: 2,
  excluded_count: 1, errors: ['Riga 30: altezza mancante.'], warnings: ['Riga 28: altezza non misurata.'] };
globalThis.fetch = async (url, options) => {
  requests.push([url, JSON.parse(options.body)]);
  exportAttempt++;
  if (exportAttempt === 1) return { ok: false, status: 500, json: async () => ({ message: 'Export failed' }) };
  if (changed) {
    changed = false;
    return { ok: false, status: 409, json: async () => ({ ...review, review_token: 'review-2' }) };
  }
  return { ok: true, status: 200,
    headers: new Headers({ 'Content-Disposition': 'attachment; filename="excluded.csv"', 'X-Import-Export-Token': 'export-2' }),
    blob: async () => new Blob(['csv rows']),
  };
};
const { showTreeImportReview } = await import(pathToFileURL(path.join(staticRoot, 'ipso/js/import-review.js')).href);
let confirmed = null;
const show = value => showTreeImportReview({ uploadId: 17, review: value,
  onConfirm: async result => { confirmed = result; } });
const button = action => modal.querySelector(`[data-action="${action}"]`);
show(review);
assert.equal(button('confirm').disabled, true, 'Import is gated before export');
assert.match(modal.querySelector('[data-field="export-required"]').textContent, /esporta.*prima/i);
assert.equal(modal.querySelector('[data-field="errors"]').textContent, review.errors.join('\n'));
assert.equal(modal.querySelector('[data-field="warnings"]').textContent, review.warnings.join('\n'));
await button('confirm').dispatch();
assert.equal(confirmed, null, 'Disabled confirmation cannot import');
await button('export').dispatch();
assert.equal(button('confirm').disabled, true, 'Failed export does not unlock import');
assert.equal(downloads, 0);
changed = true;
await button('export').dispatch();
assert.equal(button('confirm').disabled, true, 'Changed review resets export gate');
await button('export').dispatch();
assert.equal(requests.at(-1)[1].review_token, 'review-2');
assert.equal(downloads, 1, 'Successful export downloads the CSV');
assert.equal(button('confirm').disabled, false);
await button('confirm').dispatch();
assert.deepEqual(confirmed, { review_token: 'review-2', export_token: 'export-2', warnings_confirmed: true });
assert.equal(modal.classList.contains('open'), false, 'Accepted review dismisses the modal');
confirmed = null;
show({ ...review, valid_count: 0 });
await button('export').dispatch();
assert.equal(button('confirm').disabled, true, 'Export cannot enable an import with no valid rows');
await button('cancel').dispatch();
assert.equal(confirmed, null, 'Cancel never imports');
show(review);
assert.equal(button('confirm').disabled, true, 'Reopening review requires export again');
await button('cancel').dispatch();
console.log('Import review: export failure, changed review, successful download, zero-valid rows and cancellation passed');
