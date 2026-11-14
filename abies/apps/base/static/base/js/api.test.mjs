import assert from 'node:assert/strict';
import { fetchJSON } from './api.js';

let requestHeaders;
let responseStatus = 200;
const responseHeaders = new Map([
  ['ETag', '"published-file"'], ['Last-Modified', 'old-date'],
]);
globalThis.fetch = async (url, options) => {
  requestHeaders = options.headers;
  return {
    status: responseStatus, ok: responseStatus === 200,
    headers: { get: key => responseHeaders.get(key) ?? null },
    json: async () => ({ rows: [[1]] }),
  };
};

const initial = await fetchJSON('/digest');
assert.equal(initial.etag, '"published-file"');
assert.equal(initial.lastModified, 'old-date');
responseStatus = 304;
const unchanged = await fetchJSON('/digest', 'old-date', initial.etag);
assert.equal(requestHeaders['If-None-Match'], initial.etag);
assert.equal(requestHeaders['If-Modified-Since'], undefined);
assert.equal(unchanged.etag, initial.etag);
assert.equal(unchanged.data, null);

responseHeaders.delete('ETag');
assert.equal((await fetchJSON('/digest', 'old-date', initial.etag)).etag, initial.etag);
await fetchJSON('/legacy-file', 'old-date');
assert.equal(requestHeaders['If-Modified-Since'], 'old-date');
assert.equal(requestHeaders['If-None-Match'], undefined);
console.log('API validator tests passed.');

// Binary exports retain signed receipts and structured review errors.
const { postBlob } = await import('./api.js');
globalThis.document = { body: { dataset: { csrf: 'test-csrf' } } };
globalThis.fetch = async () => ({ ok: true, status: 200,
  headers: new Headers({ 'Content-Disposition': 'attachment; filename="rows.csv"', 'X-Import-Export-Token': 'receipt' }),
  blob: async () => new Blob(['rows']),
});
const exported = await postBlob('/export', {});
assert.equal(exported.filename, 'rows.csv');
assert.equal(exported.headers.get('X-Import-Export-Token'), 'receipt');
const changedReview = { status: 'import_review', review_token: 'new-token' };
globalThis.fetch = async () => ({ ok: false, status: 409, json: async () => changedReview });
await assert.rejects(postBlob('/export', {}), error => error.status === 409 && error.data === changedReview);
console.log('Binary export receipts and structured errors passed.');
