// Tests for shared form helpers in forms.js.
// Run with: node apps/base/static/base/js/forms.test.mjs (also part of `make test-js`).

let passed = 0;
let failed = 0;

function check(ok, msg) {
  if (ok) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
  }
}

function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

class MockElement {
  constructor(tag) {
    this.tagName = tag.toLowerCase();
    this.children = [];
    this.parentNode = null;
    this.type = '';
    this.name = '';
    this.value = '';
    this.textContent = '';
    this.className = '';
    this.dataset = {};
    this.fields = [];
    this.hidden = false;
    this.disabled = false;
    this._listeners = {};
    this.classList = {
      add: (...names) => {
        const classes = this._classes();
        for (const name of names) classes.add(name);
        this.className = [...classes].join(' ');
      },
      remove: (...names) => {
        const classes = this._classes();
        for (const name of names) classes.delete(name);
        this.className = [...classes].join(' ');
      },
      contains: name => this._classes().has(name),
    };
  }
  _classes() { return new Set(this.className.split(/\s+/).filter(Boolean)); }
  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  replaceChildren(...children) {
    for (const child of this.children) child.parentNode = null;
    this.children = [];
    for (const child of children) this.appendChild(child);
  }
  before(child) {
    if (!this.parentNode) return;
    const index = this.parentNode.children.indexOf(this);
    child.parentNode = this.parentNode;
    this.parentNode.children.splice(index, 0, child);
  }
  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }
  async submit(submitter = null) {
    for (const fn of this._listeners.submit || []) {
      await fn({ preventDefault() {}, submitter });
    }
  }
  _matches(sel) {
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    const inputMatch = sel.match(/^input\[name="([^"]+)"\]$/);
    if (inputMatch) return this.tagName === 'input' && this.name === inputMatch[1];
    const buttonType = sel.match(/^button\[type="([^"]+)"\]$/);
    if (buttonType) return this.tagName === 'button' && this.type === buttonType[1];
    return this.tagName === sel.toLowerCase();
  }
  querySelector(sel) {
    for (const child of this.children) {
      if (child._matches?.(sel)) return child;
      const found = child.querySelector?.(sel);
      if (found) return found;
    }
    return null;
  }
}

class MockFormData {
  constructor(form) { this.fields = form.fields || []; }
  [Symbol.iterator]() { return this.fields[Symbol.iterator](); }
}

Object.defineProperty(globalThis, 'crypto', {
  configurable: true,
  value: { randomUUID: (() => {
    let n = 0;
    return () => `nonce-${++n}`;
  })() },
});

const contentEl = new MockElement('div');
const modalContainer = new MockElement('div');

globalThis.document = {
  body: { dataset: { csrf: 'csrf-token' } },
  createElement: (tag) => new MockElement(tag),
  createDocumentFragment: () => new MockElement('fragment'),
  getElementById(id) {
    if (id === 'content') return contentEl;
    if (id === 'modal-container') return modalContainer;
    return null;
  },
  querySelector(sel) {
    if (sel === '#modal-container form') return modalContainer.querySelector('form');
    return null;
  },
  addEventListener() {},
  removeEventListener() {},
};

globalThis.DOMParser = class {
  parseFromString(html) {
    const form = new MockElement('form');
    form.dataset.source = html;
    return { body: { childNodes: [form] } };
  }
};

globalThis.FormData = MockFormData;

const {
  deleteRowWithVersion, fetchModalForm, injectNonce, interceptSubmit,
  parseHTMLFragment, renderModalForm, submitCsvImport,
} = await import('./forms.js');
const cache = await import('./cache.js');

// parseHTMLFragment: server HTML becomes a fragment with queryable children.
{
  const frag = parseHTMLFragment('<form id="x"></form>');
  check(Boolean(frag.querySelector('form')), 'parseHTMLFragment exposes parsed form');
}

// injectNonce: creates the hidden input once and refreshes its UUID on reuse.
{
  const form = new MockElement('form');
  injectNonce(form);
  const input = form.querySelector('input[name="nonce"]');
  check(Boolean(input), 'injectNonce creates nonce input');
  eq(input.type, 'hidden', 'nonce input is hidden');
  eq(input.value, 'nonce-1', 'nonce input gets first UUID');
  injectNonce(form);
  eq(form.children.length, 1, 'injectNonce reuses existing input');
  eq(input.value, 'nonce-2', 'injectNonce refreshes UUID');
}

// interceptSubmit: validation HTML can be delegated to the caller for re-render.
{
  let posted = null;
  globalThis.fetch = async (url, opts) => {
    posted = { url, body: JSON.parse(opts.body), csrf: opts.headers['X-CSRFToken'] };
    return {
      status: 400,
      json: async () => ({ html: '<form>replacement</form>' }),
    };
  };

  const form = new MockElement('form');
  form.fields = [['name', 'Abies']];
  let htmlSeen = null;
  let successCalled = false;
  interceptSubmit(form, '/save/', {
    onSuccess() { successCalled = true; },
    onHtml(html) { htmlSeen = html; },
  });
  await form.submit();

  eq(posted, {
    url: '/save/',
    body: { name: 'Abies' },
    csrf: 'csrf-token',
  }, 'interceptSubmit posts form data as JSON');
  eq(htmlSeen, '<form>replacement</form>', 'interceptSubmit delegates replacement HTML');
  check(!successCalled, 'interceptSubmit does not call success on validation HTML');
}

{
  globalThis.fetch = async () => ({ status: 200, json: async () => ({ ok: true }) });
  const form = new MockElement('form');
  const calls = [];
  interceptSubmit(form, '/save/', {
    async onSuccess() {
      await Promise.resolve();
      calls.push('success');
    },
  });
  await form.submit();
  eq(calls, ['success'], 'interceptSubmit awaits async success callbacks');
}

// interceptSubmit: conflicts surface inline and reach the conflict callback.
{
  const conflict = { status: 'conflict', message: 'Versione superata' };
  globalThis.fetch = async () => ({ status: 400, json: async () => conflict });
  const form = new MockElement('form');
  let conflictSeen = null;
  interceptSubmit(form, '/save/', {
    onConflict(data) { conflictSeen = data; },
  });

  await form.submit();

  eq(conflictSeen, conflict, 'interceptSubmit dispatches conflicts');
  eq(form.querySelector('.form-error')?.textContent, conflict.message,
     'interceptSubmit renders the conflict message inline');
}

// renderModalForm/fetchModalForm: server HTML is shown with a fresh nonce.
{
  modalContainer.className = '';
  const rendered = renderModalForm('<form>rendered modal</form>');
  check(Boolean(rendered), 'renderModalForm returns the displayed form');
  check(Boolean(rendered.querySelector('input[name="nonce"]')),
        'renderModalForm injects a nonce');
  check(modalContainer.classList.contains('open'), 'renderModalForm opens the modal');

  globalThis.fetch = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ html: '<form>fetched modal</form>' }),
  });
  const fetched = await fetchModalForm('/form/');
  check(Boolean(fetched), 'fetchModalForm returns the fetched form');
  eq(fetched.dataset.source, '<form>fetched modal</form>',
     'fetchModalForm renders response HTML');
  check(Boolean(fetched.querySelector('input[name="nonce"]')),
        'fetchModalForm injects a nonce');
}

// deleteRowWithVersion: success removes the cached row and posts its version.
{
  const dataId = 'forms-delete-success';
  cache.set(dataId, { columns: ['row_id', 'version'], rows: [[7, 3]] });
  let posted = null;
  let successSeen = null;
  globalThis.fetch = async (url, opts) => {
    posted = { url, body: JSON.parse(opts.body) };
    return { status: 200, json: async () => ({ deleted: true }) };
  };

  const deleted = await deleteRowWithVersion(dataId, 7, '/delete/', {
    confirmMessage: null,
    onSuccess(data) { successSeen = data; },
  });

  check(deleted, 'deleteRowWithVersion reports success');
  eq(posted.url, '/delete/', 'deleteRowWithVersion posts to the delete endpoint');
  eq(posted.body.row_id, '7', 'deleteRowWithVersion posts the row id');
  eq(posted.body.version, '3', 'deleteRowWithVersion posts the cached version');
  check(Boolean(posted.body.nonce), 'deleteRowWithVersion posts a nonce');
  eq(cache.get(dataId).rows, [], 'deleteRowWithVersion removes the cached row');
  eq(successSeen, { deleted: true }, 'deleteRowWithVersion dispatches success');
}

// deleteRowWithVersion: a conflict refreshes the cached row before callback.
{
  const dataId = 'forms-delete-conflict';
  cache.set(dataId, { columns: ['row_id', 'version'], rows: [[8, 2]] });
  const conflict = {
    status: 'conflict', message: 'Versione superata',
    row_id: 8, record: [8, 4],
  };
  let conflictSeen = null;
  globalThis.fetch = async () => ({ status: 400, json: async () => conflict });

  const deleted = await deleteRowWithVersion(dataId, 8, '/delete/', {
    confirmMessage: null,
    onConflict(data) { conflictSeen = data; },
  });

  check(!deleted, 'deleteRowWithVersion reports conflict as not deleted');
  eq(cache.get(dataId).rows, [[8, 4]],
     'deleteRowWithVersion applies the conflict record to cache');
  eq(conflictSeen, conflict, 'deleteRowWithVersion dispatches conflict');
}

// submitCsvImport: row errors remain visible and controls are restored.
{
  const form = new MockElement('form');
  const submit = new MockElement('button');
  submit.type = 'submit';
  form.appendChild(submit);
  const statusBox = new MockElement('p');
  const errorsBox = new MockElement('div');

  const result = await submitCsvImport({
    form, statusBox, errorsBox,
    attempt: async () => ({ errors: ['Riga 1', 'Riga 2'] }),
  });

  eq(result, { errors: ['Riga 1', 'Riga 2'] },
     'submitCsvImport returns row errors');
  check(!submit.disabled, 'submitCsvImport re-enables submit after errors');
  check(statusBox.hidden, 'submitCsvImport hides progress after errors');
  check(!errorsBox.hidden, 'submitCsvImport displays row errors');
  eq(errorsBox.querySelector('ul')?.children.map(li => li.textContent),
     ['Riga 1', 'Riga 2'], 'submitCsvImport renders each row error');
}

// submitCsvImport: success dismisses the modal and restores controls.
{
  renderModalForm('<form>CSV modal</form>');
  const form = new MockElement('form');
  const submit = new MockElement('button');
  submit.type = 'submit';
  form.appendChild(submit);
  const statusBox = new MockElement('p');
  const errorsBox = new MockElement('div');

  const result = await submitCsvImport({
    form, statusBox, errorsBox,
    attempt: async () => ({ ok: true }),
  });

  eq(result, { ok: true }, 'submitCsvImport returns success');
  check(!submit.disabled, 'submitCsvImport re-enables submit after success');
  check(statusBox.hidden, 'submitCsvImport hides progress after success');
  check(!modalContainer.classList.contains('open'),
        'submitCsvImport dismisses the modal on success');
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
