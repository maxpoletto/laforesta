// Tests for shared form helpers in forms.js.
// Run with: node test/test_forms.mjs (also part of `make test-js`).

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
    this.tagName = tag;
    this.children = [];
    this.type = '';
    this.name = '';
    this.value = '';
    this.textContent = '';
    this.className = '';
    this.dataset = {};
    this.fields = [];
    this._listeners = {};
  }
  appendChild(child) { this.children.push(child); return child; }
  before() {}
  addEventListener(type, fn) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(fn);
  }
  async submit(submitter = null) {
    for (const fn of this._listeners.submit || []) {
      await fn({ preventDefault() {}, submitter });
    }
  }
  querySelector(sel) {
    if (sel === 'form' && this.tagName === 'form') return this;
    if (sel === '.form-error' && this.className.split(/\s+/).includes('form-error')) return this;
    if (sel === '.form-actions' && this.className.split(/\s+/).includes('form-actions')) return this;
    const inputMatch = sel.match(/^input\[name="([^"]+)"\]$/);
    if (inputMatch && this.tagName === 'input' && this.name === inputMatch[1]) return this;
    for (const child of this.children) {
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

globalThis.document = {
  body: { dataset: { csrf: 'csrf-token' } },
  createElement: (tag) => new MockElement(tag),
  createDocumentFragment: () => new MockElement('fragment'),
  getElementById: () => new MockElement('div'),
  querySelector: () => null,
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

const { injectNonce, interceptSubmit, parseHTMLFragment } = await import(
  '../apps/base/static/base/js/forms.js'
);

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

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
