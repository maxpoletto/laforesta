// Tests for router.js URL stash, reset, and popstate semantics.
// Run with: node apps/base/static/base/js/router.test.mjs.

let passed = 0;
let failed = 0;

function check(ok, message) {
  if (ok) passed++;
  else {
    failed++;
    console.error(`FAIL ${message}`);
  }
}

function eq(actual, expected, message) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, `${message}: expected ${e}, got ${a}`);
}

class MockElement {
  constructor({ tab = null, href = null } = {}) {
    this.dataset = tab ? { tab } : {};
    this.href = href;
    this._listeners = {};
    this.className = '';
    this.classList = {
      add: name => this._setClass(name, true),
      remove: name => this._setClass(name, false),
      contains: name => this.className.split(/\s+/).includes(name),
      toggle: (name, force) => {
        const enabled = force === undefined ? !this.classList.contains(name) : force;
        this._setClass(name, enabled);
        return enabled;
      },
    };
  }
  _setClass(name, enabled) {
    const classes = new Set(this.className.split(/\s+/).filter(Boolean));
    if (enabled) classes.add(name);
    else classes.delete(name);
    this.className = [...classes].join(' ');
  }
  addEventListener(type, callback) { this._listeners[type] = callback; }
  getAttribute(name) { return name === 'href' ? this.href : null; }
  contains(target) { return target === this; }
  replaceChildren() {}
  async click() {
    await this._listeners.click?.({ preventDefault() {}, target: this });
  }
}

const tabs = [
  new MockElement({ tab: 'prelievi', href: '/prelievi' }),
  new MockElement({ tab: 'controllo', href: '/controllo' }),
];
const content = new MockElement();
const mobileMenu = new MockElement();
const hamburger = new MockElement();
const documentListeners = {};
const windowListeners = {};

globalThis.document = {
  querySelectorAll(selector) {
    return selector === '.tab, .mobile-tab' ? tabs : [];
  },
  getElementById(id) {
    if (id === 'content') return content;
    if (id === 'mobile-menu') return mobileMenu;
    if (id === 'hamburger') return hamburger;
    return null;
  },
  addEventListener(type, callback) { documentListeners[type] = callback; },
};

globalThis.window = {
  addEventListener(type, callback) { windowListeners[type] = callback; },
};

globalThis.location = { pathname: '/prelievi', search: '?f=uno' };
function setURL(url) {
  const parsed = new URL(url, 'https://example.test');
  location.pathname = parsed.pathname;
  location.search = parsed.search;
}

globalThis.history = {
  pushed: [],
  replaced: [],
  pushState(_state, _title, url) {
    this.pushed.push(url);
    setURL(url);
  },
  replaceState(_state, _title, url) {
    this.replaced.push(url);
    setURL(url);
  },
};

const calls = [];
function page(name) {
  return {
    mount(params) { calls.push([name, 'mount', params]); },
    unmount() { calls.push([name, 'unmount']); },
    onQueryChange(params) { calls.push([name, 'query', params]); },
  };
}

const router = await import('./router.js');
router.addRoute('prelievi', page('prelievi'));
router.addRoute('controllo', page('controllo'));
router.init();

eq(calls, [['prelievi', 'mount', { f: 'uno' }]],
   'initial URL mounts its domain with query state');
check(tabs[0].classList.contains('active'), 'initial domain tab is active');

await tabs[1].click();
eq(history.pushed.at(-1), '/controllo', 'first visit uses the target bare URL');
eq(calls.slice(-2), [['prelievi', 'unmount'], ['controllo', 'mount', {}]],
   'cross-domain click swaps page lifecycle');

router.navigate('/controllo?grafico=due');
eq(calls.at(-1), ['controllo', 'query', { grafico: 'due' }],
   'same-domain navigation updates query state in place');

await tabs[0].click();
eq(history.pushed.at(-1), '/prelievi?f=uno',
   'returning to another domain restores its stashed URL');
eq(calls.at(-1), ['prelievi', 'mount', { f: 'uno' }],
   'restored domain receives its stashed parameters');

await tabs[1].click();
eq(history.pushed.at(-1), '/controllo?grafico=due',
   'each domain keeps an independent URL stash');

await tabs[1].click();
eq(history.pushed.at(-1), '/controllo',
   'clicking the current tab resets to its bare href');
eq(calls.at(-1), ['controllo', 'query', {}],
   'same-tab reset clears page query state without remounting');

const pushesBeforeSameBareClick = history.pushed.length;
await tabs[1].click();
eq(history.pushed.length, pushesBeforeSameBareClick,
   'clicking the current tab again on its bare URL does not duplicate history');

const pushesBeforePop = history.pushed.length;
setURL('/prelievi?back=si');
await windowListeners.popstate();
eq(history.pushed.length, pushesBeforePop, 'popstate does not write browser history');
eq(calls.slice(-2), [['controllo', 'unmount'], ['prelievi', 'mount', { back: 'si' }]],
   'cross-domain popstate follows page lifecycle');

setURL('/prelievi?back=ancora');
await windowListeners.popstate();
eq(calls.at(-1), ['prelievi', 'query', { back: 'ancora' }],
   'same-domain popstate updates query state in place');

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
