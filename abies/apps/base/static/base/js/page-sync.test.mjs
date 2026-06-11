// Tests for page-sync.js and TableWrapper URL-state support.
// Run with: node apps/base/static/base/js/page-sync.test.mjs

let passed = 0;
let failed = 0;
function check(ok, msg) {
  if (ok) passed++;
  else { failed++; console.error(`FAIL ${msg}`); }
}
function eq(actual, expected, msg) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  check(a === e, `${msg}: expected ${e}, got ${a}`);
}

// ---------------------------------------------------------------------------
// Minimal browser surface for page-css, ui-widgets, modals, and router.
// ---------------------------------------------------------------------------

class MockElement {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.dataset = {};
    this.className = '';
    this.textContent = '';
    this.href = '';
    this.rel = '';
    this.removed = false;
    this._listeners = {};
    this.classList = {
      add: () => {},
      remove: () => {},
      contains: () => false,
      toggle: () => false,
    };
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  remove() { this.removed = true; }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  removeEventListener(type, fn) {
    this._listeners[type] = (this._listeners[type] || []).filter(f => f !== fn);
  }
  contains() { return false; }
}

const contentEl = new MockElement('main');
const modalEl = new MockElement('div');
const mobileMenu = new MockElement('div');
const hamburger = new MockElement('button');
const links = [];

globalThis.document = {
  body: { dataset: { csrf: 'csrf-token' } },
  head: { appendChild: el => links.push(el) },
  createElement: tag => new MockElement(tag),
  createDocumentFragment: () => new MockElement('fragment'),
  addEventListener() {},
  removeEventListener() {},
  getElementById(id) {
    if (id === 'content') return contentEl;
    if (id === 'modal-container') return modalEl;
    if (id === 'mobile-menu') return mobileMenu;
    if (id === 'hamburger') return hamburger;
    return null;
  },
  querySelector(sel) {
    const href = sel.match(/^link\[href="([^"]+)"\]$/)?.[1];
    return href ? links.find(link => link.href === href && !link.removed) || null : null;
  },
  querySelectorAll() { return []; },
};

const setLocation = (url) => {
  const u = new URL(url, 'https://example.test');
  globalThis.location = { pathname: u.pathname, search: u.search };
};
setLocation('/');

globalThis.history = {
  pushed: [],
  replaced: [],
  pushState(_state, _title, url) {
    this.pushed.push(url);
    setLocation(url);
  },
  replaceState(_state, _title, url) {
    this.replaced.push(url);
    setLocation(url);
  },
};
globalThis.window = { addEventListener() {} };

// ---------------------------------------------------------------------------
// Server model for cache.load() used by createPage.
// ---------------------------------------------------------------------------

const server = new Map();
function responseFor(url) {
  const entry = server.get(url);
  if (!entry) throw new Error(`missing server entry for ${url}`);
  return {
    status: 200,
    ok: true,
    headers: { get: h => h === 'Last-Modified' ? entry.lastModified : null },
    json: async () => entry.data,
  };
}
globalThis.fetch = async url => responseFor(url);

const cache = await import('./cache.js');
const router = await import('./router.js');
const {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableParamKeys, tableSort, writeTableState,
} = await import('./page-sync.js');
const { TableWrapper } = await import('./table.js');

// ---------------------------------------------------------------------------
// Table URL state helpers.
// ---------------------------------------------------------------------------

{
  const keys = tableParamKeys('t');
  eq(keys, { sortColumn: 'tsc', sortOrder: 'tso', search: 'tf' }, 'tableParamKeys prefixes names');

  eq(readTableState({ tsc: 'date', tso: '1', tf: 'abies' }, keys), {
    sort: { column: 'date', ascending: false }, searchText: 'abies',
  }, 'readTableState reads plain-object params');

  eq(readTableState(new URLSearchParams('tsc=name&tso=0&tf=ceduo'), keys), {
    sort: { column: 'name', ascending: true }, searchText: 'ceduo',
  }, 'readTableState reads URLSearchParams');

  eq(readTableState({}, keys), { sort: null, searchText: '' }, 'readTableState defaults blank state');
  eq(tableSort({ sort: null }, { column: 'fallback', ascending: true }),
     { column: 'fallback', ascending: true }, 'tableSort applies fallback');
}

{
  const params = new URLSearchParams();
  writeTableState(params, {
    getSort: () => ({ column: 'date', ascending: false }),
    getSearchText: () => 'needle',
  });
  eq(params.toString(), 'sc=date&so=1&f=needle', 'writeTableState serializes sort and search');

  const empty = new URLSearchParams();
  writeTableState(empty, { getSort: () => null, getSearchText: () => '' });
  eq(empty.toString(), '', 'writeTableState skips blank state');
}

{
  const calls = [];
  const table = {
    sort: { column: 'old', ascending: true },
    search: 'old text',
    getSort() { return this.sort; },
    setSort(sort) { calls.push(['sort', sort]); this.sort = sort; },
    getSearchText() { return this.search; },
    setSearchText(text) { calls.push(['search', text]); this.search = text; },
  };
  const changed = applyTableState(table, {
    sort: { column: 'new', ascending: false }, searchText: 'new text',
  });
  check(changed, 'applyTableState reports change');
  eq(calls, [
    ['sort', { column: 'new', ascending: false }],
    ['search', 'new text'],
  ], 'applyTableState updates changed sort and search');

  calls.length = 0;
  check(!applyTableState(table, {
    sort: { column: 'new', ascending: false }, searchText: 'new text',
  }), 'applyTableState is idempotent');
  eq(calls, [], 'applyTableState does not rewrite equal state');

  applyTableState(table, { sort: null, searchText: '' }, { column: 'fallback', ascending: true });
  eq(table.sort, { column: 'fallback', ascending: true }, 'applyTableState restores fallback sort when URL has no sort');
  eq(table.search, '', 'applyTableState clears search when URL has no search');
}

// TableWrapper.setSort delegates to SortableTable.sort without firing onSort.
{
  let onSortCalls = 0;
  const wrapper = Object.create(TableWrapper.prototype);
  wrapper._stColumns = [{ key: 'name', type: 'string' }, { key: 'date', type: 'date' }];
  wrapper._table = {
    currentSort: { column: 'name', ascending: true },
    onSort: () => { onSortCalls++; },
    sort(column, type, ascending) {
      this.sorted = { column, type, ascending, callbackWasDisabled: this.onSort === null };
      this.currentSort = { column, ascending };
    },
  };

  wrapper.setSort({ column: 'date', ascending: false });
  eq(wrapper._table.sorted,
     { column: 'date', type: 'date', ascending: false, callbackWasDisabled: true },
     'TableWrapper.setSort delegates with column type and suppresses onSort');
  eq(onSortCalls, 0, 'TableWrapper.setSort does not fire onSort');

  wrapper._table.sorted = null;
  wrapper.setSort({ column: 'date', ascending: false });
  eq(wrapper._table.sorted, null, 'TableWrapper.setSort skips equal sort');

  wrapper.setSort({ column: 'missing', ascending: true });
  eq(wrapper._table.sorted, null, 'TableWrapper.setSort ignores unknown columns');
}

// TableWrapper.rowForElement exposes rendered-row lookup without leaking SortableTable internals.
{
  const wrapper = Object.create(TableWrapper.prototype);
  wrapper._table = { data: [[10, 'Abete'], [20, 'Faggio']] };
  eq(wrapper.rowForElement({ dataset: { index: '1' } }), [20, 'Faggio'],
     'TableWrapper.rowForElement returns row by rendered index');
  eq(wrapper.rowForElement({ dataset: { index: 'bad' } }), null,
     'TableWrapper.rowForElement rejects invalid indexes');
  wrapper._table = null;
  eq(wrapper.rowForElement({ dataset: { index: '0' } }), null,
     'TableWrapper.rowForElement handles missing table');
}

// navigateWithParams uses replace by default and preserves empty query handling.
{
  router.init();
  navigateWithParams('/prelievi', new URLSearchParams('f=abies'));
  eq(history.replaced.at(-1), '/prelievi?f=abies', 'navigateWithParams replaces with query');

  navigateWithParams('/prelievi', new URLSearchParams(), false);
  eq(history.pushed.at(-1), '/prelievi', 'navigateWithParams pushes bare path without ?');
}

// ---------------------------------------------------------------------------
// createPage lifecycle.
// ---------------------------------------------------------------------------

{
  const id = 'page_sync_test_single';
  cache.register(id, '/api/page-sync/single/');
  server.set('/api/page-sync/single/', {
    lastModified: '1',
    data: { columns: ['row_id'], rows: [[1]] },
  });

  const events = [];
  const page = createPage({
    cssUrl: '/static/test.css',
    dataIds: [id],
    mount(el, params, data) {
      events.push(['mount', el === contentEl, params.view, data.rows[0][0]]);
    },
    unmount() { events.push(['unmount']); },
    onQueryChange(params) { events.push(['query', params.view]); },
    onUpdate: [[id, data => events.push(['update', data.rows[0][0]])]],
  });

  await page.mount({ view: 'a' });
  eq(events[0], ['mount', true, 'a', 1], 'createPage mounts with content element, params, and loaded data');
  check(links.some(link => link.href === '/static/test.css' && !link.removed), 'createPage loads page CSS');
  check(contentEl.children[0]?.className === 'loading-overlay', 'createPage shows loading before mount');

  server.set('/api/page-sync/single/', {
    lastModified: '2',
    data: { columns: ['row_id'], rows: [[2]] },
  });
  await cache.load(id);
  eq(events.at(-1), ['update', 2], 'createPage wires cache update callbacks');

  page.onQueryChange({ view: 'b' });
  eq(events.at(-1), ['query', 'b'], 'createPage forwards query changes');

  page.unmount();
  check(links.find(link => link.href === '/static/test.css')?.removed, 'createPage unloads CSS on unmount');
  eq(events.at(-1), ['unmount'], 'createPage calls unmount hook');

  server.set('/api/page-sync/single/', {
    lastModified: '3',
    data: { columns: ['row_id'], rows: [[3]] },
  });
  await cache.load(id);
  check(!events.some(e => e[0] === 'update' && e[1] === 3), 'createPage unsubscribes cache update callbacks');
}

{
  const wrapper = Object.create(TableWrapper.prototype);
  wrapper.csvFormat = { separator: ';', decimal: ',', dateFormat: 'YYYY-MM-DD' };
  wrapper.labels = { boolYes: 'si', boolNo: 'no' };
  wrapper._stColumns = [
    { key: 'name', label: 'Nome', type: 'string' },
    { key: 'value', label: 'Valore', type: 'string' },
  ];
  wrapper._table = { data: [['=cmd', '-4'], ['safe', '@cmd']] };
  eq(
    wrapper.getCSV(),
    `\ufeffNome;Valore\n'=cmd;-4\nsafe;'@cmd`,
    'TableWrapper.getCSV hardens spreadsheet formulas',
  );
}

{
  const a = 'page_sync_test_a';
  const b = 'page_sync_test_b';
  cache.register(a, '/api/page-sync/a/');
  cache.register(b, '/api/page-sync/b/');
  server.set('/api/page-sync/a/', { lastModified: '1', data: { id: 'a' } });
  server.set('/api/page-sync/b/', { lastModified: '1', data: { id: 'b' } });

  let mountedData = null;
  const page = createPage({
    dataIds: [a, b],
    mount(_el, _params, data) { mountedData = data; },
  });
  await page.mount({});
  eq(mountedData, [{ id: 'a' }, { id: 'b' }], 'createPage loads multiple data ids as an array');
  page.unmount();
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
