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
    this.style = { setProperty: (key, value) => { this.style[key] = value; } };
    this._classes = new Set();
    const syncClasses = () => {
      this._classes = new Set(String(this.className).split(/\s+/).filter(Boolean));
    };
    const writeClasses = () => { this.className = [...this._classes].join(' '); };
    this.classList = {
      add: (...names) => {
        syncClasses();
        for (const name of names) this._classes.add(name);
        writeClasses();
      },
      remove: (...names) => {
        syncClasses();
        for (const name of names) this._classes.delete(name);
        writeClasses();
      },
      contains: (name) => String(this.className).split(/\s+/).includes(name),
      toggle: (name, force) => {
        syncClasses();
        const next = force === undefined ? !this._classes.has(name) : Boolean(force);
        if (next) this._classes.add(name);
        else this._classes.delete(name);
        writeClasses();
        return next;
      },
    };
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  remove() { this.removed = true; }
  querySelectorAll() { return []; }
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

// TableWrapper drops stale URL sort columns before constructing SortableTable.
{
  let constructorSort = 'not-called';
  const previousSortableTable = window.SortableTable;
  window.SortableTable = class {
    constructor(opts) {
      constructorSort = opts.sort;
      if (opts.sort?.column === 'missing') throw new Error('unknown column');
      this.currentSort = opts.sort || { column: 'name', ascending: true };
    }
    destroy() {}
    clearFilter() {}
  };
  const digest = { columns: ['row_id', 'name'], rows: [] };

  new TableWrapper({
    container: new MockElement('div'), digest, columnDefs: {}, inlineToolbar: false,
    sort: { column: 'missing', ascending: true },
  });

  eq(constructorSort, undefined, 'TableWrapper ignores unknown initial sort columns');
  window.SortableTable = previousSortableTable;
}

// TableWrapper forwards header and cell classes from columnDefs.
{
  let columns = [];
  const previousSortableTable = window.SortableTable;
  window.SortableTable = class {
    constructor(opts) {
      columns = opts.columns;
      this.currentSort = opts.sort || { column: 'name', ascending: true };
    }
    destroy() {}
    clearFilter() {}
  };

  new TableWrapper({
    container: new MockElement('div'),
    digest: { columns: ['row_id', 'name'], rows: [] },
    columnDefs: {
      name: { className: 'col-header', cellClassName: 'col-cell' },
    },
    inlineToolbar: false,
  });

  const nameCol = columns.find(c => c.key === 'name');
  eq({ className: nameCol.className, cellClassName: nameCol.cellClassName },
     { className: 'col-header', cellClassName: 'col-cell' },
     'TableWrapper forwards header and cell classes');
  window.SortableTable = previousSortableTable;
}

// TableWrapper.setData preserves pagination across background/cache refreshes.
{
  const wrapper = Object.create(TableWrapper.prototype);
  wrapper._tableEl = new MockElement('div');
  wrapper._table = {
    currentPage: 3,
    totalPages: 3,
    setData(rows) {
      this.rows = rows;
      this.totalPages = Math.ceil(rows.length / 25);
      this.currentPage = 1;
    },
    clearFilter() {
      this.currentFilter = null;
      this.totalPages = Math.ceil(this.rows.length / 25);
      this.currentPage = 1;
    },
    updateTable() { this.updated = true; },
  };
  wrapper._searchText = '';
  wrapper._externalFilter = null;
  wrapper._selectedRowId = null;
  wrapper._digestColumns = ['row_id'];

  wrapper.setData({ columns: ['row_id'], rows: Array.from({ length: 61 }, (_, i) => [i + 1]) });
  eq(wrapper._table.currentPage, 3, 'TableWrapper.setData preserves the current page when still valid');
  check(wrapper._table.updated, 'TableWrapper.setData re-renders after restoring the page');

  wrapper._table.updated = false;
  wrapper.setData({ columns: ['row_id'], rows: Array.from({ length: 26 }, (_, i) => [i + 1]) });
  eq(wrapper._table.currentPage, 2, 'TableWrapper.setData clamps the page when rows shrink');
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

// TableWrapper marks row-click-edit tables for shared pointer styling.
{
  const editable = new TableWrapper({
    container: new MockElement('div'),
    columnDefs: {},
    inlineToolbar: false,
    canModify: true,
    actions: { onEdit: () => {} },
  });
  check(
    editable._tableEl.classList.contains('table-scroll-editable-rows'),
    'TableWrapper marks editable row-click tables',
  );

  const deleteOnly = new TableWrapper({
    container: new MockElement('div'),
    columnDefs: {},
    inlineToolbar: false,
    canModify: true,
    actions: { onDelete: () => {} },
  });
  check(
    !deleteOnly._tableEl.classList.contains('table-scroll-editable-rows'),
    'TableWrapper does not mark delete-only tables as row-click editable',
  );
}

// TableWrapper places add actions in the toolbar, immediately after export.
{
  const previousSortableTable = window.SortableTable;
  window.SortableTable = class {
    constructor(opts) {
      this.data = opts.data;
      this.columns = opts.columns;
      this.currentSort = opts.sort || { column: 'name', ascending: true };
    }
    destroy() {}
    clearFilter() {}
  };

  const container = new MockElement('div');
  new TableWrapper({
    container,
    digest: { columns: ['row_id', 'name'], rows: [] },
    columnDefs: {},
    canModify: true,
    actions: { onAdd: () => {} },
    labels: { add: 'Aggiungi', exportCSV: 'Esporta' },
  });
  const page = container.children[0];
  const toolbar = page.children[0];
  eq(
    toolbar.children.map(child => child.textContent || child.className),
    ['Filter', 'table-search', 'Esporta', '+ Aggiungi'],
    'TableWrapper add button is inside toolbar after export',
  );
  check(
    !page.children.some(child => child.className === 'action-add'),
    'TableWrapper does not render a below-table add row',
  );
  window.SortableTable = previousSortableTable;
}

// TableWrapper uses standard action-column widths by action count.
{
  const created = [];
  const previousSortableTable = window.SortableTable;
  window.SortableTable = class {
    constructor(opts) {
      this.data = opts.data;
      this.columns = opts.columns;
      this.currentSort = opts.sort || { column: 'name', ascending: true };
      created.push(this);
    }
    destroy() {}
    clearFilter() {}
  };
  const digest = { columns: ['row_id', 'name'], rows: [] };

  new TableWrapper({
    container: new MockElement('div'), digest, columnDefs: {}, inlineToolbar: false,
    canModify: true, actions: { onDelete: () => {} },
  });
  new TableWrapper({
    container: new MockElement('div'), digest, columnDefs: {}, inlineToolbar: false,
    canModify: true, actions: { onEdit: () => {}, onDelete: () => {} },
  });
  new TableWrapper({
    container: new MockElement('div'), digest, columnDefs: {}, inlineToolbar: false,
    canModify: true,
    actions: {
      onEdit: () => {},
      onDelete: () => {},
      extra: [{
        key: 'inspect', title: 'Inspect', icon: '?',
        visible: row => row[1] === 'visible',
        onClick: () => {},
      }],
    },
  });

  eq(created.map(t => t.columns.find(c => c.key === '_actions')?.width),
     ['44px', '88px', '124px'],
     'TableWrapper uses standard action-column widths by action count');

  const actionColumn = created[2].columns.find(c => c.key === '_actions');
  check(typeof actionColumn.renderCell === 'function'
        && actionColumn.formatter === undefined
        && actionColumn.trustedHTML === undefined,
        'TableWrapper renders action columns through the DOM API');
  const actionCell = new MockElement('td');
  actionColumn.renderCell(actionCell, '', [7, 'visible']);
  eq(actionCell.children.map(element => ({
    className: element.className,
    actionKey: element.dataset.actionKey || null,
    title: element.title,
    text: element.textContent,
  })), [
    { className: 'action-icon action-edit', actionKey: null, title: 'Edit', text: '\u270E' },
    { className: 'action-icon action-extra', actionKey: 'inspect', title: 'Inspect', text: '?' },
    { className: 'action-icon action-delete', actionKey: null, title: 'Delete', text: '\u{1F5D1}\u{FE0E}' },
  ], 'TableWrapper builds action icons as DOM nodes');

  const filteredActionCell = new MockElement('td');
  actionColumn.renderCell(filteredActionCell, '', [8, 'hidden']);
  eq(filteredActionCell.children.map(element => element.className), [
    'action-icon action-edit',
    'action-icon action-delete',
  ], 'TableWrapper passes the full row to action visibility predicates');
  window.SortableTable = previousSortableTable;
}

// TableWrapper delegates plain row clicks to edit and keeps explicit actions distinct.
{
  const calls = [];
  const wrapper = Object.create(TableWrapper.prototype);
  wrapper._table = { data: [[10, 'Abete']] };
  wrapper.actions = {
    onEdit: (rowId) => calls.push(['edit', rowId]),
    onDelete: (rowId) => calls.push(['delete', rowId]),
    extra: [{
      key: 'mode',
      onClick: (rowId, row) => calls.push(['mode', rowId, row[1]]),
    }],
  };
  const rowEl = { dataset: { index: '0' } };
  const target = (matches = {}) => ({
    closest: (selector) => matches[selector] || (selector === '.sortable-table-row' ? rowEl : null),
  });
  const editIcon = {
    classList: { contains: (cls) => cls === 'action-edit' },
    closest: (selector) => selector === '.sortable-table-row' ? rowEl : null,
  };
  const deleteIcon = {
    classList: { contains: (cls) => cls === 'action-delete' },
    closest: (selector) => selector === '.sortable-table-row' ? rowEl : null,
  };
  const extraIcon = {
    classList: { contains: (cls) => cls === 'action-extra' },
    dataset: { actionKey: 'mode' },
    closest: (selector) => selector === '.sortable-table-row' ? rowEl : null,
  };

  wrapper._handleTableClick({ target: target() });
  wrapper._handleTableClick({ target: target({ '.action-icon': editIcon }) });
  wrapper._handleTableClick({ target: target({ '.action-icon': deleteIcon }) });
  wrapper._handleTableClick({ target: target({ '.action-icon': extraIcon }) });
  wrapper._handleTableClick({
    target: target({ '.action-icon,a,button,input,label,select,textarea,[contenteditable="true"],[role="button"]': {} }),
  });

  eq(JSON.stringify(calls), JSON.stringify([
    ['edit', 10], ['edit', 10], ['delete', 10], ['mode', 10, 'Abete'],
  ]), 'TableWrapper row clicks edit, action icons dispatch explicitly, and controls are ignored');
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
  const id = 'page_sync_test_stale_mount';
  cache.register(id, '/api/page-sync/stale-mount/');
  server.set('/api/page-sync/stale-mount/', {
    lastModified: '1',
    data: { columns: ['row_id'], rows: [[2]] },
  });

  let resolveLoad;
  const events = [];
  const page = createPage({
    visibleIds: [id],
    load: () => new Promise(resolve => { resolveLoad = resolve; }),
    mount(_el, _params, data) { events.push(['mount', data.rows[0][0]]); },
    unmount() { events.push(['unmount']); },
    onUpdate: [[id, data => events.push(['update', data.rows[0][0]])]],
  });

  const pendingMount = page.mount({});
  page.unmount();
  resolveLoad({ columns: ['row_id'], rows: [[1]] });
  await pendingMount;

  check(!events.some(e => e[0] === 'mount'), 'createPage suppresses stale mount after unmount');

  await cache.load(id);
  check(!events.some(e => e[0] === 'update'), 'createPage does not subscribe stale mounts to cache updates');
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

// createPage surfaces mount callback exceptions instead of leaving the spinner.
{
  const id = 'page_sync_test_mount_error';
  cache.register(id, '/api/page-sync/mount-error/');
  server.set('/api/page-sync/mount-error/', {
    lastModified: '1',
    data: { columns: ['row_id'], rows: [[1]] },
  });

  const page = createPage({
    dataIds: [id],
    mount() { throw new Error('boom'); },
  });

  await page.mount({});

  check(modalEl.classList.contains('open'), 'createPage shows an error modal when mount throws');
  page.unmount();
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
