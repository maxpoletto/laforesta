/**
 * Table wrapper: search, CSV export, and action icons around SortableTable.
 *
 * Consumes digest format: { columns: string[], rows: any[][] }.
 * row_id is always the first column (hidden).
 *
 * Locale-agnostic: all user-facing strings and CSV formatting defaults to
 * English.  Callers inject a `labels` and/or `csvFormat` option to localize.
 */

import { ROW_ID } from './constants.js';
import { csvEscape } from './csv-export.js';

const DEBOUNCE_MS = 500;
const ROW_ID_COL = 0;
const ROWS_PER_PAGE = 25;
const DEFAULT_COL_WIDTH = '100';  // px fallback for columns without explicit width
const ACTION_COLUMN_SINGLE_WIDTH_PX = 44;
const ACTION_COLUMN_ICON_SLOT_WIDTH_PX = 36;
const ACTION_COLUMN_HORIZONTAL_PADDING_PX = 16;
const ROW_CLICK_IGNORE_SELECTOR = [
  '.action-icon',
  'a',
  'button',
  'input',
  'label',
  'select',
  'textarea',
  '[contenteditable="true"]',
  '[role="button"]',
].join(',');

function sameColumns(a = [], b = []) {
  if (a.length !== b.length) return false;
  return a.every((value, index) => value === b[index]);
}

/** English defaults for all user-facing strings. */
const DEFAULT_LABELS = {
  search: 'Filter',
  searchPlaceholder: 'Search...',
  exportCSV: 'Export',
  add: 'Add',
  empty: 'No results.',
  actionEdit: 'Edit',
  actionEditIcon: '\u270E',
  actionDelete: 'Delete',
  boolYes: 'Yes',
  boolNo: 'No',
  // Pagination template forwarded to SortableTable, which fills the
  // {current}/{total} placeholders with the page-number controls.
  pageInfo: 'Page {current} of {total}',
};

/**
 * CSV output format.  Defaults produce a locale-neutral CSV:
 *   - comma separator, period decimal
 *   - ISO-8601 dates (YYYY-MM-DD)
 * `dateFormat` is a template whose YYYY/MM/DD tokens are substituted from
 * incoming ISO date strings.
 */
const DEFAULT_CSV_FORMAT = {
  separator: ',',
  decimal: '.',
  dateFormat: 'YYYY-MM-DD',
};

/**
 * Wraps SortableTable with:
 *   - Built-in toolbar (search + CSV export), or external wiring for
 *     page-level filter bars via wireSearchInput() / exportCSV()
 *   - Edit/delete action icons per row (writers/admins)
 *   - Add button below table (writers/admins)
 */
export class TableWrapper {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container — element to render into
   * @param {{columns: string[], rows: any[][]}} [opts.digest]
   * @param {Object<string, {label: string, type?: string, formatter?: Function,
   *         width?: string, className?: string}>} opts.columnDefs
   *   Column metadata keyed by digest column name.
   * @param {boolean} [opts.canModify]
   * @param {{onEdit?: function(number), onDelete?: function(number),
   *         onAdd?: function, extra?: Array<{key: string, title: string,
   *         icon: string, onClick: function(number, Array), visible?: function(Array): boolean}>,
   *         editVisible?: function(Array): boolean,
   *         deleteVisible?: function(Array): boolean}} [opts.actions]
   * @param {{column: string, ascending: boolean}} [opts.sort]
   * @param {string} [opts.searchText]
   * @param {string} [opts.csvFilename]
   * @param {boolean} [opts.inlineToolbar=true] — when false, the built-in
   *   search box and CSV button are omitted.  The caller is responsible for
   *   providing them and wiring them via wireSearchInput() and exportCSV().
   * @param {Partial<typeof DEFAULT_LABELS>} [opts.labels] — overrides for the
   *   English defaults (see DEFAULT_LABELS).  Any keys omitted fall back to
   *   the English defaults.
   * @param {Partial<typeof DEFAULT_CSV_FORMAT>} [opts.csvFormat] — overrides
   *   for the default CSV format (see DEFAULT_CSV_FORMAT).
   * @param {function(string, boolean): void} [opts.onSort]
   * @param {function(string): void} [opts.onSearch]
   */
  constructor(opts) {
    this.container = opts.container;
    this.columnDefs = opts.columnDefs || {};
    this.canModify = opts.canModify || false;
    this.actions = opts.actions || {};
    this.csvFilename = opts.csvFilename || 'export.csv';
    this.inlineToolbar = opts.inlineToolbar !== false;
    this.labels = { ...DEFAULT_LABELS, ...(opts.labels || {}) };
    this.csvFormat = { ...DEFAULT_CSV_FORMAT, ...(opts.csvFormat || {}) };
    this.onSort = opts.onSort || null;
    this.onSearch = opts.onSearch || null;
    this._searchText = opts.searchText || '';
    this._externalFilter = null;
    this._debounceTimer = null;
    this._searchInputEl = null;
    this._stColumns = [];
    this._digestColumns = [];
    this._table = null;
    this._el = null;
    this._tableEl = null;
    this._selectedRowId = null;
    this._tableClickHandler = (e) => this._handleTableClick(e);
    this._tableClickWired = false;

    this._build(opts.digest, opts.sort);
  }

  // -- Public API ----------------------------------------------------------

  /** Replace digest data (e.g., after cache refresh). */
  setData(digest, columnDefs = null) {
    if (!this._table) return;
    if (columnDefs) this.columnDefs = columnDefs;
    const previousPage = this._table.currentPage || 1;
    if (!sameColumns(digest.columns, this._digestColumns)) {
      const previousSort = this.getSort();
      this._table.destroy();
      this._tableEl.replaceChildren();
      this._table = null;
      this._initTable(digest, previousSort);
      this._applyFilters();
      this._restorePage(previousPage);
      return;
    }
    this._table.setData(digest.rows);
    this._applyFilters();
    this._restorePage(previousPage);
  }

  /** Set an additional filter combined with search (e.g., year slider). */
  setExternalFilter(fn) {
    this._externalFilter = fn;
    this._applyFilters();
  }

  /** Current sort state. */
  getSort() {
    if (!this._table?.currentSort) return null;
    return {
      column: this._table.currentSort.column,
      ascending: this._table.currentSort.ascending,
    };
  }

  /** Programmatically set sort without firing the onSort callback. */
  setSort(sort) {
    if (!this._table || !sort?.column) return;
    const current = this.getSort();
    if (current?.column === sort.column && current.ascending === sort.ascending) return;
    const col = this._stColumns.find(c => c.key === sort.column);
    if (!col) return;
    const onSort = this._table.onSort;
    this._table.onSort = null;
    this._table.sort(sort.column, col.type || 'string', sort.ascending);
    this._table.onSort = onSort;
    this._applySelectedRow();
  }

  /** Current search text. */
  getSearchText() { return this._searchText; }

  /** Column defs (with formatters) for building the search haystack. */
  get searchColumns() { return this._stColumns; }

  /** Programmatically set search text (e.g., for reset). */
  setSearchText(text) {
    this._searchText = text;
    if (this._searchInputEl) this._searchInputEl.value = text;
    this._applyFilters();
  }

  /** Data row backing a rendered SortableTable row element. */
  rowForElement(rowEl) {
    if (!this._table || !rowEl?.dataset) return null;
    const index = parseInt(rowEl.dataset.index, 10);
    return Number.isInteger(index) ? this._table.data[index] || null : null;
  }

  /** Highlight the rendered row with this row_id, if it is currently visible. */
  setSelectedRow(rowId) {
    this._selectedRowId = rowId == null ? null : rowId;
    this._applySelectedRow();
  }

  /** Tear down DOM and timers. */
  destroy() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    if (this._table) this._table.destroy();
    if (this._tableClickWired) {
      this._tableEl?.removeEventListener('click', this._tableClickHandler);
      this._tableClickWired = false;
    }
    this._el?.remove();
    this._table = null;
  }

  // -- Build ---------------------------------------------------------------

  _build(digest, sort) {
    this._el = document.createElement('div');
    this._el.className = 'table-page';

    if (this.inlineToolbar) this._buildToolbar();

    this._tableEl = document.createElement('div');
    this._tableEl.className = 'table-scroll';
    if (this.canModify && this.actions.onEdit) {
      this._tableEl.classList.add('table-scroll-editable-rows');
    }
    this._el.appendChild(this._tableEl);

    if (this.canModify && this.actions.onAdd) {
      const row = document.createElement('div');
      row.className = 'action-add';
      const btn = document.createElement('button');
      btn.className = 'btn btn-create btn-add';
      btn.textContent = '+ ' + this.labels.add;
      btn.addEventListener('click', () => this.actions.onAdd());
      row.appendChild(btn);
      this._el.appendChild(row);
    }

    this.container.appendChild(this._el);
    if (digest) this._initTable(digest, sort);
  }

  _buildToolbar() {
    const bar = document.createElement('div');
    bar.className = 'table-toolbar';

    const label = document.createElement('label');
    label.className = 'table-search-label';
    label.textContent = this.labels.search;
    bar.appendChild(label);

    const search = document.createElement('input');
    search.type = 'text';
    search.className = 'table-search';
    search.placeholder = this.labels.searchPlaceholder;
    this.wireSearchInput(search);
    label.htmlFor = search.id = 'table-search-' + (++TableWrapper._idSeq);
    bar.appendChild(search);

    const csvBtn = document.createElement('button');
    csvBtn.className = 'btn btn-export ms-auto';
    csvBtn.textContent = this.labels.exportCSV;
    csvBtn.addEventListener('click', () => this.exportCSV());
    bar.appendChild(csvBtn);

    this._el.appendChild(bar);
  }

  /**
   * Attach debounced search handling to an external input element.
   * Use this when the search box lives outside the table's own toolbar
   * (e.g., a page-level filter bar).  The input's value is initialized
   * from the current search text, and setSearchText() will keep it in sync.
   */
  wireSearchInput(inputEl) {
    this._searchInputEl = inputEl;
    inputEl.value = this._searchText;
    inputEl.addEventListener('input', () => {
      clearTimeout(this._debounceTimer);
      this._debounceTimer = setTimeout(() => {
        this._searchText = inputEl.value;
        this._applyFilters();
        this.onSearch?.(this._searchText);
      }, DEBOUNCE_MS);
    });
  }

  _initTable(digest, sort) {
    this._digestColumns = [...(digest.columns || [])];
    this._stColumns = buildSTColumns(digest.columns, this.columnDefs, this.actions, this.labels);
    const initialSort = sort && this._stColumns.some(c => c.key === sort.column)
      ? sort
      : undefined;

    this._table = new window.SortableTable({
      container: this._tableEl,
      data: digest.rows,
      columns: this._stColumns,
      rowsPerPage: ROWS_PER_PAGE,
      sort: initialSort,
      emptyMessage: this.labels.empty,
      pageInfo: this.labels.pageInfo,
      onSort: (col, asc) => {
        this._applySelectedRow();
        this.onSort?.(col, asc);
      },
      onPageChange: () => this._applySelectedRow(),
    });

    // Set min-width so columns keep their specified widths and the wrapper
    // scrolls horizontally when the total exceeds the viewport.
    const totalWidth = this._stColumns.reduce((sum, col) => {
      if (col.hidden) return sum;
      return sum + parseInt(col.width || DEFAULT_COL_WIDTH, 10);
    }, 0);
    this._tableEl.style.setProperty('--st-table-min-width', totalWidth + 'px');

    // Row-action delegation on the stable container element, avoiding
    // SortableTable's onRowClick which stacks listeners on re-render.
    if (this.canModify && hasRowActions(this.actions) && !this._tableClickWired) {
      this._tableEl.addEventListener('click', this._tableClickHandler);
      this._tableClickWired = true;
    }

    if (this._searchText) this._applyFilters();
    else this._applySelectedRow();
  }

  _handleTableClick(e) {
    const icon = e.target.closest('.action-icon');
    const rowTarget = icon || e.target;
    const tr = rowTarget.closest('.sortable-table-row');
    if (!tr) return;
    const rowData = this.rowForElement(tr);
    if (!rowData) return;
    const rowId = rowData[ROW_ID_COL];

    if (icon) {
      if (icon.classList.contains('action-edit')) this.actions.onEdit?.(rowId, rowData);
      else if (icon.classList.contains('action-delete')) this.actions.onDelete?.(rowId, rowData);
      else if (icon.classList.contains('action-extra')) {
        const action = extraActions(this.actions)
          .find(a => a.key === icon.dataset.actionKey);
        action?.onClick?.(rowId, rowData);
      }
      return;
    }

    if (this.actions.onEdit && !e.target.closest(ROW_CLICK_IGNORE_SELECTOR)) {
      this.actions.onEdit(rowId, rowData);
    }
  }

  // -- Filtering -----------------------------------------------------------

  _applyFilters() {
    if (!this._table) return;
    const terms = searchTerms(this._searchText);

    if (!terms.length && !this._externalFilter) {
      this._table.clearFilter();
      this._applySelectedRow();
      return;
    }

    this._table.filter(row => {
      if (this._externalFilter && !this._externalFilter(row)) return false;
      return terms.length === 0 || matchesSearch(row, terms, this._stColumns);
    });
    this._applySelectedRow();
  }

  _applySelectedRow() {
    if (!this._tableEl || !this._table) return;
    for (const rowEl of this._tableEl.querySelectorAll('.sortable-table-row')) {
      const row = this.rowForElement(rowEl);
      const selected = this._selectedRowId != null && row?.[ROW_ID_COL] === this._selectedRowId;
      rowEl.classList.toggle('is-selected', selected);
      if (selected) rowEl.setAttribute('aria-selected', 'true');
      else rowEl.removeAttribute('aria-selected');
    }
  }

  _restorePage(page) {
    if (!this._table || !Number.isFinite(page)) return;
    const totalPages = Math.max(1, this._table.totalPages || 1);
    const nextPage = Math.min(Math.max(1, page), totalPages);
    if (this._table.currentPage === nextPage) return;
    this._table.currentPage = nextPage;
    this._table.updateTable?.();
    this._applySelectedRow();
  }

  // -- CSV export ----------------------------------------------------------

  /**
   * Build the current rows as a CSV string using the configured csvFormat.
   * Includes a UTF-8 BOM for Excel compatibility.  Returns '' if the table
   * has no data yet.
   */
  getCSV() {
    if (!this._table) return '';

    const fmt = this.csvFormat;
    const sep = fmt.separator;

    const exportCols = this._stColumns
      .map((col, i) => ({ col, i }))
      .filter(({ col }) => !col.hidden && col.key !== '_actions');

    const header = exportCols.map(({ col }) => csvEscape(col.label, sep)).join(sep);
    const body = this._table.data.map(row =>
      exportCols.map(({ col, i }) => csvEscape(formatCSV(row[i], col.type, fmt, this.labels), sep)).join(sep),
    ).join('\n');

    return '\ufeff' + header + '\n' + body;
  }

  /** Export currently-loaded rows as a CSV file. */
  exportCSV() {
    const text = this.getCSV();
    if (text) downloadText(text, this.csvFilename);
  }
}

TableWrapper._idSeq = 0;

// ---------------------------------------------------------------------------
// Column builder (module-private)
// ---------------------------------------------------------------------------

function hasRowActions(actions) {
  return !!(actions.onEdit || actions.onDelete || extraActions(actions).length);
}

function extraActions(actions) {
  return Array.isArray(actions.extra) ? actions.extra : [];
}

function actionCount(actions) {
  return (actions.onEdit ? 1 : 0) + extraActions(actions).length + (actions.onDelete ? 1 : 0);
}

function actionColumnWidth(count) {
  const width = count <= 1
    ? ACTION_COLUMN_SINGLE_WIDTH_PX
    : ACTION_COLUMN_HORIZONTAL_PADDING_PX + ACTION_COLUMN_ICON_SLOT_WIDTH_PX * count;
  return `${width}px`;
}

function appendActionIcon(cell, className, title, icon, actionKey = null) {
  const element = document.createElement('span');
  element.className = `action-icon ${className}`;
  element.title = String(title);
  element.textContent = String(icon);
  if (actionKey != null) element.dataset.actionKey = String(actionKey);
  cell.appendChild(element);
}

function renderRowActions(cell, actions, labels, row) {
  if (actions.onEdit && actionVisible(actions.editVisible, row)) {
    appendActionIcon(
      cell, 'action-edit', labels.actionEdit, labels.actionEditIcon,
    );
  }
  for (const action of extraActions(actions)) {
    if (!actionVisible(action.visible, row)) continue;
    appendActionIcon(
      cell, 'action-extra', action.title, action.icon, action.key,
    );
  }
  if (actions.onDelete && actionVisible(actions.deleteVisible, row)) {
    appendActionIcon(
      cell, 'action-delete', labels.actionDelete, '\u{1F5D1}\u{FE0E}',
    );
  }
}

function actionVisible(fn, row) {
  return !fn || fn(row) !== false;
}

function buildSTColumns(digestColumns, columnDefs, actions, labels) {
  const cols = digestColumns.map(name => {
    if (name === ROW_ID) {
      return { key: ROW_ID, label: 'ID', type: 'number', hidden: true };
    }
    const def = columnDefs[name] || {};
    return {
      key: name,
      label: def.label ?? name,
      type: def.type ?? 'string',
      hidden: def.hidden || false,
      formatter: def.formatter || (def.type === 'boolean' ? (v) => formatBool(v, labels) : undefined),
      searchFormatter: def.searchFormatter,
      width: def.width,
      className: def.className,
    };
  });

  if (hasRowActions(actions)) {
    const count = actionCount(actions);
    cols.push({
      key: '_actions', label: '', sortable: false,
      width: actionColumnWidth(count),
      className: 'col-actions',
      renderCell: (cell, _value, row) => {
        renderRowActions(cell, actions, labels, row);
      },
    });
  }

  return cols;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/**
 * Search over a row's *displayed* text: every plain term must appear
 * somewhere, independent of order.  With `columns` (the column defs), each
 * cell is rendered through its formatter so the haystack matches what the user
 * sees — the it-locale "3,14", not the raw "3.14" — and hidden columns
 * are excluded.  Columns may add row-specific search text via
 * `searchFormatter(value, row, col, index)`.
 *
 * Terms of the form `column:criterion` search a single visible column.
 * An exact key/label match wins; otherwise `column` must be a unique
 * case-insensitive key/label substring.  Quote selectors that contain spaces,
 * e.g. `"Anno previsto":2027`.  Literal criteria search that column's
 * displayed/search text.
 * Numeric columns also support `>N` and `<N`.  Ambiguous/missing columns,
 * malformed comparisons, and comparisons against non-numeric columns do not
 * match.
 *
 * Without `columns`, plain terms fall back to raw stringified row values, and
 * column terms cannot match.
 */
export function searchTerms(text) {
  const terms = [];
  let current = '';
  let inQuote = false;
  for (const char of String(text ?? '')) {
    if (char === '"') {
      inQuote = !inQuote;
    } else if (/\s/.test(char) && !inQuote) {
      if (current) {
        terms.push(current.toLowerCase());
        current = '';
      }
    } else {
      current += char;
    }
  }
  if (current) terms.push(current.toLowerCase());
  return terms;
}

export function matchesSearch(row, terms, columns = null) {
  const parsed = parseSearchTerms(Array.isArray(terms) ? terms : searchTerms(terms));

  if (parsed.plain.length) {
    const text = rowSearchText(row, columns);
    if (!parsed.plain.every(t => text.includes(t))) return false;
  }
  for (const term of parsed.column) {
    if (!matchesColumnSearch(row, columns, term)) return false;
  }
  return true;
}

function parseSearchTerms(terms) {
  const parsed = { plain: [], column: [] };
  for (const raw of terms) {
    const term = String(raw ?? '').trim().toLowerCase();
    if (!term) continue;
    const i = term.indexOf(':');
    if (i > 0 && i < term.length - 1) {
      parsed.column.push({ selector: term.slice(0, i), criterion: term.slice(i + 1) });
    } else {
      parsed.plain.push(term);
    }
  }
  return parsed;
}

function rowSearchText(row, columns) {
  return row.map((v, i) => cellSearchText(v, row, columns?.[i], i)).join(' ').toLowerCase();
}

function cellSearchText(value, row, col, index) {
  if (col?.hidden || col?.key === '_actions') return '';
  const display = col?.formatter ? String(col.formatter(value, row, col, index) ?? '') : String(value ?? '');
  const extra = col?.searchFormatter ? String(col.searchFormatter(value, row, col, index) ?? '') : '';
  return extra ? `${display} ${extra}` : display;
}

function matchesColumnSearch(row, columns, term) {
  if (!columns) return false;
  const match = findColumnMatch(columns, term.selector);
  if (!match) return false;
  const { col, index } = match;
  const criterion = term.criterion;
  const comparison = criterion.match(/^([<>])(.+)$/);
  if (comparison) {
    if (col.type !== 'number') return false;
    const value = parseSearchNumber(row[index]);
    const threshold = parseSearchNumber(comparison[2]);
    if (!Number.isFinite(value) || !Number.isFinite(threshold)) return false;
    return comparison[1] === '>' ? value > threshold : value < threshold;
  }
  if (criterion.includes('>') || criterion.includes('<')) return false;
  return cellSearchText(row[index], row, col, index).toLowerCase().includes(criterion);
}

function findColumnMatch(columns, selector) {
  const exact = [];
  const partial = [];
  for (const [index, col] of columns.entries()) {
    if (!col || col.hidden || col.key === '_actions') continue;
    const key = String(col.key ?? '').toLowerCase();
    const label = String(col.label ?? '').toLowerCase();
    const match = { col, index };
    if (key === selector || label === selector) {
      exact.push(match);
    } else if (key.includes(selector) || label.includes(selector)) {
      partial.push(match);
    }
  }
  if (exact.length === 1) return exact[0];
  if (exact.length > 1) return null;
  return partial.length === 1 ? partial[0] : null;
}

function parseSearchNumber(value) {
  if (typeof value === 'number') return value;
  const text = String(value ?? '').trim();
  if (!text) return NaN;
  const normalized = text.includes(',') && !text.includes('.')
    ? text.replace(',', '.')
    : text;
  return Number(normalized);
}

// ---------------------------------------------------------------------------
// CSV helpers
// ---------------------------------------------------------------------------

/** Format a boolean for display using the configured yes/no labels. */
function formatBool(value, labels) {
  return value ? labels.boolYes : labels.boolNo;
}

/** Format a value for CSV using the configured decimal and date formats. */
function formatCSV(value, type, csvFormat, labels) {
  if (value == null || value === '') return '';
  if (type === 'boolean') return formatBool(value, labels);
  if (type === 'number' && typeof value === 'number') {
    return csvFormat.decimal === '.' ? String(value) : String(value).replace('.', csvFormat.decimal);
  }
  if (type === 'date' && typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)) {
    const [YYYY, MM, DD] = value.split('-');
    const tokens = { YYYY, MM, DD };
    return csvFormat.dateFormat.replace(/YYYY|MM|DD/g, t => tokens[t]);
  }
  return String(value);
}

/** Trigger browser download of a text file. */
function downloadText(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
