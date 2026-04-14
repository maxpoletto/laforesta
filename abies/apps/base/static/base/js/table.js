/**
 * Table wrapper: search, CSV export, and action icons around SortableTable.
 *
 * Consumes digest format: { columns: string[], rows: any[][] }.
 * row_id is always the first column (hidden).
 *
 * Locale-agnostic: all user-facing strings and CSV formatting defaults to
 * English.  Callers inject a `labels` and/or `csvFormat` option to localize.
 */

const DEBOUNCE_MS = 500;
const ROW_ID_COL = 0;
const ROWS_PER_PAGE = 25;
const DEFAULT_COL_WIDTH = '100';  // px fallback for columns without explicit width

/** English defaults for all user-facing strings. */
const DEFAULT_LABELS = {
  search: 'Filter',
  searchPlaceholder: 'Search...',
  exportCSV: 'Export CSV',
  add: 'Add',
  empty: 'No results.',
  actionEdit: 'Edit',
  actionDelete: 'Delete',
  boolYes: 'Yes',
  boolNo: 'No',
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
   *         onAdd?: function}} [opts.actions]
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
    this._table = null;
    this._el = null;
    this._tableEl = null;

    this._build(opts.digest, opts.sort);
  }

  // -- Public API ----------------------------------------------------------

  /** Replace digest data (e.g., after cache refresh). */
  setData(digest) {
    if (!this._table) return;
    this._table.setData(digest.rows);
    this._applyFilters();
  }

  /** Set an additional filter combined with search (e.g., year slider). */
  setExternalFilter(fn) {
    this._externalFilter = fn;
    this._applyFilters();
  }

  /** Current sort state. */
  getSort() {
    if (!this._table) return null;
    return {
      column: this._table.currentSort.column,
      ascending: this._table.currentSort.ascending,
    };
  }

  /** Current search text. */
  getSearchText() { return this._searchText; }

  /** Programmatically set search text (e.g., for reset). */
  setSearchText(text) {
    this._searchText = text;
    if (this._searchInputEl) this._searchInputEl.value = text;
    this._applyFilters();
  }

  /** Tear down DOM and timers. */
  destroy() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    if (this._table) this._table.destroy();
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
    this._el.appendChild(this._tableEl);

    if (this.canModify && this.actions.onAdd) {
      const row = document.createElement('div');
      row.className = 'action-add';
      const btn = document.createElement('button');
      btn.className = 'btn btn-primary btn-add';
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
    csvBtn.className = 'btn btn-primary table-csv-btn';
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
    this._stColumns = buildSTColumns(digest.columns, this.columnDefs, this.actions, this.labels);

    this._table = new window.SortableTable({
      container: this._tableEl,
      data: digest.rows,
      columns: this._stColumns,
      rowsPerPage: ROWS_PER_PAGE,
      sort: sort || undefined,
      emptyMessage: this.labels.empty,
      onSort: (col, asc) => this.onSort?.(col, asc),
    });

    // Set min-width so columns keep their specified widths and the wrapper
    // scrolls horizontally when the total exceeds the viewport.
    const totalWidth = this._stColumns.reduce((sum, col) => {
      if (col.hidden) return sum;
      return sum + parseInt(col.width || DEFAULT_COL_WIDTH, 10);
    }, 0);
    this._tableEl.style.setProperty('--st-table-min-width', totalWidth + 'px');

    // Action-icon delegation on the stable container element, avoiding
    // SortableTable's onRowClick which stacks listeners on re-render.
    if (this.canModify) {
      this._tableEl.addEventListener('click', (e) => {
        const icon = e.target.closest('.action-icon');
        if (!icon) return;
        const tr = icon.closest('.sortable-table-row');
        if (!tr) return;
        const rowData = this._table.data[parseInt(tr.dataset.index)];
        if (!rowData) return;
        const rowId = rowData[ROW_ID_COL];
        if (icon.classList.contains('action-edit')) this.actions.onEdit?.(rowId);
        else if (icon.classList.contains('action-delete')) this.actions.onDelete?.(rowId);
      });
    }

    if (this._searchText) this._applyFilters();
  }

  // -- Filtering -----------------------------------------------------------

  _applyFilters() {
    if (!this._table) return;
    const terms = this._searchText.trim().toLowerCase().split(/\s+/).filter(Boolean);

    if (!terms.length && !this._externalFilter) {
      this._table.clearFilter();
      return;
    }

    this._table.filter(row => {
      if (this._externalFilter && !this._externalFilter(row)) return false;
      return terms.length === 0 || matchesSearch(row, terms);
    });
  }

  // -- CSV export ----------------------------------------------------------

  /** Export currently-loaded rows as a CSV using the configured csvFormat. */
  exportCSV() {
    if (!this._table) return;

    const fmt = this.csvFormat;
    const sep = fmt.separator;

    const exportCols = this._stColumns
      .map((col, i) => ({ col, i }))
      .filter(({ col }) => !col.hidden && col.key !== '_actions');

    const header = exportCols.map(({ col }) => csvEscape(col.label, sep)).join(sep);
    const body = this._table.data.map(row =>
      exportCols.map(({ col, i }) => csvEscape(formatCSV(row[i], col.type, fmt, this.labels), sep)).join(sep),
    ).join('\n');

    downloadText('\ufeff' + header + '\n' + body, this.csvFilename);
  }
}

TableWrapper._idSeq = 0;

// ---------------------------------------------------------------------------
// Column builder (module-private)
// ---------------------------------------------------------------------------

function buildSTColumns(digestColumns, columnDefs, actions, labels) {
  const cols = digestColumns.map(name => {
    if (name === 'row_id') {
      return { key: 'row_id', label: 'ID', type: 'number', hidden: true };
    }
    const def = columnDefs[name] || {};
    return {
      key: name,
      label: def.label ?? name,
      type: def.type ?? 'string',
      hidden: def.hidden || false,
      formatter: def.formatter || (def.type === 'boolean' ? (v) => formatBool(v, labels) : undefined),
      width: def.width,
      className: def.className,
    };
  });

  if (actions.onEdit || actions.onDelete) {
    const parts = [];
    if (actions.onEdit)
      parts.push(`<span class="action-icon action-edit" title="${escAttr(labels.actionEdit)}">\u270E</span>`);
    if (actions.onDelete)
      parts.push(`<span class="action-icon action-delete" title="${escAttr(labels.actionDelete)}">\u2715</span>`);
    const html = parts.join(' ');
    cols.push({
      key: '_actions', label: '', sortable: false,
      width: '61px', className: 'col-actions',
      formatter: () => html,
    });
  }

  return cols;
}

/** Minimal HTML attribute-value escape (we build title="..." by concatenation). */
function escAttr(s) {
  return String(s).replace(/[&"]/g, c => (c === '&' ? '&amp;' : '&quot;'));
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/** Ordered-term search: every term must appear in row text, in given order. */
export function matchesSearch(row, terms) {
  const text = row.map(v => String(v ?? '')).join(' ').toLowerCase();
  let pos = 0;
  for (const t of terms) {
    const idx = text.indexOf(t, pos);
    if (idx < 0) return false;
    pos = idx + t.length;
  }
  return true;
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

/** Escape a CSV field for the given separator. */
function csvEscape(value, separator) {
  const s = String(value);
  return (s.includes(separator) || s.includes('"') || s.includes('\n'))
    ? '"' + s.replace(/"/g, '""') + '"'
    : s;
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
