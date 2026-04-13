/**
 * Table wrapper: search, CSV export, and action icons around SortableTable.
 *
 * Consumes digest format: { columns: string[], rows: any[][] }.
 * row_id is always the first column (hidden).
 */

import * as S from './strings.js';

const DEBOUNCE_MS = 500;
const ROW_ID_COL = 0;
const ROWS_PER_PAGE = 25;

/**
 * Wraps SortableTable with:
 *   - Search box (upper left, 500 ms debounce)
 *   - CSV export button (upper right)
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
   * @param {function(string, boolean): void} [opts.onSort]
   * @param {function(string): void} [opts.onSearch]
   */
  constructor(opts) {
    this.container = opts.container;
    this.columnDefs = opts.columnDefs || {};
    this.canModify = opts.canModify || false;
    this.actions = opts.actions || {};
    this.csvFilename = opts.csvFilename || 'export.csv';
    this.onSort = opts.onSort || null;
    this.onSearch = opts.onSearch || null;
    this._searchText = opts.searchText || '';
    this._externalFilter = null;
    this._debounceTimer = null;
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

    this._buildToolbar();

    this._tableEl = document.createElement('div');
    this._tableEl.className = 'table-scroll';
    this._el.appendChild(this._tableEl);

    if (this.canModify && this.actions.onAdd) {
      const row = document.createElement('div');
      row.className = 'action-add';
      const btn = document.createElement('button');
      btn.className = 'action-icon';
      btn.textContent = '+';
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

    const search = document.createElement('input');
    search.type = 'text';
    search.className = 'table-search';
    search.placeholder = S.SEARCH_PLACEHOLDER;
    search.value = this._searchText;
    search.addEventListener('input', () => {
      clearTimeout(this._debounceTimer);
      this._debounceTimer = setTimeout(() => {
        this._searchText = search.value;
        this._applyFilters();
        this.onSearch?.(this._searchText);
      }, DEBOUNCE_MS);
    });
    bar.appendChild(search);

    const csvBtn = document.createElement('button');
    csvBtn.className = 'btn btn-secondary table-csv-btn';
    csvBtn.textContent = S.EXPORT_CSV;
    csvBtn.addEventListener('click', () => this._exportCSV());
    bar.appendChild(csvBtn);

    this._el.appendChild(bar);
  }

  _initTable(digest, sort) {
    this._stColumns = buildSTColumns(digest.columns, this.columnDefs, this.actions);

    this._table = new window.SortableTable({
      container: this._tableEl,
      data: digest.rows,
      columns: this._stColumns,
      rowsPerPage: ROWS_PER_PAGE,
      sort: sort || undefined,
      emptyMessage: S.NO_RESULTS,
      onSort: (col, asc) => this.onSort?.(col, asc),
    });

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

  _exportCSV() {
    if (!this._table) return;

    const exportCols = this._stColumns
      .map((col, i) => ({ col, i }))
      .filter(({ col }) => !col.hidden && col.key !== '_actions');

    const header = exportCols.map(({ col }) => csvEscape(col.label)).join(';');
    const body = this._table.data.map(row =>
      exportCols.map(({ col, i }) => csvEscape(formatCSV(row[i], col.type))).join(';'),
    ).join('\n');

    downloadText('\ufeff' + header + '\n' + body, this.csvFilename);
  }
}

// ---------------------------------------------------------------------------
// Column builder (module-private)
// ---------------------------------------------------------------------------

function buildSTColumns(digestColumns, columnDefs, actions) {
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
      formatter: def.formatter || (def.type === 'boolean' ? formatBool : undefined),
      width: def.width,
      className: def.className,
    };
  });

  if (actions.onEdit || actions.onDelete) {
    const parts = [];
    if (actions.onEdit)
      parts.push('<span class="action-icon action-edit" title="Modifica">\u270E</span>');
    if (actions.onDelete)
      parts.push('<span class="action-icon action-delete" title="Elimina">\u2715</span>');
    const html = parts.join(' ');
    cols.push({
      key: '_actions', label: '', sortable: false,
      width: '56px', className: 'col-actions',
      formatter: () => html,
    });
  }

  return cols;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/** Ordered-term search: every term must appear in row text, in given order. */
function matchesSearch(row, terms) {
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

/** Format a boolean for display. */
function formatBool(value) {
  return value ? S.BOOL_YES : S.BOOL_NO;
}

/** Format a value for Italian CSV (comma decimals, DD/MM/YYYY dates). */
function formatCSV(value, type) {
  if (value == null || value === '') return '';
  if (type === 'boolean') return formatBool(value);
  if (type === 'number' && typeof value === 'number') {
    return String(value).replace('.', ',');
  }
  if (type === 'date' && typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)) {
    const [y, m, d] = value.split('-');
    return `${d}/${m}/${y}`;
  }
  return String(value);
}

/** Escape a semicolon-delimited CSV field. */
function csvEscape(value) {
  const s = String(value);
  return (s.includes(';') || s.includes('"') || s.includes('\n'))
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
