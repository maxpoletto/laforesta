/**
 * Controllo (audit) page: read-only sortable-table of change history.
 */

import * as cache from '../../base/js/cache.js';
import { TableWrapper } from '../../base/js/table.js';
import {
  applyTableState, createPage, navigateWithParams, readTableState,
  tableSort, writeTableState,
} from '../../base/js/page-sync.js';
import * as S from '../../base/js/strings.js';

const DATA_ID = 'audit';
const DATA_URL = '/api/controllo/data/';
const PAGE_PATH = '/controllo';
const CSS_URL = '/static/controllo/css/controllo.css';
const DEFAULT_SORT = { column: S.COL_TIMESTAMP, ascending: false };
const VALUE_COLUMN = {
  width: '420px',
  cellClassName: 'col-audit-value',
};

const COLUMN_DEFS = {
  [S.COL_TIMESTAMP]: { label: S.COL_TIMESTAMP, width: '150px' },
  [S.COL_USER]: { label: S.COL_USER, width: '150px' },
  [S.COL_TABLE]: { label: S.COL_TABLE, width: '170px' },
  [S.COL_ACTION]: { label: S.COL_ACTION, width: '120px' },
  [S.COL_OLD_VALUE]: { label: S.COL_OLD_VALUE, ...VALUE_COLUMN },
  [S.COL_NEW_VALUE]: { label: S.COL_NEW_VALUE, ...VALUE_COLUMN },
};

let table = null;

cache.register(DATA_ID, DATA_URL);

const page = createPage({
  cssUrl: CSS_URL,
  dataIds: [DATA_ID],
  mount: buildPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [[DATA_ID, data => table?.setData(data)]],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

function buildPage(el, params, data) {
  el.replaceChildren();

  const state = readTableState(params);
  table = new TableWrapper({
    container: el,
    digest: data,
    columnDefs: COLUMN_DEFS,
    canModify: false,
    sort: tableSort(state, DEFAULT_SORT),
    searchText: state.searchText,
    csvFilename: S.CSV_AUDIT,
    labels: S.TABLE_LABELS,
    csvFormat: S.TABLE_CSV_FORMAT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });
}

function destroyPage() {
  if (table) { table.destroy(); table = null; }
}

function applyParams(params) {
  applyTableState(table, readTableState(params), DEFAULT_SORT);
}

function syncURL() {
  const params = new URLSearchParams();
  writeTableState(params, table);
  navigateWithParams(PAGE_PATH, params);
}
