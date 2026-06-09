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
const DEFAULT_SORT = { column: S.COL_TIMESTAMP, ascending: false };

const COLUMN_DEFS = {
  [S.COL_TIMESTAMP]: { label: S.COL_TIMESTAMP },
  [S.COL_USER]: { label: S.COL_USER },
  [S.COL_TABLE]: { label: S.COL_TABLE },
  [S.COL_ACTION]: { label: S.COL_ACTION },
  [S.COL_OLD_VALUE]: { label: S.COL_OLD_VALUE },
  [S.COL_NEW_VALUE]: { label: S.COL_NEW_VALUE },
};

let table = null;

cache.register(DATA_ID, DATA_URL);

const page = createPage({
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
