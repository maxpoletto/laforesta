/**
 * Controllo (audit) page: read-only sortable-table of change history.
 */

import * as cache from '../../base/js/cache.js';
import { TableWrapper } from '../../base/js/table.js';
import { showError } from '../../base/js/modals.js';
import * as router from '../../base/js/router.js';
import * as S from '../../base/js/strings.js';

const DATA_ID = 'audit';
const DATA_URL = '/abies/api/controllo/data/';
const PAGE_PATH = '/abies/controllo';

const COLUMN_DEFS = {
  [S.COL_TIMESTAMP]: { label: S.COL_TIMESTAMP },
  [S.COL_USER]: { label: S.COL_USER },
  [S.COL_TABLE]: { label: S.COL_TABLE },
  [S.COL_ACTION]: { label: S.COL_ACTION },
  [S.COL_OLD_VALUE]: { label: S.COL_OLD_VALUE },
  [S.COL_NEW_VALUE]: { label: S.COL_NEW_VALUE },
};

let table = null;
let unsubCache = null;

cache.register(DATA_ID, DATA_URL);

// ---------------------------------------------------------------------------
// Page lifecycle
// ---------------------------------------------------------------------------

export async function mount(params) {
  const el = document.getElementById('content');
  el.replaceChildren();

  const loading = document.createElement('div');
  loading.className = 'loading-overlay';
  loading.textContent = S.LOADING;
  el.appendChild(loading);

  let data;
  try {
    data = await cache.load(DATA_ID);
  } catch {
    showError(S.ERROR_NETWORK);
    return;
  }

  el.replaceChildren();

  const p = readParams(params);
  const sort = p.sc
    ? { column: p.sc, ascending: p.so }
    : { column: S.COL_TIMESTAMP, ascending: false };

  table = new TableWrapper({
    container: el,
    digest: data,
    columnDefs: COLUMN_DEFS,
    canModify: false,
    sort,
    searchText: p.f,
    csvFilename: S.CSV_AUDIT,
    onSort: () => syncURL(),
    onSearch: () => syncURL(),
  });

  cache.setVisible([DATA_ID]);
  unsubCache = cache.onUpdate(DATA_ID, () => {
    if (table) table.setData(cache.get(DATA_ID));
  });
}

export function unmount() {
  if (unsubCache) { unsubCache(); unsubCache = null; }
  if (table) { table.destroy(); table = null; }
  cache.setVisible([]);
}

export function onQueryChange(params) {
  // Sort and search are applied at construction; nothing to update live.
}

// ---------------------------------------------------------------------------
// URL sync
// ---------------------------------------------------------------------------

function readParams(params) {
  return {
    sc: params.sc || null,
    so: params.so !== undefined ? params.so === '0' : true,
    f: params.f || '',
  };
}

function syncURL() {
  const params = new URLSearchParams();
  if (table) {
    const sort = table.getSort();
    if (sort) {
      params.set('sc', sort.column);
      params.set('so', sort.ascending ? '0' : '1');
    }
    const f = table.getSearchText();
    if (f) params.set('f', f);
  }
  const qs = params.toString();
  router.navigate(PAGE_PATH + (qs ? '?' + qs : ''), true);
}
