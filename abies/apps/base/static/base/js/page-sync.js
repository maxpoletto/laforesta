import * as cache from './cache.js';
import * as router from './router.js';
import { showError } from './modals.js';
import { loadCSS, unloadCSS } from './page-css.js';
import * as S from './strings.js';
import { showLoadingIn } from './ui-widgets.js';

export const tableParamKeys = (prefix = '') => ({
  sortColumn: `${prefix}sc`, sortOrder: `${prefix}so`, search: `${prefix}f`,
});
export function readTableState(params, keys = tableParamKeys()) {
  const sortColumn = paramValue(params, keys.sortColumn) || null;
  const sortOrder = paramValue(params, keys.sortOrder);
  return {
    sort: sortColumn ? { column: sortColumn, ascending: sortOrder !== '1' } : null,
    searchText: paramValue(params, keys.search) || '',
  };
}
export const tableSort = (state, fallback) => state?.sort || fallback || undefined;
export function writeTableState(params, table, keys = tableParamKeys()) {
  if (!table) return;
  const sort = table.getSort();
  if (sort) {
    params.set(keys.sortColumn, sort.column);
    params.set(keys.sortOrder, sort.ascending ? '0' : '1');
  }
  const search = table.getSearchText();
  if (search) params.set(keys.search, search);
}
export function applyTableState(table, state, fallbackSort) {
  if (!table) return false;
  let changed = false;
  const nextSort = tableSort(state, fallbackSort);
  const currentSort = table.getSort();
  if (nextSort && (currentSort?.column !== nextSort.column ||
      currentSort.ascending !== nextSort.ascending)) {
    table.setSort(nextSort);
    changed = true;
  }
  if (table.getSearchText() !== state.searchText) {
    table.setSearchText(state.searchText);
    changed = true;
  }
  return changed;
}
export function navigateWithParams(path, params, replace = true) {
  const qs = params.toString();
  router.navigate(path + (qs ? '?' + qs : ''), replace);
}
export function createPage({
  cssUrl = null, dataIds = [], visibleIds = dataIds, load = null,
  mount, unmount = null, onQueryChange = null, onUpdate = [],
}) {
  let unsubscribers = [];
  return {
    async mount(params) {
      if (cssUrl) loadCSS(cssUrl);
      const el = document.getElementById('content');
      showLoadingIn(el);
      let data;
      try {
        data = load ? await load() : await loadDataIds(dataIds);
      } catch {
        showError(S.ERROR_NETWORK);
        return;
      }
      mount(el, params, data);
      cache.setVisible(visibleIds);
      unsubscribers = onUpdate.map(([dataId, callback]) =>
        cache.onUpdate(dataId, () => callback(cache.get(dataId))),
      );
    },
    unmount() {
      if (cssUrl) unloadCSS(cssUrl);
      for (const unsub of unsubscribers) unsub();
      unsubscribers = [];
      unmount?.();
      cache.setVisible([]);
    },
    onQueryChange(params) { onQueryChange?.(params); },
  };
}
async function loadDataIds(dataIds) {
  return dataIds.length === 1
    ? cache.load(dataIds[0])
    : Promise.all(dataIds.map(id => cache.load(id)));
}
function paramValue(params, key) {
  return params instanceof URLSearchParams ? params.get(key) : params[key];
}
