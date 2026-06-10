/**
 * Bosco page: map shell and URL-backed controls.
 */

import * as cache from '../../base/js/cache.js';
import * as S from '../../base/js/strings.js';
import { COLUMNS, ROWS } from '../../base/js/constants.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { sortFeaturesByArea, parcelNames } from '../../base/js/geo.js';
import { ParcelMap } from '../../base/js/parcel-map.js';
import { createPage, navigateWithParams } from '../../base/js/page-sync.js';
import {
  clearMapView, mapTypeToken, readBoscoParams, writeMapView,
} from './bosco-state.js';

const CSS_URL = '/static/bosco/css/bosco.css';
const PAGE_PATH = '/bosco';

const PARCELS_ID = 'parcels';
const PARCELS_URL = '/api/bosco/parcels/data/';
const TERRENI_ID = 'terreni';
const TERRENI_GEOJSON_URL = '/api/geo/terreni.geojson';

cache.register(PARCELS_ID, PARCELS_URL);
cache.register(TERRENI_ID, TERRENI_GEOJSON_URL);

let root = null;
let mapHost = null;
let regionSelect = null;
let modeGroup = null;
let statusEl = null;
let parcelsData = null;
let parcelsGeo = null;
let map = null;
let regions = [];
let regionById = new Map();
let currentState = null;
let suppressViewSync = false;

const page = createPage({
  cssUrl: CSS_URL,
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [[PARCELS_ID, onParcelsUpdate]],
  visibleIds: [PARCELS_ID],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

async function loadPageData() {
  const [parcels] = await Promise.all([
    cache.load(PARCELS_ID),
    cache.get(TERRENI_ID) ? Promise.resolve(cache.get(TERRENI_ID)) : cache.load(TERRENI_ID),
  ]);
  parcelsData = parcels;
  parcelsGeo = sortFeaturesByArea(cache.get(TERRENI_ID));
  rebuildRegionIndex();
}

function mountPage(el, params) {
  el.classList.add('bosco-content');
  el.replaceChildren(cloneTemplate('tmpl-bosco-page'));
  root = el.querySelector('.bosco-page');
  mapHost = el.querySelector('[data-target="map"]');
  regionSelect = el.querySelector('[data-role="region-select"]');
  modeGroup = el.querySelector('[data-role="mode-group"]');
  statusEl = el.querySelector('[data-role="status"]');

  buildRegionOptions();
  wireControls();
  applyParams(params);
}

function destroyPage() {
  destroyMap();
  const el = document.getElementById('content');
  el.classList.remove('bosco-content');
  el.replaceChildren();
  root = mapHost = regionSelect = modeGroup = statusEl = null;
  parcelsData = parcelsGeo = null;
  regions = [];
  regionById = new Map();
  currentState = null;
}

function onParcelsUpdate(data) {
  parcelsData = data;
  rebuildRegionIndex();
  buildRegionOptions();
  applyParams(Object.fromEntries(new URLSearchParams(location.search)));
}

function rebuildRegionIndex() {
  regions = [];
  regionById = new Map();
  if (!parcelsData) return;
  const cols = parcelsData[COLUMNS];
  const idxRegionId = cols.indexOf(S.COL_REGION_ID);
  const idxRegion = cols.indexOf(S.COL_REGION);
  if (idxRegionId < 0 || idxRegion < 0) return;

  const byId = new Map();
  for (const row of parcelsData[ROWS]) {
    const id = row[idxRegionId];
    if (id == null || byId.has(id)) continue;
    byId.set(id, { id, name: row[idxRegion] });
  }
  regions = [...byId.values()].sort((a, b) => a.name.localeCompare(b.name, 'it'));
  regionById = new Map(regions.map(r => [r.id, r]));
}

function buildRegionOptions() {
  if (!regionSelect) return;
  regionSelect.replaceChildren();
  for (const region of regions) {
    const opt = document.createElement('option');
    opt.value = String(region.id);
    opt.textContent = region.name;
    regionSelect.appendChild(opt);
  }
}

function wireControls() {
  regionSelect?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('c', regionSelect.value);
    clearMapView(params);
    navigateWithParams(PAGE_PATH, params, true);
  });

  modeGroup?.addEventListener('change', (e) => {
    if (e.target?.name !== 'bosco-mode') return;
    const params = new URLSearchParams(location.search);
    params.set('m', e.target.value);
    navigateWithParams(PAGE_PATH, params, true);
  });
}

function applyParams(params) {
  if (!root) return;
  const state = readBoscoParams(params, regions.map(r => r.id));
  currentState = state;

  if (regionSelect && state.regionId != null) regionSelect.value = String(state.regionId);
  const modeInput = modeGroup?.querySelector(`input[value="${state.mode}"]`);
  if (modeInput) modeInput.checked = true;
  updateModePanels(state.mode);

  const needsMapRebuild = !map
    || map._boscoRegionId !== state.regionId
    || map.wrapper?.getBasemap() !== state.basemap;
  if (needsMapRebuild) renderMap(state);
  else applyMapView(state);

  canonicalizeURL(state);
}

function updateModePanels(mode) {
  for (const panel of root.querySelectorAll('[data-mode-panel]')) {
    panel.hidden = panel.dataset.modePanel !== mode;
  }
}

function renderMap(state) {
  destroyMap();
  if (!mapHost) return;
  mapHost.replaceChildren();

  const region = regionById.get(state.regionId);
  if (!region || !parcelsGeo) {
    setStatus('Nessuna compresa');
    return;
  }

  const features = parcelsGeo.features.filter(f => parcelNames(f).compresa === region.name);
  if (!features.length) {
    setStatus(`${region.name} — nessuna geometria`);
    return;
  }

  suppressViewSync = true;
  map = new ParcelMap({
    container: mapHost,
    className: 'bosco-map',
    geojson: { type: 'FeatureCollection', features },
    basemap: state.basemap,
    tools: {
      measure: true,
      location: true,
      sidebar: { sidebarId: 'bosco-sidebar', mapId: 'bosco-map-host' },
    },
    initialView: state.center ? { center: state.center, zoom: state.zoom } : null,
    onViewChange: onMapViewChange,
    onMapClick: onMapClick,
  });
  map._boscoRegionId = state.regionId;
  map.leaflet.on('basemapchange', (e) => onBasemapChange(e.name));
  setTimeout(() => { suppressViewSync = false; }, 0);
  setStatus(`${region.name} — ${new Set(features.map(f => f.properties.name)).size} particelle`);
}

function destroyMap() {
  if (map) {
    map.destroy();
    map = null;
  }
}

function applyMapView(state) {
  if (!map?.leaflet) return;
  if (map.wrapper.getBasemap() !== state.basemap) map.wrapper.syncBasemap(state.basemap);
  if (!state.center || state.zoom == null) return;
  const c = map.leaflet.getCenter();
  const sameCenter = Math.abs(c.lat - state.center[0]) < 0.000001
    && Math.abs(c.lng - state.center[1]) < 0.000001;
  if (sameCenter && map.leaflet.getZoom() === state.zoom) return;
  suppressViewSync = true;
  map.leaflet.setView(state.center, state.zoom);
  setTimeout(() => { suppressViewSync = false; }, 0);
}

function onMapViewChange(center, zoom) {
  if (suppressViewSync || !currentState) return;
  const params = new URLSearchParams(location.search);
  writeMapView(params, center, zoom);
  navigateWithParams(PAGE_PATH, params, true);
}

function onBasemapChange(name) {
  const params = new URLSearchParams(location.search);
  params.set('mt', mapTypeToken(name));
  navigateWithParams(PAGE_PATH, params, true);
}

function onMapClick(_latlng, feature) {
  const region = regionById.get(currentState?.regionId);
  if (!feature) {
    if (region) setStatus(`${region.name} — riepilogo compresa`);
    return;
  }
  const { compresa, particella } = parcelNames(feature);
  setStatus(`${compresa} ${particella}`.trim());
}

function canonicalizeURL(state) {
  const params = new URLSearchParams(location.search);
  let changed = false;
  if (state.regionId != null && params.get('c') !== String(state.regionId)) {
    params.set('c', String(state.regionId));
    changed = true;
  }
  if (!params.get('m') || !['1', '2', '3'].includes(params.get('m'))) {
    params.set('m', state.mode);
    changed = true;
  }
  if (!params.get('mt') || !['o', 't', 's'].includes(params.get('mt'))) {
    params.set('mt', state.mt);
    changed = true;
  }
  if (changed) navigateWithParams(PAGE_PATH, params, true);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text || '';
}
