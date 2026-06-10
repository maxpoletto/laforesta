/**
 * Bosco page: map shell, URL-backed controls, and parcel characteristic styling.
 */

import * as cache from '../../base/js/cache.js';
import * as S from '../../base/js/strings.js';
import { COLUMNS, DIGEST_FUTURE_PRODUCTION, ROWS } from '../../base/js/constants.js';
import { fmtArea, fmtDecimal1, fmtDecimal2, fmtMass, fmtVolume } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { sortFeaturesByArea, parcelNames } from '../../base/js/geo.js';
import { PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { createPage, navigateWithParams } from '../../base/js/page-sync.js';
import {
  clearMapView, mapTypeToken, readBoscoParams, writeMapView,
} from './bosco-state.js';
import {
  CHARACTERISTIC_METRICS,
  Q_FUTURE_HARVEST,
  Q_HISTORICAL_HARVEST,
  Q_TYPE,
  buildParcelEntries,
  continuousDomain,
  futureHarvestByParcel,
  historicalHarvestByParcel,
  metricValue,
  normalized,
  parcelKey,
} from './bosco-characteristics.js';

const CSS_URL = '/static/bosco/css/bosco.css';
const PAGE_PATH = '/bosco';

const PARCELS_ID = 'parcels';
const PARCELS_URL = '/api/bosco/parcels/data/';
const FUTURE_ID = DIGEST_FUTURE_PRODUCTION;
const FUTURE_URL = '/api/bosco/future-production/data/';
const PRELIEVI_ID = 'prelievi';
const PRELIEVI_URL = '/api/prelievi/data/';
const TERRENI_ID = 'terreni';
const TERRENI_GEOJSON_URL = '/api/geo/terreni.geojson';

const VALID_MODES = ['1', '2', '3'];
const VALID_MAP_TYPES = ['o', 't', 's'];
const VALID_CHARACTERISTICS = ['1', '2', '3', '4', '5', '6', '7', '8'];
const SATELLITE_CHARACTERISTICS = new Set(['6', '7', '8']);

const NO_DATA_STYLE = {
  ...PARCEL_STYLE,
  color: '#777',
  weight: 1,
  opacity: 0.85,
  fillColor: '#d6d6d6',
  fillOpacity: 0.34,
};

const TYPE_STYLES = {
  [S.TYPE_HIGHFOREST]: {
    color: '#17613a', weight: 1.5, opacity: 0.95, fillColor: '#2f8f58', fillOpacity: 0.58,
  },
  [S.TYPE_COPPICE]: {
    color: '#8a6500', weight: 1.5, opacity: 0.95, fillColor: '#d7aa27', fillOpacity: 0.62,
  },
};

cache.register(PARCELS_ID, PARCELS_URL);
cache.register(FUTURE_ID, FUTURE_URL);
cache.register(PRELIEVI_ID, PRELIEVI_URL);
cache.register(TERRENI_ID, TERRENI_GEOJSON_URL);

let root = null;
let mapHost = null;
let regionSelect = null;
let modeGroup = null;
let characteristicSelect = null;
let cadastralToggle = null;
let perHaToggle = null;
let perHaRow = null;
let legendEl = null;
let statusEl = null;
let parcelsData = null;
let futureData = null;
let prelieviData = null;
let prelieviLoad = null;
let parcelsGeo = null;
let map = null;
let regions = [];
let regionById = new Map();
let currentState = null;
let suppressViewSync = false;
let characteristicRenderSeq = 0;

const page = createPage({
  cssUrl: CSS_URL,
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [
    [PARCELS_ID, onParcelsUpdate],
    [FUTURE_ID, onFutureUpdate],
    [PRELIEVI_ID, onPrelieviUpdate],
  ],
  visibleIds: [PARCELS_ID, FUTURE_ID, PRELIEVI_ID],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

async function loadPageData() {
  const [parcels, future] = await Promise.all([
    cache.load(PARCELS_ID),
    cache.load(FUTURE_ID),
    cache.get(TERRENI_ID) ? Promise.resolve(cache.get(TERRENI_ID)) : cache.load(TERRENI_ID),
  ]);
  parcelsData = parcels;
  futureData = future;
  prelieviData = cache.get(PRELIEVI_ID);
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
  characteristicSelect = el.querySelector('[data-role="characteristic-select"]');
  cadastralToggle = el.querySelector('[data-role="cadastral-toggle"]');
  perHaToggle = el.querySelector('[data-role="per-ha-toggle"]');
  perHaRow = el.querySelector('[data-role="per-ha-row"]');
  legendEl = el.querySelector('[data-target="legend"]');
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
  characteristicSelect = cadastralToggle = perHaToggle = perHaRow = legendEl = null;
  parcelsData = futureData = prelieviData = parcelsGeo = null;
  prelieviLoad = null;
  regions = [];
  regionById = new Map();
  currentState = null;
  characteristicRenderSeq++;
}

function onParcelsUpdate(data) {
  parcelsData = data;
  rebuildRegionIndex();
  buildRegionOptions();
  applyParams(Object.fromEntries(new URLSearchParams(location.search)));
}

function onFutureUpdate(data) {
  futureData = data;
  refreshCharacteristicLayer();
}

function onPrelieviUpdate(data) {
  prelieviData = data;
  refreshCharacteristicLayer();
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

  characteristicSelect?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('q', characteristicSelect.value);
    if (!isHarvestMetric(characteristicSelect.value)) params.delete('fh');
    navigateWithParams(PAGE_PATH, params, true);
  });

  cadastralToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    setFlagParam(params, 'fc', cadastralToggle.checked);
    navigateWithParams(PAGE_PATH, params, true);
  });

  perHaToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    setFlagParam(params, 'fh', perHaToggle.checked);
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
  updateCharacteristicControls(state);

  const needsMapRebuild = !map
    || map._boscoRegionId !== state.regionId
    || map.wrapper?.getBasemap() !== state.basemap;
  if (needsMapRebuild) renderMap(state);
  else {
    applyMapView(state);
    updateMapDisplayAreas(state);
    refreshCharacteristicLayer();
  }

  canonicalizeURL(state);
}

function updateModePanels(mode) {
  for (const panel of root.querySelectorAll('[data-mode-panel]')) {
    panel.hidden = panel.dataset.modePanel !== mode;
  }
}

function updateCharacteristicControls(state) {
  if (characteristicSelect) characteristicSelect.value = state.q;
  if (cadastralToggle) cadastralToggle.checked = state.useCadastralArea;
  const harvest = isHarvestMetric(state.q);
  if (perHaRow) perHaRow.hidden = !harvest;
  if (perHaToggle) {
    perHaToggle.checked = harvest && state.harvestPerHa;
    perHaToggle.disabled = !harvest;
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
  buildMapParcelEntries(state);
  refreshCharacteristicLayer();
  setStatus(`${region.name} — ${map._boscoEntries.length} particelle`);
}

function buildMapParcelEntries(state) {
  if (!map || !parcelsData) return;
  const region = regionById.get(state.regionId);
  const entries = buildParcelEntries(parcelsData)
    .filter(e => e.regionId === state.regionId || e.region === region?.name)
    .map(e => ({ ...e, layers: [], geoAreaHa: 0, displayAreaHa: null }));
  const byKey = new Map(entries.map(e => [e.key, e]));

  map.parcelLayer.eachLayer(layer => {
    const { compresa, particella } = parcelNames(layer.feature);
    const entry = byKey.get(parcelKey(compresa, particella));
    if (!entry) return;
    entry.layers.push(layer);
    entry.geoAreaHa += (layer.feature?.properties?._areaM2 || 0) / 10000;
  });

  map._boscoEntries = entries;
  map._boscoEntriesByKey = byKey;
  updateMapDisplayAreas(state);
}

function updateMapDisplayAreas(state) {
  if (!map?._boscoEntries) return;
  for (const entry of map._boscoEntries) {
    entry.displayAreaHa = displayAreaHa(entry, state);
  }
}

function displayAreaHa(entry, state) {
  if (state.useCadastralArea) {
    return firstNumber(entry.cadastralAreaHa, entry.areaHa, entry.geoAreaHa);
  }
  return firstNumber(entry.geoAreaHa, entry.areaHa, entry.cadastralAreaHa);
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

function refreshCharacteristicLayer() {
  const seq = ++characteristicRenderSeq;
  if (!map?._boscoEntries || !currentState) return;
  if (currentState.mode !== '1') {
    resetParcelStyles();
    clearLegend();
    return;
  }

  updateCharacteristicControls(currentState);
  if (SATELLITE_CHARACTERISTICS.has(currentState.q)) {
    resetParcelStyles();
    renderMessageLegend('Dati satellitari: layer in preparazione.');
    return;
  }

  if (currentState.q === Q_HISTORICAL_HARVEST && !prelieviData) {
    resetParcelStyles();
    renderMessageLegend('Caricamento prelievi...');
    loadPrelievi().then(() => {
      if (seq === characteristicRenderSeq) refreshCharacteristicLayer();
    }).catch(() => {
      if (seq === characteristicRenderSeq) renderMessageLegend('Prelievi non disponibili.');
    });
    return;
  }

  const context = characteristicContext(currentState.q);
  const entries = map._boscoEntries;
  if (currentState.q === Q_TYPE) {
    for (const entry of entries) {
      const value = metricValue(entry, currentState.q, context);
      applyEntryStyle(entry, TYPE_STYLES[value] || NO_DATA_STYLE, value);
    }
    renderTypeLegend();
    return;
  }

  const values = entries.map(entry => metricValue(entry, currentState.q, context));
  const domain = continuousDomain(values);
  for (const entry of entries) {
    const value = metricValue(entry, currentState.q, context);
    const t = normalized(value, domain);
    applyEntryStyle(entry, t == null ? NO_DATA_STYLE : continuousStyle(t), value);
  }
  renderContinuousLegend(domain, currentState.q);
}

function characteristicContext(metricId) {
  return {
    historical: metricId === Q_HISTORICAL_HARVEST ? historicalHarvestByParcel(prelieviData) : undefined,
    future: metricId === Q_FUTURE_HARVEST ? futureHarvestByParcel(futureData) : undefined,
    perHa: currentState?.harvestPerHa,
  };
}

function loadPrelievi() {
  if (!prelieviLoad) {
    prelieviLoad = cache.load(PRELIEVI_ID).then(data => {
      prelieviData = data;
      return data;
    }).finally(() => { prelieviLoad = null; });
  }
  return prelieviLoad;
}

function resetParcelStyles() {
  if (!map?.parcelLayer) return;
  map.parcelLayer.eachLayer(layer => {
    layer.setStyle(PARCEL_STYLE);
    if (layer.feature) setLayerTooltip(layer, safeTooltip(layer.feature));
  });
}

function applyEntryStyle(entry, style, value) {
  for (const layer of entry.layers) {
    layer.setStyle(style);
    setLayerTooltip(layer, buildTooltip(entry, value));
  }
}

function setLayerTooltip(layer, content) {
  if (layer.getTooltip?.()) layer.setTooltipContent(content);
  else layer.bindTooltip(content, { sticky: true, direction: 'top' });
}

function continuousStyle(t) {
  return {
    color: '#244126',
    weight: 1.4,
    opacity: 0.92,
    fillColor: interpolateColor([245, 222, 91], [32, 113, 75], t),
    fillOpacity: 0.64,
  };
}

function buildTooltip(entry, value) {
  const el = document.createElement('div');
  el.className = 'bosco-tooltip';

  const title = document.createElement('div');
  title.className = 'bosco-tooltip-title';
  title.textContent = `${entry.region} ${entry.parcel}`.trim();
  el.appendChild(title);

  const meta = document.createElement('div');
  const area = entry.displayAreaHa ? fmtArea(entry.displayAreaHa) : '';
  meta.textContent = [entry.type, area].filter(Boolean).join(' · ');
  el.appendChild(meta);

  const metric = document.createElement('div');
  metric.textContent = metricDisplay(currentState?.q, value);
  el.appendChild(metric);
  return el;
}

function safeTooltip(feature) {
  const el = document.createElement('span');
  const { compresa, particella } = parcelNames(feature);
  el.textContent = `${compresa} ${particella}`.trim();
  return el;
}

function metricDisplay(metricId, value) {
  if (value == null || value === '') return 'n.d.';
  if (metricId === Q_TYPE) return value;
  const perHa = currentState?.harvestPerHa && isHarvestMetric(metricId);
  if (metricId === Q_HISTORICAL_HARVEST) return perHa ? `${fmtDecimal2(value)} q/ha` : fmtMass(value);
  if (metricId === Q_FUTURE_HARVEST) return perHa ? `${fmtDecimal2(value)} m³/ha` : fmtVolume(value);
  const unit = CHARACTERISTIC_METRICS[metricId]?.unit;
  return unit ? `${fmtDecimal1(value)} ${unit}` : fmtDecimal1(value);
}

function renderTypeLegend() {
  if (!legendEl) return;
  legendEl.replaceChildren();
  legendEl.appendChild(legendRow(TYPE_STYLES[S.TYPE_HIGHFOREST].fillColor, S.TYPE_HIGHFOREST));
  legendEl.appendChild(legendRow(TYPE_STYLES[S.TYPE_COPPICE].fillColor, S.TYPE_COPPICE));
  legendEl.appendChild(legendRow(NO_DATA_STYLE.fillColor, 'n.d.'));
}

function renderContinuousLegend(domain, metricId) {
  if (!legendEl) return;
  legendEl.replaceChildren();
  if (!domain) {
    renderMessageLegend('Nessun dato disponibile.');
    return;
  }

  const title = document.createElement('div');
  title.className = 'bosco-legend-title';
  title.textContent = selectedCharacteristicLabel();
  legendEl.appendChild(title);

  const gradient = document.createElement('div');
  gradient.className = 'bosco-gradient';
  legendEl.appendChild(gradient);

  const labels = document.createElement('div');
  labels.className = 'bosco-legend-labels';
  const min = document.createElement('span');
  min.textContent = metricDisplay(metricId, domain.min);
  const max = document.createElement('span');
  max.textContent = metricDisplay(metricId, domain.max);
  labels.append(min, max);
  legendEl.appendChild(labels);
}

function renderMessageLegend(message) {
  if (!legendEl) return;
  legendEl.replaceChildren();
  const p = document.createElement('div');
  p.className = 'bosco-legend-note';
  p.textContent = message;
  legendEl.appendChild(p);
}

function clearLegend() {
  legendEl?.replaceChildren();
}

function legendRow(color, label) {
  const row = document.createElement('div');
  row.className = 'bosco-legend-row';
  const dot = document.createElement('span');
  dot.className = 'bosco-legend-dot';
  dot.style.backgroundColor = color;
  const text = document.createElement('span');
  text.textContent = label;
  row.append(dot, text);
  return row;
}

function selectedCharacteristicLabel() {
  return characteristicSelect?.selectedOptions?.[0]?.textContent || '';
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
  const entry = map?._boscoEntriesByKey?.get(parcelKey(compresa, particella));
  const context = currentState ? characteristicContext(currentState.q) : {};
  const value = entry && currentState ? metricValue(entry, currentState.q, context) : null;
  const metric = currentState?.mode === '1' ? ` — ${metricDisplay(currentState.q, value)}` : '';
  setStatus(`${compresa} ${particella}${metric}`.trim());
}

function canonicalizeURL(state) {
  const params = new URLSearchParams(location.search);
  let changed = false;
  if (state.regionId != null && params.get('c') !== String(state.regionId)) {
    params.set('c', String(state.regionId));
    changed = true;
  }
  if (!params.get('m') || !VALID_MODES.includes(params.get('m'))) {
    params.set('m', state.mode);
    changed = true;
  }
  if (!params.get('mt') || !VALID_MAP_TYPES.includes(params.get('mt'))) {
    params.set('mt', state.mt);
    changed = true;
  }
  if (!params.get('q') || !VALID_CHARACTERISTICS.includes(params.get('q'))) {
    params.set('q', state.q);
    changed = true;
  }
  changed = canonicalizeFlag(params, 'fc', state.useCadastralArea) || changed;
  changed = canonicalizeFlag(params, 'fh', state.harvestPerHa && isHarvestMetric(state.q)) || changed;
  if (changed) navigateWithParams(PAGE_PATH, params, true);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text || '';
}

function setFlagParam(params, key, enabled) {
  if (enabled) params.set(key, '1');
  else params.delete(key);
}

function canonicalizeFlag(params, key, enabled) {
  if (enabled) {
    if (params.get(key) === '1') return false;
    params.set(key, '1');
    return true;
  }
  if (!params.has(key)) return false;
  params.delete(key);
  return true;
}

function isHarvestMetric(metricId) {
  return Boolean(CHARACTERISTIC_METRICS[metricId]?.harvest);
}

function firstNumber(...values) {
  return values.find(v => v != null && Number.isFinite(v) && v > 0) ?? null;
}

function interpolateColor(start, end, t) {
  const rgb = start.map((v, i) => Math.round(v + (end[i] - v) * t));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
