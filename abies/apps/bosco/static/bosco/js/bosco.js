/**
 * Bosco page: map shell, URL-backed controls, and parcel characteristic styling.
 */

import * as cache from '../../base/js/cache.js';
import * as S from '../../base/js/strings.js';
import {
  COLUMNS, DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY,
  DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, ROWS,
} from '../../base/js/constants.js';
import { fmtArea, fmtDecimal1, fmtDecimal2, fmtInt, fmtMass, fmtVolume } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { findContainingParcel, sortFeaturesByArea, parcelNames } from '../../base/js/geo.js';
import { PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { createPage, navigateWithParams } from '../../base/js/page-sync.js';
import { renderStackedBar } from '../../prelievi/js/charts.js';
import {
  deleteRowWithVersion, fetchModalForm, interceptSubmit, renderModalForm,
  showFormError,
} from '../../base/js/forms.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
import { dismiss as dismissModal } from '../../base/js/modals.js';
import { canModify } from '../../base/js/roles.js';
import { wireCancelButtons, wireCollapsibleToggle } from '../../base/js/ui-widgets.js';
import {
  clearDetailParams, clearMapView, mapTypeToken, readBoscoParams,
  writeMapView, writeOptionalIdList, writeSectionTokens,
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
  isHarvestMetric,
  metricValue,
  normalized,
  parcelKey,
} from './bosco-characteristics.js';
import {
  EVOLUTION_METRICS, SATELLITE_LAYERS, availableMonths, characteristicSatelliteLayer, dateFromMonthValue,
  dateParam, diffColor, divergingDomain, monthValue, pickDate, satelliteColor,
  satelliteDiffPngUrl, satelliteDiffValue, satelliteValue,
} from './bosco-satellite.js';
import {
  aggregateDendrometry, dendrometryBarChartData, dendrometryHeightPoints,
  dendrometryLineChartData, dendrometryScatterChartData, dendrometrySpecies,
  dendrometryTreeTotal, regionMetadata,
} from './bosco-detail.js';
import {
  buildPreservedTrees, filterPaiTrees, paiParcelItems, paiSpeciesItems, speciesColorMap,
} from './bosco-pai.js';
import { aggregateProduction, prelieviUrlForScope } from './bosco-production.js';

const CSS_URL = '/static/bosco/css/bosco.css';
const PAGE_PATH = '/bosco';

const PARCELS_ID = 'parcels';
const PARCELS_URL = '/api/bosco/parcels/data/';
const FUTURE_ID = DIGEST_FUTURE_PRODUCTION;
const FUTURE_URL = '/api/bosco/future-production/data/';
const DENDROMETRY_ID = DIGEST_PARCEL_DENDROMETRY;
const DENDROMETRY_URL = '/api/bosco/parcel-dendrometry/data/';
const DENDROMETRY_POINTS_ID = DIGEST_PARCEL_DENDROMETRY_POINTS;
const DENDROMETRY_POINTS_URL = '/api/bosco/parcel-dendrometry-points/data/';
const PARCEL_METADATA_FORM_URL = '/api/bosco/parcels/metadata/form/';
const PARCEL_METADATA_SAVE_URL = '/api/bosco/parcels/metadata/save/';
const PRESERVED_ID = DIGEST_PRESERVED_TREES;
const PRESERVED_URL = '/api/bosco/preserved-trees/data/';
const PAI_FORM_URL = '/api/bosco/pai/form/';
const PAI_SAVE_URL = '/api/bosco/pai/save/';
const PAI_DELETE_URL = '/api/bosco/pai/delete/';
const PRELIEVI_ID = 'prelievi';
const PRELIEVI_URL = '/api/prelievi/data/';
const TERRENI_ID = 'terreni';
const TERRENI_GEOJSON_URL = '/api/geo/terreni.geojson';

const VALID_MODES = ['1', '2', '3'];
const VALID_MAP_TYPES = ['o', 't', 's'];
const VALID_CHARACTERISTICS = ['1', '2', '3', '4', '5', '6', '7', '8'];
const VALID_EVOLUTION_METRICS = ['1', '2', '3', '4'];
const SATELLITE_CHARACTERISTICS = new Set(['6', '7', '8']);
const DETAIL_SECTIONS = ['m', 'd', 'p'];
const EVOLUTION_OVERLAY_OPACITY = 0.85;

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
cache.register(DENDROMETRY_ID, DENDROMETRY_URL);
cache.register(DENDROMETRY_POINTS_ID, DENDROMETRY_POINTS_URL);
cache.register(PRESERVED_ID, PRESERVED_URL);
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
let evolutionSelect = null;
let date1Input = null;
let date2Input = null;
let parcelAverageToggle = null;
let evolutionCadastralToggle = null;
let diffLegendEl = null;
let legendEl = null;
let statusEl = null;
let detailOverlay = null;
let detailTitle = null;
let detailScopeLabel = null;
let metadataHost = null;
let metadataActions = null;
let metadataEditButton = null;
let dendrometryHost = null;
let dendrometrySpeciesHost = null;
let dendrometryPerHa = null;
let dendrometryStatus = null;
let dendrometryChartGrid = null;
let dendrometryTreeCanvas = null;
let dendrometryVolumeCanvas = null;
let dendrometryBasalAreaCanvas = null;
let dendrometryHeightCanvas = null;
let dendrometryIncrementCanvas = null;
let dendrometryCharts = {};
let productionHost = null;
let productionCanvas = null;
let productionSummary = null;
let productionLink = null;
let productionPerHa = null;
let productionMonthly = null;
let productionChart = null;
let paiParcelsHost = null;
let paiSpeciesHost = null;
let detailSections = {};
let parcelsData = null;
let futureData = null;
let prelieviData = null;
let prelieviLoad = null;
let dendrometryData = null;
let dendrometryPointsData = null;
let dendrometryLoad = null;
let dendrometryPointsLoad = null;
let preservedData = null;
let preservedLoad = null;
let satelliteData = null;
let satelliteRegionId = null;
let satelliteLoad = null;
let evolutionOverlay = null;
let paiMarkerLayer = null;
let parcelsGeo = null;
let map = null;
let regions = [];
let regionById = new Map();
let currentState = null;
let suppressViewSync = false;
let characteristicRenderSeq = 0;
let evolutionRenderSeq = 0;

const page = createPage({
  cssUrl: CSS_URL,
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [
    [PARCELS_ID, onParcelsUpdate],
    [FUTURE_ID, onFutureUpdate],
    [DENDROMETRY_ID, onDendrometryUpdate],
    [DENDROMETRY_POINTS_ID, onDendrometryPointsUpdate],
    [PRESERVED_ID, onPreservedUpdate],
    [PRELIEVI_ID, onPrelieviUpdate],
  ],
  visibleIds: [
    PARCELS_ID, FUTURE_ID, DENDROMETRY_ID, DENDROMETRY_POINTS_ID,
    PRESERVED_ID, PRELIEVI_ID,
  ],
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
  dendrometryData = cache.get(DENDROMETRY_ID);
  dendrometryPointsData = cache.get(DENDROMETRY_POINTS_ID);
  preservedData = cache.get(PRESERVED_ID);
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
  evolutionSelect = el.querySelector('[data-role="evolution-select"]');
  date1Input = el.querySelector('[data-role="date1"]');
  date2Input = el.querySelector('[data-role="date2"]');
  parcelAverageToggle = el.querySelector('[data-role="parcel-average-toggle"]');
  evolutionCadastralToggle = el.querySelector('[data-role="evolution-cadastral-toggle"]');
  diffLegendEl = el.querySelector('[data-target="diff-legend"]');
  legendEl = el.querySelector('[data-target="legend"]');
  statusEl = el.querySelector('[data-role="status"]');
  detailOverlay = el.querySelector('[data-target="detail-overlay"]');
  detailTitle = el.querySelector('[data-target="detail-title"]');
  detailScopeLabel = el.querySelector('[data-target="detail-scope"]');
  metadataHost = el.querySelector('[data-target="metadata"]');
  metadataActions = el.querySelector('[data-target="metadata-actions"]');
  metadataEditButton = el.querySelector('[data-action="edit-parcel-metadata"]');
  dendrometryHost = el.querySelector('[data-target="dendrometry"]');
  dendrometrySpeciesHost = el.querySelector('[data-target="dendrometry-species"]');
  dendrometryPerHa = el.querySelector('[data-role="dendrometry-per-ha"]');
  dendrometryStatus = el.querySelector('[data-target="dendrometry-status"]');
  dendrometryChartGrid = el.querySelector('[data-target="dendrometry-chart-grid"]');
  dendrometryTreeCanvas = el.querySelector('[data-target="dendrometry-tree-count-chart"]');
  dendrometryVolumeCanvas = el.querySelector('[data-target="dendrometry-volume-chart"]');
  dendrometryBasalAreaCanvas = el.querySelector('[data-target="dendrometry-basal-area-chart"]');
  dendrometryHeightCanvas = el.querySelector('[data-target="dendrometry-height-chart"]');
  dendrometryIncrementCanvas = el.querySelector('[data-target="dendrometry-increment-chart"]');
  productionHost = el.querySelector('[data-target="production-chart-host"]');
  productionCanvas = el.querySelector('[data-target="production-chart"]');
  productionSummary = el.querySelector('[data-target="production-summary"]');
  productionLink = el.querySelector('[data-target="production-link"]');
  productionPerHa = el.querySelector('[data-role="production-per-ha"]');
  productionMonthly = el.querySelector('[data-role="production-monthly"]');
  paiParcelsHost = el.querySelector('[data-target="pai-parcels"]');
  paiSpeciesHost = el.querySelector('[data-target="pai-species"]');

  buildRegionOptions();
  wireControls();
  wireDetailControls();
  applyParams(params);
}

function destroyPage() {
  destroyMap();
  const el = document.getElementById('content');
  el.classList.remove('bosco-content');
  el.replaceChildren();
  document.removeEventListener('keydown', onDetailKeyDown);
  root = mapHost = regionSelect = modeGroup = statusEl = null;
  characteristicSelect = cadastralToggle = perHaToggle = perHaRow = null;
  evolutionSelect = date1Input = date2Input = parcelAverageToggle = null;
  evolutionCadastralToggle = diffLegendEl = legendEl = null;
  detailOverlay = detailTitle = detailScopeLabel = metadataHost = null;
  metadataActions = metadataEditButton = null;
  dendrometryHost = dendrometrySpeciesHost = dendrometryPerHa = null;
  dendrometryStatus = dendrometryChartGrid = null;
  dendrometryTreeCanvas = dendrometryVolumeCanvas = null;
  dendrometryBasalAreaCanvas = dendrometryHeightCanvas = null;
  dendrometryIncrementCanvas = null;
  destroyDendrometryCharts();
  productionHost = productionCanvas = productionSummary = productionLink = null;
  productionPerHa = productionMonthly = null;
  destroyProductionChart();
  paiParcelsHost = paiSpeciesHost = null;
  detailSections = {};
  parcelsData = futureData = prelieviData = dendrometryData = null;
  dendrometryPointsData = preservedData = parcelsGeo = null;
  prelieviLoad = dendrometryLoad = dendrometryPointsLoad = preservedLoad = null;
  satelliteData = satelliteRegionId = satelliteLoad = null;
  evolutionOverlay = paiMarkerLayer = null;
  regions = [];
  regionById = new Map();
  currentState = null;
  characteristicRenderSeq++;
  evolutionRenderSeq++;
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
  renderProduction();
}

function onDendrometryUpdate(data) {
  dendrometryData = data;
  renderDendrometry();
}

function onDendrometryPointsUpdate(data) {
  dendrometryPointsData = data;
  renderDendrometry();
}

function onPreservedUpdate(data) {
  preservedData = data;
  renderPaiMode();
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

  evolutionSelect?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('m', '2');
    params.set('q', evolutionSelect.value);
    navigateWithParams(PAGE_PATH, params, true);
  });

  date1Input?.addEventListener('change', () => updateEvolutionDateParam('d1', date1Input));
  date2Input?.addEventListener('change', () => updateEvolutionDateParam('d2', date2Input));

  parcelAverageToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('m', '2');
    setFlagParam(params, 'fa', parcelAverageToggle.checked);
    navigateWithParams(PAGE_PATH, params, true);
  });

  evolutionCadastralToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    setFlagParam(params, 'fc', evolutionCadastralToggle.checked);
    navigateWithParams(PAGE_PATH, params, true);
  });
}

function updateEvolutionDateParam(key, input) {
  const params = new URLSearchParams(location.search);
  const value = dateParam(dateFromMonthValue(input.value));
  if (value) params.set(key, value);
  else params.delete(key);
  navigateWithParams(PAGE_PATH, params, true);
}

function wireDetailControls() {
  detailOverlay?.querySelector('[data-action="close-detail"]')
    ?.addEventListener('click', closeDetailOverlay);
  document.addEventListener('keydown', onDetailKeyDown);

  detailSections = {};
  for (const key of DETAIL_SECTIONS) {
    const section = detailOverlay?.querySelector(`[data-detail-section="${key}"]`);
    const header = section?.querySelector('.collapsible-header');
    const body = section?.querySelector('.collapsible-body');
    detailSections[key] = { header, body };
    if (header && body) {
      wireCollapsibleToggle(header, body, (open) => {
        if (!currentState?.detailMode) return;
        const openSections = DETAIL_SECTIONS.filter(k =>
          detailSections[k].header?.classList.contains('open'));
        const params = new URLSearchParams(location.search);
        writeSectionTokens(params, openSections);
        navigateWithParams(PAGE_PATH, params, true);
        if (open && key === 'd') renderDendrometry();
        if (open && key === 'p') renderProduction();
      });
    }
  }

  dendrometryPerHa?.addEventListener('change', renderDendrometry);
  productionPerHa?.addEventListener('change', renderProduction);
  productionMonthly?.addEventListener('change', renderProduction);
  root?.querySelector('[data-action="show-all-parcels"]')
    ?.addEventListener('click', () => setPaiFilter('pp', null, paiParcelsHost));
  root?.querySelector('[data-action="hide-all-parcels"]')
    ?.addEventListener('click', () => setPaiFilter('pp', [], paiParcelsHost));
  root?.querySelector('[data-action="show-all-species"]')
    ?.addEventListener('click', () => setPaiFilter('ps', null, paiSpeciesHost));
  root?.querySelector('[data-action="hide-all-species"]')
    ?.addEventListener('click', () => setPaiFilter('ps', [], paiSpeciesHost));
  root?.querySelector('[data-action="show-all-dendrometry-species"]')
    ?.addEventListener('click', () => setDendrometrySpeciesFilter(null));
  root?.querySelector('[data-action="hide-all-dendrometry-species"]')
    ?.addEventListener('click', () => setDendrometrySpeciesFilter([]));
  root?.querySelector('[data-action="add-pai"]')
    ?.addEventListener('click', () => showPaiForm());
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
  updateEvolutionControls(state);

  const needsMapRebuild = !map
    || map._boscoRegionId !== state.regionId
    || map.wrapper?.getBasemap() !== state.basemap;
  if (needsMapRebuild) renderMap(state);
  else {
    applyMapView(state);
    updateMapDisplayAreas(state);
    refreshCharacteristicLayer();
  }
  if (state.mode === '2') renderEvolutionMode();
  else {
    clearEvolutionOverlay();
    clearEvolutionLegend();
  }
  syncDetailOverlay(state);
  if (state.mode === '3') renderPaiMode();
  else clearPaiMarkers();

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

function updateEvolutionControls(state) {
  if (evolutionSelect) evolutionSelect.value = state.evolutionMetric;
  if (parcelAverageToggle) {
    const supportsRaster = Boolean(EVOLUTION_METRICS[state.evolutionMetric]?.satellite);
    parcelAverageToggle.checked = supportsRaster && state.parcelAverage;
    parcelAverageToggle.disabled = !supportsRaster;
  }
  if (evolutionCadastralToggle) evolutionCadastralToggle.checked = state.useCadastralArea;

  const dates = satelliteRegionId === state.regionId ? satelliteData?.timeseries?.dates : null;
  const date1 = dates ? pickDate(dates, state.evolutionDate1, 'earliest') : state.evolutionDate1;
  const date2 = dates ? pickDate(dates, state.evolutionDate2, 'latest') : state.evolutionDate2;
  updateEvolutionDateControls(date1, date2, dates);
}

function updateEvolutionDateControls(date1, date2, dates = null) {
  const months = availableMonths(dates || []);
  populateEvolutionDateSelect(date1Input, months, monthValue(date1));
  populateEvolutionDateSelect(date2Input, months, monthValue(date2));
}

function populateEvolutionDateSelect(select, months, selectedMonth) {
  if (!select) return;
  const previous = select.value;
  select.replaceChildren();
  if (!months.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Nessuna data';
    select.appendChild(opt);
    select.disabled = true;
    return;
  }

  select.disabled = false;
  const selected = months.includes(selectedMonth) ? selectedMonth : previous;
  for (const month of months) {
    const opt = document.createElement('option');
    opt.value = month;
    opt.textContent = month;
    opt.selected = month === selected;
    select.appendChild(opt);
  }
  if (!select.value) select.value = months[0];
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
    clearEvolutionOverlay();
    clearPaiMarkers();
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
    renderSatelliteCharacteristic(seq);
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

function renderSatelliteCharacteristic(seq) {
  const layer = characteristicSatelliteLayer(currentState?.q);
  if (!layer || !map?._boscoEntries || !currentState) return;
  if (!satelliteReady(currentState.regionId)) {
    resetParcelStyles();
    renderMessageLegend('Caricamento dati satellitari...');
    loadSatellite(currentState.regionId).then(() => {
      if (seq === characteristicRenderSeq) refreshCharacteristicLayer();
    }).catch(() => {
      if (seq === characteristicRenderSeq) renderMessageLegend('Dati satellitari non disponibili.');
    });
    return;
  }

  const date = pickDate(satelliteData.timeseries?.dates, null, 'latest');
  if (!date) {
    resetParcelStyles();
    renderMessageLegend('Dati satellitari non disponibili.');
    return;
  }

  const entries = map._boscoEntries;
  const values = entries.map(entry => satelliteValue(satelliteData.timeseries, entry.key, layer, date));
  if (!continuousDomain(values)) {
    resetParcelStyles();
    renderMessageLegend('Nessun dato satellitare disponibile.');
    return;
  }

  for (const entry of entries) {
    const value = satelliteValue(satelliteData.timeseries, entry.key, layer, date);
    applyEntryStyle(entry, value == null ? NO_DATA_STYLE : satelliteValueStyle(value), value);
  }
  renderSatelliteLegend(legendEl, layer, date);
}

function renderEvolutionMode() {
  const seq = ++evolutionRenderSeq;
  if (!map?._boscoEntries || currentState?.mode !== '2') return;
  updateEvolutionControls(currentState);

  const metric = EVOLUTION_METRICS[currentState.evolutionMetric];
  if (!metric?.satellite) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, 'Prelievo in preparazione.');
    return;
  }

  if (!satelliteReady(currentState.regionId)) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, 'Caricamento dati satellitari...');
    loadSatellite(currentState.regionId).then(() => {
      if (seq === evolutionRenderSeq) renderEvolutionMode();
    }).catch(() => {
      if (seq === evolutionRenderSeq) renderLegendMessage(diffLegendEl, 'Dati satellitari non disponibili.');
    });
    return;
  }

  const dates = satelliteData.timeseries?.dates || [];
  const date1 = pickDate(dates, currentState.evolutionDate1, 'earliest');
  const date2 = pickDate(dates, currentState.evolutionDate2, 'latest');
  updateEvolutionDateControls(date1, date2, dates);
  if (!date1 || !date2) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, 'Dati satellitari non disponibili.');
    return;
  }

  if (currentState.parcelAverage) {
    renderEvolutionParcelAverages(metric, date1, date2);
  } else {
    renderEvolutionRaster(seq, metric, date1, date2);
  }
  canonicalizeEvolutionDates(currentState, date1, date2);
}

function renderEvolutionParcelAverages(metric, date1, date2) {
  clearEvolutionOverlay();
  const values = map._boscoEntries.map(entry => (
    satelliteDiffValue(satelliteData.timeseries, entry.key, metric.layer, date1, date2)
  ));
  const domain = divergingDomain(values);
  if (!domain) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, 'Nessun dato satellitare disponibile.');
    return;
  }

  for (const entry of map._boscoEntries) {
    const value = satelliteDiffValue(satelliteData.timeseries, entry.key, metric.layer, date1, date2);
    applyEntryStyle(
      entry,
      value == null ? NO_DATA_STYLE : satelliteDiffStyle(value, domain.maxAbs),
      value,
      v => evolutionMetricDisplay(metric, v),
    );
  }
  renderDiffLegend(metric, date1, date2, domain);
}

function renderEvolutionRaster(seq, metric, date1, date2) {
  clearEvolutionOverlay();
  resetParcelStyles();
  renderLegendMessage(diffLegendEl, 'Caricamento raster...');
  const url = satelliteDiffPngUrl(currentState.regionId, metric.layer, date1, date2);
  fetchImageDataURL(url).then(({ dataURL, maxAbs }) => {
    if (seq !== evolutionRenderSeq || currentState?.mode !== '2' || currentState?.parcelAverage) return;
    clearEvolutionOverlay();
    evolutionOverlay = L.imageOverlay(dataURL, satelliteData.manifest.bbox, {
      opacity: EVOLUTION_OVERLAY_OPACITY,
    }).addTo(map.leaflet);
    map.parcelLayer.bringToFront();
    renderDiffLegend(metric, date1, date2, { maxAbs });
    setStatus(`${metric.label} ${date2.slice(0, 7)} - ${date1.slice(0, 7)}`);
  }).catch(() => {
    if (seq === evolutionRenderSeq) renderLegendMessage(diffLegendEl, 'Raster non disponibile.');
  });
}

async function fetchImageDataURL(url) {
  if (!url) throw new Error('Missing image URL');
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`GET ${url} failed: ${resp.status}`);
  const maxAbs = Number(resp.headers.get('X-Bosco-Max-Abs')) || 1;
  return { dataURL: await blobToDataURL(await resp.blob()), maxAbs };
}

function blobToDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function clearEvolutionOverlay() {
  if (!evolutionOverlay || !map?.leaflet) {
    evolutionOverlay = null;
    return;
  }
  map.leaflet.removeLayer(evolutionOverlay);
  evolutionOverlay = null;
}

function clearEvolutionLegend() {
  diffLegendEl?.replaceChildren();
}

function loadSatellite(regionId) {
  if (satelliteReady(regionId)) return Promise.resolve(satelliteData);
  if (!satelliteLoad || satelliteRegionId !== regionId) {
    const requestedRegionId = regionId;
    satelliteRegionId = regionId;
    satelliteData = null;
    satelliteLoad = Promise.all([
      fetchJSON(`/api/bosco/satellite/${regionId}/manifest/`),
      fetchJSON(`/api/bosco/satellite/${regionId}/timeseries/`),
    ]).then(([manifest, timeseries]) => {
      if (satelliteRegionId !== requestedRegionId) return null;
      satelliteData = { manifest, timeseries };
      return satelliteData;
    }).finally(() => {
      if (satelliteRegionId === requestedRegionId) satelliteLoad = null;
    });
  }
  return satelliteLoad;
}

function satelliteReady(regionId) {
  return satelliteData && satelliteRegionId === regionId;
}

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`GET ${url} failed: ${resp.status}`);
  return resp.json();
}

function satelliteValueStyle(value) {
  return {
    color: '#244126',
    weight: 1.4,
    opacity: 0.92,
    fillColor: satelliteColor(value),
    fillOpacity: 0.64,
  };
}

function satelliteDiffStyle(value, maxAbs) {
  return {
    color: '#333',
    weight: 1.4,
    opacity: 0.92,
    fillColor: diffColor(value, maxAbs),
    fillOpacity: 0.66,
  };
}

function evolutionMetricDisplay(metric, value) {
  if (value == null || !Number.isFinite(value)) return 'n.d.';
  const sign = value > 0 ? '+' : '';
  return `${metric.label}: ${sign}${fmtDecimal2(value)}`;
}

function canonicalizeEvolutionDates(state, date1, date2) {
  if (!state || state.mode !== '2') return;
  const params = new URLSearchParams(location.search);
  let changed = false;
  const d1 = dateParam(date1);
  const d2 = dateParam(date2);
  if (d1 && params.get('d1') !== d1) {
    params.set('d1', d1);
    changed = true;
  }
  if (d2 && params.get('d2') !== d2) {
    params.set('d2', d2);
    changed = true;
  }
  if (changed) navigateWithParams(PAGE_PATH, params, true);
}

function resetParcelStyles() {
  if (!map?.parcelLayer) return;
  map.parcelLayer.eachLayer(layer => {
    layer.setStyle(PARCEL_STYLE);
    if (layer.feature) setLayerTooltip(layer, safeTooltip(layer.feature));
  });
}

function applyEntryStyle(entry, style, value, displayFn = null) {
  for (const layer of entry.layers) {
    layer.setStyle(style);
    setLayerTooltip(layer, buildTooltip(entry, value, displayFn));
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

function buildTooltip(entry, value, displayFn = null) {
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
  metric.textContent = displayFn ? displayFn(value) : metricDisplay(currentState?.q, value);
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
  if (characteristicSatelliteLayer(metricId)) return fmtDecimal2(value);
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

function renderSatelliteLegend(target, layer, date) {
  if (!target) return;
  target.replaceChildren();

  const title = document.createElement('div');
  title.className = 'bosco-legend-title';
  title.textContent = `${SATELLITE_LAYERS[layer]?.label || layer.toUpperCase()} - ${date}`;
  target.appendChild(title);

  const gradient = document.createElement('div');
  gradient.className = 'bosco-gradient satellite';
  target.appendChild(gradient);

  const labels = document.createElement('div');
  labels.className = 'bosco-legend-labels';
  for (const text of ['-1,0', '-0,5', '0', '+0,5', '+1,0']) {
    const span = document.createElement('span');
    span.textContent = text;
    labels.appendChild(span);
  }
  target.appendChild(labels);
}

function renderDiffLegend(metric, date1, date2, domain) {
  if (!diffLegendEl) return;
  diffLegendEl.replaceChildren();

  const title = document.createElement('div');
  title.className = 'bosco-legend-title';
  title.textContent = `${metric.label} ${date2.slice(0, 7)} - ${date1.slice(0, 7)}`;
  diffLegendEl.appendChild(title);

  const gradient = document.createElement('div');
  gradient.className = 'bosco-gradient diff';
  diffLegendEl.appendChild(gradient);

  const max = domain.maxAbs || 1;
  const labels = document.createElement('div');
  labels.className = 'bosco-legend-labels';
  for (const text of [
    `-${fmtDecimal2(max)}`, `-${fmtDecimal2(max / 2)}`, '0',
    `+${fmtDecimal2(max / 2)}`, `+${fmtDecimal2(max)}`,
  ]) {
    const span = document.createElement('span');
    span.textContent = text;
    labels.appendChild(span);
  }
  diffLegendEl.appendChild(labels);
}

function renderMessageLegend(message) {
  renderLegendMessage(legendEl, message);
}

function renderLegendMessage(target, message) {
  if (!target) return;
  target.replaceChildren();
  const p = document.createElement('div');
  p.className = 'bosco-legend-note';
  p.textContent = message;
  target.appendChild(p);
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


function syncDetailOverlay(state) {
  if (!detailOverlay) return;
  const scope = detailScopeForState(state);
  if (!scope) {
    detailOverlay.hidden = true;
    destroyProductionChart();
    return;
  }

  detailOverlay.hidden = false;
  detailTitle.textContent = scope.title;
  detailScopeLabel.textContent = scope.type === 'parcel' ? S.COL_PARCEL : S.COL_REGION;
  applyDetailSections(state.openSections);
  renderMetadata(scope);
  if (state.openSections.includes('d')) renderDendrometry();
  if (state.openSections.includes('p')) renderProduction();
}

function detailScopeForState(state = currentState) {
  if (!state?.detailMode || !map?._boscoEntries) return null;
  const region = regionById.get(state.regionId);
  if (state.detailMode === '1') {
    const entry = map._boscoEntries.find(e => e.id === state.parcelId);
    if (!entry) return null;
    return {
      type: 'parcel',
      title: `${entry.region} ${entry.parcel}`.trim(),
      regionId: state.regionId,
      region: entry.region,
      parcelId: entry.id,
      areaHa: entry.displayAreaHa,
      entries: [entry],
      entry,
    };
  }
  if (state.detailMode === '2' && region) {
    const entries = map._boscoEntries;
    return {
      type: 'region',
      title: region.name,
      regionId: region.id,
      region: region.name,
      parcelId: null,
      areaHa: entries.reduce((total, e) => total + (e.displayAreaHa || 0), 0),
      entries,
    };
  }
  return null;
}

function applyDetailSections(openSections) {
  for (const key of DETAIL_SECTIONS) {
    const open = openSections.includes(key);
    detailSections[key]?.header?.classList.toggle('open', open);
    detailSections[key]?.body?.classList.toggle('open', open);
  }
}

function renderMetadata(scope) {
  if (!metadataHost) return;
  metadataHost.replaceChildren();
  const editable = scope.type === 'parcel' && canModify();
  if (metadataActions) metadataActions.hidden = !editable;
  if (metadataEditButton) metadataEditButton.onclick = editable
    ? () => showParcelMetadataForm(scope.entry.id)
    : null;
  if (scope.type === 'parcel') renderParcelMetadata(scope.entry);
  else renderRegionMetadata(scope.entries);
}

function renderParcelMetadata(entry) {
  appendMetadataField(S.COL_LOCATION, entry.location);
  appendMetadataField(S.COL_AREA_HA, entry.displayAreaHa ? fmtArea(entry.displayAreaHa) : '');
  appendMetadataField(S.COL_AREA_CAD_HA, entry.cadastralAreaHa ? fmtArea(entry.cadastralAreaHa) : '');
  appendMetadataField(S.COL_AVE_AGE, fmtDecimal1(entry.aveAge));
  appendMetadataField(S.COL_CLASS, entry.className);
  appendMetadataField(S.COL_TYPE, entry.type);
  appendMetadataField(S.COL_ALT_MIN, fmtDecimal1(entry.altMin));
  appendMetadataField(S.COL_ALT_MAX, fmtDecimal1(entry.altMax));
  appendMetadataField(S.COL_ASPECT, entry.aspect);
  appendMetadataField(S.COL_GRADE_PCT, fmtDecimal1(entry.gradePct));
  appendMetadataField(S.COL_DESC_VEG, entry.descVeg, true);
  appendMetadataField(S.COL_DESC_GEO, entry.descGeo, true);
}


async function showParcelMetadataForm(parcelId) {
  const form = await fetchModalForm(`${PARCEL_METADATA_FORM_URL}${parcelId}/`);
  if (!form) return;
  wireParcelMetadataForm(form);
}

function wireParcelMetadataForm(form) {
  wireCancelButtons(form, dismissModal);
  interceptSubmit(form, PARCEL_METADATA_SAVE_URL, {
    onSuccess: (data) => {
      applyParcelMetadataResponse(data);
      dismissModal();
    },
    onConflict: applyParcelMetadataResponse,
    onHtml: (html, data) => {
      const newForm = renderModalForm(html);
      if (newForm) {
        wireParcelMetadataForm(newForm);
        showFormError(newForm, data.message || S.ERROR_GENERIC);
      }
    },
  });
}

function applyParcelMetadataResponse(data) {
  cache.applyResponseChanges(data);
  parcelsData = cache.get(PARCELS_ID);
  rebuildRegionIndex();
  if (map && currentState) {
    buildMapParcelEntries(currentState);
    refreshCharacteristicLayer();
    syncDetailOverlay(currentState);
  }
}

function renderRegionMetadata(entries) {
  const meta = regionMetadata(entries);
  appendMetadataField('Particelle', fmtInt(meta.count));
  appendMetadataField(S.COL_AREA_HA, fmtArea(meta.areaHa));
  appendMetadataField(S.COL_AREA_CAD_HA, fmtArea(meta.cadastralAreaHa));
  appendMetadataField(S.COL_AVE_AGE, fmtDecimal1(meta.aveAge));
  appendMetadataField(S.COL_ALT_MIN, fmtDecimal1(meta.altMin));
  appendMetadataField(S.COL_ALT_MAX, fmtDecimal1(meta.altMax));
  const types = [...meta.typeCounts.entries()].map(([name, count]) => `${name}: ${count}`).join(' · ');
  appendMetadataField(S.COL_TYPE, types);
}

function appendMetadataField(label, value, wide = false) {
  const item = document.createElement('div');
  item.className = wide ? 'bosco-metadata-item wide' : 'bosco-metadata-item';
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value || 'n.d.';
  item.append(dt, dd);
  metadataHost.appendChild(item);
}

function renderDendrometry() {
  if (!dendrometryHost) return;
  const scope = detailScopeForState();
  if (!scope || detailOverlay?.hidden) return;
  if (!dendrometryData || !dendrometryPointsData) {
    destroyDendrometryCharts();
    if (dendrometryChartGrid) dendrometryChartGrid.hidden = true;
    if (dendrometryStatus) dendrometryStatus.textContent = S.LOADING;
    Promise.all([loadDendrometry(), loadDendrometryPoints()])
      .then(renderDendrometry)
      .catch(() => {
        if (dendrometryStatus) dendrometryStatus.textContent = 'Dendrometria non disponibile.';
      });
    return;
  }

  renderDendrometrySpecies(scope);
  const baseScope = { region: scope.region, parcelId: scope.parcelId };
  const filter = currentState?.detailSpeciesIds;
  const rows = aggregateDendrometry(dendrometryData, baseScope, {
    areaHa: scope.areaHa,
    perHa: dendrometryPerHa?.checked !== false,
    speciesIds: filter,
  });
  const rawRows = aggregateDendrometry(dendrometryData, baseScope, {
    perHa: false,
    speciesIds: filter,
  });
  const heightPoints = dendrometryHeightPoints(dendrometryPointsData, baseScope, {
    speciesIds: filter,
  });
  renderDendrometryCharts(rows, rawRows, heightPoints);
}

function loadDendrometry() {
  if (!dendrometryLoad) {
    dendrometryLoad = cache.load(DENDROMETRY_ID).then(data => {
      dendrometryData = data;
      return data;
    }).finally(() => { dendrometryLoad = null; });
  }
  return dendrometryLoad;
}

function loadDendrometryPoints() {
  if (!dendrometryPointsLoad) {
    dendrometryPointsLoad = cache.load(DENDROMETRY_POINTS_ID).then(data => {
      dendrometryPointsData = data;
      return data;
    }).finally(() => { dendrometryPointsLoad = null; });
  }
  return dendrometryPointsLoad;
}

function renderDendrometrySpecies(scope) {
  if (!dendrometrySpeciesHost) return;
  const species = dendrometrySpecies(dendrometryData, { region: scope.region, parcelId: scope.parcelId });
  dendrometrySpeciesHost.replaceChildren();
  if (!species.length) {
    dendrometrySpeciesHost.textContent = 'Nessun dato dendrometrico.';
    return;
  }
  const selectedIds = currentState?.detailSpeciesIds;
  const selected = new Set(selectedIds || []);
  for (const item of species) {
    const label = document.createElement('label');
    label.className = 'bosco-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = String(item.id);
    input.checked = selectedIds == null || selected.has(item.id);
    input.addEventListener('change', () => updateDendrometrySpeciesFilter(species));
    const text = document.createElement('span');
    text.textContent = `${item.name} (${fmtInt(item.count)})`;
    label.append(input, text);
    dendrometrySpeciesHost.appendChild(label);
  }
}

function updateDendrometrySpeciesFilter(allSpecies) {
  const checked = [...dendrometrySpeciesHost.querySelectorAll('input:checked')]
    .map(input => Number(input.value));
  setDendrometrySpeciesFilter(checked.length === allSpecies.length ? null : checked);
}

function setDendrometrySpeciesFilter(selected) {
  const params = new URLSearchParams(location.search);
  if (selected == null) params.delete('ds');
  else params.set('ds', selected.join(','));
  navigateWithParams(PAGE_PATH, params, true);
}

function renderDendrometryCharts(rows, rawRows, heightPoints) {
  if (!dendrometryStatus || !dendrometryChartGrid) return;
  if (!rows.length) {
    destroyDendrometryCharts();
    dendrometryChartGrid.hidden = true;
    dendrometryStatus.textContent = 'Nessun dato dendrometrico.';
    return;
  }

  dendrometryChartGrid.hidden = false;
  dendrometryStatus.textContent = `${fmtInt(dendrometryTreeTotal(rawRows))} alberi`;
  const perHa = dendrometryPerHa?.checked !== false;
  dendrometryCharts.treeCount = renderStackedBar(
    dendrometryTreeCanvas,
    dendrometryBarChartData(rows, 'treeCount', perHa ? 'Numero alberi/ha' : 'Numero alberi'),
    dendrometryCharts.treeCount,
  );
  dendrometryCharts.volume = renderStackedBar(
    dendrometryVolumeCanvas,
    dendrometryBarChartData(rows, 'volumeM3', perHa ? 'Volume (m³/ha)' : S.COL_VOLUME_M3),
    dendrometryCharts.volume,
  );
  dendrometryCharts.basalArea = renderStackedBar(
    dendrometryBasalAreaCanvas,
    dendrometryBarChartData(rows, 'basalAreaM2', perHa ? 'Area bas. (m²/ha)' : S.COL_BASAL_AREA_M2),
    dendrometryCharts.basalArea,
  );
  dendrometryCharts.height = renderScatterChart(
    dendrometryHeightCanvas,
    dendrometryScatterChartData(heightPoints, S.COL_H_M),
    dendrometryCharts.height,
  );
  dendrometryCharts.increment = renderLineChart(
    dendrometryIncrementCanvas,
    dendrometryLineChartData(rows, 'incrementPct', S.COL_INCREMENT_PCT),
    dendrometryCharts.increment,
  );
}

function renderScatterChart(canvas, chartData, existing) {
  if (!canvas) return existing || null;
  if (existing) {
    existing.data.datasets = chartData.datasets;
    if (existing.options?.scales?.y?.title) existing.options.scales.y.title.text = chartData.yTitle;
    existing.update('none');
    return existing;
  }

  return new window.Chart(canvas, {
    type: 'scatter',
    data: { datasets: chartData.datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          beginAtZero: true,
          title: { display: true, text: S.COL_D_CM },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: chartData.yTitle },
        },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function renderLineChart(canvas, chartData, existing) {
  if (!canvas) return existing || null;
  if (existing) {
    existing.data.labels = chartData.labels;
    existing.data.datasets = chartData.datasets;
    if (existing.options?.scales?.y?.title) existing.options.scales.y.title.text = chartData.yTitle;
    existing.update('none');
    return existing;
  }

  return new window.Chart(canvas, {
    type: 'line',
    data: { labels: chartData.labels, datasets: chartData.datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: chartData.yTitle },
        },
      },
      plugins: { legend: { position: 'bottom' } },
    },
  });
}

function destroyDendrometryCharts() {
  for (const chart of Object.values(dendrometryCharts)) chart?.destroy?.();
  dendrometryCharts = {};
}

function renderProduction() {
  if (!productionHost || !productionSummary) return;
  const scope = detailScopeForState();
  if (!scope || detailOverlay?.hidden) return;
  if (productionLink) productionLink.href = prelieviUrlForScope(scope);
  if (!prelieviData) {
    destroyProductionChart();
    productionHost.hidden = true;
    productionSummary.textContent = S.LOADING;
    loadPrelievi().then(renderProduction).catch(() => {
      if (productionSummary) productionSummary.textContent = 'Produzione storica non disponibile.';
    });
    return;
  }

  const perHa = productionPerHa?.checked !== false;
  const result = aggregateProduction(prelieviData, {
    region: scope.region,
    parcel: scope.type === 'parcel' ? scope.entry.parcel : null,
  }, {
    areaHa: scope.areaHa,
    perHa,
    byMonth: productionMonthly?.checked === true,
  });

  if (!result.labels.length) {
    destroyProductionChart();
    productionHost.hidden = true;
    productionSummary.textContent = 'Nessun prelievo storico.';
    return;
  }

  productionHost.hidden = false;
  productionChart = renderStackedBar(productionCanvas, result.chartData, productionChart);
  const area = scope.areaHa || 0;
  const total = perHa && area > 0
    ? `${fmtDecimal2(result.totalQuintals / area)} q/ha`
    : fmtMass(result.totalQuintals);
  productionSummary.textContent = `${total} - ${fmtInt(result.rowCount)} interventi`;
}

function destroyProductionChart() {
  if (productionChart) {
    productionChart.destroy();
    productionChart = null;
  }
}


function renderPaiMode() {
  if (!map?._boscoEntries || currentState?.mode !== '3') return;
  if (!preservedData) {
    if (paiParcelsHost) paiParcelsHost.textContent = S.LOADING;
    if (paiSpeciesHost) paiSpeciesHost.textContent = S.LOADING;
    clearPaiMarkers();
    loadPreservedTrees().then(renderPaiMode).catch(() => {
      if (paiParcelsHost) paiParcelsHost.textContent = S.ERROR_NETWORK;
      if (paiSpeciesHost) paiSpeciesHost.textContent = S.ERROR_NETWORK;
    });
    return;
  }

  const region = regionById.get(currentState.regionId)?.name;
  const allTrees = filterPaiTrees(buildPreservedTrees(preservedData), { region });
  const parcelItems = paiParcelItems(map._boscoEntries, allTrees);
  const speciesItems = paiSpeciesItems(allTrees);
  const colors = speciesColorMap(speciesItems);
  renderPaiCheckboxes(paiParcelsHost, parcelItems, currentState.paiParcelIds, 'pp');
  renderPaiCheckboxes(paiSpeciesHost, speciesItems, currentState.paiSpeciesIds, 'ps', colors);

  const trees = filterPaiTrees(allTrees, {
    parcelIds: currentState.paiParcelIds,
    speciesIds: currentState.paiSpeciesIds,
  });
  renderPaiMarkers(trees, colors);
}

function loadPreservedTrees() {
  if (!preservedLoad) {
    preservedLoad = cache.load(PRESERVED_ID).then(data => {
      preservedData = data;
      return data;
    }).finally(() => { preservedLoad = null; });
  }
  return preservedLoad;
}

async function showPaiForm(rowId = null) {
  const url = rowId ? `${PAI_FORM_URL}${rowId}/` : paiAddFormUrl();
  const form = await fetchModalForm(url);
  if (!form) return;
  wirePaiForm(form);
}

function paiAddFormUrl() {
  const params = new URLSearchParams();
  if (currentState?.regionId) params.set('region_id', String(currentState.regionId));
  return params.toString() ? `${PAI_FORM_URL}?${params}` : PAI_FORM_URL;
}

function wirePaiForm(form) {
  wireCancelButtons(form, dismissModal);
  const latEl = form.querySelector('#id_pai_lat');
  const lonEl = form.querySelector('#id_pai_lon');
  mountUseLocationButton(latEl, lonEl, { appendTo: form.querySelector('.bosco-pai-latlon-row') });
  latEl?.addEventListener('change', () => selectPaiParcelFromLatLon(form));
  lonEl?.addEventListener('change', () => selectPaiParcelFromLatLon(form));

  interceptSubmit(form, PAI_SAVE_URL, {
    validate: validatePaiForm,
    onSuccess: (data, isSaveAndAdd) => {
      applyPaiResponse(data);
      dismissModal();
      if (isSaveAndAdd) showPaiForm();
    },
    onConflict: applyPaiResponse,
    onHtml: (html, data) => {
      const newForm = renderModalForm(html);
      if (newForm) {
        wirePaiForm(newForm);
        showFormError(newForm, data.message || S.ERROR_GENERIC);
      }
    },
  });
}

function validatePaiForm(body) {
  if (!body.lat || !body.lon) return 'Lat e Lon obbligatorie.';
  return null;
}

function selectPaiParcelFromLatLon(form) {
  const lat = parseFormFloat(form.querySelector('#id_pai_lat')?.value);
  const lon = parseFormFloat(form.querySelector('#id_pai_lon')?.value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !parcelsGeo?.features) return;
  const feature = findContainingParcel(lon, lat, parcelsGeo.features);
  if (!feature) return;
  const { compresa, particella } = parcelNames(feature);
  const entry = map?._boscoEntriesByKey?.get(parcelKey(compresa, particella));
  const select = form.querySelector('#id_pai_parcel');
  if (entry && select) select.value = String(entry.id);
}

function parseFormFloat(value) {
  if (value == null || value === '') return NaN;
  return Number(String(value).replace(',', '.'));
}

function applyPaiResponse(data) {
  cache.applyResponseChanges(data);
  preservedData = cache.get(PRESERVED_ID);
  renderPaiMode();
}

async function deletePai(rowId) {
  await deleteRowWithVersion(PRESERVED_ID, rowId, PAI_DELETE_URL, {
    onSuccess: applyPaiResponse,
    onConflict: applyPaiResponse,
  });
}

function renderPaiCheckboxes(host, items, selectedIds, paramName, colors = null) {
  if (!host) return;
  host.replaceChildren();
  if (!items.length) {
    host.textContent = 'Nessuna pianta.';
    return;
  }
  const selected = selectedIds == null ? null : new Set(selectedIds);
  for (const item of items) {
    const label = document.createElement('label');
    label.className = 'bosco-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = String(item.id);
    input.checked = selected == null || selected.has(item.id);
    input.addEventListener('change', () => updatePaiFilter(paramName, host, items));
    label.appendChild(input);
    if (colors) {
      const dot = document.createElement('span');
      dot.className = 'bosco-legend-dot';
      dot.style.backgroundColor = colors.get(item.id) || '#777';
      label.appendChild(dot);
    }
    const text = document.createElement('span');
    text.textContent = `${item.name} (${fmtInt(item.count)})`;
    label.appendChild(text);
    host.appendChild(label);
  }
}

function updatePaiFilter(paramName, host, items) {
  const checked = [...host.querySelectorAll('input:checked')].map(input => Number(input.value));
  setPaiFilter(paramName, checked, host, items.map(item => item.id));
}

function setPaiFilter(paramName, selected, host, allIds = null) {
  const params = new URLSearchParams(location.search);
  const ids = allIds || [...(host?.querySelectorAll('input') || [])].map(input => Number(input.value));
  writeOptionalIdList(params, paramName, selected, ids);
  navigateWithParams(PAGE_PATH, params, true);
}

function renderPaiMarkers(trees, colors) {
  clearPaiMarkers();
  if (!map?.leaflet) return;
  paiMarkerLayer = L.layerGroup().addTo(map.leaflet);
  for (const tree of trees) {
    const marker = L.circleMarker([tree.lat, tree.lon], {
      radius: 6,
      color: '#222',
      weight: 1,
      opacity: 0.9,
      fillColor: colors.get(tree.speciesId) || '#777',
      fillOpacity: 0.88,
      bubblingMouseEvents: false,
    });
    marker.bindTooltip(paiTooltip(tree), { direction: 'top', offset: [0, -5] });
    marker.bindPopup(paiPopup(tree));
    marker.addTo(paiMarkerLayer);
  }
}

function clearPaiMarkers() {
  if (paiMarkerLayer) {
    paiMarkerLayer.remove();
    paiMarkerLayer = null;
  }
}

function paiTooltip(tree) {
  const el = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'bosco-tooltip-title';
  title.textContent = tree.species;
  const meta = document.createElement('div');
  meta.textContent = `${tree.region} ${tree.parcel} · ${fmtInt(tree.year)}`;
  el.append(title, meta);
  return el;
}

function paiPopup(tree) {
  const el = document.createElement('div');
  el.className = 'bosco-pai-popup';
  const rows = [
    [S.COL_SPECIES, tree.species],
    [S.COL_YEAR, fmtInt(tree.year)],
    [S.COL_PARCEL, `${tree.region} ${tree.parcel}`.trim()],
    [S.COL_LAT, fmtDecimal2(tree.lat)],
    [S.COL_LON, fmtDecimal2(tree.lon)],
  ];
  for (const [label, value] of rows) {
    const div = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = `${label}: `;
    const span = document.createElement('span');
    span.textContent = value || '';
    div.append(strong, span);
    el.appendChild(div);
  }
  if (canModify()) {
    const actions = document.createElement('div');
    actions.className = 'bosco-pai-popup-actions';
    const edit = document.createElement('button');
    edit.type = 'button';
    edit.className = 'btn';
    edit.textContent = 'Modifica';
    edit.addEventListener('click', () => showPaiForm(tree.id));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn btn-delete';
    del.textContent = 'Elimina';
    del.addEventListener('click', () => deletePai(tree.id));
    actions.append(edit, del);
    el.appendChild(actions);
  }
  return el;
}

function closeDetailOverlay() {
  const params = new URLSearchParams(location.search);
  clearDetailParams(params);
  navigateWithParams(PAGE_PATH, params, true);
}

function onDetailKeyDown(e) {
  if (e.key === 'Escape' && detailOverlay && !detailOverlay.hidden) closeDetailOverlay();
}

function openParcelDetail(entry) {
  const params = new URLSearchParams(location.search);
  params.set('v', '1');
  params.set('pa', String(entry.id));
  params.delete('vo');
  params.delete('ds');
  navigateWithParams(PAGE_PATH, params, true);
}

function openRegionDetail() {
  const params = new URLSearchParams(location.search);
  params.set('v', '2');
  params.delete('pa');
  params.delete('vo');
  params.delete('ds');
  navigateWithParams(PAGE_PATH, params, true);
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
    openRegionDetail();
    return;
  }
  const { compresa, particella } = parcelNames(feature);
  const entry = map?._boscoEntriesByKey?.get(parcelKey(compresa, particella));
  const context = currentState ? characteristicContext(currentState.q) : {};
  const value = entry && currentState ? metricValue(entry, currentState.q, context) : null;
  const metric = currentState?.mode === '1' ? ` — ${metricDisplay(currentState.q, value)}` : '';
  setStatus(`${compresa} ${particella}${metric}`.trim());
  if (entry) openParcelDetail(entry);
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
  const validQ = state.mode === '2' ? VALID_EVOLUTION_METRICS : VALID_CHARACTERISTICS;
  if (!params.get('q') || !validQ.includes(params.get('q'))) {
    params.set('q', state.q);
    changed = true;
  }
  changed = canonicalizeFlag(params, 'fc', state.useCadastralArea) || changed;
  changed = canonicalizeFlag(
    params, 'fh', state.mode === '1' && state.harvestPerHa && isHarvestMetric(state.q),
  ) || changed;
  changed = canonicalizeEvolutionParams(params, state) || changed;
  changed = canonicalizeDetailParams(params, state) || changed;
  changed = canonicalizePaiParams(params, state) || changed;
  if (changed) navigateWithParams(PAGE_PATH, params, true);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text || '';
}

function canonicalizeEvolutionParams(params, state) {
  if (state.mode !== '2') {
    if (!params.has('d1') && !params.has('d2') && !params.has('fa')) return false;
    params.delete('d1');
    params.delete('d2');
    params.delete('fa');
    return true;
  }
  return canonicalizeFlag(
    params, 'fa', state.parcelAverage && Boolean(EVOLUTION_METRICS[state.evolutionMetric]?.satellite),
  );
}

function canonicalizePaiParams(params, state) {
  if (state.mode !== '3') {
    if (!params.has('pp') && !params.has('ps')) return false;
    params.delete('pp');
    params.delete('ps');
    return true;
  }

  let changed = false;
  const parcelIds = (map?._boscoEntries || []).map(e => e.id);
  changed = canonicalizeOptionalIdParam(params, 'pp', state.paiParcelIds, parcelIds) || changed;

  if (preservedData) {
    const region = regionById.get(state.regionId)?.name;
    const speciesIds = paiSpeciesItems(filterPaiTrees(buildPreservedTrees(preservedData), { region }))
      .map(item => item.id);
    changed = canonicalizeOptionalIdParam(params, 'ps', state.paiSpeciesIds, speciesIds) || changed;
  }
  return changed;
}

function canonicalizeOptionalIdParam(params, key, selectedIds, allIds) {
  if (selectedIds == null) {
    if (!params.has(key)) return false;
    params.delete(key);
    return true;
  }
  const allowed = new Set(allIds);
  const clean = selectedIds.filter(id => allowed.has(id));
  const before = params.get(key);
  const had = params.has(key);
  writeOptionalIdList(params, key, clean, allIds);
  return params.get(key) !== before || params.has(key) !== had;
}

function canonicalizeDetailParams(params, state) {
  const scope = detailScopeForState(state);
  if (!state.detailMode || !scope) {
    if (!params.has('v') && !params.has('pa') && !params.has('vo') && !params.has('ds')) return false;
    clearDetailParams(params);
    return true;
  }
  let changed = false;
  if (params.get('v') !== state.detailMode) {
    params.set('v', state.detailMode);
    changed = true;
  }
  if (state.detailMode === '1') {
    if (params.get('pa') !== String(scope.parcelId)) {
      params.set('pa', String(scope.parcelId));
      changed = true;
    }
  } else if (params.has('pa')) {
    params.delete('pa');
    changed = true;
  }

  const beforeVo = params.get('vo');
  const hadVo = params.has('vo');
  writeSectionTokens(params, state.openSections);
  if (params.get('vo') !== beforeVo || params.has('vo') !== hadVo) changed = true;

  const species = state.detailSpeciesIds.join(',');
  if (species) {
    if (params.get('ds') !== species) {
      params.set('ds', species);
      changed = true;
    }
  } else if (params.has('ds')) {
    params.delete('ds');
    changed = true;
  }
  return changed;
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


function firstNumber(...values) {
  return values.find(v => v != null && Number.isFinite(v) && v > 0) ?? null;
}

function interpolateColor(start, end, t) {
  const rgb = start.map((v, i) => Math.round(v + (end[i] - v) * t));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
