/**
 * Bosco page: map shell, URL-backed controls, and parcel characteristic styling.
 */

import * as cache from '../../base/js/cache.js';
import * as S from '../../base/js/strings.js';
import {
  COLUMNS, COL_REGION_ID, DIGEST_FUTURE_PRODUCTION, DIGEST_PARCELS, DIGEST_PARCEL_DENDROMETRY,
  DIGEST_PARCEL_DENDROMETRY_POINTS, DIGEST_PRESERVED_TREES, FIELD_LAT, FIELD_LON,
  FIELD_PARCEL_ID, FIELD_REGION_ID, FIELD_SPECIES, M2_PER_HA, ROWS,
} from '../../base/js/constants.js';
import { fetchJSON } from '../../base/js/api.js';
import {
  renderLineChart, renderScatterChart, renderStackedBar, speciesNamesFromDigest,
} from '../../base/js/charts.js';
import {
  fmtArea, fmtCoord, fmtDecimal1, fmtDecimal2, fmtInt, fmtMass, fmtVolume, parseDecimal,
} from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { findContainingParcel, sortFeaturesByArea, parcelNames } from '../../base/js/geo.js';
import { PARCEL_STYLE, ParcelMap, parcelTooltipContent } from '../../base/js/parcel-map.js';
import { createPage, navigateWithParams } from '../../base/js/page-sync.js';
import {
  deleteRowWithVersion, fetchModalForm, interceptSubmit, renderModalForm,
  showFormError,
} from '../../base/js/forms.js';
import { mountUseLocationButton } from '../../base/js/latlng-input.js';
import { dismiss as dismissModal } from '../../base/js/modals.js';
import { canModify } from '../../base/js/roles.js';
import {
  showConfirmModal, wireCancelButtons, wireCollapsibleToggle,
} from '../../base/js/ui-widgets.js';
import {
  BOSCO_MODES, MODE_CHARACTERISTICS, MODE_EVOLUTION, MODE_PAI, clearDetailParams,
  clearMapView, harvestPerHaAllowed, mapTypeToken, parcelAverageAllowed, readBoscoParams,
  writeMapView, writeOptionalIdList, writeSectionTokens,
} from './bosco-state.js';
import {
  CHARACTERISTIC_METRICS,
  CHARACTERISTIC_METRIC_IDS,
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
  perHaArea,
} from './bosco-characteristics.js';
import {
  BYTE_MIDPOINT, EVOLUTION_METRIC_IDS, EVOLUTION_METRICS, SATELLITE_LAYERS,
  availableMonths, characteristicSatelliteLayer, dateFromMonthValue, dateParam,
  diffColor, diffRgb, divergingDomain, interpolateRgb, monthValue, pickDate, rgbString,
  satelliteColor, satelliteDiffValue, satelliteMaskRawUrl, satelliteRawUrl, satelliteRgb, satelliteValue,
} from './bosco-satellite.js';
import {
  aggregateDendrometry, dendrometryBarChartData, dendrometryHeightPoints,
  dendrometryLineChartData, dendrometryScatterChartData, dendrometrySpecies,
  dendrometrySpeciesColor,
  dendrometryTreeTotal, parcelNavigation, regionMetadata,
} from './bosco-detail.js';
import {
  buildPreservedTrees, filterPaiTrees, paiParcelItems, paiSpeciesItems, speciesColorMap,
} from './bosco-pai.js';
import {
  aggregateProduction, harvestYear, pickProductionYear, prelieviUrlForScope,
  productionDeltaByParcel, productionYears,
} from './bosco-production.js';
import { E_HARVEST, Q_AGE, Q_ALTITUDE, Q_EVI, Q_NDMI, Q_NDVI } from './bosco-metrics.js';

const CSS_URL = '/static/bosco/css/bosco.css';
const PAGE_PATH = '/bosco';

const PARCELS_ID = DIGEST_PARCELS;
const PARCELS_URL = '/api/bosco/parcels/data/';
const FUTURE_ID = DIGEST_FUTURE_PRODUCTION;
const FUTURE_URL = '/api/bosco/future-production/data/';
const SPECIES_ID = FIELD_SPECIES;
const SPECIES_URL = '/api/species/data/';
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

const VALID_MODES = BOSCO_MODES;
const VALID_MAP_TYPES = ['o', 't', 's'];
const VALID_CHARACTERISTICS = CHARACTERISTIC_METRIC_IDS;
const VALID_EVOLUTION_METRICS = EVOLUTION_METRIC_IDS;
const DETAIL_SECTIONS = ['m', 'd', 'p'];
const SATELLITE_OVERLAY_OPACITY = 0.85;
const SATELLITE_INSIDE_ALPHA = 210;
const SATELLITE_OUTSIDE_ALPHA = 60;
const SATELLITE_OVERLAY_CLASS = 'bosco-satellite-raster-overlay';
const TYPE_HIGHFOREST_KEY = 'highforest';
const TYPE_COPPICE_KEY = 'coppice';
const RASTER_TOOLTIP_HANDLERS = Symbol('rasterTooltipHandlers');

const NO_DATA_STYLE = {
  ...PARCEL_STYLE,
  color: '#777',
  weight: 1,
  opacity: 0.85,
  fillColor: '#d6d6d6',
  fillOpacity: 0.34,
};

const TYPE_STYLES = {
  [TYPE_HIGHFOREST_KEY]: {
    color: '#17613a', weight: 1.5, opacity: 0.95, fillColor: '#2f8f58', fillOpacity: 0.58,
  },
  [TYPE_COPPICE_KEY]: {
    color: '#8a6500', weight: 1.5, opacity: 0.95, fillColor: '#d7aa27', fillOpacity: 0.62,
  },
};

cache.register(PARCELS_ID, PARCELS_URL);
cache.register(FUTURE_ID, FUTURE_URL);
cache.register(SPECIES_ID, SPECIES_URL);
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
let characteristicParcelAverageToggle = null;
let characteristicParcelAverageRow = null;
let evolutionSelect = null;
let date1Input = null;
let date2Input = null;
let parcelAverageToggle = null;
let parcelAverageRow = null;
let evolutionPerHaToggle = null;
let evolutionPerHaRow = null;
let evolutionCadastralToggle = null;
let diffLegendEl = null;
let legendEl = null;
let statusEl = null;
let detailOverlay = null;
let detailTitle = null;
let detailScopeLabel = null;
let detailPrevButton = null;
let detailNextButton = null;
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
let speciesNames = [];
let prelieviData = null;
let dendrometryData = null;
let dendrometryPointsData = null;
let preservedData = null;
let satelliteData = null;
let satelliteRegionId = null;
let satelliteLoad = null;
let satelliteRasterOverlay = null;
let rasterTooltipContext = null;
let rawRasterCache = new Map();
let paiMarkerLayer = null;
let parcelsGeo = null;
let map = null;
let mapRegionId = null;
let mapEntries = [];
let mapEntriesByKey = new Map();
let regions = [];
let regionById = new Map();
let currentState = null;
let suppressViewSync = false;
let characteristicRenderSeq = 0;
let evolutionRenderSeq = 0;

const loadPrelievi = makeLoader(PRELIEVI_ID, data => { prelieviData = data; });
const loadDendrometry = makeLoader(DENDROMETRY_ID, data => { dendrometryData = data; });
const loadDendrometryPoints = makeLoader(
  DENDROMETRY_POINTS_ID, data => { dendrometryPointsData = data; },
);
const loadPreservedTrees = makeLoader(PRESERVED_ID, data => { preservedData = data; });

const page = createPage({
  cssUrl: CSS_URL,
  load: loadPageData,
  mount: mountPage,
  unmount: destroyPage,
  onQueryChange: applyParams,
  onUpdate: [
    [PARCELS_ID, onParcelsUpdate],
    [FUTURE_ID, onFutureUpdate],
    [SPECIES_ID, onSpeciesUpdate],
    [DENDROMETRY_ID, onDendrometryUpdate],
    [DENDROMETRY_POINTS_ID, onDendrometryPointsUpdate],
    [PRESERVED_ID, onPreservedUpdate],
    [PRELIEVI_ID, onPrelieviUpdate],
  ],
  visibleIds: [
    PARCELS_ID, FUTURE_ID, SPECIES_ID, DENDROMETRY_ID, DENDROMETRY_POINTS_ID,
    PRESERVED_ID, PRELIEVI_ID,
  ],
});

export const mount = page.mount;
export const unmount = page.unmount;
export const onQueryChange = page.onQueryChange;

function makeLoader(dataId, assign) {
  let pending = null;
  return () => {
    if (!pending) {
      pending = cache.load(dataId).then(data => {
        assign(data);
        return data;
      }).finally(() => { pending = null; });
    }
    return pending;
  };
}

async function loadPageData() {
  const [parcels, future, species] = await Promise.all([
    cache.load(PARCELS_ID),
    cache.load(FUTURE_ID),
    cache.load(SPECIES_ID),
    cache.get(TERRENI_ID) ? Promise.resolve(cache.get(TERRENI_ID)) : cache.load(TERRENI_ID),
  ]);
  parcelsData = parcels;
  futureData = future;
  speciesNames = speciesNamesFromDigest(species);
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
  characteristicParcelAverageToggle = el.querySelector('[data-role="characteristic-parcel-average-toggle"]');
  characteristicParcelAverageRow = el.querySelector('[data-role="characteristic-parcel-average-row"]');
  evolutionSelect = el.querySelector('[data-role="evolution-select"]');
  date1Input = el.querySelector('[data-role="date1"]');
  date2Input = el.querySelector('[data-role="date2"]');
  parcelAverageToggle = el.querySelector('[data-role="parcel-average-toggle"]');
  parcelAverageRow = el.querySelector('[data-role="parcel-average-row"]');
  evolutionPerHaToggle = el.querySelector('[data-role="evolution-per-ha-toggle"]');
  evolutionPerHaRow = el.querySelector('[data-role="evolution-per-ha-row"]');
  evolutionCadastralToggle = el.querySelector('[data-role="evolution-cadastral-toggle"]');
  diffLegendEl = el.querySelector('[data-target="diff-legend"]');
  legendEl = el.querySelector('[data-target="legend"]');
  statusEl = el.querySelector('[data-role="status"]');
  detailOverlay = el.querySelector('[data-target="detail-overlay"]');
  detailTitle = el.querySelector('[data-target="detail-title"]');
  detailScopeLabel = el.querySelector('[data-target="detail-scope"]');
  detailPrevButton = el.querySelector('[data-action="detail-prev-parcel"]');
  detailNextButton = el.querySelector('[data-action="detail-next-parcel"]');
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
  characteristicParcelAverageToggle = characteristicParcelAverageRow = null;
  evolutionSelect = date1Input = date2Input = parcelAverageToggle = null;
  parcelAverageRow = evolutionPerHaToggle = evolutionPerHaRow = null;
  evolutionCadastralToggle = diffLegendEl = legendEl = null;
  detailOverlay = detailTitle = detailScopeLabel = metadataHost = null;
  detailPrevButton = detailNextButton = null;
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
  speciesNames = [];
  dendrometryPointsData = preservedData = parcelsGeo = null;
  satelliteData = satelliteRegionId = satelliteLoad = null;
  satelliteRasterOverlay = rasterTooltipContext = paiMarkerLayer = null;
  rawRasterCache = new Map();
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

function onSpeciesUpdate(data) {
  speciesNames = speciesNamesFromDigest(data);
  renderDendrometry();
  renderPaiMode();
}

function onPrelieviUpdate(data) {
  prelieviData = data;
  refreshCharacteristicLayer();
  if (currentState?.mode === MODE_EVOLUTION && currentState.evolutionMetric === E_HARVEST) {
    renderEvolutionMode();
  }
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
  const idxRegionId = cols.indexOf(COL_REGION_ID);
  const idxRegion = cols.indexOf(S.COL_REGION);
  if (idxRegionId < 0 || idxRegion < 0) return;

  const byId = new Map();
  for (const row of parcelsData[ROWS]) {
    const id = row[idxRegionId];
    if (id == null || byId.has(id)) continue;
    byId.set(id, { id, name: row[idxRegion] });
  }
  regions = [...byId.values()].sort((a, b) => a.name.localeCompare(b.name, S.LOCALE));
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
    if (!characteristicSatelliteLayer(characteristicSelect.value)) params.delete('fa');
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

  characteristicParcelAverageToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    setFlagParam(params, 'fa', characteristicParcelAverageToggle.checked);
    navigateWithParams(PAGE_PATH, params, true);
  });

  evolutionSelect?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('m', MODE_EVOLUTION);
    params.set('q', evolutionSelect.value);
    navigateWithParams(PAGE_PATH, params, true);
  });

  date1Input?.addEventListener('change', () => updateEvolutionDateParam('d1', date1Input));
  date2Input?.addEventListener('change', () => updateEvolutionDateParam('d2', date2Input));

  parcelAverageToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('m', MODE_EVOLUTION);
    setFlagParam(params, 'fa', parcelAverageToggle.checked);
    navigateWithParams(PAGE_PATH, params, true);
  });

  evolutionPerHaToggle?.addEventListener('change', () => {
    const params = new URLSearchParams(location.search);
    params.set('m', MODE_EVOLUTION);
    params.set('q', E_HARVEST);
    setFlagParam(params, 'fh', evolutionPerHaToggle.checked);
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
  const value = currentState?.mode === MODE_EVOLUTION && currentState.evolutionMetric === E_HARVEST
    ? harvestYear(input.value)
    : dateParam(dateFromMonthValue(input.value));
  if (value) params.set(key, value);
  else params.delete(key);
  navigateWithParams(PAGE_PATH, params, true);
}

function wireDetailControls() {
  detailOverlay?.querySelector('[data-action="close-detail"]')
    ?.addEventListener('click', closeDetailOverlay);
  detailPrevButton?.addEventListener('click', () => openAdjacentParcel(-1));
  detailNextButton?.addEventListener('click', () => openAdjacentParcel(1));
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
    || mapRegionId !== state.regionId
    || map.wrapper?.getBasemap() !== state.basemap;
  if (needsMapRebuild) renderMap(state);
  else {
    applyMapView(state);
    updateMapDisplayAreas(state);
    refreshCharacteristicLayer();
  }
  if (state.mode === MODE_EVOLUTION) renderEvolutionMode();
  else clearEvolutionLegend();
  syncDetailOverlay(state);
  if (state.mode === MODE_PAI) renderPaiMode();
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
  const satellite = Boolean(characteristicSatelliteLayer(state.q));
  setControlVisible(perHaRow, harvest);
  if (perHaToggle) {
    perHaToggle.checked = harvest && state.harvestPerHa;
    perHaToggle.disabled = !harvest;
  }
  setControlVisible(characteristicParcelAverageRow, satellite);
  if (characteristicParcelAverageToggle) {
    characteristicParcelAverageToggle.checked = satellite && state.parcelAverage;
    characteristicParcelAverageToggle.disabled = !satellite;
  }
}

function updateEvolutionControls(state) {
  if (evolutionSelect) evolutionSelect.value = state.evolutionMetric;
  const supportsRaster = Boolean(EVOLUTION_METRICS[state.evolutionMetric]?.satellite);
  const supportsHarvestPerHa = state.evolutionMetric === E_HARVEST;
  setControlVisible(parcelAverageRow, supportsRaster);
  if (parcelAverageToggle) {
    parcelAverageToggle.checked = supportsRaster && state.parcelAverage;
    parcelAverageToggle.disabled = !supportsRaster;
  }
  setControlVisible(evolutionPerHaRow, supportsHarvestPerHa);
  if (evolutionPerHaToggle) {
    evolutionPerHaToggle.checked = supportsHarvestPerHa && state.harvestPerHa;
    evolutionPerHaToggle.disabled = !supportsHarvestPerHa;
  }
  if (evolutionCadastralToggle) evolutionCadastralToggle.checked = state.useCadastralArea;

  if (state.evolutionMetric === E_HARVEST) {
    const years = productionYears(prelieviData);
    const year1 = pickProductionYear(years, state.evolutionDate1, 'earliest');
    const year2 = pickProductionYear(years, state.evolutionDate2, 'latest');
    updateEvolutionYearControls(year1, year2, years);
    return;
  }

  const dates = satelliteRegionId === state.regionId ? satelliteData?.timeseries?.dates : null;
  const date1 = dates ? pickDate(dates, state.evolutionDate1, 'earliest') : state.evolutionDate1;
  const date2 = dates ? pickDate(dates, state.evolutionDate2, 'latest') : state.evolutionDate2;
  updateEvolutionDateControls(date1, date2, dates);
}

function setControlVisible(el, visible) {
  if (!el) return;
  el.classList.toggle('hidden', !visible);
  el.setAttribute('aria-hidden', visible ? 'false' : 'true');
}

function updateEvolutionDateControls(date1, date2, dates = null) {
  const months = availableMonths(dates || []);
  populateEvolutionSelect(date1Input, months, monthValue(date1));
  populateEvolutionSelect(date2Input, months, monthValue(date2));
}

function updateEvolutionYearControls(year1, year2, years = []) {
  populateEvolutionSelect(date1Input, years, year1);
  populateEvolutionSelect(date2Input, years, year2);
}

function populateEvolutionSelect(select, values, selectedValue) {
  if (!select) return;
  const previous = select.value;
  select.replaceChildren();
  if (!values.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = S.BOSCO_NO_DATE;
    select.appendChild(opt);
    select.disabled = true;
    return;
  }

  select.disabled = false;
  const selected = values.includes(selectedValue) ? selectedValue : previous;
  for (const value of values) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    opt.selected = value === selected;
    select.appendChild(opt);
  }
  if (!select.value) select.value = values[0];
}

function renderMap(state) {
  destroyMap();
  if (!mapHost) return;
  mapHost.replaceChildren();

  const region = regionById.get(state.regionId);
  if (!region || !parcelsGeo) {
    setStatus(S.BOSCO_NO_REGION);
    return;
  }

  const features = parcelsGeo.features.filter(f => parcelNames(f).compresa === region.name);
  if (!features.length) {
    setStatus(S.BOSCO_NO_GEOMETRY(region.name));
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
  mapRegionId = state.regionId;
  map.leaflet.on('basemapchange', (e) => onBasemapChange(e.name));
  setTimeout(() => { suppressViewSync = false; }, 0);
  buildMapParcelEntries(state);
  refreshCharacteristicLayer();
  setStatus(S.BOSCO_PARCELS(region.name, fmtInt(mapEntries.length)));
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
    entry.geoAreaHa += (layer.feature?.properties?._areaM2 || 0) / M2_PER_HA;
    bindRasterTooltipHandlers(layer, entry);
  });

  mapEntries = entries;
  mapEntriesByKey = byKey;
  updateMapDisplayAreas(state);
}

function bindRasterTooltipHandlers(layer, entry) {
  const previous = layer[RASTER_TOOLTIP_HANDLERS];
  if (previous) {
    layer.off?.('mousemove', previous.move);
    layer.off?.('mouseout', previous.out);
  }
  const handlers = {
    move: e => onRasterTooltipMove(entry, layer, e.latlng),
    out: () => onRasterTooltipOut(entry, layer),
  };
  layer.on('mousemove', handlers.move);
  layer.on('mouseout', handlers.out);
  layer[RASTER_TOOLTIP_HANDLERS] = handlers;
}

function updateMapDisplayAreas(state) {
  if (!map) return;
  for (const entry of mapEntries) {
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
    clearSatelliteRasterOverlay();
    clearPaiMarkers();
    map.destroy();
    map = null;
  }
  mapRegionId = null;
  mapEntries = [];
  mapEntriesByKey = new Map();
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
  if (!map || !currentState) return;
  if (currentState.mode !== MODE_CHARACTERISTICS) {
    clearSatelliteRasterOverlay();
    resetParcelStyles();
    clearLegend();
    return;
  }

  updateCharacteristicControls(currentState);
  if (characteristicSatelliteLayer(currentState.q)) {
    renderSatelliteCharacteristic(seq);
    return;
  }

  if (currentState.q === Q_HISTORICAL_HARVEST && !prelieviData) {
    resetParcelStyles();
    renderMessageLegend(S.BOSCO_LOADING_HARVESTS);
    loadPrelievi().then(() => {
      if (seq === characteristicRenderSeq) refreshCharacteristicLayer();
    }).catch(() => {
      if (seq === characteristicRenderSeq) renderMessageLegend(S.BOSCO_HARVESTS_UNAVAILABLE);
    });
    return;
  }

  clearSatelliteRasterOverlay();
  const context = characteristicContext(currentState.q);
  const entries = mapEntries;
  if (currentState.q === Q_TYPE) {
    let hasNoData = false;
    for (const entry of entries) {
      const value = metricValue(entry, currentState.q, context);
      const key = standTypeKey(entry);
      if (!TYPE_STYLES[key]) hasNoData = true;
      applyEntryStyle(entry, TYPE_STYLES[key] || NO_DATA_STYLE, value);
    }
    renderTypeLegend(hasNoData);
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

function renderSatelliteCharacteristic(seq) {
  const layer = characteristicSatelliteLayer(currentState?.q);
  if (!layer || !map || !currentState) return;
  if (!satelliteReady(currentState.regionId)) {
    clearSatelliteRasterOverlay();
    resetParcelStyles();
    renderMessageLegend(S.BOSCO_LOADING_SATELLITE);
    loadSatellite(currentState.regionId).then(() => {
      if (seq === characteristicRenderSeq) refreshCharacteristicLayer();
    }).catch(() => {
      if (seq === characteristicRenderSeq) renderMessageLegend(S.BOSCO_SATELLITE_UNAVAILABLE);
    });
    return;
  }

  const date = pickDate(satelliteData.timeseries?.dates, null, 'latest');
  if (!date) {
    clearSatelliteRasterOverlay();
    resetParcelStyles();
    renderMessageLegend(S.BOSCO_SATELLITE_UNAVAILABLE);
    return;
  }

  if (!currentState.parcelAverage) {
    renderCharacteristicRaster(seq, layer, date);
    return;
  }

  clearSatelliteRasterOverlay();
  const entries = mapEntries;
  const values = entries.map(entry => satelliteValue(satelliteData.timeseries, entry.key, layer, date));
  if (!continuousDomain(values)) {
    resetParcelStyles();
    renderMessageLegend(S.BOSCO_NO_SATELLITE);
    return;
  }

  for (const entry of entries) {
    const value = satelliteValue(satelliteData.timeseries, entry.key, layer, date);
    applyEntryStyle(entry, value == null ? NO_DATA_STYLE : satelliteValueStyle(value), value);
  }
  renderSatelliteLegend(legendEl, layer, date);
}

function renderCharacteristicRaster(seq, layer, date) {
  clearSatelliteRasterOverlay();
  resetParcelStyles();
  renderMessageLegend(S.BOSCO_LOADING_RASTER);
  const rawUrl = satelliteRawUrl(currentState.regionId, layer, date);
  const maskUrl = satelliteMaskRawUrl(currentState.regionId);
  Promise.all([loadRawRaster(rawUrl), loadRawRaster(maskUrl)]).then(([raster, mask]) => {
    if (seq !== characteristicRenderSeq || currentState?.mode !== MODE_CHARACTERISTICS || currentState?.parcelAverage) return;
    clearSatelliteRasterOverlay();
    satelliteRasterOverlay = L.imageOverlay(renderSatelliteValueOverlay(raster, mask), raster.bbox, {
      className: SATELLITE_OVERLAY_CLASS,
      opacity: SATELLITE_OVERLAY_OPACITY,
    }).addTo(map.leaflet);
    map.parcelLayer.bringToFront();
    setRasterTooltipContext({
      sampleValue: latlng => rawRasterValue(raster, latlng),
      displayFn: v => metricTooltipDisplay(currentState.q, v),
    });
    renderSatelliteLegend(legendEl, layer, date);
    setStatus(`${SATELLITE_LAYERS[layer]?.label || layer.toUpperCase()} ${date.slice(0, 7)}`);
  }).catch(() => {
    if (seq === characteristicRenderSeq) renderMessageLegend(S.BOSCO_RASTER_UNAVAILABLE);
  });
}

function renderEvolutionMode() {
  const seq = ++evolutionRenderSeq;
  if (!map || currentState?.mode !== MODE_EVOLUTION) return;
  updateEvolutionControls(currentState);

  const metric = EVOLUTION_METRICS[currentState.evolutionMetric];
  if (!metric?.satellite) {
    renderEvolutionHarvest(seq);
    return;
  }

  if (!satelliteReady(currentState.regionId)) {
    clearSatelliteRasterOverlay();
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_LOADING_SATELLITE);
    loadSatellite(currentState.regionId).then(() => {
      if (seq === evolutionRenderSeq) renderEvolutionMode();
    }).catch(() => {
      if (seq === evolutionRenderSeq) renderLegendMessage(diffLegendEl, S.BOSCO_SATELLITE_UNAVAILABLE);
    });
    return;
  }

  const dates = satelliteData.timeseries?.dates || [];
  const date1 = pickDate(dates, currentState.evolutionDate1, 'earliest');
  const date2 = pickDate(dates, currentState.evolutionDate2, 'latest');
  updateEvolutionDateControls(date1, date2, dates);
  if (!date1 || !date2) {
    clearSatelliteRasterOverlay();
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_SATELLITE_UNAVAILABLE);
    return;
  }

  if (currentState.parcelAverage) {
    renderEvolutionParcelAverages(metric, date1, date2);
  } else {
    renderEvolutionRaster(seq, metric, date1, date2);
  }
  canonicalizeEvolutionDates(currentState, date1, date2);
}

function renderEvolutionHarvest(seq) {
  clearSatelliteRasterOverlay();
  if (!prelieviData) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_LOADING_HARVESTS);
    loadPrelievi().then(() => {
      if (seq === evolutionRenderSeq) renderEvolutionMode();
    }).catch(() => {
      if (seq === evolutionRenderSeq) renderLegendMessage(diffLegendEl, S.BOSCO_HARVESTS_UNAVAILABLE);
    });
    return;
  }

  const scope = evolutionProductionScope(currentState);
  const years = productionYears(prelieviData);
  const fromYear = pickProductionYear(years, currentState.evolutionDate1, 'earliest');
  const toYear = pickProductionYear(years, currentState.evolutionDate2, 'latest');
  updateEvolutionYearControls(fromYear, toYear, years);
  if (!fromYear || !toYear) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_NO_DATA_AVAILABLE);
    return;
  }

  const deltas = productionDeltaByParcel(prelieviData, scope, fromYear, toYear);
  const values = mapEntries.map(entry => harvestDeltaValue(entry, deltas));
  const domain = divergingDomain(values);
  if (!domain) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_NO_DATA_AVAILABLE);
    return;
  }

  for (const entry of mapEntries) {
    const value = harvestDeltaValue(entry, deltas);
    applyEntryStyle(
      entry,
      value == null ? NO_DATA_STYLE : satelliteDiffStyle(value, domain.maxAbs),
      value,
      v => `${S.BOSCO_HARVEST_METRIC}: ${signedHarvestDeltaDisplay(v)}`,
    );
  }
  renderHarvestDeltaLegend(fromYear, toYear, domain);
  setStatus(`${S.BOSCO_HARVEST_METRIC} ${toYear} - ${fromYear}`);
  canonicalizeEvolutionDates(currentState, fromYear, toYear);
}

function evolutionProductionScope(state) {
  const region = regionById.get(state?.regionId);
  return region ? { region: region.name } : {};
}

function harvestDeltaValue(entry, deltas) {
  const raw = deltas.get(entry.key) || 0;
  if (!currentState?.harvestPerHa) return raw;
  const area = perHaArea(entry);
  return area ? raw / area : null;
}

function signedHarvestDeltaDisplay(value) {
  if (value == null || !Number.isFinite(value)) return S.BOSCO_NO_DATA;
  return signedDisplay(value, v => metricDisplay(Q_HISTORICAL_HARVEST, v));
}

function renderEvolutionParcelAverages(metric, date1, date2) {
  clearSatelliteRasterOverlay();
  const values = mapEntries.map(entry => (
    satelliteDiffValue(satelliteData.timeseries, entry.key, metric.layer, date1, date2)
  ));
  const domain = divergingDomain(values);
  if (!domain) {
    resetParcelStyles();
    renderLegendMessage(diffLegendEl, S.BOSCO_NO_SATELLITE);
    return;
  }

  for (const entry of mapEntries) {
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
  clearSatelliteRasterOverlay();
  resetParcelStyles();
  renderLegendMessage(diffLegendEl, S.BOSCO_LOADING_RASTER);
  const rawUrl1 = satelliteRawUrl(currentState.regionId, metric.layer, date1);
  const rawUrl2 = satelliteRawUrl(currentState.regionId, metric.layer, date2);
  const maskUrl = satelliteMaskRawUrl(currentState.regionId);
  Promise.all([
    loadRawRaster(rawUrl1), loadRawRaster(rawUrl2), loadRawRaster(maskUrl),
  ]).then(([raster1, raster2, mask]) => {
    if (seq !== evolutionRenderSeq || currentState?.mode !== MODE_EVOLUTION || currentState?.parcelAverage) return;
    const maxAbs = rawRasterDiffMaxAbs(raster1, raster2, mask);
    clearSatelliteRasterOverlay();
    const overlayUrl = renderSatelliteDiffOverlay(raster1, raster2, mask, maxAbs);
    satelliteRasterOverlay = L.imageOverlay(overlayUrl, raster1.bbox, {
      className: SATELLITE_OVERLAY_CLASS,
      opacity: SATELLITE_OVERLAY_OPACITY,
    }).addTo(map.leaflet);
    map.parcelLayer.bringToFront();
    setRasterTooltipContext({
      sampleValue: latlng => rawRasterDiffValue(raster1, raster2, latlng),
      displayFn: v => evolutionMetricDisplay(metric, v),
    });
    renderDiffLegend(metric, date1, date2, { maxAbs });
    setStatus(`${evolutionMetricLabel(metric)} ${date2.slice(0, 7)} - ${date1.slice(0, 7)}`);
  }).catch(() => {
    if (seq === evolutionRenderSeq) renderLegendMessage(diffLegendEl, S.BOSCO_RASTER_UNAVAILABLE);
  });
}

function renderSatelliteValueOverlay(raster, mask) {
  assertRasterCompatible(raster, mask);
  return rasterDataUrl(raster, (idx) => {
    const rgb = satelliteRgb(rawByteValue(raster.values[idx]));
    return [rgb[0], rgb[1], rgb[2], maskAlpha(mask, idx)];
  });
}

function renderSatelliteDiffOverlay(raster1, raster2, mask, maxAbs) {
  assertRasterCompatible(raster1, raster2, mask);
  return rasterDataUrl(raster1, (idx) => {
    const value = rawByteDiffValue(raster1.values[idx], raster2.values[idx]);
    const rgb = diffRgb(value, maxAbs);
    return [rgb[0], rgb[1], rgb[2], maskAlpha(mask, idx)];
  });
}

function rasterDataUrl(raster, pixelFn) {
  if (!validRaster(raster)) throw new Error('Invalid raster');
  const canvas = document.createElement('canvas');
  canvas.width = raster.width;
  canvas.height = raster.height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas rendering unavailable');
  const imageData = context.createImageData(raster.width, raster.height);
  for (let idx = 0; idx < raster.values.length; idx++) {
    const offset = idx * 4;
    const rgba = pixelFn(idx);
    imageData.data[offset] = rgba[0];
    imageData.data[offset + 1] = rgba[1];
    imageData.data[offset + 2] = rgba[2];
    imageData.data[offset + 3] = rgba[3];
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL('image/png');
}

function rawRasterDiffMaxAbs(raster1, raster2, mask) {
  assertRasterCompatible(raster1, raster2, mask);
  let maxAbs = 0;
  for (let idx = 0; idx < raster1.values.length; idx++) {
    if (mask?.values && mask.values[idx] <= 0) continue;
    const value = Math.abs(rawByteDiffValue(raster1.values[idx], raster2.values[idx]));
    maxAbs = Math.max(maxAbs, value);
  }
  return maxAbs || 1;
}

function assertRasterCompatible(base, ...rasters) {
  if (!validRaster(base)) throw new Error('Invalid raster');
  for (const raster of rasters) {
    if (!raster) continue;
    if (!validRaster(raster) || raster.width !== base.width || raster.height !== base.height) {
      throw new Error('Mismatched raster dimensions');
    }
  }
}

function validRaster(raster) {
  return Boolean(
    raster?.values
      && raster.width > 0
      && raster.height > 0
      && raster.values.length === raster.width * raster.height,
  );
}

function maskAlpha(mask, idx) {
  return !mask?.values || mask.values[idx] > 0 ? SATELLITE_INSIDE_ALPHA : SATELLITE_OUTSIDE_ALPHA;
}

function clearSatelliteRasterOverlay() {
  setRasterTooltipContext(null);
  if (!satelliteRasterOverlay || !map?.leaflet) {
    satelliteRasterOverlay = null;
    return;
  }
  map.leaflet.removeLayer(satelliteRasterOverlay);
  satelliteRasterOverlay = null;
}

function setRasterTooltipContext(context) {
  rasterTooltipContext = context;
}

function onRasterTooltipMove(entry, layer, latlng) {
  const context = rasterTooltipContext;
  if (!context || !latlng) return;
  setLayerTooltip(layer, buildTooltip(entry, context.sampleValue(latlng), context.displayFn));
}

function onRasterTooltipOut(entry, layer) {
  if (!rasterTooltipContext) return;
  setLayerTooltip(layer, buildTooltip(entry));
}

function loadRawRaster(url) {
  if (!url) return Promise.reject(new Error('Missing raw raster URL'));
  if (!rawRasterCache.has(url)) {
    rawRasterCache.set(url, fetchJSON(url).then(result => decodeRawRaster(result.data)));
  }
  return rawRasterCache.get(url);
}

function decodeRawRaster(data) {
  const bytes = decodeBase64Bytes(data.data || '');
  return {
    width: Number(data.width) || 0,
    height: Number(data.height) || 0,
    bbox: data.bbox,
    values: bytes,
  };
}

function decodeBase64Bytes(encoded) {
  const binary = atob(encoded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

function rawRasterValue(raster, latlng) {
  const raw = rawRasterByte(raster, latlng);
  return raw == null ? null : rawByteValue(raw);
}

function rawRasterDiffValue(raster1, raster2, latlng) {
  const raw1 = rawRasterByte(raster1, latlng);
  const raw2 = rawRasterByte(raster2, latlng);
  return raw1 == null || raw2 == null ? null : rawByteDiffValue(raw1, raw2);
}

function rawByteValue(raw) {
  return raw / BYTE_MIDPOINT - 1;
}

function rawByteDiffValue(raw1, raw2) {
  return (raw2 - raw1) / BYTE_MIDPOINT;
}

function rawRasterByte(raster, latlng) {
  if (!raster?.values || !Array.isArray(raster.bbox) || raster.width <= 0 || raster.height <= 0) return null;
  const [[south, west], [north, east]] = raster.bbox;
  if (latlng.lat < south || latlng.lat > north || latlng.lng < west || latlng.lng > east) return null;
  const col = Math.min(raster.width - 1, Math.max(0, Math.floor(((latlng.lng - west) / (east - west)) * raster.width)));
  const row = Math.min(raster.height - 1, Math.max(0, Math.floor(((north - latlng.lat) / (north - south)) * raster.height)));
  const idx = row * raster.width + col;
  return idx >= 0 && idx < raster.values.length ? raster.values[idx] : null;
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
      fetchJSON(`/api/bosco/satellite/${regionId}/manifest/`).then(result => result.data),
      fetchJSON(`/api/bosco/satellite/${regionId}/timeseries/`).then(result => result.data),
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
  const label = evolutionMetricLabel(metric);
  if (value == null || !Number.isFinite(value)) return `${label}: ${S.BOSCO_NO_DATA}`;
  return `${label}: ${signedDisplay(value, fmtDecimal2)}`;
}

/** Prefix a positive number with '+' (negatives already carry '-'). */
function signedDisplay(value, format) {
  return value > 0 ? `+${format(value)}` : format(value);
}

function evolutionMetricLabel(metric) {
  if (!metric) return '';
  if (metric.layer === 'prelievo') return S.BOSCO_HARVEST_METRIC;
  return metric.label || SATELLITE_LAYERS[metric.layer]?.label || String(metric.layer || '').toUpperCase();
}

function canonicalizeEvolutionDates(state, date1, date2) {
  if (!state || state.mode !== MODE_EVOLUTION) return;
  const params = new URLSearchParams(location.search);
  let changed = false;
  const d1 = state.evolutionMetric === E_HARVEST ? harvestYear(date1) : dateParam(date1);
  const d2 = state.evolutionMetric === E_HARVEST ? harvestYear(date2) : dateParam(date2);
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
    if (!layer.feature) return;
    const entry = entryForFeature(layer.feature);
    setLayerTooltip(layer, entry ? buildTooltip(entry) : parcelTooltipContent(layer.feature));
  });
}

function applyEntryStyle(entry, style, value, displayFn = null) {
  for (const layer of entry.layers) {
    layer.setStyle(style);
    setLayerTooltip(layer, buildTooltip(entry, value, displayFn));
  }
}

function setLayerTooltip(layer, content) {
  if (!content) {
    layer.unbindTooltip?.();
    return;
  }
  if (layer.getTooltip?.()) layer.setTooltipContent(content);
  else layer.bindTooltip(content, { sticky: true, direction: 'top' });
}

function continuousStyle(t) {
  return {
    color: '#244126',
    weight: 1.4,
    opacity: 0.92,
    fillColor: rgbString(interpolateRgb([245, 222, 91], [32, 113, 75], t)),
    fillOpacity: 0.64,
  };
}

function buildTooltip(entry, value = undefined, displayFn = null) {
  const el = document.createElement('div');
  el.className = 'bosco-tooltip';

  const title = document.createElement('div');
  title.className = 'parcel-tooltip-title';
  title.textContent = `${entry.region} ${entry.parcel}`.trim();
  el.appendChild(title);

  const meta = document.createElement('div');
  const area = entry.displayAreaHa ? fmtArea(entry.displayAreaHa) : '';
  meta.textContent = [entry.type, area].filter(Boolean).join(' · ');
  el.appendChild(meta);

  if (value !== undefined || displayFn) {
    const metric = document.createElement('div');
    metric.textContent = displayFn ? displayFn(value) : metricTooltipDisplay(currentState?.q, value);
    el.appendChild(metric);
  }
  return el;
}

function entryForFeature(feature) {
  const { compresa, particella } = parcelNames(feature);
  return mapEntriesByKey?.get(parcelKey(compresa, particella)) || null;
}

function metricDisplay(metricId, value) {
  if (value == null || value === '') return S.BOSCO_NO_DATA;
  if (metricId === Q_TYPE) return value;
  if (characteristicSatelliteLayer(metricId)) return fmtDecimal2(value);
  const perHa = currentState?.harvestPerHa && isHarvestMetric(metricId);
  if (metricId === Q_HISTORICAL_HARVEST) {
    return perHa ? S.BOSCO_QUINTALS_PER_HA_VALUE(fmtDecimal2(value)) : fmtMass(value);
  }
  if (metricId === Q_FUTURE_HARVEST) {
    return perHa ? S.BOSCO_VOLUME_PER_HA_VALUE(fmtDecimal2(value)) : fmtVolume(value);
  }
  const unit = CHARACTERISTIC_METRICS[metricId]?.unit;
  const display = wholeNumberMetric(metricId) ? fmtRoundedInt(value) : fmtDecimal1(value);
  return unit ? `${display} ${unit}` : display;
}

function wholeNumberMetric(metricId) {
  return [Q_AGE, Q_ALTITUDE].includes(String(metricId));
}

function fmtRoundedInt(value) {
  if (value == null || value === '') return '';
  const n = Number(value);
  return Number.isFinite(n) ? fmtInt(Math.round(n)) : fmtInt(value);
}

function metricTooltipDisplay(metricId, value) {
  const label = characteristicTooltipLabel(metricId);
  const text = metricDisplay(metricId, value);
  return label ? `${label}: ${text}` : text;
}

function characteristicTooltipLabel(metricId) {
  return {
    [Q_TYPE]: S.BOSCO_METRIC_TYPE,
    [Q_HISTORICAL_HARVEST]: S.BOSCO_METRIC_HISTORICAL_HARVEST,
    [Q_FUTURE_HARVEST]: S.BOSCO_METRIC_FUTURE_HARVEST,
    [Q_AGE]: S.BOSCO_METRIC_AGE,
    [Q_ALTITUDE]: S.BOSCO_METRIC_ALTITUDE,
    [Q_NDVI]: 'NDVI',
    [Q_NDMI]: 'NDMI',
    [Q_EVI]: 'EVI',
  }[String(metricId)] || '';
}

function standTypeKey(entry) {
  if (entry.coppice === true) return TYPE_COPPICE_KEY;
  if (entry.coppice === false) return TYPE_HIGHFOREST_KEY;
  return null;
}

function renderTypeLegend(showNoData = false) {
  if (!legendEl) return;
  legendEl.replaceChildren();
  legendEl.appendChild(legendRow(TYPE_STYLES[TYPE_HIGHFOREST_KEY].fillColor, S.TYPE_HIGHFOREST));
  legendEl.appendChild(legendRow(TYPE_STYLES[TYPE_COPPICE_KEY].fillColor, S.TYPE_COPPICE));
  if (showNoData) legendEl.appendChild(legendRow(NO_DATA_STYLE.fillColor, S.BOSCO_NO_DATA));
}

function renderContinuousLegend(domain, metricId) {
  renderContinuousLegendTarget(
    legendEl, selectedCharacteristicLabel(), domain, v => metricDisplay(metricId, v),
  );
}

function renderContinuousLegendTarget(target, titleText, domain, displayFn) {
  if (!target) return;
  target.replaceChildren();
  if (!domain) {
    renderLegendMessage(target, S.BOSCO_NO_DATA_AVAILABLE);
    return;
  }

  const title = document.createElement('div');
  title.className = 'bosco-legend-title';
  title.textContent = titleText;
  target.appendChild(title);

  const gradient = document.createElement('div');
  gradient.className = 'bosco-gradient';
  target.appendChild(gradient);

  const labels = document.createElement('div');
  labels.className = 'bosco-legend-labels';
  const min = document.createElement('span');
  min.textContent = displayFn(domain.min);
  const max = document.createElement('span');
  max.textContent = displayFn(domain.max);
  labels.append(min, max);
  target.appendChild(labels);
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
  for (const value of [-1, -0.5, 0, 0.5, 1]) {
    const span = document.createElement('span');
    span.textContent = signedDisplay(value, fmtDecimal1);
    labels.appendChild(span);
  }
  target.appendChild(labels);
}

function renderHarvestDeltaLegend(fromYear, toYear, domain) {
  renderDivergingLegend(
    diffLegendEl,
    `${S.BOSCO_HARVEST_METRIC} ${toYear} - ${fromYear}`,
    domain,
    signedHarvestDeltaDisplay,
  );
}

// Symmetric -max..0..+max legend (satellite diff, harvest delta).  `displayFn`
// formats each of the five tick values.
function renderDivergingLegend(target, titleText, domain, displayFn) {
  if (!target) return;
  target.replaceChildren();

  const title = document.createElement('div');
  title.className = 'bosco-legend-title';
  title.textContent = titleText;
  target.appendChild(title);

  const gradient = document.createElement('div');
  gradient.className = 'bosco-gradient diff';
  target.appendChild(gradient);

  const max = domain.maxAbs || 1;
  const labels = document.createElement('div');
  labels.className = 'bosco-legend-labels';
  for (const value of [-max, -max / 2, 0, max / 2, max]) {
    const span = document.createElement('span');
    span.textContent = displayFn(value);
    labels.appendChild(span);
  }
  target.appendChild(labels);
}

function renderDiffLegend(metric, date1, date2, domain) {
  renderDivergingLegend(
    diffLegendEl,
    `${evolutionMetricLabel(metric)} ${date2.slice(0, 7)} - ${date1.slice(0, 7)}`,
    domain,
    v => signedDisplay(v, fmtDecimal2),
  );
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
    updateDetailNavigation(null);
    destroyProductionChart();
    return;
  }

  detailOverlay.hidden = false;
  detailTitle.textContent = scope.title;
  detailScopeLabel.textContent = scope.type === 'parcel' ? S.COL_PARCEL : S.COL_REGION;
  updateDetailNavigation(scope);
  applyDetailSections(state.openSections);
  renderMetadata(scope);
  if (state.openSections.includes('d')) renderDendrometry();
  if (state.openSections.includes('p')) renderProduction();
}

function detailScopeForState(state = currentState) {
  if (!state?.detailMode || !map) return null;
  const region = regionById.get(state.regionId);
  if (state.detailMode === '1') {
    const entry = mapEntries.find(e => e.id === state.parcelId);
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
    const entries = mapEntries;
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

function updateDetailNavigation(scope) {
  const nav = scope?.type === 'parcel'
    ? parcelNavigation(mapEntries, scope.parcelId)
    : { previous: null, next: null };
  updateDetailNavigationButton(detailPrevButton, nav.previous);
  updateDetailNavigationButton(detailNextButton, nav.next);
}

function updateDetailNavigationButton(button, entry) {
  if (!button) return;
  button.disabled = !entry;
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
  appendMetadataField(S.COL_AVE_AGE, fmtRoundedInt(entry.aveAge));
  if (entry.className) appendMetadataField(S.COL_CLASS, entry.className);
  appendMetadataField(S.COL_TYPE, entry.type);
  appendMetadataField(S.COL_ALT_MIN, fmtRoundedInt(entry.altMin));
  appendMetadataField(S.COL_ALT_MAX, fmtRoundedInt(entry.altMax));
  appendMetadataField(S.COL_ASPECT, entry.aspect);
  appendMetadataField(S.COL_GRADE_PCT, fmtRoundedInt(entry.gradePct));
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
  appendMetadataField(S.BOSCO_REGION_PARCELS, fmtInt(meta.count));
  appendMetadataField(S.COL_AREA_HA, fmtArea(meta.areaHa));
  appendMetadataField(S.COL_AREA_CAD_HA, fmtArea(meta.cadastralAreaHa));
  appendMetadataField(S.COL_AVE_AGE, fmtRoundedInt(meta.aveAge));
  appendMetadataField(S.COL_ALT_MIN, fmtRoundedInt(meta.altMin));
  appendMetadataField(S.COL_ALT_MAX, fmtRoundedInt(meta.altMax));
  const types = [...meta.typeCounts.entries()].map(([name, count]) => `${name}: ${count}`).join(' · ');
  appendMetadataField(S.COL_TYPE, types);
}

function appendMetadataField(label, value, wide = false) {
  const item = document.createElement('div');
  item.className = wide ? 'bosco-metadata-item wide' : 'bosco-metadata-item';
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value || S.BOSCO_NO_DATA;
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
        if (dendrometryStatus) dendrometryStatus.textContent = S.BOSCO_DENDROMETRY_UNAVAILABLE;
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
    allSpeciesNames: speciesNames,
  });
  const rawRows = aggregateDendrometry(dendrometryData, baseScope, {
    areaHa: scope.areaHa,
    perHa: false,
    speciesIds: filter,
    allSpeciesNames: speciesNames,
  });
  const heightPoints = dendrometryHeightPoints(dendrometryPointsData, baseScope, {
    speciesIds: filter,
    allSpeciesNames: speciesNames,
  });
  renderDendrometryCharts(rows, rawRows, heightPoints);
}

function renderDendrometrySpecies(scope) {
  if (!dendrometrySpeciesHost) return;
  const species = dendrometrySpecies(
    dendrometryData, { region: scope.region, parcelId: scope.parcelId }, { allSpeciesNames: speciesNames },
  );
  dendrometrySpeciesHost.replaceChildren();
  if (!species.length) {
    dendrometrySpeciesHost.textContent = S.BOSCO_NO_DENDROMETRY;
    return;
  }
  const selectedIds = currentState?.detailSpeciesIds;
  const selected = new Set(selectedIds || []);
  for (const [idx, item] of species.entries()) {
    const label = document.createElement('label');
    label.className = 'bosco-check';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.value = String(item.id);
    input.checked = selectedIds == null || selected.has(item.id);
    input.addEventListener('change', () => updateDendrometrySpeciesFilter(species));
    const dot = document.createElement('span');
    dot.className = 'bosco-legend-dot';
    dot.style.backgroundColor = item.color || dendrometrySpeciesColor(idx);
    const text = document.createElement('span');
    text.textContent = `${item.name} (${fmtInt(item.count)})`;
    label.append(input, dot, text);
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
  writeOptionalIdList(params, 'ds', selected);
  navigateWithParams(PAGE_PATH, params, true);
}

function renderDendrometryCharts(rows, rawRows, heightPoints) {
  if (!dendrometryStatus || !dendrometryChartGrid) return;
  if (!rows.length) {
    destroyDendrometryCharts();
    dendrometryChartGrid.hidden = true;
    dendrometryStatus.textContent = S.BOSCO_NO_DENDROMETRY;
    return;
  }

  dendrometryChartGrid.hidden = false;
  dendrometryStatus.textContent = S.BOSCO_TREES(fmtInt(dendrometryTreeTotal(rawRows)));
  const perHa = dendrometryPerHa?.checked !== false;
  dendrometryCharts.treeCount = renderStackedBar(
    dendrometryTreeCanvas,
    dendrometryBarChartData(rows, 'treeCount', perHa ? S.BOSCO_TREE_COUNT_PER_HA : S.BOSCO_TREE_COUNT),
    dendrometryCharts.treeCount,
  );
  dendrometryCharts.volume = renderStackedBar(
    dendrometryVolumeCanvas,
    dendrometryBarChartData(rows, 'volumeM3', perHa ? S.BOSCO_VOLUME_PER_HA : S.COL_VOLUME_M3),
    dendrometryCharts.volume,
  );
  dendrometryCharts.basalArea = renderStackedBar(
    dendrometryBasalAreaCanvas,
    dendrometryBarChartData(rows, 'basalAreaM2', perHa ? S.BOSCO_BASAL_AREA_PER_HA : S.COL_BASAL_AREA_M2),
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
      if (productionSummary) productionSummary.textContent = S.BOSCO_HISTORICAL_PRODUCTION_UNAVAILABLE;
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
    productionSummary.textContent = S.BOSCO_NO_HISTORICAL_HARVEST;
    return;
  }

  productionHost.hidden = false;
  productionChart = renderStackedBar(productionCanvas, result.chartData, productionChart);
  const area = scope.areaHa || 0;
  const total = perHa && area > 0
    ? S.BOSCO_QUINTALS_PER_HA_VALUE(fmtDecimal2(result.totalQuintals / area))
    : fmtMass(result.totalQuintals);
  productionSummary.textContent = `${total} - ${S.BOSCO_INTERVENTIONS(fmtInt(result.rowCount))}`;
}

function destroyProductionChart() {
  if (productionChart) {
    productionChart.destroy();
    productionChart = null;
  }
}


function renderPaiMode() {
  if (!map || currentState?.mode !== MODE_PAI) return;
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
  const parcelItems = paiParcelItems(mapEntries, allTrees);
  const speciesItems = paiSpeciesItems(allTrees);
  const colors = speciesColorMap(speciesItems, speciesNames);
  renderPaiCheckboxes(paiParcelsHost, parcelItems, currentState.paiParcelIds, 'pp');
  renderPaiCheckboxes(paiSpeciesHost, speciesItems, currentState.paiSpeciesIds, 'ps', colors);

  const trees = filterPaiTrees(allTrees, {
    parcelIds: currentState.paiParcelIds,
    speciesIds: currentState.paiSpeciesIds,
  });
  renderPaiMarkers(trees, colors);
}

async function showPaiForm(rowId = null, defaults = {}) {
  const url = rowId ? `${PAI_FORM_URL}${rowId}/` : paiAddFormUrl(defaults);
  const form = await fetchModalForm(url);
  if (!form) return;
  wirePaiForm(form);
}

function paiAddFormUrl(defaults = {}) {
  const params = new URLSearchParams();
  if (currentState?.regionId) params.set(FIELD_REGION_ID, String(currentState.regionId));
  if (defaults.parcelId != null) params.set(FIELD_PARCEL_ID, String(defaults.parcelId));
  if (defaults.lat != null) params.set(FIELD_LAT, fmtCoord(defaults.lat));
  if (defaults.lon != null) params.set(FIELD_LON, fmtCoord(defaults.lon));
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
  if (!body.lat || !body.lon) return S.BOSCO_LAT_LON_REQUIRED;
  return null;
}

function selectPaiParcelFromLatLon(form) {
  const lat = parseDecimal(form.querySelector('#id_pai_lat')?.value);
  const lon = parseDecimal(form.querySelector('#id_pai_lon')?.value);
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || !parcelsGeo?.features) return;
  const feature = findContainingParcel(lon, lat, parcelsGeo.features);
  if (!feature) return;
  const { compresa, particella } = parcelNames(feature);
  const entry = mapEntriesByKey?.get(parcelKey(compresa, particella));
  const select = form.querySelector('#id_pai_parcel');
  if (entry && select) select.value = String(entry.id);
}


function applyPaiResponse(data) {
  cache.applyResponseChanges(data);
  preservedData = cache.get(PRESERVED_ID);
  renderPaiMode();
}

function confirmDeletePai(rowId) {
  showConfirmModal(S.DELETE_CONFIRM, () => deletePai(rowId));
}

async function deletePai(rowId) {
  await deleteRowWithVersion(PRESERVED_ID, rowId, PAI_DELETE_URL, {
    confirmMessage: null,
    onSuccess: applyPaiResponse,
    onConflict: applyPaiResponse,
  });
}

function renderPaiCheckboxes(host, items, selectedIds, paramName, colors = null) {
  if (!host) return;
  host.replaceChildren();
  if (!items.length) {
    host.textContent = S.BOSCO_NO_PAI_TREES;
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

function paiParcelLabel(tree) {
  return [tree.region, tree.parcel].filter(v => v != null && String(v).trim() !== '')
    .map(v => String(v).trim()).join(' ');
}

function paiTooltip(tree) {
  const el = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'parcel-tooltip-title';
  title.textContent = tree.species;
  const meta = document.createElement('div');
  const parcel = paiParcelLabel(tree);
  meta.textContent = tree.number
    ? S.BOSCO_PAI_TREE_META(parcel, fmtInt(tree.number))
    : parcel;
  el.append(title, meta);
  return el;
}

function paiPopup(tree) {
  const el = document.createElement('div');
  el.className = 'bosco-pai-popup';
  const rows = [
    [S.COL_SPECIES, tree.species],
    [S.COL_NUMBER, fmtInt(tree.number)],
    [S.COL_PARCEL, paiParcelLabel(tree)],
    [S.COL_SURVEY_DATE, tree.date],
    [S.COL_ESTIMATED_BIRTH_YEAR, fmtInt(tree.estimatedBirthYear)],
    [S.COL_D_CM, fmtInt(tree.dCm)],
    [S.COL_H_M, tree.hM === '' || tree.hM == null ? '' : fmtDecimal2(tree.hM)],
    [S.COL_LAT, fmtCoord(tree.lat)],
    [S.COL_LON, fmtCoord(tree.lon)],
    [S.COL_NOTE, tree.note],
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
    edit.textContent = S.ACTION_EDIT;
    edit.addEventListener('click', () => showPaiForm(tree.id));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn btn-delete';
    del.textContent = S.ACTION_DELETE;
    del.addEventListener('click', () => confirmDeletePai(tree.id));
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

function openAdjacentParcel(direction) {
  const scope = detailScopeForState();
  if (!scope || scope.type !== 'parcel') return;
  const nav = parcelNavigation(mapEntries, scope.parcelId);
  const entry = direction < 0 ? nav.previous : nav.next;
  if (entry) openParcelDetail(entry, { resetSections: false });
}

function openParcelDetail(entry, { resetSections = true, resetSpecies = true } = {}) {
  const params = new URLSearchParams(location.search);
  params.set('v', '1');
  params.set('pa', String(entry.id));
  if (resetSections) params.delete('vo');
  if (resetSpecies) params.delete('ds');
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

function promptNewPaiTreeAt(latlng, feature) {
  if (!canModify()) return;
  const defaults = { lat: latlng.lat, lon: latlng.lng };
  if (feature) {
    const { compresa, particella } = parcelNames(feature);
    const entry = mapEntriesByKey?.get(parcelKey(compresa, particella));
    if (entry) defaults.parcelId = entry.id;
  }
  showConfirmModal(
    S.BOSCO_INSERT_PAI_TREE_HERE,
    () => showPaiForm(null, defaults),
    { confirmLabel: S.CONFIRM, intent: 'confirm' },
  );
}

function onMapClick(latlng, feature) {
  if (currentState?.mode === MODE_PAI) {
    promptNewPaiTreeAt(latlng, feature);
    return;
  }
  const region = regionById.get(currentState?.regionId);
  if (!feature) {
    if (region) setStatus(S.BOSCO_REGION_SUMMARY(region.name));
    openRegionDetail();
    return;
  }
  const { compresa, particella } = parcelNames(feature);
  const entry = mapEntriesByKey?.get(parcelKey(compresa, particella));
  const context = currentState ? characteristicContext(currentState.q) : {};
  const value = entry && currentState ? metricValue(entry, currentState.q, context) : null;
  const metric = currentState?.mode === MODE_CHARACTERISTICS ? ` — ${metricDisplay(currentState.q, value)}` : '';
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
  const validQ = state.mode === MODE_EVOLUTION ? VALID_EVOLUTION_METRICS : VALID_CHARACTERISTICS;
  if (!params.get('q') || !validQ.includes(params.get('q'))) {
    params.set('q', state.q);
    changed = true;
  }
  changed = canonicalizeFlag(params, 'fc', state.useCadastralArea) || changed;
  changed = canonicalizeFlag(params, 'fh', state.harvestPerHa && harvestPerHaAllowed(
    state.mode, state.q, state.evolutionMetric,
  )) || changed;
  changed = canonicalizeFlag(params, 'fa', state.parcelAverage && parcelAverageAllowed(
    state.mode, state.q, state.evolutionMetric,
  )) || changed;
  changed = canonicalizeEvolutionParams(params, state) || changed;
  changed = canonicalizeDetailParams(params, state) || changed;
  changed = canonicalizePaiParams(params, state) || changed;
  if (changed) navigateWithParams(PAGE_PATH, params, true);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text || '';
}

function canonicalizeEvolutionParams(params, state) {
  if (state.mode !== MODE_EVOLUTION) {
    if (!params.has('d1') && !params.has('d2')) return false;
    params.delete('d1');
    params.delete('d2');
    return true;
  }
  return false;
}

function canonicalizePaiParams(params, state) {
  if (state.mode !== MODE_PAI) {
    if (!params.has('pp') && !params.has('ps')) return false;
    params.delete('pp');
    params.delete('ps');
    return true;
  }

  let changed = false;
  const parcelIds = mapEntries.map(e => e.id);
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

  const beforeDs = params.get('ds');
  const hadDs = params.has('ds');
  writeOptionalIdList(params, 'ds', state.detailSpeciesIds);
  if (params.get('ds') !== beforeDs || params.has('ds') !== hadDs) changed = true;
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
