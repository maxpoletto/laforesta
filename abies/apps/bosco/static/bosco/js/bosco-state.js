import {
  DEFAULT_EVOLUTION_METRIC, evolutionMetricId, normalizeDateParam,
} from './bosco-satellite.js';

const DEFAULT_MODE = '1';
const DEFAULT_MAP_TYPE_TOKEN = 's';
const DEFAULT_CHARACTERISTIC = '1';
const DEFAULT_DETAIL_SECTIONS = ['m'];
const VALID_DETAIL_SECTIONS = new Set(['m', 'd', 'p']);
const VALID_CHARACTERISTICS = new Set(['1', '2', '3', '4', '5', '6', '7', '8']);

export const MAP_TYPE_TOKENS = { o: 'osm', t: 'topo', s: 'satellite' };
export const MAP_TYPE_BY_NAME = { osm: 'o', topo: 't', satellite: 's' };

function paramValue(params, key) {
  return params instanceof URLSearchParams ? params.get(key) : params[key];
}

function hasParam(params, key) {
  return params instanceof URLSearchParams ? params.has(key) : Object.hasOwn(params, key);
}

function intParam(params, key) {
  const raw = paramValue(params, key);
  if (raw == null || raw === '') return null;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) ? n : null;
}

export function mapTypeName(token) {
  return MAP_TYPE_TOKENS[token] || MAP_TYPE_TOKENS[DEFAULT_MAP_TYPE_TOKEN];
}

export function mapTypeToken(name) {
  return MAP_TYPE_BY_NAME[name] || DEFAULT_MAP_TYPE_TOKEN;
}

export function parseCenter(raw) {
  if (!raw) return null;
  const parts = String(raw).split(',');
  if (parts.length !== 2) return null;
  const lat = Number(parts[0]);
  const lng = Number(parts[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  return [lat, lng];
}

export function formatCenter(center) {
  return `${center[0].toFixed(6)},${center[1].toFixed(6)}`;
}

export function readBoscoParams(params, regionIds = []) {
  const validRegions = new Set(regionIds);
  const requestedRegion = intParam(params, 'c');
  const fallbackRegion = regionIds.length ? regionIds[0] : null;
  const regionId = validRegions.has(requestedRegion) ? requestedRegion : fallbackRegion;

  const modeRaw = String(paramValue(params, 'm') || DEFAULT_MODE);
  const mode = ['1', '2', '3'].includes(modeRaw) ? modeRaw : DEFAULT_MODE;

  const mtRaw = String(paramValue(params, 'mt') || DEFAULT_MAP_TYPE_TOKEN);
  const mt = MAP_TYPE_TOKENS[mtRaw] ? mtRaw : DEFAULT_MAP_TYPE_TOKEN;

  const zoom = intParam(params, 'mz');
  const center = parseCenter(paramValue(params, 'mc'));

  const qRaw = String(paramValue(params, 'q') || DEFAULT_CHARACTERISTIC);
  const characteristic = VALID_CHARACTERISTICS.has(qRaw) ? qRaw : DEFAULT_CHARACTERISTIC;
  const evolutionMetric = evolutionMetricId(qRaw || DEFAULT_EVOLUTION_METRIC);
  const q = mode === '2' ? evolutionMetric : characteristic;

  const detailRaw = String(paramValue(params, 'v') || '');
  const detailMode = ['1', '2'].includes(detailRaw) ? detailRaw : null;

  return {
    regionId,
    mode,
    mt,
    basemap: mapTypeName(mt),
    center: center && zoom !== null ? center : null,
    zoom: center && zoom !== null ? zoom : null,
    q,
    evolutionMetric,
    evolutionDate1: normalizeDateParam(paramValue(params, 'd1')) || null,
    evolutionDate2: normalizeDateParam(paramValue(params, 'd2')) || null,
    parcelAverage: true,
    useCadastralArea: paramValue(params, 'fc') === '1',
    harvestPerHa: paramValue(params, 'fh') === '1',
    detailMode,
    parcelId: intParam(params, 'pa'),
    openSections: parseSectionTokens(paramValue(params, 'vo')),
    detailSpeciesIds: parseIdList(paramValue(params, 'ds')),
    paiParcelIds: parseOptionalIdList(paramValue(params, 'pp')),
    paiSpeciesIds: parseOptionalIdList(paramValue(params, 'ps')),
    hasRegionParam: hasParam(params, 'c'),
  };
}

export function parseSectionTokens(raw) {
  if (raw == null) return [...DEFAULT_DETAIL_SECTIONS];
  const out = [];
  for (const token of String(raw)) {
    if (VALID_DETAIL_SECTIONS.has(token) && !out.includes(token)) out.push(token);
  }
  return out;
}

export function parseIdList(raw) {
  if (!raw) return [];
  const out = [];
  const seen = new Set();
  for (const part of String(raw).split(',')) {
    const n = Number(part);
    if (!Number.isInteger(n) || n <= 0 || seen.has(n)) continue;
    seen.add(n);
    out.push(n);
  }
  return out;
}

export function parseOptionalIdList(raw) {
  if (raw == null) return null;
  if (raw === '') return [];
  return parseIdList(raw);
}

export function writeOptionalIdList(params, key, selected, allIds) {
  if (selected == null || selected.length === allIds.length) {
    params.delete(key);
  } else {
    params.set(key, selected.join(','));
  }
}

export function writeSectionTokens(params, sections) {
  const clean = (sections || []).filter(s => VALID_DETAIL_SECTIONS.has(s));
  if (clean.length === DEFAULT_DETAIL_SECTIONS.length
      && clean.every((s, i) => s === DEFAULT_DETAIL_SECTIONS[i])) {
    params.delete('vo');
  } else {
    params.set('vo', clean.join(''));
  }
}

export function clearDetailParams(params) {
  params.delete('v');
  params.delete('pa');
  params.delete('vo');
  params.delete('ds');
}

export function writeMapView(params, center, zoom) {
  if (!center || zoom == null) return;
  params.set('mc', formatCenter(center));
  params.set('mz', String(zoom));
}

export function clearMapView(params) {
  params.delete('mc');
  params.delete('mz');
}
