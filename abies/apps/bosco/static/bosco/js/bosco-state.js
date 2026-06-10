const DEFAULT_MODE = '1';
const DEFAULT_MAP_TYPE_TOKEN = 's';

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

  return {
    regionId,
    mode,
    mt,
    basemap: mapTypeName(mt),
    center: center && zoom !== null ? center : null,
    zoom: center && zoom !== null ? zoom : null,
    hasRegionParam: hasParam(params, 'c'),
  };
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
