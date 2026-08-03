import * as S from '../../base/js/strings.js';
import {
  COLUMNS, FIELD_CATEGORIES, FIELD_CATEGORY_IDS, FIELD_ID, FIELD_LAT,
  FIELD_LON, FIELD_NAME, FIELD_ORIGINAL_FILENAME, FIELD_PHOTO_COUNT,
  FIELD_REGION_ID, FIELD_SIZE_BYTES, ROW_ID, ROWS, VERSION,
} from '../../base/js/constants.js';
import { columnMap, toNumber } from '../../base/js/digests.js';
import { fmtInt } from '../../base/js/format.js';
import { distanceMeters, findContainingParcel, parcelNames } from '../../base/js/geo.js';

export function buildObservations(digest) {
  if (!digest) return [];
  const c = columnMap(digest);
  return digest[ROWS].map(row => {
    const categoryIds = intArray(row[c[FIELD_CATEGORY_IDS]]);
    const categoryNames = categoryNameArray(
      row[c[S.COL_OBSERVATION_CATEGORIES]], categoryIds.length,
    );
    const date = row[c[S.COL_DATE]] || '';
    const lat = toNumber(row[c[S.COL_LAT]], NaN);
    const lon = toNumber(row[c[S.COL_LON]], NaN);
    return {
      id: row[c[ROW_ID]],
      version: row[c[VERSION]],
      regionId: toNumber(row[c[FIELD_REGION_ID]], null),
      categoryIds,
      categoryNames,
      date,
      year: observationYear(date),
      text: row[c[S.COL_TEXT]] || '',
      lat,
      lon,
      accM: toNumber(row[c[S.CSV_COL_ACC_M]], null),
      operator: row[c[S.COL_OPERATOR]] || '',
      photoCount: toNumber(row[c[FIELD_PHOTO_COUNT]], 0) || 0,
      region: '',
      parcel: '',
    };
  }).filter(obs => Number.isFinite(obs.lat) && Number.isFinite(obs.lon));
}

export function attributeObservationParcels(observations, features) {
  const source = Array.isArray(features) ? features : [];
  return observations.map(obs => {
    const feature = findContainingParcel(obs.lon, obs.lat, source);
    if (!feature) return { ...obs };
    const names = parcelNames(feature);
    return { ...obs, region: names.compresa || '', parcel: names.particella || '' };
  });
}

export function filterObservations(
  observations, {
    regionId = null, region = '', categoryIds = null, yearFrom = null, yearTo = null,
  } = {},
) {
  const categories = categoryIds == null ? null : new Set(categoryIds);
  return observations.filter(obs => {
    if (Number.isInteger(regionId)) {
      if (Number.isInteger(obs.regionId)) {
        if (obs.regionId !== regionId) return false;
      } else if (region && obs.region !== region) return false;
    } else if (region && obs.region !== region) return false;
    if (categories && !obs.categoryIds.some(id => categories.has(id))) return false;
    if (Number.isInteger(yearFrom) && (!Number.isInteger(obs.year) || obs.year < yearFrom)) return false;
    if (Number.isInteger(yearTo) && (!Number.isInteger(obs.year) || obs.year > yearTo)) return false;
    return true;
  });
}

export function buildObservationCategories(digest) {
  const rows = Array.isArray(digest?.[FIELD_CATEGORIES])
    ? digest[FIELD_CATEGORIES] : [];
  return rows.map(row => ({
    id: toNumber(row?.[FIELD_ID], null),
    name: String(row?.[FIELD_NAME] || ''),
  })).filter(item => Number.isInteger(item.id) && item.name);
}

export function observationCategoryItems(observations, categories = []) {
  const byId = new Map();
  for (const category of categories) {
    byId.set(category.id, { id: category.id, name: category.name, count: 0 });
  }
  for (const obs of observations) {
    for (const [idx, id] of obs.categoryIds.entries()) {
      const name = obs.categoryNames[idx] || String(id);
      const item = byId.get(id) || { id, name, count: 0 };
      item.count += 1;
      byId.set(id, item);
    }
  }
  return [...byId.values()].sort((a, b) => a.name.localeCompare(b.name, S.LOCALE));
}

export function observationYears(observations) {
  const observed = observations.map(obs => obs.year).filter(Number.isInteger);
  if (!observed.length) return [];
  const first = Math.min(...observed);
  const last = Math.max(...observed);
  const years = [];
  for (let year = first; year <= last; year += 1) years.push(year);
  return years;
}

export function normalizeObservationYearRange(yearFrom, yearTo, years) {
  if (!years.length) return { from: null, to: null };
  let from = years.includes(yearFrom) ? yearFrom : years[0];
  let to = years.includes(yearTo) ? yearTo : years[years.length - 1];
  if (from > to) [from, to] = [to, from];
  return { from, to };
}

export function observationPhotoMapItems(observation, photos, thresholdM = 10) {
  const obsLat = coordinateNumber(observation?.[FIELD_LAT]);
  const obsLon = coordinateNumber(observation?.[FIELD_LON]);
  if (!Number.isFinite(obsLat) || !Number.isFinite(obsLon)) {
    return { items: [], hasDistant: false };
  }
  const items = (Array.isArray(photos) ? photos : []).map((photo, index) => {
    const lat = coordinateNumber(photo?.[FIELD_LAT]);
    const lon = coordinateNumber(photo?.[FIELD_LON]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
    const distanceM = distanceMeters(obsLat, obsLon, lat, lon);
    return { photo, lat, lon, distanceM, caption: index + 1 };
  }).filter(Boolean);
  return {
    items,
    hasDistant: items.some(item => item.distanceM > thresholdM),
  };
}

export function groupObservationPhotoMapItems(items, overlapM = 5) {
  const groups = [];
  for (const item of Array.isArray(items) ? items : []) {
    let group = null;
    for (const candidate of groups) {
      if (candidate.items.some(existing => (
        distanceMeters(item.lat, item.lon, existing.lat, existing.lon) <= overlapM
      ))) {
        group = candidate;
        break;
      }
    }
    if (!group) {
      group = { items: [] };
      groups.push(group);
    }
    group.items.push(item);
  }
  return groups.map(group => {
    const lat = average(group.items.map(item => item.lat));
    const lon = average(group.items.map(item => item.lon));
    return {
      lat,
      lon,
      items: group.items,
      caption: group.items.length > 1 ? '+' : String(group.items[0].caption),
    };
  });
}

export function observationCategoryLabel(count) {
  return count === 1
    ? S.BOSCO_OBSERVATION_CATEGORY
    : S.COL_OBSERVATION_CATEGORIES;
}

export function observationPhotoTitle(photo) {
  return [
    photo?.[FIELD_ORIGINAL_FILENAME] || S.COL_PHOTO_COUNT,
    photo?.[FIELD_SIZE_BYTES] ? `${fmtInt(photo[FIELD_SIZE_BYTES])} B` : '',
  ].filter(Boolean).join(' · ');
}

export function shouldPreviewObservationPhoto(file) {
  const type = String(file?.type || '').toLowerCase();
  return !type || type.startsWith('image/');
}

function intArray(value) {
  return Array.isArray(value) ? value.filter(v => Number.isInteger(v)) : [];
}

function categoryNameArray(value, expectedLength) {
  if (Array.isArray(value)) return value.map(v => String(v || ''));
  if (typeof value !== 'string' || !value.trim()) return [];
  const names = value.split(',').map(v => v.trim()).filter(Boolean);
  if (expectedLength && names.length !== expectedLength) return names;
  return names;
}

function coordinateNumber(value) {
  if (value == null || value === '') return NaN;
  return Number(value);
}

function average(values) {
  if (!values.length) return NaN;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function observationYear(date) {
  const match = String(date || '').match(/^(\d{4})-/);
  if (!match) return null;
  const year = Number(match[1]);
  return Number.isInteger(year) ? year : null;
}
