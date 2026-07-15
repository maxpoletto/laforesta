/**
 * TreePointsMap — parcel map + reusable tree point overlay.
 *
 * This is intentionally thin: ParcelMap owns basemap controls, parcel borders,
 * parcel hover labels, measure/location tools, fit/teardown, and marker layer
 * lifetime.  TreePointsMap only maps tree-like digest rows to dark-green
 * circle markers with a shared tooltip.
 */

import { ParcelMap } from './parcel-map.js';
import { ROW_ID } from './constants.js';
import { fmtDecimal2, fmtInt } from './format.js';
import * as S from './strings.js';

const TREE_MARKER_RADIUS = 5;
const TREE_MARKER_COLOR = '#2d5d2c'; // abies dark green
const TREE_MARKER_BORDER = '#000';
const TREE_MARKER_BORDER_OPACITY = 0.8;
const TREE_MARKER_FILL_OPACITY = 0.85;

const DEFAULT_COLUMN_NAMES = {
  id: ROW_ID,
  number: S.COL_NUMBER,
  species: S.COL_SPECIES,
  diameter: S.COL_D_CM,
  height: S.COL_H_M,
  lat: S.COL_LAT,
  lon: S.COL_LON,
};

export class TreePointsMap extends ParcelMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — terreni.geojson, sorted-by-area
   * @param {string} [opts.className]
   * @param {string} [opts.basemap]
   * @param {{measure?:boolean, location?:boolean}} [opts.tools]
   * @param {function(object): void} [opts.onTreeClick] — clicked tree row.
   */
  constructor(opts) {
    super({
      container: opts.container,
      className: opts.className || 'tree-points-map',
      geojson: opts.geojson,
      basemap: opts.basemap,
      tools: opts.tools || { measure: true, location: true },
    });
    this.onTreeClick = opts.onTreeClick || null;
  }

  /**
   * Render one dark-green dot for every tree with finite Lat/Lon.
   * @param {Array<{id, row, number, species, diameter, height, lat, lon}>} trees
   */
  setTrees(trees) {
    this._clearMarkers();
    for (const tree of trees || []) {
      const lat = numericValue(tree.lat);
      const lon = numericValue(tree.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const marker = L.circleMarker([lat, lon], {
        radius: TREE_MARKER_RADIUS,
        color: TREE_MARKER_BORDER,
        weight: 1,
        opacity: TREE_MARKER_BORDER_OPACITY,
        fillColor: TREE_MARKER_COLOR,
        fillOpacity: TREE_MARKER_FILL_OPACITY,
      });
      marker.bindTooltip(treeTooltipContent(tree), { sticky: true, direction: 'top' });
      if (this.onTreeClick) {
        marker.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          this.onTreeClick(tree);
        });
      }
      marker.addTo(this.markerLayer);
    }
  }
}

export function treePointsFromDigest(rows, columns, names = DEFAULT_COLUMN_NAMES) {
  const index = (name) => columns.indexOf(name);
  const iId = index(names.id);
  const iNumber = index(names.number);
  const iSpecies = index(names.species);
  const iDiameter = index(names.diameter);
  const iHeight = index(names.height);
  const iLat = index(names.lat);
  const iLon = index(names.lon);
  return (rows || []).map(row => ({
    id: valueAt(row, iId),
    row,
    number: valueAt(row, iNumber),
    species: valueAt(row, iSpecies),
    diameter: valueAt(row, iDiameter),
    height: valueAt(row, iHeight),
    lat: valueAt(row, iLat),
    lon: valueAt(row, iLon),
  }));
}

export function treeTooltipContent(tree) {
  const el = document.createElement('div');
  el.className = 'tree-tooltip';
  appendTooltipLine(el, S.COL_NUMBER, fmtIntOrDash(tree.number));
  appendTooltipLine(el, S.COL_SPECIES, textOrDash(tree.species));
  appendTooltipLine(el, S.COL_D_CM, fmtIntOrDash(tree.diameter));
  appendTooltipLine(el, S.COL_H_M, fmtDecimal2OrDash(tree.height));
  return el;
}

function appendTooltipLine(root, label, value) {
  const line = document.createElement('div');
  line.className = 'tree-tooltip-line';
  const labelEl = document.createElement('strong');
  labelEl.className = 'tree-tooltip-label';
  labelEl.textContent = `${label}: `;
  line.append(labelEl, document.createTextNode(value));
  root.appendChild(line);
}

function valueAt(row, index) {
  return index >= 0 ? row[index] : null;
}

function numericValue(value) {
  if (value == null || value === '') return NaN;
  return typeof value === 'number' ? value : Number(value);
}

function textOrDash(value) {
  return value == null || value === '' ? '—' : String(value);
}

function fmtIntOrDash(value) {
  return value == null || value === '' ? '—' : fmtInt(value);
}

function fmtDecimal2OrDash(value) {
  return value == null || value === '' ? '—' : fmtDecimal2(value);
}
