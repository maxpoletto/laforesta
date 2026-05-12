/**
 * Section 2 (Rilevamenti) map.
 *
 * Renders parcel borders + sample-area markers for the active
 * survey, with visited / unvisited coloring sourced from
 * samples.json.  Click a marker to set the active sample area
 * (URL `a=N`); click empty map space to clear.
 */

import MapCommon from '../../base/js/map-common.js';

const VISITED_COLOR = '#2d5d2c';     // abies dark green
const UNVISITED_COLOR = '#8fbf8e';   // abies light green
const MARKER_RADIUS = 7;

export class RilevamentiMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — particelle.geojson (sorted-by-area)
   * @param {function(number|null): void} opts.onAreaSelect — called with
   *   sample_area_id, or null when the user clicks empty space.
   */
  constructor(opts) {
    this.container = opts.container;
    this.geojson = opts.geojson;
    this.onAreaSelect = opts.onAreaSelect;
    this.activeAreaId = null;

    // Build the map div.
    this.mapEl = document.createElement('div');
    this.mapEl.className = 'rilevamenti-map';
    this.container.appendChild(this.mapEl);

    this.wrapper = MapCommon.create(this.mapEl, {
      basemap: opts.basemap || 'satellite',
    });
    this.leaflet = this.wrapper.getLeafletMap();

    // Empty-space click → clear active area.
    this.leaflet.on('click', (e) => {
      if (e.originalEvent.target === this.mapEl ||
          this.mapEl.contains(e.originalEvent.target)) {
        if (this.activeAreaId !== null) {
          this.activeAreaId = null;
          this.onAreaSelect(null);
        }
      }
    });

    this._renderParcels();
    this.markerLayer = L.layerGroup().addTo(this.leaflet);
    this.markers = new Map();    // sample_area_id → CircleMarker

    // Fit to particelle extent on initial render.
    if (this.parcelLayer) {
      this.leaflet.fitBounds(this.parcelLayer.getBounds(), {
        padding: [20, 20],
      });
    }
  }

  _renderParcels() {
    const layer = L.geoJSON(this.geojson, {
      style: {
        color: '#444',
        weight: 1,
        opacity: 0.7,
        fillColor: '#888',
        fillOpacity: 0.05,
      },
    }).addTo(this.leaflet);
    this.parcelLayer = layer;
  }

  /**
   * Render markers for the given sample-area rows.
   *
   * @param {Array<{id, lat, lng, compresa, particella, numero}>} areas
   * @param {Map<number, {nAlberi: number}>} visitedById — sample_area_id →
   *   visit info from samples.json.  Missing entries = unvisited.
   */
  setAreas(areas, visitedById) {
    this.markerLayer.clearLayers();
    this.markers.clear();

    for (const area of areas) {
      const visited = visitedById.get(area.id);
      const isVisited = !!visited;
      const m = L.circleMarker([area.lat, area.lng], {
        radius: MARKER_RADIUS,
        color: '#000',
        weight: 1,
        opacity: 0.8,
        fillColor: isVisited ? VISITED_COLOR : UNVISITED_COLOR,
        fillOpacity: isVisited ? 0.85 : 0.5,
      });

      const tooltip = isVisited
        ? `${area.compresa} ${area.particella} / adc ${area.numero} / ${visited.nAlberi} alberi`
        : `${area.compresa} ${area.particella} / adc ${area.numero}`;
      m.bindTooltip(tooltip, { direction: 'top', offset: [0, -5] });

      m.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        this.activeAreaId = area.id;
        this.onAreaSelect(area.id);
        this._refreshHighlight();
      });

      this.markers.set(area.id, m);
      m.addTo(this.markerLayer);
    }
    this._refreshHighlight();
  }

  setActiveAreaId(areaId) {
    if (this.activeAreaId === areaId) return;
    this.activeAreaId = areaId;
    this._refreshHighlight();
  }

  _refreshHighlight() {
    for (const [id, m] of this.markers) {
      const isActive = id === this.activeAreaId;
      m.setStyle({
        weight: isActive ? 3 : 1,
        color: isActive ? '#ffcc00' : '#000',
      });
      if (isActive) m.bringToFront();
    }
  }

  destroy() {
    if (this.leaflet) {
      this.leaflet.remove();
      this.leaflet = null;
    }
    this.mapEl?.remove();
    this.mapEl = null;
    this.wrapper = null;
    this.parcelLayer = null;
    this.markerLayer = null;
    this.markers.clear();
  }
}
