/**
 * Section 1 (Griglie) map.
 *
 * Renders parcel borders + sample-area markers for the active grid.
 * No visited / unvisited coloring (that lives on Rilevamenti); the
 * Griglie map is purely a grid-management surface.
 *
 * M3d-read: hover tooltips + popovers on click.  Writer affordances
 * (new-area click, edit/delete on popover) ship in M3d-write.
 */

import MapCommon from '../../base/js/map-common.js';

const MARKER_COLOR = '#2d5d2c';
const MARKER_RADIUS = 7;

export class GriglieMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — particelle.geojson
   * @param {function(number): void} [opts.onAreaClick]
   *   Called with sample_area_id when an area is clicked.
   */
  constructor(opts) {
    this.container = opts.container;
    this.geojson = opts.geojson;
    this.onAreaClick = opts.onAreaClick || null;
    this.markers = new Map();
    this.activeAreaId = null;

    this.mapEl = document.createElement('div');
    this.mapEl.className = 'griglie-map';
    this.container.appendChild(this.mapEl);

    this.wrapper = MapCommon.create(this.mapEl, { basemap: 'satellite' });
    this.leaflet = this.wrapper.getLeafletMap();

    this._renderParcels();
    this.markerLayer = L.layerGroup().addTo(this.leaflet);

    if (this.parcelLayer) {
      this.leaflet.fitBounds(this.parcelLayer.getBounds(), { padding: [20, 20] });
    }
  }

  _renderParcels() {
    this.parcelLayer = L.geoJSON(this.geojson, {
      style: {
        color: '#444',
        weight: 1,
        opacity: 0.7,
        fillColor: '#888',
        fillOpacity: 0.05,
      },
    }).addTo(this.leaflet);
  }

  /**
   * Render markers for the active grid's sample areas.
   *
   * @param {Array<{id, lat, lng, compresa, particella, numero}>} areas
   */
  setAreas(areas) {
    this.markerLayer.clearLayers();
    this.markers.clear();

    for (const area of areas) {
      const m = L.circleMarker([area.lat, area.lng], {
        radius: MARKER_RADIUS,
        color: '#000',
        weight: 1,
        opacity: 0.8,
        fillColor: MARKER_COLOR,
        fillOpacity: 0.85,
      });
      m.bindTooltip(
        `${area.compresa} ${area.particella} / adc ${area.numero}`,
        { direction: 'top', offset: [0, -5] },
      );
      m.on('click', (e) => {
        L.DomEvent.stopPropagation(e);
        this.activeAreaId = area.id;
        this._refreshHighlight();
        if (this.onAreaClick) this.onAreaClick(area.id);
      });
      this.markers.set(area.id, m);
      m.addTo(this.markerLayer);
    }
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

  /** Invalidate Leaflet's cached container size (call after un-collapsing). */
  invalidateSize() {
    if (this.leaflet) this.leaflet.invalidateSize();
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
