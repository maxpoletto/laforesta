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
import { parcelLabel } from '../../base/js/geo.js';

const MARKER_COLOR = '#2d5d2c';
const MARKER_RADIUS = 7;

export class GriglieMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container
   * @param {object} opts.geojson — terreni.geojson (sorted-by-area)
   * @param {function(object): void} [opts.onAreaClick]
   *   Called with the area row ({id, lat, lon, compresa, particella, numero,
   *   altitude, r_m, note}) when an area marker is clicked.  The handler
   *   typically opens a popover.
   * @param {function(number, number): void} [opts.onEmptyClick]
   *   Called with (lat, lon) when the user clicks empty map space.
   *   Used by writers to open the "Inserire una nuova area qui?" prompt.
   * @param {{center: [number,number], zoom: number}} [opts.initialView]
   *   If given, the map opens at this view instead of fit-to-parcels.
   *   Used to keep the user's pan/zoom across re-renders.
   * @param {function([number,number], number): void} [opts.onViewChange]
   *   Called on `moveend` / `zoomend` so callers can stash the current
   *   center + zoom.
   */
  constructor(opts) {
    this.container = opts.container;
    this.geojson = opts.geojson;
    this.onAreaClick = opts.onAreaClick || null;
    this.onEmptyClick = opts.onEmptyClick || null;
    this.onViewChange = opts.onViewChange || null;
    this.markers = new Map();
    this.activeAreaId = null;

    this.mapEl = document.createElement('div');
    this.mapEl.className = 'griglie-map';
    this.container.appendChild(this.mapEl);

    this.wrapper = MapCommon.create(this.mapEl, {
      basemap: opts.basemap || 'satellite',
    });
    this.leaflet = this.wrapper.getLeafletMap();

    // Empty-space click → callback.  Markers stop propagation, so this
    // only fires for clicks on the basemap / parcel polygons.
    this.leaflet.on('click', (e) => {
      if (this.onEmptyClick) {
        this.onEmptyClick(e.latlng.lat, e.latlng.lng);
      }
    });

    this._renderParcels();
    this.markerLayer = L.layerGroup().addTo(this.leaflet);

    if (opts.initialView) {
      this.leaflet.setView(opts.initialView.center, opts.initialView.zoom);
    } else if (this.parcelLayer) {
      this.leaflet.fitBounds(this.parcelLayer.getBounds(), { padding: [20, 20] });
    }

    if (this.onViewChange) {
      const report = () => {
        const c = this.leaflet.getCenter();
        this.onViewChange([c.lat, c.lng], this.leaflet.getZoom());
      };
      this.leaflet.on('moveend', report);
      this.leaflet.on('zoomend', report);
    }
  }

  _renderParcels() {
    this.parcelLayer = L.geoJSON(this.geojson, {
      style: MapCommon.PARCEL_STYLE,
      onEachFeature: (feature, lyr) => {
        const label = parcelLabel(feature);
        if (label) lyr.bindTooltip(label, { sticky: true, direction: 'top' });
      },
    }).addTo(this.leaflet);
  }

  /**
   * Render markers for the active grid's sample areas.
   *
   * @param {Array<{id, lat, lon, compresa, particella, numero}>} areas
   */
  setAreas(areas) {
    this.markerLayer.clearLayers();
    this.markers.clear();

    for (const area of areas) {
      const m = L.circleMarker([area.lat, area.lon], {
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
        if (this.onAreaClick) this.onAreaClick(area);
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
