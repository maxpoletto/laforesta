/**
 * ParcelMap — the general abies map abstraction, layered on the MapCommon
 * Leaflet shim.  It provides:
 *
 *   - a map div (created inside a host container) with the chosen basemap;
 *   - opt-in tools (measure / location / sidebar) via the `tools` option;
 *   - parcel polygons rendered with the shared style + hover labels;
 *   - fit-to-parcels / saved-view restore + view-change reporting;
 *   - a single unified click callback, `onMapClick(latlng, feature|null)`;
 *   - a sample-area marker layer (add/clear + active-highlight) for the maps
 *     that place markers.
 *
 * Subclass it or compose it — see "Using ParcelMap" in docs/ui-maps.md.
 * Leaflet is referenced as the runtime global `L`.
 */
import MapCommon from './map-common.js';
import { parcelLabel } from './geo.js';
import { attachMeasure, attachLocation, attachSidebarToggle } from './map-tools.js';

export const MARKER_RADIUS = 7;
const MARKER_BORDER = '#000';
const MARKER_BORDER_OPACITY = 0.8;
const ACTIVE_COLOR = '#ffcc00';
const ACTIVE_WEIGHT = 3;
const INACTIVE_WEIGHT = 1;

// Shared Leaflet style for parcel polygons.  Warm yellow against satellite
// green/brown gives strong border contrast on every basemap; fillOpacity stays
// low so imagery shows through.
export const PARCEL_STYLE = {
  color: '#ffd54f', weight: 2, opacity: 0.9, fillColor: '#fff', fillOpacity: 0.04,
};

export function parcelTooltipContent(feature) {
  const label = parcelLabel(feature);
  if (!label) return null;
  const el = document.createElement('div');
  el.className = 'parcel-tooltip';
  const title = document.createElement('strong');
  title.className = 'parcel-tooltip-title';
  title.textContent = label.title;
  el.appendChild(title);
  if (label.type) {
    const type = document.createElement('div');
    type.textContent = label.type;
    el.appendChild(type);
  }
  return el;
}

// Padding (px) when fitting the map to the parcel layer's bounds, so the
// parcels never sit flush against the map edge.  Shared by every map that
// frames the parcels so they all open with the same framing.
export const FIT_PADDING = [20, 20];

export class ParcelMap {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.container — host; ParcelMap appends the map div.
   * @param {string} [opts.className] — CSS class for the map div.
   * @param {object} opts.geojson — sorted parcel FeatureCollection.
   * @param {string} [opts.basemap]
   * @param {{measure?:boolean, location?:boolean, sidebar?:boolean|object}} [opts.tools]
   *   — opt-in tools, off by default.  `sidebar` may be `true` or an
   *   `{sidebarId, mapId}` object passed through to `attachSidebarToggle`.
   * @param {{center:[number,number], zoom:number}} [opts.initialView] — open at
   *   this view instead of fit-to-parcels (keeps pan/zoom across re-renders).
   * @param {function([number,number], number): void} [opts.onViewChange] —
   *   called on moveend / zoomend with the current center + zoom.
   * @param {function(object, object|null): void} [opts.onMapClick] — called with
   *   (latlng, feature) for a parcel click and (latlng, null) for empty space.
   *   Marker clicks do NOT reach here (they stop propagation).
   */
  constructor(opts) {
    this.container = opts.container;
    this.geojson = opts.geojson;
    this.onViewChange = opts.onViewChange || null;
    this.onMapClick = opts.onMapClick || null;
    this.markers = new Map();      // sample_area_id → CircleMarker
    this.activeAreaId = null;

    this.mapEl = document.createElement('div');
    if (opts.className) this.mapEl.className = opts.className;
    this.container.appendChild(this.mapEl);

    this.wrapper = MapCommon.create(this.mapEl, {
      basemap: opts.basemap || 'satellite',
    });
    this.leaflet = this.wrapper.getLeafletMap();

    // Opt-in tools (off by default).  Each returns a teardown handle.
    const tools = opts.tools || {};
    this._toolHandles = [];
    if (tools.measure) this._toolHandles.push(attachMeasure(this.leaflet));
    if (tools.location) this._toolHandles.push(attachLocation(this.leaflet));
    if (tools.sidebar) {
      this._toolHandles.push(attachSidebarToggle(
        this.leaflet, tools.sidebar === true ? undefined : tools.sidebar));
    }

    // Empty-space click → onMapClick(latlng, null).  Markers and parcel
    // polygons stop propagation, so this only fires off every feature.
    this.leaflet.on('click', (e) => {
      if (this._toolCapturesClicks()) return;   // a tool owns clicks while active
      if (this.onMapClick) this.onMapClick(e.latlng, null);
    });

    this._renderParcels();
    this.markerLayer = L.layerGroup().addTo(this.leaflet);

    if (opts.initialView) {
      this.leaflet.setView(opts.initialView.center, opts.initialView.zoom);
    } else {
      this.fitParcels();
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

  /** True while any tool is capturing map clicks (so onMapClick is suppressed). */
  _toolCapturesClicks() {
    return this._toolHandles.some(h => h.capturesClicks?.());
  }

  /** Frame the map on the parcel layer — the default view (no saved view). */
  fitParcels() {
    if (this.parcelLayer && this.parcelLayer.getBounds().isValid()) {
      this.leaflet.fitBounds(this.parcelLayer.getBounds(),
                             { padding: FIT_PADDING });
    }
  }

  _renderParcels() {
    this.parcelLayer = L.geoJSON(this.geojson, {
      style: PARCEL_STYLE,
      onEachFeature: (feature, lyr) => {
        const label = parcelTooltipContent(feature);
        if (label) lyr.bindTooltip(label, { sticky: true, direction: 'top' });
        // Click on a parcel → onMapClick(latlng, feature).  Leaflet identifies
        // the parcel via its own hit-testing (the layer that drives the
        // tooltip); stopPropagation keeps the empty-space handler quiet.
        lyr.on('click', (e) => {
          // Let a click-capturing tool (measure) take this: fall through to the
          // map without stopPropagation, rather than open the page's prompt.
          if (this._toolCapturesClicks()) return;
          if (!this.onMapClick) return;
          L.DomEvent.stopPropagation(e);
          this.onMapClick(e.latlng, feature);
        });
      },
    }).addTo(this.leaflet);
  }

  _clearMarkers() {
    this.markerLayer.clearLayers();
    this.markers.clear();
  }

  /**
   * Add a sample-area circle marker.  Subclasses pass the fill style, tooltip,
   * and a click handler (receiving the area row).  The marker click sets the
   * active area *directly* (not via the idempotent `setActiveAreaId`) and
   * re-fires `onClick` even for the already-active marker, matching prior
   * behaviour.
   */
  _addAreaMarker(area, { fillColor, fillOpacity, tooltip, onClick }) {
    const m = L.circleMarker([area.lat, area.lon], {
      radius: MARKER_RADIUS, color: MARKER_BORDER, weight: INACTIVE_WEIGHT,
      opacity: MARKER_BORDER_OPACITY, fillColor, fillOpacity,
    });
    if (tooltip) m.bindTooltip(tooltip, { direction: 'top', offset: [0, -5] });
    m.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      this.activeAreaId = area.id;
      this._refreshHighlight();
      if (onClick) onClick(area);
    });
    this.markers.set(area.id, m);
    m.addTo(this.markerLayer);
  }

  /**
   * Fill a sample-area tooltip formatter's shared area data.  Subclasses call
   * it from `setAreas`; Rilevamenti passes the visited-tree count as an extra
   * argument.
   */
  _areaTooltip(formatter, area, ...args) {
    return formatter(area, ...args);
  }

  /** Page-driven active-area setter (idempotent — no work when unchanged). */
  setActiveAreaId(areaId) {
    if (this.activeAreaId === areaId) return;
    this.activeAreaId = areaId;
    this._refreshHighlight();
  }

  _refreshHighlight() {
    for (const [id, m] of this.markers) {
      const isActive = id === this.activeAreaId;
      m.setStyle({ weight: isActive ? ACTIVE_WEIGHT : INACTIVE_WEIGHT,
                   color: isActive ? ACTIVE_COLOR : MARKER_BORDER });
      if (isActive) m.bringToFront();
    }
  }

  /** Invalidate Leaflet's cached container size (call after un-collapsing). */
  invalidateSize() {
    if (this.leaflet) this.leaflet.invalidateSize();
  }

  destroy() {
    // Tear down tools (stops the geolocation watch) before removing the map.
    if (this._toolHandles) {
      this._toolHandles.forEach(h => h.destroy());
      this._toolHandles = [];
    }
    if (this.leaflet) { this.leaflet.remove(); this.leaflet = null; }
    this.mapEl?.remove();
    this.mapEl = null;
    this.wrapper = null;
    this.parcelLayer = null;
    this.markerLayer = null;
    this.markers.clear();
  }

  /** Override in subclasses: map data rows → markers via `_addAreaMarker`. */
  setAreas() {}
}
