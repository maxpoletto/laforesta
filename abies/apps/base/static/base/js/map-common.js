// Thin Leaflet shim for abies maps.
//
// Provides map creation with the standard chrome every abies map shares — the
// basemap layer + switcher, the zoom control, and an optional coordinate
// readout — plus basemap get/set/sync (firing `basemapchange` for cross-map
// sync).  The heavier opt-in tools (measure, location, sidebar toggle) live in
// map-tools.js; the parcel/marker abstraction in parcel-map.js.

import * as S from './strings.js';
import { fmtCoord } from './format.js';

const $ = id => document.getElementById(id);

const MapCommon = (function() {
    'use strict';

    const BASEMAP_ICON_BASE = '/static/base/img/';
    const BASEMAPS = {
        osm: {
            url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution: '© OpenStreetMap',
            label: S.BASEMAP_OSM,
            icon: BASEMAP_ICON_BASE + 'basemap-osm.png',
        },
        satellite: {
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attribution: '© Esri',
            label: S.BASEMAP_SAT,
            icon: BASEMAP_ICON_BASE + 'basemap-sat.png',
        },
        topo: {
            url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attribution: '© OpenTopoMap',
            label: S.BASEMAP_TOPO,
            icon: BASEMAP_ICON_BASE + 'basemap-topo.png',
        },
    };
    // Display order in the BasemapControl row.
    const BASEMAP_ORDER = ['osm', 'topo', 'satellite'];

    // Create a map wrapper that abstracts the underlying map library.
    function create(elementId, options = {}) {
        const config = {
            basemap: options.basemap || 'satellite',
            coordsElement: options.coordsElement || 'coords',
            leafletOptions: options.leafletOptions || { preferCanvas: true, zoomControl: false },
        };

        const leafletMap = L.map(elementId, config.leafletOptions);

        // Track current basemap layer + key.
        let activeBasemapName = BASEMAPS[config.basemap] ? config.basemap : 'satellite';
        let currentBasemap = createBasemapLayer(activeBasemapName);
        currentBasemap.addTo(leafletMap);
        let basemapControlInstance = null;

        // --- Basemap functions ---
        function createBasemapLayer(name) {
            const basemap = BASEMAPS[name] || BASEMAPS.satellite;
            return L.tileLayer(basemap.url, { attribution: basemap.attribution });
        }

        function setBasemap(name) {
            if (!BASEMAPS[name]) return;
            leafletMap.removeLayer(currentBasemap);
            currentBasemap = createBasemapLayer(name);
            currentBasemap.addTo(leafletMap);
            activeBasemapName = name;
        }

        // Keep the basemap layer and the control's visual state in lockstep,
        // *without* firing a `basemapchange` event.  Called by page code to
        // mirror a basemap change from one map onto its siblings (or from a
        // URL/back-button sync) — firing here would loop.
        function syncBasemap(name) {
            if (!BASEMAPS[name] || name === activeBasemapName) return;
            setBasemap(name);
            basemapControlInstance?.setActive(name);
        }

        // --- Coordinate readout (no-op if the page has no coords element) ---
        function setupCoords() {
            const coordsEl = $(config.coordsElement);
            if (!coordsEl) return;
            leafletMap.on('mousemove', e => {
                coordsEl.textContent = `(${fmtCoord(e.latlng.lat)}, ${fmtCoord(e.latlng.lng)})`;
            });
        }

        // --- Zoom control with localized labels ---
        function setupZoom() {
            L.control.zoom({
                position: 'topleft',
                zoomInTitle: S.MAP_ZOOM_IN,
                zoomOutTitle: S.MAP_ZOOM_OUT,
            }).addTo(leafletMap);
        }

        // --- Basemap switcher control ---
        // Collapses to the active thumbnail at rest; expands horizontally on
        // hover (mouse) or on a tap of the active thumbnail (touch).  First
        // click on a collapsed control expands without selecting — this is
        // what makes it work on phones without a hover fallback.
        function setupBasemapControl() {
            const Control = L.Control.extend({
                options: { position: 'topright' },
                initialize: function () {
                    this._items = [];
                    this._expanded = false;
                },
                onAdd: function () {
                    const c = L.DomUtil.create('div', 'mc-basemap-switcher');
                    L.DomEvent.disableClickPropagation(c);
                    L.DomEvent.on(c, 'mouseenter', () => this._setExpanded(true));
                    L.DomEvent.on(c, 'mouseleave', () => this._setExpanded(false));
                    for (const key of BASEMAP_ORDER) {
                        const meta = BASEMAPS[key];
                        const el = L.DomUtil.create('button', 'mc-basemap-item', c);
                        el.type = 'button';
                        el.title = meta.label;
                        el.style.backgroundImage = `url(${meta.icon})`;
                        const lbl = L.DomUtil.create('span', 'mc-basemap-label', el);
                        lbl.textContent = meta.label;
                        L.DomEvent.on(el, 'click', (e) => {
                            L.DomEvent.stop(e);
                            if (!this._expanded) { this._setExpanded(true); return; }
                            this._setExpanded(false);
                            if (key !== activeBasemapName) {
                                setBasemap(key);
                                this._render();
                                leafletMap.fire('basemapchange', { name: key });
                            }
                        });
                        this._items.push({ key, el });
                    }
                    this._render();
                    return c;
                },
                // External hook used by `syncBasemap` to refresh visuals
                // after the page has already swapped the layer.
                setActive: function () { this._render(); },
                _setExpanded: function (v) { this._expanded = v; this._render(); },
                _render: function () {
                    for (const it of this._items) {
                        it.el.classList.toggle('mc-active', it.key === activeBasemapName);
                        it.el.classList.toggle(
                            'mc-collapsed',
                            !this._expanded && it.key !== activeBasemapName);
                    }
                },
            });
            basemapControlInstance = new Control();
            leafletMap.addControl(basemapControlInstance);
        }

        setupCoords();
        setupZoom();
        setupBasemapControl();

        return {
            getLeafletMap() { return leafletMap; },
            getBasemap() { return activeBasemapName; },
            setBasemap,
            syncBasemap,
        };
    }

    // Public API
    return {
        create,
        BASEMAPS,
    };
})();

if (typeof module !== 'undefined') module.exports = MapCommon;

export default MapCommon;
