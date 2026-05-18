// Shared map functionality for bosco apps
// Provides: map creation, basemaps, coordinate display, measurement, location tracking

import * as S from './strings.js';

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

    // Create a map wrapper that abstracts the underlying map library
    function create(elementId, options = {}) {
        const config = {
            basemap: options.basemap || 'satellite',
            enableMeasure: options.enableMeasure !== false,
            enableLocation: options.enableLocation !== false,
            enableCoords: options.enableCoords !== false,
            enableBasemapControl: options.enableBasemapControl !== false,
            coordsElement: options.coordsElement || 'coords',
            leafletOptions: options.leafletOptions || { preferCanvas: true, zoomControl: false }
        };

        // Create Leaflet map
        const leafletMap = L.map(elementId, config.leafletOptions);

        // Track current basemap layer + key
        let activeBasemapName = BASEMAPS[config.basemap] ? config.basemap : 'satellite';
        let currentBasemap = createBasemapLayer(activeBasemapName);
        currentBasemap.addTo(leafletMap);
        let basemapControlInstance = null;

        // State for features
        let measureMode = false;
        let measurePoints = [];
        let measureLayer = null;

        let locationMode = false;
        let locationMarker = null;
        let locationCircle = null;

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

        // --- Coordinate display ---
        function setupCoords() {
            const coordsEl = $(config.coordsElement);
            if (!coordsEl) return;

            leafletMap.on('mousemove', e => {
                coordsEl.textContent = `(${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)})`;
            });
        }

        // --- Measurement tool ---
        function toggleMeasure() {
            measureMode = !measureMode;
            const btn = $('mc-measure-button');

            if (measureMode) {
                if (btn) btn.classList.add('mc-active');
                leafletMap.getContainer().style.cursor = 'crosshair';
            } else {
                if (btn) btn.classList.remove('mc-active');
                leafletMap.getContainer().style.cursor = '';
                clearMeasure();
            }
        }

        function clearMeasure() {
            measurePoints = [];
            if (measureLayer) {
                leafletMap.removeLayer(measureLayer);
                measureLayer = null;
            }
        }

        function addMeasurePoint(latlng) {
            if (!measureMode) return;

            measurePoints.push(latlng);

            if (!measureLayer) {
                measureLayer = L.layerGroup().addTo(leafletMap);
            }

            // Draw marker
            L.circleMarker(latlng, {
                radius: 4,
                fillColor: '#ff0000',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(measureLayer);

            // Draw line and label if multiple points
            if (measurePoints.length > 1) {
                L.polyline(measurePoints, {
                    color: '#ff0000',
                    weight: 2,
                    opacity: 0.7
                }).addTo(measureLayer);

                // Calculate total distance
                let totalDistance = 0;
                for (let i = 1; i < measurePoints.length; i++) {
                    totalDistance += leafletMap.distance(measurePoints[i - 1], measurePoints[i]);
                }

                const distText = totalDistance >= 1000
                    ? `${(totalDistance / 1000).toFixed(2)} km`
                    : `${totalDistance.toFixed(1)} m`;

                L.marker(latlng, {
                    icon: L.divIcon({
                        className: 'mc-measure-label',
                        html: `<div class="mc-measure-label-content">${distText}</div>`,
                        iconSize: null
                    })
                }).addTo(measureLayer);
            }
        }

        function setupMeasure() {
            const MeasureControl = L.Control.extend({
                options: { position: 'topleft' },
                onAdd: function() {
                    const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                    container.innerHTML = `
                        <a href="#" id="mc-measure-button" class="mc-control-button" title="Misura distanza">📏</a>
                    `;
                    L.DomEvent.on(container, 'click', function(e) {
                        L.DomEvent.stopPropagation(e);
                        L.DomEvent.preventDefault(e);
                        toggleMeasure();
                    });
                    return container;
                }
            });
            leafletMap.addControl(new MeasureControl());

            leafletMap.on('click', function(e) {
                if (measureMode) {
                    addMeasurePoint(e.latlng);
                }
            });
        }

        // --- Sidebar control ---
        function toggleSidebar() {
            const sidebar = $('sidebar');
            const mapContainer = $('map');
            if (!sidebar || !mapContainer) return;

            const isHidden = sidebar.classList.toggle('hidden');
            mapContainer.classList.toggle('sidebar-hidden', isHidden);

            // Wait for CSS transition to complete, then resize map without panning
            setTimeout(() => {
                leafletMap.invalidateSize({ pan: false });
            }, 300);
        }

        function setupSidebarToggle() {
            const SidebarControl = L.Control.extend({
                options: { position: 'topleft' },
                onAdd: function() {
                    const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                    container.innerHTML = `
                        <a href="#" id="mc-sidebar-button" class="mc-control-button" title="Mostra/nascondi pannello">☰</a>
                    `;
                    L.DomEvent.on(container, 'click', function(e) {
                        L.DomEvent.stopPropagation(e);
                        L.DomEvent.preventDefault(e);
                        toggleSidebar();
                    });
                    return container;
                }
            });
            leafletMap.addControl(new SidebarControl());
        }

        // --- Location tracking ---
        function toggleLocation() {
            locationMode = !locationMode;
            const btn = $('mc-location-button');

            if (locationMode) {
                if (btn) btn.classList.add('mc-active-location');
                leafletMap.locate({ watch: true, enableHighAccuracy: true });
            } else {
                if (btn) btn.classList.remove('mc-active-location');
                leafletMap.stopLocate();
                if (locationMarker) {
                    leafletMap.removeLayer(locationMarker);
                    locationMarker = null;
                }
                if (locationCircle) {
                    leafletMap.removeLayer(locationCircle);
                    locationCircle = null;
                }
            }
        }

        function onLocationFound(e) {
            const radius = e.accuracy;

            if (locationMarker) leafletMap.removeLayer(locationMarker);
            if (locationCircle) leafletMap.removeLayer(locationCircle);

            locationCircle = L.circle(e.latlng, {
                radius: radius,
                color: '#4CAF50',
                fillColor: '#4CAF50',
                fillOpacity: 0.15,
                weight: 2
            }).addTo(leafletMap);

            locationMarker = L.circleMarker(e.latlng, {
                radius: 8,
                fillColor: '#4CAF50',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 1
            }).addTo(leafletMap);

            locationMarker.bindTooltip(
                `<b>Posizione attuale</b><br>Precisione: ±${radius.toFixed(0)} m`,
                { permanent: false, direction: 'top' }
            );
        }

        function onLocationError(e) {
            alert('Impossibile determinare la posizione: ' + e.message);
            locationMode = false;
            const btn = $('mc-location-button');
            if (btn) btn.classList.remove('mc-active-location');
        }

        function setupLocation() {
            const LocationControl = L.Control.extend({
                options: { position: 'topleft' },
                onAdd: function() {
                    const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                    container.innerHTML = `
                        <a href="#" id="mc-location-button" class="mc-control-button" title="Mostra posizione">📍</a>
                    `;
                    L.DomEvent.on(container, 'click', function(e) {
                        L.DomEvent.stopPropagation(e);
                        L.DomEvent.preventDefault(e);
                        toggleLocation();
                    });
                    return container;
                }
            });
            leafletMap.addControl(new LocationControl());

            leafletMap.on('locationfound', onLocationFound);
            leafletMap.on('locationerror', onLocationError);
        }

        function setupZoom() {
            // Add zoom control with Italian labels
            L.control.zoom({
                position: 'topleft',
                zoomInTitle: 'Ingrandisci',
                zoomOutTitle: 'Rimpicciolisci'
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

        // --- Initialize features ---
        if (config.enableCoords) {
            setupCoords();
        }
        if ($('sidebar')) {
            setupSidebarToggle();
        }
        if (config.enableLocation) {
            setupLocation();
        }
        setupZoom();
        if (config.enableMeasure) {
            setupMeasure();
        }
        if (config.enableBasemapControl) {
            setupBasemapControl();
        }

        return {
            getLeafletMap() { return leafletMap; },
            getBasemap() { return activeBasemapName; },
            setBasemap,
            syncBasemap,
        };
    }

    const DEG_TO_RAD = Math.PI / 180;
    const EARTH_RADIUS = 6378137.0; // WGS84 semi-major axis in meters

    // Geodesic area of a polygon. Takes [{lat, lng}, ...], returns area in m².
    // Same algorithm as Leaflet.draw's L.GeometryUtil.geodesicArea.
    function geodesicArea(latlngs) {
        const n = latlngs.length;
        let area = 0;
        for (let i = 0; i < n; i++) {
            const p1 = latlngs[i];
            const p2 = latlngs[(i + 1) % n];
            area += (p2.lng - p1.lng) * DEG_TO_RAD *
                    (2 + Math.sin(p1.lat * DEG_TO_RAD) + Math.sin(p2.lat * DEG_TO_RAD));
        }
        return Math.abs(area * EARTH_RADIUS * EARTH_RADIUS / 2);
    }

    // Convert GeoJSON coordinate ring [[lng, lat], ...] to [{lat, lng}, ...]
    function ringToLatLngs(ring) {
        return ring.map(c => ({ lat: c[1], lng: c[0] }));
    }

    // Geodesic area of a GeoJSON feature in m² (exterior minus holes).
    function geoJSONFeatureArea(feature) {
        const geom = feature.geometry;
        if (!geom) return 0;
        const polygons = geom.type === 'Polygon' ? [geom.coordinates]
            : geom.type === 'MultiPolygon' ? geom.coordinates : [];
        let total = 0;
        for (const rings of polygons) {
            total += geodesicArea(ringToLatLngs(rings[0]));
            for (let i = 1; i < rings.length; i++) {
                total -= geodesicArea(ringToLatLngs(rings[i]));
            }
        }
        return total;
    }

    // Precompute geodesic area (m²) on each feature, sort largest-first
    // so smaller polygons render on top.
    function sortFeaturesByArea(geojson) {
        geojson.features.forEach(f => {
            f.properties._areaM2 = geoJSONFeatureArea(f);
        });
        geojson.features.sort((a, b) => b.properties._areaM2 - a.properties._areaM2);
        return geojson;
    }

    // Shared Leaflet style for parcel polygons.  Warm yellow against
    // satellite green/brown gives strong border contrast on every
    // basemap; fillOpacity stays low so imagery shows through.
    const PARCEL_STYLE = {
        color: '#ffd54f',
        weight: 2,
        opacity: 0.9,
        fillColor: '#fff',
        fillOpacity: 0.04,
    };

    // Public API
    return {
        create,
        geodesicArea,
        sortFeaturesByArea,
        BASEMAPS,
        PARCEL_STYLE,
    };
})();

if (typeof module !== 'undefined') module.exports = MapCommon;

export default MapCommon;
