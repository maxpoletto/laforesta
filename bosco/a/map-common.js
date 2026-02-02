// Shared map functionality for bosco apps
// Provides: map creation, basemaps, coordinate display, measurement, location tracking
const MapCommon = (function() {
    'use strict';

    const BASEMAPS = {
        osm: {
            url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attribution: '© OpenStreetMap'
        },
        satellite: {
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attribution: '© Esri'
        },
        topo: {
            url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attribution: '© OpenTopoMap'
        }
    };

    // Create a map wrapper that abstracts the underlying map library
    function create(elementId, options = {}) {
        const config = {
            basemap: options.basemap || 'satellite',
            enableMeasure: options.enableMeasure !== false,
            enableLocation: options.enableLocation !== false,
            enableCoords: options.enableCoords !== false,
            coordsElement: options.coordsElement || 'coords',
            leafletOptions: options.leafletOptions || { preferCanvas: true }
        };

        // Create Leaflet map
        const leafletMap = L.map(elementId, config.leafletOptions);

        // Track current basemap layer
        let currentBasemap = createBasemapLayer(config.basemap);
        currentBasemap.addTo(leafletMap);

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
            leafletMap.removeLayer(currentBasemap);
            currentBasemap = createBasemapLayer(name);
            currentBasemap.addTo(leafletMap);
        }

        // --- Coordinate display ---
        function setupCoords() {
            const coordsEl = document.getElementById(config.coordsElement);
            if (!coordsEl) return;

            leafletMap.on('mousemove', e => {
                coordsEl.textContent = `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
            });
        }

        // --- Measurement tool ---
        function toggleMeasure() {
            measureMode = !measureMode;
            const btn = document.querySelector('.mc-measure-button');

            if (measureMode) {
                if (btn) btn.style.backgroundColor = '#ffd700';
                leafletMap.getContainer().style.cursor = 'crosshair';
            } else {
                if (btn) btn.style.backgroundColor = '';
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
                        html: `<div style="background: white; padding: 2px 6px; border: 2px solid #ff0000; border-radius: 3px; font-size: 12px; font-weight: bold; white-space: nowrap;">${distText}</div>`,
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
                        <a href="#" class="mc-measure-button" title="Misura distanza"
                           style="width: 30px; height: 30px; line-height: 30px; text-align: center; font-size: 18px; text-decoration: none;">📏</a>
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

        // --- Location tracking ---
        function toggleLocation() {
            locationMode = !locationMode;
            const btn = document.querySelector('.mc-location-button');

            if (locationMode) {
                if (btn) btn.style.backgroundColor = '#4CAF50';
                leafletMap.locate({ watch: true, enableHighAccuracy: true });
            } else {
                if (btn) btn.style.backgroundColor = '';
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
            const btn = document.querySelector('.mc-location-button');
            if (btn) btn.style.backgroundColor = '';
        }

        function setupLocation() {
            const LocationControl = L.Control.extend({
                options: { position: 'topleft' },
                onAdd: function() {
                    const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
                    container.innerHTML = `
                        <a href="#" class="mc-location-button" title="Mostra posizione"
                           style="width: 30px; height: 30px; line-height: 30px; text-align: center; font-size: 18px; text-decoration: none;">📍</a>
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

        // --- Initialize features ---
        if (config.enableCoords) {
            setupCoords();
        }
        if (config.enableMeasure) {
            setupMeasure();
        }
        if (config.enableLocation) {
            setupLocation();
        }

        // --- Return wrapper object ---
        // This abstraction layer allows future migration to other map libraries
        return {
            // Get underlying Leaflet map for app-specific operations
            getLeafletMap() {
                return leafletMap;
            },

            // Basemap control
            setBasemap,

            // Layer operations
            addLayerGroup() {
                const group = L.layerGroup().addTo(leafletMap);
                return group;
            },

            addGeoJSON(data, options = {}) {
                const layer = L.geoJSON(data, options).addTo(leafletMap);
                return layer;
            },

            // View operations
            fitBounds(bounds, options) {
                leafletMap.fitBounds(bounds, options);
            },

            setView(center, zoom) {
                leafletMap.setView(center, zoom);
            },

            // Event handling
            on(event, handler) {
                leafletMap.on(event, handler);
            },

            // Marker creation
            addCircleMarker(latlng, options) {
                return L.circleMarker(latlng, options).addTo(leafletMap);
            },

            // Utility
            distance(latlng1, latlng2) {
                return leafletMap.distance(latlng1, latlng2);
            }
        };
    }

    // Public API
    return {
        create,
        BASEMAPS
    };
})();
