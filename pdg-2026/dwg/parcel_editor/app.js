// Parcel Editor - encapsulated module
const ParcelEditor = (function() {
    'use strict';

    // State
    let map = null;
    let currentBasemap = null;
    let referenceLayer = null;
    let drawnItems = null;
    let drawControl = null;

    let originalReferenceData = null;
    let currentOffsetEW = 0;
    let currentOffsetNS = 0;

    let visibleLayers = new Set();
    let allLayers = new Set();
    let parcelCounter = 0;
    let selectedLayer = null;

    // Constants
    const basemaps = {
        osm: () => L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }),
        satellite: () => L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri'
        }),
        topo: () => L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenTopoMap'
        })
    };

    const crsPresets = {
        'none': { ew: 0, ns: 0 },
        'wgs84': { ew: 0, ns: 0 },
        'ed50': { ew: -90, ns: -130 },
        'monte-mario': { ew: -50, ns: -70 },
        'custom': null
    };

    const styles = {
        default: { color: '#3388ff', weight: 2, fillOpacity: 0.2 },
        loaded: { color: '#ff6600', weight: 2, fillOpacity: 0.2 },
        selected: { color: '#00ff00', weight: 3, fillOpacity: 0.4 }
    };

    // DOM helpers
    const $ = id => document.getElementById(id);

    function updateStatus(msg) {
        $('status').innerHTML = msg;
    }

    // Offset handling
    function offsetCoordinates(coords, offsetLon, offsetLat) {
        if (typeof coords[0] === 'number') {
            return [coords[0] + offsetLon, coords[1] + offsetLat];
        }
        return coords.map(c => offsetCoordinates(c, offsetLon, offsetLat));
    }

    function getOffsetDegrees() {
        return {
            lon: currentOffsetEW / 87000,
            lat: currentOffsetNS / 111000
        };
    }

    function applyOffsetToData(data) {
        const { lon, lat } = getOffsetDegrees();
        const offsetData = JSON.parse(JSON.stringify(data));
        for (const feature of offsetData.features || []) {
            if (feature.geometry?.coordinates) {
                feature.geometry.coordinates = offsetCoordinates(
                    feature.geometry.coordinates, lon, lat
                );
            }
        }
        return offsetData;
    }

    function updateReferenceWithOffset() {
        if (!originalReferenceData) return;
        referenceLayer.clearLayers();
        referenceLayer.addData(applyOffsetToData(originalReferenceData));
        updateStatus(`Offset: ${currentOffsetEW}m E, ${currentOffsetNS}m N`);
    }

    function setOffsetUI(ew, ns) {
        currentOffsetEW = ew;
        currentOffsetNS = ns;
        $('offset-ew').value = ew;
        $('offset-ns').value = ns;
        $('offset-ew-value').textContent = `${ew}m`;
        $('offset-ns-value').textContent = `${ns}m`;
    }

    // Parcel selection
    function addLayerClickHandler(layer) {
        layer.on('click', function(e) {
            L.DomEvent.stopPropagation(e);
            selectParcel(layer._leaflet_id);
        });
    }

    function selectParcel(leafletId) {
        if (selectedLayer?.editing) {
            selectedLayer.editing.disable();
        }
        if (selectedLayer) {
            selectedLayer.setStyle(selectedLayer.isLoaded ? styles.loaded : styles.default);
        }

        selectedLayer = null;
        drawnItems.eachLayer(layer => {
            if (layer._leaflet_id === leafletId) {
                selectedLayer = layer;
                layer.setStyle(styles.selected);
                layer.editing.enable();
                updateStatus(`Selected: ${layer.parcelName} - drag vertices to edit`);
            }
        });
        updateParcelList();
    }

    function deselectParcel() {
        if (selectedLayer) {
            selectedLayer.editing?.disable();
            selectedLayer.setStyle(selectedLayer.isLoaded ? styles.loaded : styles.default);
            selectedLayer = null;
            updateParcelList();
            updateStatus('Ready');
        }
    }

    // Layer filtering
    function updateLayerFilter() {
        const container = $('layer-filter-container');
        const filter = $('layer-filter');

        if (allLayers.size === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';
        filter.innerHTML = '';

        for (const layerName of Array.from(allLayers).sort()) {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = visibleLayers.has(layerName);
            checkbox.onchange = () => toggleLayer(layerName, checkbox.checked);
            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(layerName));
            filter.appendChild(label);
        }
    }

    function toggleLayer(layerName, visible) {
        visible ? visibleLayers.add(layerName) : visibleLayers.delete(layerName);
        updateParcelVisibility();
        updateParcelList();
    }

    function updateParcelVisibility() {
        drawnItems.eachLayer(layer => {
            const layerName = layer.originalProperties?.layer || '';
            const shouldShow = visibleLayers.size === 0 || visibleLayers.has(layerName);

            if (shouldShow) {
                layer.setStyle({ ...layer.options, opacity: 1, fillOpacity: 0.2 });
            } else {
                layer.setStyle({ ...layer.options, opacity: 0, fillOpacity: 0 });
                if (selectedLayer?._leaflet_id === layer._leaflet_id) {
                    deselectParcel();
                }
            }
        });
    }

    function updateParcelList() {
        const list = $('parcel-list');
        list.innerHTML = '';

        drawnItems.eachLayer(layer => {
            const layerName = layer.originalProperties?.layer || '';
            if (visibleLayers.size > 0 && !visibleLayers.has(layerName)) return;

            const isSelected = selectedLayer?._leaflet_id === layer._leaflet_id;
            const loadedClass = layer.isLoaded ? ' loaded' : '';

            const item = document.createElement('div');
            item.className = 'parcel-item' + (isSelected ? ' selected' : '');
            item.dataset.leafletId = layer._leaflet_id;
            item.innerHTML = `
                <span class="parcel-name${loadedClass}" onclick="ParcelEditor.selectParcel(${layer._leaflet_id})">${layer.parcelName || 'Unnamed'}</span>
                <span class="parcel-actions">
                    <span class="edit-btn" onclick="ParcelEditor.startRename(${layer._leaflet_id})" title="Rename">✎</span>
                    <span class="delete-btn" onclick="ParcelEditor.deleteParcel(${layer._leaflet_id})" title="Delete">✕</span>
                </span>
            `;
            list.appendChild(item);
        });
    }

    // Rename functionality
    function startRename(leafletId) {
        const item = document.querySelector(`.parcel-item[data-leaflet-id="${leafletId}"]`);
        if (!item) return;

        let layer = null;
        drawnItems.eachLayer(l => { if (l._leaflet_id === leafletId) layer = l; });
        if (!layer) return;

        const currentName = layer.parcelName || 'Unnamed';
        item.innerHTML = `
            <input type="text" class="rename-input" value="${currentName}"
                   onkeydown="ParcelEditor.handleRenameKey(event, ${leafletId})" />
            <span class="parcel-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRename(${leafletId})" title="Save">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateParcelList()" title="Cancel">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRenameKey(event, leafletId) {
        if (event.key === 'Enter') finishRename(leafletId);
        else if (event.key === 'Escape') updateParcelList();
    }

    function finishRename(leafletId) {
        const input = document.querySelector(`.parcel-item[data-leaflet-id="${leafletId}"] .rename-input`);
        if (!input) return;

        const newName = input.value.trim() || 'Unnamed';
        drawnItems.eachLayer(layer => {
            if (layer._leaflet_id === leafletId) {
                layer.parcelName = newName;
                layer.bindPopup(`<b>${newName}</b>`);
                updateStatus(`Renamed to "${newName}"`);
            }
        });
        updateParcelList();
    }

    function deleteParcel(leafletId) {
        drawnItems.eachLayer(layer => {
            if (layer._leaflet_id === leafletId) {
                if (selectedLayer?._leaflet_id === leafletId) deselectParcel();
                drawnItems.removeLayer(layer);
            }
        });
        updateParcelList();
        updateStatus('Parcel deleted');
    }

    // Export functions
    function collectFeatures(filterByVisibility) {
        const features = [];
        drawnItems.eachLayer(layer => {
            if (filterByVisibility) {
                const layerName = layer.originalProperties?.layer || '';
                if (visibleLayers.size > 0 && !visibleLayers.has(layerName)) return;
            }

            const geojson = layer.toGeoJSON();
            if (layer.isLoaded && layer.originalProperties) {
                geojson.properties = { ...layer.originalProperties };
            }
            geojson.properties = geojson.properties || {};
            geojson.properties.name = layer.parcelName;
            geojson.properties.id = layer.parcelId;
            features.push(geojson);
        });
        return features;
    }

    function downloadGeoJSON(features, filename) {
        const geojson = { type: 'FeatureCollection', features };
        const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    // Public API
    return {
        init() {
            // Initialize map
            map = L.map('map').setView([38.65, 16.3], 12);
            currentBasemap = basemaps.satellite().addTo(map);

            // Reference layer
            referenceLayer = L.geoJSON(null, {
                style: styles.loaded,
                onEachFeature(feature, layer) {
                    if (feature.properties) {
                        const props = Object.entries(feature.properties)
                            .map(([k, v]) => `<b>${k}:</b> ${v}`)
                            .join('<br>');
                        layer.bindPopup(props || 'No properties');
                    }
                }
            }).addTo(map);

            // Drawn items layer
            drawnItems = new L.FeatureGroup().addTo(map);

            // Drawing controls
            drawControl = new L.Control.Draw({
                position: 'topleft',
                draw: {
                    polygon: { allowIntersection: false, shapeOptions: styles.default },
                    polyline: false, rectangle: false, circle: false,
                    marker: false, circlemarker: false
                },
                edit: false
            });
            map.addControl(drawControl);

            // Event handlers
            map.on(L.Draw.Event.CREATED, e => {
                const layer = e.layer;
                parcelCounter++;
                layer.parcelId = parcelCounter;
                layer.parcelName = `Parcel ${parcelCounter}`;
                layer.isLoaded = false;
                layer.bindPopup(`<b>${layer.parcelName}</b>`);
                addLayerClickHandler(layer);
                drawnItems.addLayer(layer);
                updateParcelList();
                selectParcel(layer._leaflet_id);
                updateStatus(`Created ${layer.parcelName}`);
            });

            map.on('click', e => {
                if (e.originalEvent.target === map._container ||
                    e.originalEvent.target.classList.contains('leaflet-tile')) {
                    deselectParcel();
                }
            });

            map.on('mousemove', e => {
                $('coords').innerHTML = `Lat: ${e.latlng.lat.toFixed(6)}, Lng: ${e.latlng.lng.toFixed(6)}`;
            });

            // Offset slider handlers
            $('offset-ew').addEventListener('input', e => {
                currentOffsetEW = parseInt(e.target.value);
                $('offset-ew-value').textContent = `${currentOffsetEW}m`;
                updateReferenceWithOffset();
            });

            $('offset-ns').addEventListener('input', e => {
                currentOffsetNS = parseInt(e.target.value);
                $('offset-ns-value').textContent = `${currentOffsetNS}m`;
                updateReferenceWithOffset();
            });

            // File loader
            $('load-reference').addEventListener('change', e => {
                const file = e.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = evt => {
                    try {
                        originalReferenceData = JSON.parse(evt.target.result);
                        this.resetOffset();
                        referenceLayer.clearLayers();
                        referenceLayer.addData(originalReferenceData);

                        if (referenceLayer.getBounds().isValid()) {
                            map.fitBounds(referenceLayer.getBounds());
                        }
                        const count = originalReferenceData.features?.length || 0;
                        updateStatus(`Loaded ${count} reference parcels. Adjust offset, then click "Import" to edit.`);
                    } catch (err) {
                        updateStatus('Error loading GeoJSON: ' + err.message);
                    }
                };
                reader.readAsText(file);
            });

            // Auto-load all_parcels.json if available
            fetch('all_parcels.json')
                .then(r => r.ok ? r.json() : Promise.reject())
                .then(data => {
                    originalReferenceData = data;
                    referenceLayer.clearLayers();
                    referenceLayer.addData(data);
                    if (referenceLayer.getBounds().isValid()) {
                        map.fitBounds(referenceLayer.getBounds());
                    }
                    const count = data.features?.length || 0;
                    updateStatus(`Auto-loaded ${count} parcels from all_parcels.json. Adjust offset, then "Import to Edit".`);
                })
                .catch(() => {});
        },

        setBasemap(name) {
            map.removeLayer(currentBasemap);
            currentBasemap = basemaps[name]().addTo(map);
        },

        resetOffset() {
            setOffsetUI(0, 0);
            $('crs-preset').value = 'none';
            updateReferenceWithOffset();
        },

        applyCrsPreset() {
            const preset = $('crs-preset').value;
            if (preset === 'custom' || !crsPresets[preset]) return;
            const { ew, ns } = crsPresets[preset];
            setOffsetUI(ew, ns);
            updateReferenceWithOffset();
        },

        applyOffset() {
            if (!originalReferenceData) {
                updateStatus('No reference data loaded');
                return;
            }
            const offsetData = applyOffsetToData(originalReferenceData);
            offsetData.properties = offsetData.properties || {};
            offsetData.properties.offset_applied = {
                east_meters: currentOffsetEW,
                north_meters: currentOffsetNS
            };
            downloadGeoJSON(offsetData.features, 'parcels_corrected.geojson');
            updateStatus(`Exported with offset: ${currentOffsetEW}m E, ${currentOffsetNS}m N`);
        },

        importReferenceToEdit() {
            if (!originalReferenceData) {
                updateStatus('No reference data loaded');
                return;
            }

            const offsetData = applyOffsetToData(originalReferenceData);
            let loadedCount = 0;
            allLayers.clear();

            L.geoJSON(offsetData, {
                onEachFeature(feature, layer) {
                    if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
                        parcelCounter++;
                        layer.parcelId = parcelCounter;
                        layer.parcelName = feature.properties?.name ||
                            (feature.properties?.parcel_index !== undefined
                                ? `Parcel ${feature.properties.parcel_index}`
                                : `Imported ${parcelCounter}`);
                        layer.isLoaded = true;
                        layer.originalProperties = { ...feature.properties };
                        layer.bindPopup(`<b>${layer.parcelName}</b>`);
                        addLayerClickHandler(layer);
                        drawnItems.addLayer(layer);
                        loadedCount++;

                        if (feature.properties?.layer) {
                            allLayers.add(feature.properties.layer);
                        }
                    }
                },
                style: styles.loaded
            });

            visibleLayers = new Set(allLayers);
            referenceLayer.clearLayers();
            originalReferenceData = null;
            this.resetOffset();
            updateLayerFilter();
            updateParcelList();
            updateStatus(`Imported ${loadedCount} parcels for editing.`);
        },

        clearReference() {
            referenceLayer.clearLayers();
            originalReferenceData = null;
            this.resetOffset();
            updateStatus('Reference data cleared');
        },

        clearAllParcels() {
            if (confirm('Delete all parcels?')) {
                deselectParcel();
                drawnItems.clearLayers();
                parcelCounter = 0;
                allLayers.clear();
                visibleLayers.clear();
                updateLayerFilter();
                updateParcelList();
                updateStatus('All parcels cleared');
            }
        },

        exportGeoJSON() {
            const features = collectFeatures(false);
            downloadGeoJSON(features, 'parcels_edited.geojson');
            updateStatus(`Exported ${features.length} parcels`);
        },

        exportFilteredGeoJSON() {
            const features = collectFeatures(true);
            const layerSuffix = visibleLayers.size > 0
                ? '_' + Array.from(visibleLayers).join('_').replace(/[^a-zA-Z0-9_]/g, '')
                : '';
            downloadGeoJSON(features, `parcels${layerSuffix}.geojson`);
            updateStatus(`Exported ${features.length} filtered parcels`);
        },

        // Exposed for HTML onclick handlers
        selectParcel,
        startRename,
        handleRenameKey,
        finishRename,
        deleteParcel,
        updateParcelList
    };
})();

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => ParcelEditor.init());
