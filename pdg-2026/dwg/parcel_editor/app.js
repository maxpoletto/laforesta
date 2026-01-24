// Parcel Editor - Layer-based editing
const ParcelEditor = (function() {
    'use strict';

    // State
    let map = null;
    let currentBasemap = null;
    let drawnItems = null;
    let drawControl = null;

    // Layer data: { name: { parcels: [...], offset: {ew, ns}, visible: true } }
    let layers = {};
    let selectedLayerName = null;
    let selectedParcel = null;
    let parcelCounter = 0;

    // Constants
    const basemaps = {
        osm: () => L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap'
        }),
        satellite: () => L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: '© Esri'
        }),
        topo: () => L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenTopoMap'
        })
    };

    const styles = {
        default: { color: '#3388ff', weight: 2, opacity: 1, fillOpacity: 0.2 },
        otherLayer: { color: '#ff6600', weight: 1, opacity: 1, fillOpacity: 0.1 },
        selected: { color: '#00ff00', weight: 3, opacity: 1, fillOpacity: 0.4 },
        hidden: { opacity: 0, fillOpacity: 0 }
    };

    const $ = id => document.getElementById(id);

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    // Coordinate offset utilities
    function offsetCoordinates(coords, offsetLon, offsetLat) {
        if (typeof coords[0] === 'number') {
            return [coords[0] + offsetLon, coords[1] + offsetLat];
        }
        return coords.map(c => offsetCoordinates(c, offsetLon, offsetLat));
    }

    function getOffsetDegrees(ew, ns) {
        return { lon: ew / 87000, lat: ns / 111000 };
    }

    // Layer management
    function getLayer(name) {
        return layers[name] || null;
    }

    function getSelectedLayer() {
        return selectedLayerName ? layers[selectedLayerName] : null;
    }

    function createLayer(name) {
        if (layers[name]) return false;
        layers[name] = { parcels: [], offset: { ew: 0, ns: 0 }, visible: true };
        return true;
    }

    function deleteLayer(name) {
        const layer = layers[name];
        if (!layer) return;

        // Remove parcels from map
        layer.parcels.forEach(p => {
            if (p.mapLayer) drawnItems.removeLayer(p.mapLayer);
        });

        delete layers[name];

        // Select another layer or none
        const remaining = Object.keys(layers);
        selectLayer(remaining.length > 0 ? remaining[0] : null);
        updateLayerSelector();
    }

    function selectLayer(name) {
        // Deselect current parcel
        if (selectedParcel) {
            deselectParcel();
        }

        selectedLayerName = name;

        // Update offset controls to show this layer's values
        const layer = getSelectedLayer();
        if (layer) {
            $('offset-ew').value = layer.offset.ew;
            $('offset-ns').value = layer.offset.ns;
            $('offset-ew-value').textContent = `${layer.offset.ew}m`;
            $('offset-ns-value').textContent = `${layer.offset.ns}m`;
            $('layer-visible').checked = layer.visible;
        }

        updateParcelStyles();
        updateParcelList();
        updateStatus(name ? `Compresa attiva: ${name}` : 'Nessuna compresa selezionata');
    }

    // Parcel management
    function addParcelToLayer(layerName, feature, mapLayer) {
        const layer = layers[layerName];
        if (!layer) return null;

        parcelCounter++;
        const parcel = {
            id: parcelCounter,
            name: feature.properties?.name || `Particella ${parcelCounter}`,
            feature: feature,
            mapLayer: mapLayer
        };

        // Store reference back to parcel data
        mapLayer.parcelData = parcel;
        mapLayer.layerName = layerName;

        layer.parcels.push(parcel);
        return parcel;
    }

    function deleteParcel(parcel) {
        const layer = layers[parcel.mapLayer.layerName];
        if (!layer) return;

        if (selectedParcel === parcel) {
            deselectParcel();
        }

        drawnItems.removeLayer(parcel.mapLayer);
        layer.parcels = layer.parcels.filter(p => p !== parcel);
        updateParcelList();
        updateStatus('Particella eliminata');
    }

    function selectParcel(parcel) {
        deselectParcel();
        selectedParcel = parcel;
        parcel.mapLayer.setStyle(styles.selected);
        parcel.mapLayer.editing.enable();
        updateParcelList();
        updateStatus(`Particella selezionata: ${parcel.name}`);
    }

    function deselectParcel() {
        if (selectedParcel) {
            selectedParcel.mapLayer.editing?.disable();
            updateParcelStyle(selectedParcel);
            selectedParcel = null;
            updateParcelList();
        }
    }

    function updateParcelStyle(parcel) {
        const layer = layers[parcel.mapLayer.layerName];
        if (!layer) return;

        if (!layer.visible) {
            parcel.mapLayer.setStyle(styles.hidden);
        } else if (parcel.mapLayer.layerName === selectedLayerName) {
            parcel.mapLayer.setStyle(styles.default);
        } else {
            parcel.mapLayer.setStyle(styles.otherLayer);
        }
    }

    function updateParcelStyles() {
        Object.values(layers).forEach(layer => {
            layer.parcels.forEach(parcel => {
                if (parcel !== selectedParcel) {
                    updateParcelStyle(parcel);
                }
            });
        });
    }

    // Apply offset to a layer's parcels on the map
    function applyLayerOffset(layerName) {
        const layer = layers[layerName];
        if (!layer) return;

        const { lon, lat } = getOffsetDegrees(layer.offset.ew, layer.offset.ns);

        layer.parcels.forEach(parcel => {
            // Get original coordinates from stored feature
            const origCoords = parcel.feature.geometry.coordinates;
            const offsetCoords = offsetCoordinates(origCoords, lon, lat);

            // Update map layer with offset coordinates
            const newLatLngs = L.GeoJSON.coordsToLatLngs(offsetCoords,
                parcel.feature.geometry.type === 'Polygon' ? 1 : 2);
            parcel.mapLayer.setLatLngs(newLatLngs);
        });
    }

    // UI updates
    function updateLayerSelector() {
        const selector = $('layer-selector');
        selector.innerHTML = '';

        const names = Object.keys(layers).sort();
        if (names.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '(nessuna compresa)';
            selector.appendChild(opt);
            $('layer-controls').style.display = 'none';
            return;
        }

        $('layer-controls').style.display = 'block';

        names.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === selectedLayerName) opt.selected = true;
            selector.appendChild(opt);
        });
    }

    function updateParcelList() {
        const list = $('parcel-list');
        list.innerHTML = '';

        const layer = getSelectedLayer();
        if (!layer) return;

        layer.parcels.forEach(parcel => {
            const isSelected = selectedParcel === parcel;
            const item = document.createElement('div');
            item.className = 'parcel-item' + (isSelected ? ' selected' : '');
            item.innerHTML = `
                <span class="parcel-name" onclick="ParcelEditor.onParcelClick(${parcel.id})">${parcel.name}</span>
                <span class="parcel-actions">
                    <span class="edit-btn" onclick="ParcelEditor.startRename(${parcel.id})" title="Rinomina">✎</span>
                    <span class="delete-btn" onclick="ParcelEditor.onDeleteParcel(${parcel.id})" title="Elimina">✕</span>
                </span>
            `;
            list.appendChild(item);
        });
    }

    function findParcelById(id) {
        for (const layer of Object.values(layers)) {
            const parcel = layer.parcels.find(p => p.id === id);
            if (parcel) return parcel;
        }
        return null;
    }

    // Load/Export
    function loadGeoJSON(data) {
        // Clear existing
        clearAll(false);

        // Group features by layer property
        const featuresByLayer = {};
        (data.features || []).forEach(feature => {
            const layerName = feature.properties?.layer || 'Default';
            if (!featuresByLayer[layerName]) {
                featuresByLayer[layerName] = [];
            }
            featuresByLayer[layerName].push(feature);
        });

        // Create layers and add parcels
        Object.entries(featuresByLayer).forEach(([layerName, features]) => {
            createLayer(layerName);

            features.forEach(feature => {
                if (feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon') {
                    const mapLayer = L.geoJSON(feature, { style: styles.otherLayer }).getLayers()[0];
                    if (mapLayer) {
                        drawnItems.addLayer(mapLayer);
                        addParcelToLayer(layerName, feature, mapLayer);
                        addParcelClickHandler(mapLayer);
                    }
                }
            });
        });

        // Select first layer
        const firstLayer = Object.keys(layers).sort()[0];
        updateLayerSelector();
        selectLayer(firstLayer);

        // Fit bounds
        if (drawnItems.getBounds().isValid()) {
            map.fitBounds(drawnItems.getBounds());
        } else {
            map.setView([38.65, 16.3], 12); // Default to center of Calabria
        }

        const parcelCount = Object.values(layers).reduce((sum, l) => sum + l.parcels.length, 0);
        updateStatus(`Caricate ${parcelCount} particelle in ${Object.keys(layers).length} comprese`);
    }

    function exportGeoJSON() {
        const features = [];

        Object.entries(layers).forEach(([layerName, layer]) => {
            const { lon, lat } = getOffsetDegrees(layer.offset.ew, layer.offset.ns);

            layer.parcels.forEach(parcel => {
                // Get current coordinates from map (includes any edits)
                const geojson = parcel.mapLayer.toGeoJSON();

                // Apply offset to get final coordinates
                // Note: mapLayer already has offset applied visually, but toGeoJSON
                // returns the current visual coordinates, so we're good

                geojson.properties = geojson.properties || {};
                geojson.properties.name = parcel.name;
                geojson.properties.layer = layerName;
                geojson.properties.id = parcel.id;
                features.push(geojson);
            });
        });

        const geojson = { type: 'FeatureCollection', features };
        const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'particelle.geojson';
        a.click();
        URL.revokeObjectURL(url);

        updateStatus(`Esportate ${features.length} particelle`);
    }

    function clearAll(confirm_needed = true) {
        if (confirm_needed && Object.keys(layers).length > 0) {
            if (!confirm('Cancellare tutte le comprese e particelle?')) return;
        }

        deselectParcel();
        drawnItems.clearLayers();
        layers = {};
        selectedLayerName = null;
        parcelCounter = 0;
        updateLayerSelector();
        updateParcelList();
        updateStatus('Tutte comprese cancellate');
    }

    function addParcelClickHandler(mapLayer) {
        mapLayer.on('click', function(e) {
            L.DomEvent.stopPropagation(e);
            const parcel = mapLayer.parcelData;
            if (parcel && mapLayer.layerName === selectedLayerName) {
                selectParcel(parcel);
            }
        });
    }

    // Rename functionality
    function startRename(parcelId) {
        const parcel = findParcelById(parcelId);
        if (!parcel) return;

        const items = $('parcel-list').querySelectorAll('.parcel-item');
        const layer = getSelectedLayer();
        const index = layer.parcels.indexOf(parcel);
        if (index < 0 || !items[index]) return;

        const item = items[index];
        item.innerHTML = `
            <input type="text" class="rename-input" value="${parcel.name}"
                   onkeydown="ParcelEditor.handleRenameKey(event, ${parcelId})" />
            <span class="parcel-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRename(${parcelId})" title="Salva">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateParcelList()" title="Annulla">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRenameKey(event, parcelId) {
        if (event.key === 'Enter') finishRename(parcelId);
        else if (event.key === 'Escape') updateParcelList();
    }

    function finishRename(parcelId) {
        const parcel = findParcelById(parcelId);
        if (!parcel) return;

        const input = $('parcel-list').querySelector('.rename-input');
        if (!input) return;

        parcel.name = input.value.trim() || 'Senza nome';
        parcel.mapLayer.bindPopup(`<b>${parcel.name}</b>`);
        updateParcelList();
        updateStatus(`Particella rinominata a "${parcel.name}"`);
    }

    // Public API
    return {
        init(filename = null) {
            map = L.map('map');
            currentBasemap = basemaps.satellite().addTo(map);

            drawnItems = new L.FeatureGroup().addTo(map);

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

            // New parcel drawn
            map.on(L.Draw.Event.CREATED, e => {
                if (!selectedLayerName) {
                    updateStatus('Seleziona o crea una compresa prima di disegnare una particella');
                    return;
                }

                const mapLayer = e.layer;
                const feature = mapLayer.toGeoJSON();

                // Store original feature (without offset)
                const layer = getSelectedLayer();
                const { lon, lat } = getOffsetDegrees(layer.offset.ew, layer.offset.ns);

                // Remove offset from stored coordinates (reverse the visual offset)
                feature.geometry.coordinates = offsetCoordinates(
                    feature.geometry.coordinates, -lon, -lat
                );

                drawnItems.addLayer(mapLayer);
                const parcel = addParcelToLayer(selectedLayerName, feature, mapLayer);
                addParcelClickHandler(mapLayer);
                mapLayer.bindPopup(`<b>${parcel.name}</b>`);

                selectParcel(parcel);
                updateParcelList();
                updateStatus(`Creata particella ${parcel.name}`);
            });

            // Click background to deselect
            map.on('click', e => {
                if (e.originalEvent.target === map._container ||
                    e.originalEvent.target.classList.contains('leaflet-tile')) {
                    deselectParcel();
                }
            });

            // Coordinates display
            map.on('mousemove', e => {
                $('coords').textContent = `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
            });

            // Layer selector
            $('layer-selector').addEventListener('change', e => {
                selectLayer(e.target.value);
            });

            // Visibility checkbox
            $('layer-visible').addEventListener('change', e => {
                const layer = getSelectedLayer();
                if (layer) {
                    layer.visible = e.target.checked;
                    updateParcelStyles();
                }
            });

            // Offset controls
            $('offset-ew').addEventListener('input', e => {
                const layer = getSelectedLayer();
                if (layer) {
                    layer.offset.ew = parseInt(e.target.value);
                    $('offset-ew-value').textContent = `${layer.offset.ew}m`;
                    applyLayerOffset(selectedLayerName);
                }
            });

            $('offset-ns').addEventListener('input', e => {
                const layer = getSelectedLayer();
                if (layer) {
                    layer.offset.ns = parseInt(e.target.value);
                    $('offset-ns-value').textContent = `${layer.offset.ns}m`;
                    applyLayerOffset(selectedLayerName);
                }
            });

            // File loader
            $('load-file').addEventListener('change', e => {
                const file = e.target.files[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = evt => {
                    try {
                        const data = JSON.parse(evt.target.result);
                        loadGeoJSON(data);
                    } catch (err) {
                        updateStatus('Errore nel caricamento del GeoJSON: ' + err.message);
                    }
                };
                reader.readAsText(file);
            });

            if (filename) {
                fetch(filename)
                    .then(r => r.ok ? r.json() : Promise.reject())
                    .then(data => loadGeoJSON(data))
                    .catch(() => {});
            }
            updateLayerSelector();
        },

        setBasemap(name) {
            map.removeLayer(currentBasemap);
            currentBasemap = basemaps[name]().addTo(map);
        },

        resetOffset() {
            const layer = getSelectedLayer();
            if (!layer) return;

            layer.offset.ew = 0;
            layer.offset.ns = 0;
            $('offset-ew').value = 0;
            $('offset-ns').value = 0;
            $('offset-ew-value').textContent = '0m';
            $('offset-ns-value').textContent = '0m';
            applyLayerOffset(selectedLayerName);
        },

        createNewLayer() {
            const name = prompt('Nome della compresa:');
            if (!name || !name.trim()) return;

            if (layers[name.trim()]) {
                updateStatus('La compresa esiste già');
                return;
            }

            createLayer(name.trim());
            updateLayerSelector();
            selectLayer(name.trim());
            updateStatus(`Creata compresa: ${name.trim()}`);
        },

        deleteCurrentLayer() {
            if (!selectedLayerName) return;
            if (!confirm(`Elimina la compresa "${selectedLayerName}" e tutte le sue particelle?`)) return;

            const name = selectedLayerName;
            deleteLayer(name);
            updateStatus(`Eliminata compresa: ${name}`);
        },

        exportGeoJSON,
        clearAll,
        updateParcelList,

        onParcelClick(id) {
            const parcel = findParcelById(id);
            if (parcel) selectParcel(parcel);
        },

        onDeleteParcel(id) {
            const parcel = findParcelById(id);
            if (parcel) deleteParcel(parcel);
        },

        startRename,
        handleRenameKey,
        finishRename
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelEditor.init("all_parcels.json"));
