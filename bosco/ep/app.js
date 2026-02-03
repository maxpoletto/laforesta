// Parcel Editor - Layer-based editing
const ParcelEditor = (function() {
    'use strict';

    const OFFSET_ROADS = false;

    // State
    let mapWrapper = null;
    let map = null;
    let drawnItems = null;
    let drawControl = null;

    // Layer data: { name: { parcels: [...], roads: [...], offset: {ew, ns}, oldOffset: {ew, ns}, visible: true } }
    let layers = {};
    let selectedLayerName = null;
    let selectedParcel = null;
    let selectedRoad = null;
    let parcelCounter = 0;
    let roadCounter = 0;

    // Undo state: saves GeoJSON when editing starts
    let undoState = null;  // { type: 'parcel'|'road', feature, geojson }

    const styles = {
        default: { color: '#3388ff', weight: 2, opacity: 1, fillOpacity: 0.2 },
        otherLayer: { color: '#ff6600', weight: 1, opacity: 1, fillOpacity: 0.1 },
        selected: { color: '#00ff00', weight: 3, opacity: 1, fillOpacity: 0.4 },
        hidden: { opacity: 0, fillOpacity: 0 },
        road: { color: '#cc3333', weight: 3, opacity: 0.8 },
        roadOtherLayer: { color: '#cc6666', weight: 2, opacity: 0.5 },
        roadSelected: { color: '#ffff00', weight: 4, opacity: 1 }
    };

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

    function sortParcels(layer) {
        layer.parcels.sort((a, b) => a.name.localeCompare(b.name));
    }

    function sortRoads(layer) {
        layer.roads.sort((a, b) => a.name.localeCompare(b.name));
    }

    function createLayer(name) {
        if (layers[name]) return false;
        layers[name] = {
            parcels: [],
            roads: [],
            offset: { ew: 0, ns: 0 },
            oldOffset: { ew: 0, ns: 0 },  // Track what's already applied to coordinates
            visible: true
        };
        return true;
    }

    function deleteLayer(name) {
        const layer = layers[name];
        if (!layer) return;

        // Remove parcels and roads from map
        layer.parcels.forEach(p => {
            if (p.mapLayer) drawnItems.removeLayer(p.mapLayer);
        });
        layer.roads.forEach(r => {
            if (r.mapLayer) drawnItems.removeLayer(r.mapLayer);
        });

        delete layers[name];

        // Select another layer or none
        const remaining = Object.keys(layers);
        selectLayer(remaining.length > 0 ? remaining[0] : null);
        updateLayerSelector();
    }

    function selectLayer(name) {
        // Deselect current parcel and road
        if (selectedParcel) {
            deselectParcel();
        }
        if (selectedRoad) {
            deselectRoad();
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
        updateRoadStyles();
        updateParcelList();
        updateRoadList();
        updateStatus(name ? `Compresa attiva: ${name}` : 'Nessuna compresa selezionata');
    }

    // Parcel management
    function addParcelToLayer(layerName, mapLayer) {
        const layer = layers[layerName];
        if (!layer) return null;

        parcelCounter++;
        const parcel = {
            id: parcelCounter,
            name: mapLayer.parcelName || `Particella ${parcelCounter}`,
            mapLayer: mapLayer  // Single source of truth for coordinates
        };

        // Store reference back to parcel data
        mapLayer.parcelData = parcel;
        mapLayer.layerName = layerName;

        layer.parcels.push(parcel);
        sortParcels(layer);
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
        if (selectedRoad) {
            deselectRoad();
        }
        selectedParcel = parcel;

        // Save state for undo
        undoState = {
            type: 'parcel',
            feature: parcel,
            geojson: parcel.mapLayer.toGeoJSON()
        };

        parcel.mapLayer.setStyle(styles.selected);
        parcel.mapLayer.editing.enable();
        updateParcelList();
        updateStatus(`Particella selezionata: ${parcel.name} (Ctrl+Z per annullare)`);
    }

    function deselectParcel() {
        if (selectedParcel) {
            selectedParcel.mapLayer.editing?.disable();
            updateParcelStyle(selectedParcel);
            selectedParcel = null;
            undoState = null;  // Clear undo state
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

    // Road management
    function addRoadToLayer(layerName, mapLayer) {
        const layer = layers[layerName];
        if (!layer) return null;

        roadCounter++;
        const road = {
            id: roadCounter,
            name: mapLayer.roadName || `Strada ${roadCounter}`,
            mapLayer: mapLayer
        };

        mapLayer.roadData = road;
        mapLayer.layerName = layerName;

        layer.roads.push(road);
        sortRoads(layer);
        return road;
    }

    function deleteRoad(road) {
        const layer = layers[road.mapLayer.layerName];
        if (!layer) return;

        if (selectedRoad === road) {
            deselectRoad();
        }

        drawnItems.removeLayer(road.mapLayer);
        layer.roads = layer.roads.filter(r => r !== road);
        updateRoadList();
        updateStatus('Strada eliminata');
    }

    function selectRoad(road) {
        deselectRoad();
        if (selectedParcel) {
            deselectParcel();
        }
        selectedRoad = road;

        // Save state for undo
        undoState = {
            type: 'road',
            feature: road,
            geojson: road.mapLayer.toGeoJSON()
        };

        road.mapLayer.setStyle(styles.roadSelected);
        road.mapLayer.editing.enable();
        updateRoadList();
        updateStatus(`Strada selezionata: ${road.name} (Ctrl+Z per annullare)`);
    }

    function deselectRoad() {
        if (selectedRoad) {
            selectedRoad.mapLayer.editing?.disable();
            updateRoadStyle(selectedRoad);
            selectedRoad = null;
            undoState = null;  // Clear undo state
            updateRoadList();
        }
    }

    function undoEdit() {
        if (!undoState) return;

        const { type, feature, geojson } = undoState;
        const layerName = feature.mapLayer.layerName;

        // Restore coordinates from saved GeoJSON
        const coords = geojson.geometry.coordinates;
        let newLatLngs, newLayer;

        if (type === 'parcel') {
            const depth = geojson.geometry.type === 'Polygon' ? 1 : 2;
            newLatLngs = L.GeoJSON.coordsToLatLngs(coords, depth);
            newLayer = L.polygon(newLatLngs, feature.mapLayer.options);
        } else {  // road
            newLatLngs = L.GeoJSON.coordsToLatLngs(coords, 0);
            newLayer = L.polyline(newLatLngs, feature.mapLayer.options);
        }

        // Replace the layer
        drawnItems.removeLayer(feature.mapLayer);
        feature.mapLayer = newLayer;
        drawnItems.addLayer(newLayer);

        // Restore metadata
        if (type === 'parcel') {
            newLayer.parcelData = feature;
            newLayer.layerName = layerName;
            addParcelClickHandler(newLayer);
            selectParcel(feature);  // Re-select with new undo state
        } else {
            newLayer.roadData = feature;
            newLayer.layerName = layerName;
            addRoadClickHandler(newLayer);
            selectRoad(feature);  // Re-select with new undo state
        }

        updateStatus('Modifiche annullate');
    }

    function updateRoadStyle(road) {
        const layer = layers[road.mapLayer.layerName];
        if (!layer) return;

        if (!layer.visible) {
            road.mapLayer.setStyle(styles.hidden);
        } else if (road.mapLayer.layerName === selectedLayerName) {
            road.mapLayer.setStyle(styles.road);
        } else {
            road.mapLayer.setStyle(styles.roadOtherLayer);
        }
    }

    function updateRoadStyles() {
        Object.values(layers).forEach(layer => {
            layer.roads.forEach(road => {
                if (road !== selectedRoad) {
                    updateRoadStyle(road);
                }
            });
        });
    }

    // Apply offset delta to a layer's parcels and roads
    function applyLayerOffset(layerName) {
        const layer = layers[layerName];
        if (!layer) return;

        const deltaEw = layer.offset.ew - layer.oldOffset.ew;
        const deltaNs = layer.offset.ns - layer.oldOffset.ns;
        if (deltaEw === 0 && deltaNs === 0) return;

        const { lon: deltaLon, lat: deltaLat } = getOffsetDegrees(deltaEw, deltaNs);
        const wasEditingParcel = selectedParcel?.mapLayer.layerName === layerName ? selectedParcel : null;
        const wasEditingRoad = selectedRoad?.mapLayer.layerName === layerName ? selectedRoad : null;

        // Apply to parcels
        layer.parcels.forEach(parcel => {
            const geom = parcel.mapLayer.toGeoJSON().geometry;
            const newCoords = offsetCoordinates(geom.coordinates, deltaLon, deltaLat);
            const depth = geom.type === 'Polygon' ? 1 : 2;
            const newLatLngs = L.GeoJSON.coordsToLatLngs(newCoords, depth);
            const style = parcel.mapLayer.options;

            drawnItems.removeLayer(parcel.mapLayer);
            const newLayer = L.polygon(newLatLngs, style);
            newLayer.parcelData = parcel;
            newLayer.layerName = layerName;
            parcel.mapLayer = newLayer;
            drawnItems.addLayer(newLayer);
            addParcelClickHandler(newLayer);
        });

        // Apply to roads
        if (OFFSET_ROADS) {
            layer.roads.forEach(road => {
                const geom = road.mapLayer.toGeoJSON().geometry;
                const newCoords = offsetCoordinates(geom.coordinates, deltaLon, deltaLat);
                const newLatLngs = L.GeoJSON.coordsToLatLngs(newCoords, 0);
                const style = road.mapLayer.options;

                drawnItems.removeLayer(road.mapLayer);
                const newLayer = L.polyline(newLatLngs, style);
                newLayer.roadData = road;
                newLayer.layerName = layerName;
                road.mapLayer = newLayer;
                drawnItems.addLayer(newLayer);
                addRoadClickHandler(newLayer);
            });
        }

        if (wasEditingParcel) {
            wasEditingParcel.mapLayer.setStyle(styles.selected);
            wasEditingParcel.mapLayer.editing.enable();
        }
        if (wasEditingRoad) {
            wasEditingRoad.mapLayer.setStyle(styles.roadSelected);
            wasEditingRoad.mapLayer.editing.enable();
        }

        layer.oldOffset = { ew: layer.offset.ew, ns: layer.offset.ns };
    }

    // UI updates
    function updateLayerSelector() {
        const selector = $('layer-selector');
        selector.innerHTML = '';

        const names = Object.keys(layers).sort((a, b) => b.localeCompare(a));
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
            item.className = 'list-item' + (isSelected ? ' selected' : '');
            item.innerHTML = `
                <span class="item-name" onclick="ParcelEditor.onParcelClick(${parcel.id})">${parcel.name}</span>
                <span class="item-actions">
                    <span class="edit-btn" onclick="ParcelEditor.startRename(${parcel.id})" title="Rinomina">✎</span>
                    <span class="delete-btn" onclick="ParcelEditor.onDeleteParcel(${parcel.id})" title="Elimina">✕</span>
                </span>
            `;
            list.appendChild(item);
        });
    }

    function updateRoadList() {
        const list = $('road-list');
        list.innerHTML = '';

        const layer = getSelectedLayer();
        if (!layer) return;

        layer.roads.forEach(road => {
            const isSelected = selectedRoad === road;
            const item = document.createElement('div');
            item.className = 'list-item' + (isSelected ? ' selected' : '');
            item.innerHTML = `
                <span class="item-name" onclick="ParcelEditor.onRoadClick(${road.id})">${road.name}</span>
                <span class="item-actions">
                    <span class="edit-btn" onclick="ParcelEditor.startRoadRename(${road.id})" title="Rinomina">✎</span>
                    <span class="delete-btn" onclick="ParcelEditor.onDeleteRoad(${road.id})" title="Elimina">✕</span>
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

    function findRoadById(id) {
        for (const layer of Object.values(layers)) {
            const road = layer.roads.find(r => r.id === id);
            if (road) return road;
        }
        return null;
    }

    // Load/Export
    function loadGeoJSON(data) {
        // Group features by layer property
        const featuresByLayer = {};
        (data.features || []).forEach(feature => {
            const layerName = feature.properties?.layer || 'Default';
            if (!featuresByLayer[layerName]) {
                featuresByLayer[layerName] = [];
            }
            featuresByLayer[layerName].push(feature);
        });

        let newParcels = 0;
        let newRoads = 0;
        const newLayers = [];

        // Merge features into existing layers or create new ones
        Object.entries(featuresByLayer).forEach(([layerName, features]) => {
            const isNewLayer = !layers[layerName];
            if (isNewLayer) {
                createLayer(layerName);
                newLayers.push(layerName);
            }

            features.forEach(feature => {
                if (feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon') {
                    // Handle polygons (parcels)
                    const depth = feature.geometry.type === 'Polygon' ? 1 : 2;
                    const latlngs = L.GeoJSON.coordsToLatLngs(feature.geometry.coordinates, depth);
                    const mapLayer = L.polygon(latlngs, styles.otherLayer);
                    mapLayer.parcelName = feature.properties?.name;
                    drawnItems.addLayer(mapLayer);
                    addParcelToLayer(layerName, mapLayer);
                    addParcelClickHandler(mapLayer);
                    newParcels++;
                } else if (feature.geometry?.type === 'LineString') {
                    // Handle LineStrings (roads)
                    const latlngs = L.GeoJSON.coordsToLatLngs(feature.geometry.coordinates, 0);
                    const mapLayer = L.polyline(latlngs, styles.roadOtherLayer);
                    mapLayer.roadName = feature.properties?.name;
                    drawnItems.addLayer(mapLayer);
                    addRoadToLayer(layerName, mapLayer);
                    addRoadClickHandler(mapLayer);
                    newRoads++;
                }
            });
        });

        // Update UI
        updateLayerSelector();

        // If this is the first load or we added new layers, select appropriately
        if (!selectedLayerName && Object.keys(layers).length > 0) {
            const firstLayer = Object.keys(layers).sort((a, b) => b.localeCompare(a))[0];
            selectLayer(firstLayer);
        } else if (selectedLayerName && layers[selectedLayerName]) {
            // Refresh current layer to show new items
            selectLayer(selectedLayerName);
        }

        // Fit bounds if this looks like an initial load
        if (Object.keys(layers).length === newLayers.length && drawnItems.getBounds().isValid()) {
            map.fitBounds(drawnItems.getBounds());
        }

        const totalParcels = Object.values(layers).reduce((sum, l) => sum + l.parcels.length, 0);
        const totalRoads = Object.values(layers).reduce((sum, l) => sum + l.roads.length, 0);
        const msg = newLayers.length > 0
            ? `Aggiunte ${newParcels} particelle e ${newRoads} strade (${newLayers.length} nuove comprese). Totale: ${totalParcels} particelle, ${totalRoads} strade in ${Object.keys(layers).length} comprese`
            : `Aggiunte ${newParcels} particelle e ${newRoads} strade alle comprese esistenti. Totale: ${totalParcels} particelle, ${totalRoads} strade`;
        updateStatus(msg);
    }

    function exportGeoJSON() {
        const features = [];

        Object.entries(layers).sort((a, b) => b[0].localeCompare(a[0])).forEach(([layerName, layer]) => {
            // Export parcels
            layer.parcels.forEach(parcel => {
                const geojson = parcel.mapLayer.toGeoJSON();
                geojson.properties = geojson.properties || {};
                geojson.properties.name = parcel.name;
                geojson.properties.layer = layerName;
                geojson.properties.id = parcel.id;
                geojson.properties.type = 'parcel';
                features.push(geojson);
            });

            // Export roads
            layer.roads.forEach(road => {
                const geojson = road.mapLayer.toGeoJSON();
                geojson.properties = geojson.properties || {};
                geojson.properties.name = road.name;
                geojson.properties.layer = layerName;
                geojson.properties.id = road.id;
                geojson.properties.type = 'viabilita';
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

        updateStatus(`Esportate ${features.length} feature (particelle e strade)`);
    }

    function clearAll(confirm_needed = true) {
        if (confirm_needed && Object.keys(layers).length > 0) {
            if (!confirm('Cancellare tutte le comprese e particelle?')) return;
        }

        deselectParcel();
        deselectRoad();
        drawnItems.clearLayers();
        layers = {};
        selectedLayerName = null;
        parcelCounter = 0;
        roadCounter = 0;
        updateLayerSelector();
        updateParcelList();
        updateRoadList();
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

    function addRoadClickHandler(mapLayer) {
        mapLayer.on('click', function(e) {
            L.DomEvent.stopPropagation(e);
            const road = mapLayer.roadData;
            if (road && mapLayer.layerName === selectedLayerName) {
                selectRoad(road);
            }
        });
    }

    // Rename functionality
    function startRename(parcelId) {
        const parcel = findParcelById(parcelId);
        if (!parcel) return;

        const items = $('parcel-list').querySelectorAll('.list-item');
        const layer = getSelectedLayer();
        const index = layer.parcels.indexOf(parcel);
        if (index < 0 || !items[index]) return;

        const item = items[index];
        item.innerHTML = `
            <input type="text" class="rename-input" value="${parcel.name}"
                   onkeydown="ParcelEditor.handleRenameKey(event, ${parcelId})" />
            <span class="item-actions">
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
        sortParcels(parcel.mapLayer);
        updateParcelList();
        updateStatus(`Particella rinominata a "${parcel.name}"`);
    }

    // Road rename functionality
    function startRoadRename(roadId) {
        const road = findRoadById(roadId);
        if (!road) return;

        const items = $('road-list').querySelectorAll('.list-item');
        const layer = getSelectedLayer();
        const index = layer.roads.indexOf(road);
        if (index < 0 || !items[index]) return;

        const item = items[index];
        item.innerHTML = `
            <input type="text" class="rename-input" value="${road.name}"
                   onkeydown="ParcelEditor.handleRoadRenameKey(event, ${roadId})" />
            <span class="item-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRoadRename(${roadId})" title="Salva">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateRoadList()" title="Annulla">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRoadRenameKey(event, roadId) {
        if (event.key === 'Enter') finishRoadRename(roadId);
        else if (event.key === 'Escape') updateRoadList();
    }

    function finishRoadRename(roadId) {
        const road = findRoadById(roadId);
        if (!road) return;

        const input = $('road-list').querySelector('.rename-input');
        if (!input) return;

        road.name = input.value.trim() || 'Senza nome';
        road.mapLayer.bindPopup(`<b>${road.name}</b>`);
        sortRoads(road.mapLayer);
        updateRoadList();
        updateStatus(`Strada rinominata a "${road.name}"`);
    }

    // Public API
    return {
        init(filename = null) {
            // Create map with shared features (measure, location, coords)
            mapWrapper = MapCommon.create('map', {
                basemap: 'satellite',
                enableMeasure: false,
            });
            map = mapWrapper.getLeafletMap();

            drawnItems = new L.FeatureGroup().addTo(map);

            drawControl = new L.Control.Draw({
                position: 'topleft',
                draw: {
                    polygon: { allowIntersection: false, shapeOptions: styles.default },
                    polyline: true, rectangle: false, circle: false,
                    marker: false, circlemarker: false
                },
                edit: false
            });
            map.addControl(drawControl);

            // New feature drawn
            map.on(L.Draw.Event.CREATED, e => {
                if (!selectedLayerName) {
                    updateStatus('Seleziona o crea una compresa prima di disegnare');
                    return;
                }

                const mapLayer = e.layer;
                drawnItems.addLayer(mapLayer);

                // Check if it's a polyline (road) or polygon (parcel)
                if (e.layerType === 'polyline') {
                    const road = addRoadToLayer(selectedLayerName, mapLayer);
                    addRoadClickHandler(mapLayer);
                    mapLayer.bindPopup(`<b>${road.name}</b>`);
                    selectRoad(road);
                    updateRoadList();
                    updateStatus(`Creata strada ${road.name}`);
                } else {
                    const parcel = addParcelToLayer(selectedLayerName, mapLayer);
                    addParcelClickHandler(mapLayer);
                    mapLayer.bindPopup(`<b>${parcel.name}</b>`);
                    selectParcel(parcel);
                    updateParcelList();
                    updateStatus(`Creata particella ${parcel.name}`);
                }
            });

            // Click map to deselect (feature clicks stop propagation)
            map.on('click', () => {
                deselectParcel();
                deselectRoad();
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', e => {
                if (e.key === 'Escape') {
                    deselectParcel();
                    deselectRoad();
                } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    undoEdit();
                }
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
            mapWrapper.setBasemap(name);
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
        updateRoadList,

        onParcelClick(id) {
            const parcel = findParcelById(id);
            if (parcel) selectParcel(parcel);
        },

        onDeleteParcel(id) {
            const parcel = findParcelById(id);
            if (parcel) deleteParcel(parcel);
        },

        onRoadClick(id) {
            const road = findRoadById(id);
            if (road) selectRoad(road);
        },

        onDeleteRoad(id) {
            const road = findRoadById(id);
            if (road) deleteRoad(road);
        },

        startRename,
        handleRenameKey,
        finishRename,

        startRoadRename,
        handleRoadRenameKey,
        finishRoadRename
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelEditor.init("particelle.geojson"));
