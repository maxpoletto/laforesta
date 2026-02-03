// Parcel Editor - Layer-based editing
const ParcelEditor = (function() {
    'use strict';

    const OFFSET_ROADS = true;

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

    // Tool state for polygon/line manipulation
    let toolState = {
        active: null,        // 'snip' | 'close' | 'complete' | null
        step: 0,             // Current workflow step
        sourceFeature: null, // Line or polygon being modified
        targetFeature: null, // Polygon providing boundary (for 'complete')
        vertices: [],        // Collected {latlng, index} objects
        vertexMarkers: []    // Visual feedback markers
    };

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
            name: mapLayer.parcelName || `Poligono ${parcelCounter}`,
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
            name: mapLayer.roadName || `Linea ${roadCounter}`,
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

    // Tool workflow management
    const toolSteps = {
        snip: [
            'Clicca un poligono da tagliare',
            'Clicca il primo estremo (Sv1)',
            'Clicca un vertice da eliminare (Sv2)',
            'Clicca il secondo estremo (Sv3)'
        ],
        close: [
            'Clicca una linea da chiudere',
            'Clicca il primo vertice (Lv1)',
            'Clicca il secondo vertice (Lv2)'
        ],
        complete: [
            'Clicca una linea da completare',
            'Clicca il primo vertice della linea (Lv1)',
            'Clicca il secondo vertice della linea (Lv2)',
            'Clicca un poligono di confine',
            'Clicca il primo vertice del confine (Pv1)',
            'Clicca i vertici intermedio (Pv2) e finale (Pv3)'
        ],
        split: [
            'Clicca una linea da dividere',
            'Clicca il vertice di divisione (verrà eliminato)'
        ],
        join: [
            'Clicca la prima linea',
            'Clicca il primo vertice (L1v1)',
            'Clicca il secondo vertice (L1v2)',
            'Clicca la seconda linea',
            'Clicca il primo vertice (L2v1)',
            'Clicca il secondo vertice (L2v2)'
        ]
    };

    function startTool(toolName) {
        if (!selectedLayerName) {
            updateStatus('Seleziona una compresa prima di usare gli strumenti');
            return;
        }

        // Cancel any existing tool
        cancelTool();

        // Deselect current selections
        deselectParcel();
        deselectRoad();

        toolState.active = toolName;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.targetFeature = null;
        toolState.vertices = [];

        updateToolStatus();
        $('tool-workflow').style.display = 'block';
        $('tool-active-name').textContent = {
            snip: 'Poligono → linea',
            close: 'Linea → poligono',
            complete: 'Linea+confine → poligono',
            split: 'Linea → 2 linee',
            join: '2 linee → linea'
        }[toolName];

        updateStatus(`Strumento attivo: ${toolState.active}`);
    }

    function cancelTool() {
        if (!toolState.active) return;

        // Remove vertex markers
        toolState.vertexMarkers.forEach(m => map.removeLayer(m));
        toolState.vertexMarkers = [];

        // Disable editing on features if enabled
        if (toolState.sourceFeature) {
            const mapLayer = toolState.sourceFeature.mapLayer;
            if (mapLayer.editing) mapLayer.editing.disable();
            updateFeatureStyle(toolState.sourceFeature);
        }
        if (toolState.targetFeature) {
            const mapLayer = toolState.targetFeature.mapLayer;
            if (mapLayer.editing) mapLayer.editing.disable();
            updateFeatureStyle(toolState.targetFeature);
        }

        toolState.active = null;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.targetFeature = null;
        toolState.vertices = [];

        $('tool-workflow').style.display = 'none';
        updateStatus('Strumento annullato');
    }

    function updateToolStatus() {
        if (!toolState.active) return;

        const steps = toolSteps[toolState.active];
        const stepText = steps[toolState.step] || 'Esecuzione...';
        $('tool-step-text').textContent = `Passo ${toolState.step + 1}/${steps.length}: ${stepText}`;
    }

    function updateFeatureStyle(feature) {
        if (feature.mapLayer.parcelData) {
            updateParcelStyle(feature);
        } else if (feature.mapLayer.roadData) {
            updateRoadStyle(feature);
        }
    }

    function advanceToolStep() {
        toolState.step++;
        updateToolStatus();

        const steps = toolSteps[toolState.active];
        if (toolState.step >= steps.length) {
            executeTool();
        }
    }

    function executeTool() {
        try {
            if (toolState.active === 'snip') {
                executeSnip();
            } else if (toolState.active === 'close') {
                executeClose();
            } else if (toolState.active === 'complete') {
                executeComplete();
            } else if (toolState.active === 'split') {
                executeSplit();
            } else if (toolState.active === 'join') {
                executeJoin();
            }
        } catch (err) {
            updateStatus('Errore: ' + err.message);
            console.error(err);
        }

        // Clean up
        toolState.vertexMarkers.forEach(m => map.removeLayer(m));
        toolState.vertexMarkers = [];
        toolState.active = null;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.targetFeature = null;
        toolState.vertices = [];
        $('tool-workflow').style.display = 'none';

        updateElementList();
    }

    // Get indices for a path from 'from' to 'to' around a polygon
    // direction: 1 = clockwise, -1 = counter-clockwise
    function getPathIndices(from, to, n, direction) {
        const indices = [from];
        let current = from;
        while (current !== to) {
            current = (current + direction + n) % n;
            indices.push(current);
        }
        return indices;
    }

    // Create a vertex highlight marker
    function createVertexMarker(latlng, label) {
        const marker = L.circleMarker(latlng, {
            radius: 10,
            color: '#ff00ff',
            fillColor: '#ff00ff',
            fillOpacity: 0.5,
            weight: 2
        });
        marker.bindTooltip(label, { permanent: true, direction: 'top', offset: [0, -10] });
        marker.addTo(map);
        toolState.vertexMarkers.push(marker);
        return marker;
    }

    // Handle feature click during tool workflow
    function handleToolFeatureClick(feature, isPolygon) {
        if (!toolState.active) return false;

        const tool = toolState.active;
        const step = toolState.step;

        // Step 0: Select source feature
        if (step === 0) {
            if (tool === 'snip' && isPolygon) {
                toolState.sourceFeature = feature;
                feature.mapLayer.setStyle(styles.selected);
                enableVertexClicks(feature.mapLayer, 'source');
                advanceToolStep();
                return true;
            } else if ((tool === 'close' || tool === 'complete' || tool === 'split' || tool === 'join') && !isPolygon) {
                toolState.sourceFeature = feature;
                feature.mapLayer.setStyle(styles.roadSelected);
                enableVertexClicks(feature.mapLayer, 'source');
                advanceToolStep();
                return true;
            }
        }

        // Step 3 for 'complete': Select target polygon
        if (tool === 'complete' && step === 3 && isPolygon) {
            toolState.targetFeature = feature;
            feature.mapLayer.setStyle(styles.selected);
            enableVertexClicks(feature.mapLayer, 'target');
            advanceToolStep();
            return true;
        }

        // Step 3 for 'join': Select second line
        if (tool === 'join' && step === 3 && !isPolygon) {
            toolState.targetFeature = feature;
            feature.mapLayer.setStyle(styles.roadSelected);
            enableVertexClicks(feature.mapLayer, 'target');
            advanceToolStep();
            return true;
        }

        return false;
    }

    // Create clickable vertex markers for tool selection (not using Leaflet edit mode)
    function enableVertexClicks(mapLayer, featureRole) {
        // Get vertices from the layer
        const latlngs = mapLayer.getLatLngs();

        // Flatten for polygons (which have nested arrays)
        let vertices;
        if (Array.isArray(latlngs[0]) && latlngs[0].length && latlngs[0][0].lat !== undefined) {
            // Polygon: [[latlng, latlng, ...]]
            vertices = latlngs[0];
        } else if (latlngs[0] && latlngs[0].lat !== undefined) {
            // LineString: [latlng, latlng, ...]
            vertices = latlngs;
        } else {
            vertices = latlngs.flat();
        }

        // Create clickable markers for each vertex
        vertices.forEach((latlng, idx) => {
            const marker = L.circleMarker(latlng, {
                radius: 8,
                color: '#0066ff',
                fillColor: '#0066ff',
                fillOpacity: 0.3,
                weight: 2,
                className: 'vertex-select-marker'
            });

            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                handleVertexClick(latlng, idx, featureRole);
            });

            marker.addTo(map);
            toolState.vertexMarkers.push(marker);
        });
    }

    // Handle vertex click during tool workflow
    function handleVertexClick(latlng, index, featureRole) {
        if (!toolState.active) return;

        const tool = toolState.active;
        const step = toolState.step;

        // Determine expected vertex based on tool and step
        let label = '';
        let shouldAdvance = true;

        if (tool === 'snip') {
            if (step === 1) label = 'Sv1';
            else if (step === 2) label = 'Sv2';
            else if (step === 3) label = 'Sv3';
        } else if (tool === 'close') {
            if (step === 1) label = 'Lv1';
            else if (step === 2) label = 'Lv2';
        } else if (tool === 'complete') {
            if (step === 1) label = 'Lv1';
            else if (step === 2) label = 'Lv2';
            else if (step === 4) label = 'Pv1';
            else if (step === 5) {
                // Pv2 and Pv3 on same step
                // At this point vertices = [lv1, lv2, pv1], length = 3
                if (toolState.vertices.length === 3) {
                    label = 'Pv2';
                    shouldAdvance = false;  // Wait for Pv3
                } else {
                    label = 'Pv3';
                }
            }
        } else if (tool === 'split') {
            if (step === 1) label = 'V';
        } else if (tool === 'join') {
            if (step === 1) label = 'L1v1';
            else if (step === 2) label = 'L1v2';
            else if (step === 4) label = 'L2v1';
            else if (step === 5) label = 'L2v2';
        }

        if (!label) return;

        toolState.vertices.push({ latlng, index });
        createVertexMarker(latlng, label);
        updateStatus(`Vertice ${label} selezionato`);

        if (shouldAdvance) {
            advanceToolStep();
        }
    }

    // Execute snip: polygon -> line
    function executeSnip() {
        const feature = toolState.sourceFeature;
        const [sv1, sv2, sv3] = toolState.vertices;
        const mapLayer = feature.mapLayer;
        const layerName = mapLayer.layerName;
        const layer = layers[layerName];

        // Get polygon coordinates (outer ring only)
        const geom = mapLayer.toGeoJSON().geometry;
        const coords = geom.coordinates[0].slice(0, -1);  // Remove closing point
        const n = coords.length;

        // Edge case: all three vertices are the same -> delete that vertex
        if (sv1.index === sv2.index && sv2.index === sv3.index) {
            const newCoords = coords.filter((_, i) => i !== sv1.index);
            if (newCoords.length < 2) {
                throw new Error('Troppo pochi vertici rimanenti');
            }

            // Convert to line
            const latlngs = L.GeoJSON.coordsToLatLngs(newCoords, 0);
            const newLine = L.polyline(latlngs, styles.road);
            newLine.roadName = feature.name;

            // Remove old polygon, add new line
            drawnItems.removeLayer(mapLayer);
            layer.parcels = layer.parcels.filter(p => p !== feature);
            drawnItems.addLayer(newLine);
            addRoadToLayer(layerName, newLine);
            addRoadClickHandler(newLine);

            updateStatus(`Vertice eliminato, poligono convertito in linea`);
            return;
        }

        // Sv1 and Sv3 are the endpoints, Sv2 indicates which path to REMOVE
        // Keep the path that does NOT include Sv2
        // Path 1: sv1 -> sv3 clockwise
        // Path 2: sv1 -> sv3 counter-clockwise
        const pathCW = getPathIndices(sv1.index, sv3.index, n, 1);
        const pathCCW = getPathIndices(sv1.index, sv3.index, n, -1);

        let keepPath;
        if (pathCW.includes(sv2.index)) {
            // Sv2 is on clockwise path, so keep counter-clockwise
            keepPath = pathCCW;
        } else if (pathCCW.includes(sv2.index)) {
            // Sv2 is on counter-clockwise path, so keep clockwise
            keepPath = pathCW;
        } else {
            throw new Error('Sv2 non trovato su nessun percorso');
        }

        // Extract coordinates for the kept path
        const lineCoords = keepPath.map(i => coords[i]);

        // Create line
        const latlngs = L.GeoJSON.coordsToLatLngs(lineCoords, 0);
        const newLine = L.polyline(latlngs, styles.road);
        newLine.roadName = feature.name;

        // Remove old polygon, add new line
        drawnItems.removeLayer(mapLayer);
        layer.parcels = layer.parcels.filter(p => p !== feature);
        drawnItems.addLayer(newLine);
        addRoadToLayer(layerName, newLine);
        addRoadClickHandler(newLine);

        updateStatus(`Poligono tagliato in linea con ${lineCoords.length} vertici`);
    }

    // Execute close: line -> polygon
    function executeClose() {
        const feature = toolState.sourceFeature;
        const [lv1, lv2] = toolState.vertices;
        const mapLayer = feature.mapLayer;
        const layerName = mapLayer.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const geom = mapLayer.toGeoJSON().geometry;
        const coords = geom.coordinates;

        // Ensure lv1 comes before lv2
        let idx1 = lv1.index;
        let idx2 = lv2.index;
        if (idx1 > idx2) {
            [idx1, idx2] = [idx2, idx1];
        }

        // Extract segment to close as polygon
        const polygonCoords = coords.slice(idx1, idx2 + 1);
        if (polygonCoords.length < 3) {
            throw new Error('Troppo pochi vertici per creare un poligono');
        }

        // Create polygon (close the ring)
        const closedCoords = [...polygonCoords, polygonCoords[0]];
        const polyLatLngs = L.GeoJSON.coordsToLatLngs([closedCoords], 1);
        const newPolygon = L.polygon(polyLatLngs, styles.default);
        newPolygon.parcelName = feature.name;

        // Remove old line
        drawnItems.removeLayer(mapLayer);
        layer.roads = layer.roads.filter(r => r !== feature);

        // Add new polygon
        drawnItems.addLayer(newPolygon);
        addParcelToLayer(layerName, newPolygon);
        addParcelClickHandler(newPolygon);

        // Create leftover lines if any
        if (idx1 > 0) {
            const leftCoords = coords.slice(0, idx1 + 1);
            if (leftCoords.length >= 2) {
                const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.road);
                leftLine.roadName = feature.name + ' (rimanente 1)';
                drawnItems.addLayer(leftLine);
                addRoadToLayer(layerName, leftLine);
                addRoadClickHandler(leftLine);
            }
        }

        if (idx2 < coords.length - 1) {
            const rightCoords = coords.slice(idx2);
            if (rightCoords.length >= 2) {
                const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.road);
                rightLine.roadName = feature.name + ' (rimanente 2)';
                drawnItems.addLayer(rightLine);
                addRoadToLayer(layerName, rightLine);
                addRoadClickHandler(rightLine);
            }
        }

        updateStatus(`Linea chiusa in poligono con ${polygonCoords.length} vertici`);
    }

    // Execute complete: line + polygon boundary -> polygon
    function executeComplete() {
        const lineFeature = toolState.sourceFeature;
        const polyFeature = toolState.targetFeature;
        const [lv1, lv2, pv1, pv2, pv3] = toolState.vertices;
        const lineLayer = lineFeature.mapLayer;
        const polyLayer = polyFeature.mapLayer;
        const layerName = lineLayer.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const lineGeom = lineLayer.toGeoJSON().geometry;
        const lineCoords = lineGeom.coordinates;

        // Ensure lv1 comes before lv2
        let idx1 = lv1.index;
        let idx2 = lv2.index;
        if (idx1 > idx2) {
            [idx1, idx2] = [idx2, idx1];
        }

        // Get polygon coordinates (outer ring, without closing point)
        const polyGeom = polyLayer.toGeoJSON().geometry;
        const polyCoords = polyGeom.coordinates[0].slice(0, -1);
        const n = polyCoords.length;

        // Determine path from pv1 to pv3 through pv2
        const pathCW = getPathIndices(pv1.index, pv3.index, n, 1);
        const pathCCW = getPathIndices(pv1.index, pv3.index, n, -1);

        let boundaryPath;
        if (pathCW.includes(pv2.index)) {
            boundaryPath = pathCW;
        } else if (pathCCW.includes(pv2.index)) {
            boundaryPath = pathCCW;
        } else {
            throw new Error('Pv2 non trovato su nessun percorso');
        }

        // Extract boundary coordinates: [Pv1, ..., Pv2, ..., Pv3]
        const boundaryCoords = boundaryPath.map(i => polyCoords[i]);

        // Build new polygon: line segment + reversed boundary
        // Line from idx1 to idx2: [Lv1, ..., Lv2]
        const lineSegment = lineCoords.slice(idx1, idx2 + 1);

        // Boundary needs to be reversed so Pv3 connects to Lv2, and Pv1 connects back to Lv1
        // Reversed: [Pv3, ..., Pv2, ..., Pv1]
        const reversedBoundary = boundaryCoords.slice().reverse();
        const newPolygonCoords = [...lineSegment, ...reversedBoundary];

        if (newPolygonCoords.length < 3) {
            throw new Error('Troppo pochi vertici per creare un poligono');
        }

        // Close the ring
        const closedCoords = [...newPolygonCoords, newPolygonCoords[0]];
        const polyLatLngs = L.GeoJSON.coordsToLatLngs([closedCoords], 1);
        const newPolygon = L.polygon(polyLatLngs, styles.default);
        newPolygon.parcelName = lineFeature.name;

        // Remove old line
        drawnItems.removeLayer(lineLayer);
        layer.roads = layer.roads.filter(r => r !== lineFeature);

        // Disable editing on target polygon
        polyLayer.editing.disable();
        updateParcelStyle(polyFeature);

        // Add new polygon
        drawnItems.addLayer(newPolygon);
        addParcelToLayer(layerName, newPolygon);
        addParcelClickHandler(newPolygon);

        // Create leftover lines if any
        if (idx1 > 0) {
            const leftCoords = lineCoords.slice(0, idx1 + 1);
            if (leftCoords.length >= 2) {
                const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.road);
                leftLine.roadName = lineFeature.name + ' (rimanente 1)';
                drawnItems.addLayer(leftLine);
                addRoadToLayer(layerName, leftLine);
                addRoadClickHandler(leftLine);
            }
        }

        if (idx2 < lineCoords.length - 1) {
            const rightCoords = lineCoords.slice(idx2);
            if (rightCoords.length >= 2) {
                const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.road);
                rightLine.roadName = lineFeature.name + ' (rimanente 2)';
                drawnItems.addLayer(rightLine);
                addRoadToLayer(layerName, rightLine);
                addRoadClickHandler(rightLine);
            }
        }

        updateStatus(`Linea completata in poligono con ${newPolygonCoords.length} vertici`);
    }

    // Execute split: line -> 2 lines
    function executeSplit() {
        const feature = toolState.sourceFeature;
        const [splitVertex] = toolState.vertices;
        const mapLayer = feature.mapLayer;
        const layerName = mapLayer.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const geom = mapLayer.toGeoJSON().geometry;
        const coords = geom.coordinates;
        const idx = splitVertex.index;

        // Can't split at endpoints
        if (idx === 0 || idx === coords.length - 1) {
            throw new Error('Non puoi dividere agli estremi della linea');
        }

        // Create two lines, excluding the split vertex
        const leftCoords = coords.slice(0, idx);
        const rightCoords = coords.slice(idx + 1);

        if (leftCoords.length < 2 || rightCoords.length < 2) {
            throw new Error('La divisione creerebbe linee troppo corte');
        }

        // Remove old line
        drawnItems.removeLayer(mapLayer);
        layer.roads = layer.roads.filter(r => r !== feature);

        // Create left line
        const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.road);
        leftLine.roadName = feature.name + ' (1)';
        drawnItems.addLayer(leftLine);
        addRoadToLayer(layerName, leftLine);
        addRoadClickHandler(leftLine);

        // Create right line
        const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.road);
        rightLine.roadName = feature.name + ' (2)';
        drawnItems.addLayer(rightLine);
        addRoadToLayer(layerName, rightLine);
        addRoadClickHandler(rightLine);

        updateStatus(`Linea divisa in 2 linee`);
    }

    // Execute join: 2 lines -> line
    function executeJoin() {
        const line1Feature = toolState.sourceFeature;
        const line2Feature = toolState.targetFeature;
        const [l1v1, l1v2, l2v1, l2v2] = toolState.vertices;
        const line1Layer = line1Feature.mapLayer;
        const line2Layer = line2Feature.mapLayer;
        const layerName = line1Layer.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const coords1 = line1Layer.toGeoJSON().geometry.coordinates;
        const coords2 = line2Layer.toGeoJSON().geometry.coordinates;

        // Extract segments respecting user's pick order (L1v1→L1v2→L2v1→L2v2)
        let segment1, segment2;
        let idx1_min, idx1_max, idx2_min, idx2_max;

        if (l1v1.index <= l1v2.index) {
            segment1 = coords1.slice(l1v1.index, l1v2.index + 1);
            idx1_min = l1v1.index;
            idx1_max = l1v2.index;
        } else {
            segment1 = coords1.slice(l1v2.index, l1v1.index + 1).reverse();
            idx1_min = l1v2.index;
            idx1_max = l1v1.index;
        }

        if (l2v1.index <= l2v2.index) {
            segment2 = coords2.slice(l2v1.index, l2v2.index + 1);
            idx2_min = l2v1.index;
            idx2_max = l2v2.index;
        } else {
            segment2 = coords2.slice(l2v2.index, l2v1.index + 1).reverse();
            idx2_min = l2v2.index;
            idx2_max = l2v1.index;
        }

        if (segment1.length < 1 || segment2.length < 1) {
            throw new Error('Segmenti troppo corti');
        }

        // Join: L1v1..L1v2..L2v1..L2v2
        const joinedCoords = [...segment1, ...segment2];

        if (joinedCoords.length < 2) {
            throw new Error('La linea risultante è troppo corta');
        }

        // Remove old lines
        drawnItems.removeLayer(line1Layer);
        drawnItems.removeLayer(line2Layer);
        layer.roads = layer.roads.filter(r => r !== line1Feature && r !== line2Feature);

        // Create joined line
        const joinedLine = L.polyline(L.GeoJSON.coordsToLatLngs(joinedCoords, 0), styles.road);
        joinedLine.roadName = line1Feature.name;
        drawnItems.addLayer(joinedLine);
        addRoadToLayer(layerName, joinedLine);
        addRoadClickHandler(joinedLine);

        // Create leftover lines from line 1
        if (idx1_min > 0) {
            const leftCoords = coords1.slice(0, idx1_min + 1);
            if (leftCoords.length >= 2) {
                const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.road);
                leftLine.roadName = line1Feature.name + ' (rimanente)';
                drawnItems.addLayer(leftLine);
                addRoadToLayer(layerName, leftLine);
                addRoadClickHandler(leftLine);
            }
        }
        if (idx1_max < coords1.length - 1) {
            const rightCoords = coords1.slice(idx1_max);
            if (rightCoords.length >= 2) {
                const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.road);
                rightLine.roadName = line1Feature.name + ' (rimanente)';
                drawnItems.addLayer(rightLine);
                addRoadToLayer(layerName, rightLine);
                addRoadClickHandler(rightLine);
            }
        }

        // Create leftover lines from line 2
        if (idx2_min > 0) {
            const leftCoords = coords2.slice(0, idx2_min + 1);
            if (leftCoords.length >= 2) {
                const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.road);
                leftLine.roadName = line2Feature.name + ' (rimanente)';
                drawnItems.addLayer(leftLine);
                addRoadToLayer(layerName, leftLine);
                addRoadClickHandler(leftLine);
            }
        }
        if (idx2_max < coords2.length - 1) {
            const rightCoords = coords2.slice(idx2_max);
            if (rightCoords.length >= 2) {
                const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.road);
                rightLine.roadName = line2Feature.name + ' (rimanente)';
                drawnItems.addLayer(rightLine);
                addRoadToLayer(layerName, rightLine);
                addRoadClickHandler(rightLine);
            }
        }

        updateStatus(`2 linee unite in una linea con ${joinedCoords.length} vertici`);
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
        // For compatibility, just update the element list
        updateElementList();
    }

    function updateRoadList() {
        // For compatibility, just update the element list
        updateElementList();
    }

    function updateElementList() {
        const list = $('element-list');
        if (!list) return;
        list.innerHTML = '';
        list.className = 'feature-list';

        const layer = getSelectedLayer();
        if (!layer) return;

        // Combine parcels and roads, sort by name
        const elements = [
            ...layer.parcels.map(p => ({ type: 'parcel', item: p })),
            ...layer.roads.map(r => ({ type: 'road', item: r }))
        ];
        elements.sort((a, b) => a.item.name.localeCompare(b.item.name));

        elements.forEach(({ type, item }) => {
            const isSelected = (type === 'parcel' && selectedParcel === item) ||
                              (type === 'road' && selectedRoad === item);
            const div = document.createElement('div');
            div.className = 'list-item' + (isSelected ? ' selected' : '');

            const typeIcon = type === 'parcel' ? '▢' : '─';
            const clickFn = type === 'parcel' ? 'onParcelClick' : 'onRoadClick';
            const renameFn = type === 'parcel' ? 'startRename' : 'startRoadRename';
            const deleteFn = type === 'parcel' ? 'onDeleteParcel' : 'onDeleteRoad';

            div.innerHTML = `
                <span class="item-type-icon">${typeIcon}</span>
                <span class="item-name" onclick="ParcelEditor.${clickFn}(${item.id})">${item.name}</span>
                <span class="item-actions">
                    <span class="edit-btn" onclick="ParcelEditor.${renameFn}(${item.id})" title="Rinomina">✎</span>
                    <span class="delete-btn" onclick="ParcelEditor.${deleteFn}(${item.id})" title="Elimina">✕</span>
                </span>
            `;
            list.appendChild(div);
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
            if (!parcel || mapLayer.layerName !== selectedLayerName) return;

            // Check if tool workflow is active
            if (toolState.active) {
                if (!handleToolFeatureClick(parcel, true)) {
                    updateStatus('Questa feature non è valida per questo passo');
                }
                return;
            }

            selectParcel(parcel);
        });
    }

    function addRoadClickHandler(mapLayer) {
        mapLayer.on('click', function(e) {
            L.DomEvent.stopPropagation(e);
            const road = mapLayer.roadData;
            if (!road || mapLayer.layerName !== selectedLayerName) return;

            // Check if tool workflow is active
            if (toolState.active) {
                if (!handleToolFeatureClick(road, false)) {
                    updateStatus('Questa feature non è valida per questo passo');
                }
                return;
            }

            selectRoad(road);
        });
    }

    // Rename functionality - find item in unified element list
    function findElementListItem(id, type) {
        const list = $('element-list');
        if (!list) return null;

        const items = list.querySelectorAll('.list-item');
        for (const item of items) {
            const nameEl = item.querySelector('.item-name');
            if (!nameEl) continue;

            const onclick = nameEl.getAttribute('onclick') || '';
            const fn = type === 'parcel' ? 'onParcelClick' : 'onRoadClick';
            if (onclick.includes(`${fn}(${id})`)) {
                return item;
            }
        }
        return null;
    }

    function startRename(parcelId) {
        const parcel = findParcelById(parcelId);
        if (!parcel) return;

        const item = findElementListItem(parcelId, 'parcel');
        if (!item) return;

        item.innerHTML = `
            <span class="item-type-icon">▢</span>
            <input type="text" class="rename-input" value="${parcel.name}"
                   onkeydown="ParcelEditor.handleRenameKey(event, ${parcelId})" />
            <span class="item-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRename(${parcelId})" title="Salva">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateElementList()" title="Annulla">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRenameKey(event, parcelId) {
        if (event.key === 'Enter') finishRename(parcelId);
        else if (event.key === 'Escape') updateElementList();
    }

    function finishRename(parcelId) {
        const parcel = findParcelById(parcelId);
        if (!parcel) return;

        const input = $('element-list').querySelector('.rename-input');
        if (!input) return;

        parcel.name = input.value.trim() || 'Senza nome';
        parcel.mapLayer.bindPopup(`<b>${parcel.name}</b>`);
        const layer = layers[parcel.mapLayer.layerName];
        if (layer) sortParcels(layer);
        updateElementList();
        updateStatus(`Particella rinominata a "${parcel.name}"`);
    }

    // Road rename functionality
    function startRoadRename(roadId) {
        const road = findRoadById(roadId);
        if (!road) return;

        const item = findElementListItem(roadId, 'road');
        if (!item) return;

        item.innerHTML = `
            <span class="item-type-icon">─</span>
            <input type="text" class="rename-input" value="${road.name}"
                   onkeydown="ParcelEditor.handleRoadRenameKey(event, ${roadId})" />
            <span class="item-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRoadRename(${roadId})" title="Salva">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateElementList()" title="Annulla">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRoadRenameKey(event, roadId) {
        if (event.key === 'Enter') finishRoadRename(roadId);
        else if (event.key === 'Escape') updateElementList();
    }

    function finishRoadRename(roadId) {
        const road = findRoadById(roadId);
        if (!road) return;

        const input = $('element-list').querySelector('.rename-input');
        if (!input) return;

        road.name = input.value.trim() || 'Senza nome';
        road.mapLayer.bindPopup(`<b>${road.name}</b>`);
        const layer = layers[road.mapLayer.layerName];
        if (layer) sortRoads(layer);
        updateElementList();
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
                    if (toolState.active) {
                        cancelTool();
                    } else {
                        deselectParcel();
                        deselectRoad();
                    }
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
        finishRoadRename,

        // Tool functions
        startTool,
        cancelTool,
        updateElementList
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelEditor.init("particelle.geojson"));
