// Element Editor - Layer-based editing with unified polygon/line handling
const ParcelEditor = (function() {
    'use strict';

    const OFFSET_LINES = true;

    let layersVisible = true;

    // State
    let mapWrapper = null;
    let map = null;
    let drawnItems = null;
    let drawControl = null;

    // Layer data: { name: { elements: [...], offset: {ew, ns}, oldOffset: {ew, ns}, visible: true } }
    let layers = {};
    let selectedLayerName = null;
    let selectedElement = null;
    let elementCounter = 0;

    // Undo state: saves GeoJSON when editing starts
    let undoState = null;  // { element, geojson }

    // Tool state for polygon/line manipulation
    let toolState = {
        active: null,        // 'snip' | 'close' | 'split' | 'join' | 'newpoly' | 'newline' | null
        step: 0,             // Current workflow step
        sourceFeature: null, // Line or polygon being modified
        targetFeature: null, // Second line (for 'join')
        vertices: [],        // Collected {latlng, index} objects
        vertexMarkers: [],   // Visual feedback markers
        // For newpoly/newline tools:
        accumulatedCoords: [],   // [lng, lat] coordinates for new shape
        previewLayer: null,      // Leaflet polyline showing accumulated path
        closeMarker: null,       // Special marker on first point to close (newpoly only)
        currentObjectType: null, // 'line' or 'polygon' for current selection
        segmentCount: 0          // Number of segments added
    };

    const styles = {
        default: { color: '#3388ff', weight: 2, opacity: 1, fillOpacity: 0.2 },
        otherLayer: { color: '#ff6600', weight: 1, opacity: 1, fillOpacity: 0.1 },
        selected: { color: '#00ff00', weight: 3, opacity: 1, fillOpacity: 0.4 },
        hidden: { opacity: 0, fillOpacity: 0 },
        line: { color: '#cc3333', weight: 3, opacity: 0.8 },
        lineOtherLayer: { color: '#cc6666', weight: 2, opacity: 0.5 },
        lineSelected: { color: '#ffff00', weight: 4, opacity: 1 }
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

    function sortElements(layer) {
        layer.elements.sort((a, b) => a.name.localeCompare(b.name));
    }

    function createLayer(name) {
        if (layers[name]) return false;
        layers[name] = {
            elements: [],
            offset: { ew: 0, ns: 0 },
            oldOffset: { ew: 0, ns: 0 },  // Track what's already applied to coordinates
            visible: true
        };
        return true;
    }

    function deleteLayer(name) {
        const layer = layers[name];
        if (!layer) return;

        // Remove elements from map
        layer.elements.forEach(el => {
            drawnItems.removeLayer(el);
        });

        delete layers[name];

        // Select another layer or none
        const remaining = Object.keys(layers).sort();
        selectLayer(remaining.length > 0 ? remaining[0] : null);
    }

    function selectLayer(name) {
        // Deselect current element
        if (selectedElement) {
            deselectElement();
        }

        selectedLayerName = name;

        // Update offset controls to show this layer's values
        const layer = getSelectedLayer();
        if (layer) {
            $('offset-ew').value = layer.offset.ew;
            $('offset-ns').value = layer.offset.ns;
            $('offset-ew-value').textContent = `${layer.offset.ew}m`;
            $('offset-ns-value').textContent = `${layer.offset.ns}m`;
        }

        // Show/hide sections based on whether a layer is selected
        const elementsSection = $('elements-section');
        const offsetSection = $('offset-section');
        if (elementsSection) elementsSection.style.display = name ? 'block' : 'none';
        if (offsetSection) offsetSection.style.display = name ? 'block' : 'none';

        updateElementStyles();
        updateLayerList();
        updateElementList();
        updateStatus(name ? `Strato attivo: ${name}` : 'Nessuno strato selezionato');
    }

    // Element management (unified for polygons and lines)
    function addElementToLayer(layerName, mapLayer, type) {
        const layer = layers[layerName];
        if (!layer) return null;

        elementCounter++;
        const defaultName = type === 'line'
            ? (mapLayer.elementName || `Linea ${elementCounter}`)
            : (mapLayer.elementName || `Poligono ${elementCounter}`);

        // Store properties directly on the Leaflet layer
        mapLayer.id = elementCounter;
        mapLayer.name = defaultName;
        mapLayer.type = type;  // 'polygon' or 'line'
        mapLayer.layerName = layerName;
        mapLayer.visible = true;

        layer.elements.push(mapLayer);
        sortElements(layer);
        return mapLayer;
    }

    function deleteElement(element, confirmed = false) {
        if (!confirmed && !confirm(`Elimina "${element.name}"?`)) return;

        const layer = layers[element.layerName];
        if (!layer) return;

        if (selectedElement === element) {
            deselectElement();
        }

        drawnItems.removeLayer(element);
        layer.elements = layer.elements.filter(el => el !== element);
        updateElementList();
        updateStatus('Elemento eliminato');
    }

    function selectElement(element) {
        deselectElement();
        selectedElement = element;

        // Save state for undo
        undoState = {
            element: element,
            geojson: element.toGeoJSON()
        };

        const style = element.type === 'line' ? styles.lineSelected : styles.selected;
        element.setStyle(style);
        element.editing.enable();
        updateElementList();

        const typeLabel = element.type === 'line' ? 'Linea' : 'Poligono';
        updateStatus(`${typeLabel} selezionato: ${element.name} (Ctrl+Z per annullare)`);
    }

    function deselectElement() {
        if (selectedElement) {
            selectedElement.editing?.disable();
            updateElementStyle(selectedElement);
            selectedElement = null;
            undoState = null;  // Clear undo state
            updateElementList();
        }
    }

    function updateElementStyle(element) {
        const layer = layers[element.layerName];
        if (!layer) return;

        if (!layer.visible || element.visible === false) {
            element.setStyle(styles.hidden);
        } else if (element.layerName === selectedLayerName) {
            element.setStyle(element.type === 'line' ? styles.line : styles.default);
        } else {
            element.setStyle(element.type === 'line' ? styles.lineOtherLayer : styles.otherLayer);
        }
    }

    function updateElementStyles() {
        Object.values(layers).forEach(layer => {
            layer.elements.forEach(element => {
                if (element !== selectedElement) {
                    updateElementStyle(element);
                }
            });
        });
    }

    function undoEdit() {
        if (!undoState) return;

        const { element, geojson } = undoState;
        const layerName = element.layerName;
        const layer = layers[layerName];
        if (!layer) return;

        // Restore coordinates from saved GeoJSON
        const coords = geojson.geometry.coordinates;
        let newLatLngs, newLayer;

        if (element.type === 'polygon') {
            const depth = geojson.geometry.type === 'Polygon' ? 1 : 2;
            newLatLngs = L.GeoJSON.coordsToLatLngs(coords, depth);
            newLayer = L.polygon(newLatLngs, element.options);
        } else {  // line
            newLatLngs = L.GeoJSON.coordsToLatLngs(coords, 0);
            newLayer = L.polyline(newLatLngs, element.options);
        }

        // Copy properties to new layer
        newLayer.id = element.id;
        newLayer.name = element.name;
        newLayer.type = element.type;
        newLayer.layerName = layerName;
        newLayer.visible = element.visible;

        // Replace the layer in the elements array
        const idx = layer.elements.indexOf(element);
        if (idx !== -1) {
            layer.elements[idx] = newLayer;
        }

        // Replace on map
        drawnItems.removeLayer(element);
        drawnItems.addLayer(newLayer);

        // Set up click handler and re-select
        addElementClickHandler(newLayer);
        selectElement(newLayer);  // Re-select with new undo state

        updateStatus('Modifiche annullate');
    }

    // Tool workflow management
    const toolSteps = {
        snip: [
            'Clicca un poligono da aprire',
            'Clicca il vertice da eliminare'
        ],
        close: [
            'Clicca una linea da chiudere',
            'Clicca il primo vertice (Lv1)',
            'Clicca il secondo vertice (Lv2)'
        ],
        split: [
            'Clicca una linea da dividere',
            'Clicca il vertice di divisione (verrà eliminato)'
        ],
        join: [
            'Clicca la prima linea',
            'Clicca un estremo da collegare',
            'Clicca la seconda linea',
            'Clicca un estremo da collegare'
        ],
        // newpoly and newline have dynamic steps, handled separately
        newpoly: [],
        newline: []
    };

    function startTool(toolName) {
        if (!selectedLayerName) {
            updateStatus('Seleziona uno strato prima di usare gli strumenti');
            return;
        }

        // Cancel any existing tool
        cancelTool();

        // Deselect current selection
        deselectElement();

        toolState.active = toolName;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.targetFeature = null;
        toolState.vertices = [];

        // Reset newpoly-specific state
        toolState.accumulatedCoords = [];
        toolState.previewLayer = null;
        toolState.closeMarker = null;
        toolState.currentObjectType = null;
        toolState.segmentCount = 0;

        updateToolStatus();
        $('tool-workflow').style.display = 'block';
        $('tool-active-name').textContent = {
            snip: 'Poligono → linea',
            close: 'Linea → poligono',
            split: 'Linea → 2 linee',
            join: '2 linee → linea',
            newpoly: 'Nuovo poligono',
            newline: 'Nuova linea'
        }[toolName];

        updateStatus(`Strumento attivo: ${toolState.active}`);
    }

    function cancelTool() {
        if (!toolState.active) return;

        // Remove vertex markers (with event cleanup)
        toolState.vertexMarkers.forEach(m => removeMarker(m));
        toolState.vertexMarkers = [];

        // Disable editing on features if enabled
        if (toolState.sourceFeature) {
            if (toolState.sourceFeature.editing) toolState.sourceFeature.editing.disable();
            updateElementStyle(toolState.sourceFeature);
        }
        if (toolState.targetFeature) {
            if (toolState.targetFeature.editing) toolState.targetFeature.editing.disable();
            updateElementStyle(toolState.targetFeature);
        }

        // Clean up newpoly-specific elements
        if (toolState.previewLayer) {
            map.removeLayer(toolState.previewLayer);
            toolState.previewLayer = null;
        }
        if (toolState.closeMarker) {
            removeMarker(toolState.closeMarker);
            toolState.closeMarker = null;
        }

        toolState.active = null;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.targetFeature = null;
        toolState.vertices = [];
        toolState.accumulatedCoords = [];
        toolState.currentObjectType = null;
        toolState.segmentCount = 0;

        $('tool-workflow').style.display = 'none';
        updateStatus('Strumento annullato');
    }

    function updateToolStatus() {
        if (!toolState.active) return;

        // Show/hide the finish button for newline tool
        const finishBtn = $('tool-finish-btn');
        if (finishBtn) {
            const showFinish = toolState.active === 'newline' && toolState.step === 0 && toolState.segmentCount > 0;
            finishBtn.style.display = showFinish ? 'inline-block' : 'none';
        }

        // Handle newpoly/newline tools with dynamic steps
        if (toolState.active === 'newpoly' || toolState.active === 'newline') {
            const stepText = getNewShapeStepText();
            const segInfo = toolState.segmentCount > 0
                ? ` (${toolState.segmentCount} segmenti, ${toolState.accumulatedCoords.length} punti)`
                : '';
            $('tool-step-text').textContent = stepText + segInfo;
            return;
        }

        const steps = toolSteps[toolState.active];
        const stepText = steps[toolState.step] || 'Esecuzione...';
        $('tool-step-text').textContent = `Passo ${toolState.step + 1}/${steps.length}: ${stepText}`;
    }

    function getNewShapeStepText() {
        // step 0: waiting for object selection
        // step 1: picked object, waiting for first vertex
        // step 2: picked first vertex, waiting for second
        // step 3: (polygon only) picked second vertex, waiting for third
        const { active, step, currentObjectType, segmentCount } = toolState;
        const isNewpoly = active === 'newpoly';

        if (step === 0) {
            if (segmentCount === 0) {
                return 'Clicca un oggetto (linea o poligono)';
            } else if (isNewpoly) {
                return 'Clicca un oggetto, oppure clicca il punto iniziale per chiudere';
            } else {
                return 'Clicca un oggetto, oppure premi Invio/Fine per terminare';
            }
        }

        if (currentObjectType === 'line') {
            if (step === 1) return 'Clicca il primo vertice della linea';
            if (step === 2) return 'Clicca il secondo vertice della linea';
        } else if (currentObjectType === 'polygon') {
            if (step === 1) return 'Clicca il primo vertice del poligono';
            if (step === 2) return 'Clicca il vertice di direzione';
            if (step === 3) return 'Clicca il vertice finale';
        }

        return 'Selezione...';
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
            } else if (toolState.active === 'split') {
                executeSplit();
            } else if (toolState.active === 'join') {
                executeJoin();
            }
        } catch (err) {
            updateStatus('Errore: ' + err.message);
            console.error(err);
        }

        // Clean up (with event cleanup)
        toolState.vertexMarkers.forEach(m => removeMarker(m));
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
        marker.addTo(map);
        toolState.vertexMarkers.push(marker);
        return marker;
    }

    // Handle feature click during tool workflow
    function handleToolFeatureClick(element, isPolygon) {
        if (!toolState.active) return false;

        const tool = toolState.active;
        const step = toolState.step;

        // Handle newpoly/newline tools separately (they share the same logic)
        if (tool === 'newpoly' || tool === 'newline') {
            return handleNewShapeFeatureClick(element, isPolygon);
        }

        // Step 0: Select source feature
        if (step === 0) {
            if (tool === 'snip' && isPolygon) {
                toolState.sourceFeature = element;
                element.setStyle(styles.selected);
                enableVertexClicks(element, 'source');
                advanceToolStep();
                return true;
            } else if ((tool === 'close' || tool === 'split') && !isPolygon) {
                toolState.sourceFeature = element;
                element.setStyle(styles.lineSelected);
                enableVertexClicks(element, 'source');
                advanceToolStep();
                return true;
            } else if (tool === 'join' && !isPolygon) {
                toolState.sourceFeature = element;
                element.setStyle(styles.lineSelected);
                enableEndpointClicks(element, 'source');
                advanceToolStep();
                return true;
            }
        }

        // Step 2 for 'join': Select second line
        if (tool === 'join' && step === 2 && !isPolygon) {
            toolState.targetFeature = element;
            element.setStyle(styles.lineSelected);
            enableEndpointClicks(element, 'target');
            advanceToolStep();
            return true;
        }

        return false;
    }

    // Handle feature click for newpoly/newline tools
    function handleNewShapeFeatureClick(element, isPolygon) {
        if (toolState.step !== 0) return false;

        // Clear previous vertex markers (but keep close marker and preview)
        toolState.vertexMarkers.forEach(m => {
            if (m !== toolState.closeMarker) {
                removeMarker(m);
            }
        });
        toolState.vertexMarkers = toolState.closeMarker ? [toolState.closeMarker] : [];

        // Reset current selection state
        if (toolState.sourceFeature) {
            updateElementStyle(toolState.sourceFeature);
        }
        toolState.sourceFeature = element;
        toolState.currentObjectType = isPolygon ? 'polygon' : 'line';
        toolState.vertices = [];

        // Highlight selected feature
        if (isPolygon) {
            element.setStyle(styles.selected);
        } else {
            element.setStyle(styles.lineSelected);
        }

        // Enable vertex clicks
        enableVertexClicks(element, 'source');

        toolState.step = 1;
        updateToolStatus();
        return true;
    }

    // Create clickable vertex markers for tool selection (not using Leaflet edit mode)
    function enableVertexClicks(element, featureRole) {
        // Get vertices from the layer
        const latlngs = element.getLatLngs();

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
                bubblingMouseEvents: false  // Don't let events bubble to layers below
            });

            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                handleVertexClick(latlng, idx, featureRole);
            });

            marker.addTo(map);
            toolState.vertexMarkers.push(marker);
        });
    }

    // Create clickable markers for line endpoints only (for join tool)
    function enableEndpointClicks(element, featureRole) {
        const latlngs = element.getLatLngs();

        // For lines, latlngs is [latlng, latlng, ...]
        const vertices = latlngs;
        if (vertices.length < 2) return;

        // Only create markers for first and last vertex
        const endpoints = [
            { latlng: vertices[0], idx: 0 },
            { latlng: vertices[vertices.length - 1], idx: vertices.length - 1 }
        ];

        endpoints.forEach(({ latlng, idx }) => {
            const marker = L.circleMarker(latlng, {
                radius: 10,
                color: '#ff6600',
                fillColor: '#ff6600',
                fillOpacity: 0.5,
                weight: 2,
                bubblingMouseEvents: false
            });

            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
                handleVertexClick(latlng, idx, featureRole);
            });

            marker.addTo(map);
            toolState.vertexMarkers.push(marker);
        });
    }

    // Remove a marker from the map and clean up its events
    function removeMarker(marker) {
        if (!marker) return;
        marker.off();  // Remove all event listeners
        if (map.hasLayer(marker)) {
            map.removeLayer(marker);
        }
    }

    // Handle vertex click during tool workflow
    function handleVertexClick(latlng, index, featureRole) {
        if (!toolState.active) return;

        const tool = toolState.active;
        const step = toolState.step;

        // Handle newpoly/newline tools separately (they share the same logic)
        if (tool === 'newpoly' || tool === 'newline') {
            handleNewShapeVertexClick(latlng, index);
            return;
        }

        // Determine expected vertex based on tool and step
        let label = '';

        if (tool === 'snip') {
            if (step === 1) label = 'V';  // Single vertex to delete
        } else if (tool === 'close') {
            if (step === 1) label = 'Lv1';
            else if (step === 2) label = 'Lv2';
        } else if (tool === 'split') {
            if (step === 1) label = 'V';
        } else if (tool === 'join') {
            if (step === 1) label = 'E1';  // Endpoint from first line
            else if (step === 3) label = 'E2';  // Endpoint from second line
        }

        if (!label) return;

        toolState.vertices.push({ latlng, index });
        createVertexMarker(latlng, label);
        updateStatus(`Vertice selezionato`);

        advanceToolStep();
    }

    // Handle vertex click for newpoly/newline tools
    function handleNewShapeVertexClick(latlng, index) {
        const { step, currentObjectType, vertices } = toolState;

        if (currentObjectType === 'line') {
            // Lines need 2 vertices
            if (step === 1) {
                toolState.vertices.push({ latlng, index });
                createVertexMarker(latlng, `L${toolState.segmentCount + 1}v1`);
                toolState.step = 2;
                updateToolStatus();
            } else if (step === 2) {
                toolState.vertices.push({ latlng, index });
                createVertexMarker(latlng, `L${toolState.segmentCount + 1}v2`);
                // Collect segment from line
                addNewpolySegmentFromLine();
            }
        } else if (currentObjectType === 'polygon') {
            // Polygons need 3 vertices
            if (step === 1) {
                toolState.vertices.push({ latlng, index });
                createVertexMarker(latlng, `P${toolState.segmentCount + 1}v1`);
                toolState.step = 2;
                updateToolStatus();
            } else if (step === 2) {
                toolState.vertices.push({ latlng, index });
                createVertexMarker(latlng, `P${toolState.segmentCount + 1}v2`);
                toolState.step = 3;
                updateToolStatus();
            } else if (step === 3) {
                toolState.vertices.push({ latlng, index });
                createVertexMarker(latlng, `P${toolState.segmentCount + 1}v3`);
                // Collect segment from polygon
                addNewpolySegmentFromPolygon();
            }
        }
    }

    // Add segment from a line to the new polygon
    function addNewpolySegmentFromLine() {
        const [v1, v2] = toolState.vertices;
        const element = toolState.sourceFeature;
        const coords = element.toGeoJSON().geometry.coordinates;

        // Extract segment v1 to v2 (inclusive)
        let segmentCoords;
        if (v1.index <= v2.index) {
            segmentCoords = coords.slice(v1.index, v2.index + 1);
        } else {
            segmentCoords = coords.slice(v2.index, v1.index + 1).reverse();
        }

        finishNewpolySegment(segmentCoords);
    }

    // Add segment from a polygon to the new polygon
    function addNewpolySegmentFromPolygon() {
        const [v1, v2, v3] = toolState.vertices;
        const element = toolState.sourceFeature;
        const coords = element.toGeoJSON().geometry.coordinates[0].slice(0, -1); // Remove closing point
        const n = coords.length;

        // Determine path from v1 to v3 through v2
        const pathCW = getPathIndices(v1.index, v3.index, n, 1);
        const pathCCW = getPathIndices(v1.index, v3.index, n, -1);

        let pathIndices;
        if (pathCW.includes(v2.index)) {
            pathIndices = pathCW;
        } else if (pathCCW.includes(v2.index)) {
            pathIndices = pathCCW;
        } else {
            updateStatus('Errore: v2 non trovato su nessun percorso');
            return;
        }

        const segmentCoords = pathIndices.map(i => coords[i]);
        finishNewpolySegment(segmentCoords);
    }

    // Common logic after collecting a segment
    function finishNewpolySegment(segmentCoords) {
        // Add to accumulated coordinates
        toolState.accumulatedCoords.push(...segmentCoords);
        toolState.segmentCount++;

        // Reset feature style
        updateElementStyle(toolState.sourceFeature);
        toolState.sourceFeature = null;
        toolState.currentObjectType = null;
        toolState.vertices = [];

        // Clear vertex markers except close marker (with event cleanup)
        toolState.vertexMarkers.forEach(m => {
            if (m !== toolState.closeMarker) {
                removeMarker(m);
            }
        });
        toolState.vertexMarkers = toolState.closeMarker ? [toolState.closeMarker] : [];

        // Update preview
        updateNewShapePreview();

        // Create close marker for newpoly only (not for newline)
        if (toolState.active === 'newpoly' && !toolState.closeMarker && toolState.accumulatedCoords.length > 0) {
            createCloseMarker();
        }

        // Go back to step 0 to select next object
        toolState.step = 0;
        updateToolStatus();
        updateStatus(`Segmento ${toolState.segmentCount} aggiunto (${toolState.accumulatedCoords.length} punti totali)`);
    }

    // Update the preview polyline
    function updateNewShapePreview() {
        if (toolState.previewLayer) {
            map.removeLayer(toolState.previewLayer);
        }

        if (toolState.accumulatedCoords.length >= 2) {
            const latlngs = L.GeoJSON.coordsToLatLngs(toolState.accumulatedCoords, 0);
            toolState.previewLayer = L.polyline(latlngs, {
                color: '#9900ff',
                weight: 3,
                opacity: 0.8,
                dashArray: '8, 8',
                interactive: false  // Don't capture mouse events
            });
            toolState.previewLayer.addTo(map);
        }
    }

    // Create the close marker on the first point
    function createCloseMarker() {
        const firstCoord = toolState.accumulatedCoords[0];
        const latlng = L.GeoJSON.coordsToLatLng(firstCoord);

        toolState.closeMarker = L.circleMarker(latlng, {
            radius: 12,
            color: '#00cc00',
            fillColor: '#00ff00',
            fillOpacity: 0.7,
            weight: 3,
            bubblingMouseEvents: false
        });


        toolState.closeMarker.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            executeNewpoly();
        });

        toolState.closeMarker.addTo(map);
        toolState.vertexMarkers.push(toolState.closeMarker);
    }

    // Execute newpoly: create the polygon from accumulated coords
    function executeNewpoly() {
        if (toolState.accumulatedCoords.length < 3) {
            updateStatus('Servono almeno 3 punti per creare un poligono');
            return;
        }

        const layerName = selectedLayerName;

        // Close the ring
        const closedCoords = [...toolState.accumulatedCoords, toolState.accumulatedCoords[0]];
        const polyLatLngs = L.GeoJSON.coordsToLatLngs([closedCoords], 1);
        const newPolygon = L.polygon(polyLatLngs, styles.default);

        // Add to layer
        drawnItems.addLayer(newPolygon);
        addElementToLayer(layerName, newPolygon, 'polygon');
        addElementClickHandler(newPolygon);

        updateStatus(`Nuovo poligono creato con ${toolState.accumulatedCoords.length} vertici`);

        // Clean up (with event cleanup)
        if (toolState.previewLayer) {
            map.removeLayer(toolState.previewLayer);
        }
        toolState.vertexMarkers.forEach(m => removeMarker(m));
        toolState.vertexMarkers = [];
        toolState.previewLayer = null;
        toolState.closeMarker = null;
        toolState.accumulatedCoords = [];
        toolState.segmentCount = 0;
        toolState.active = null;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.currentObjectType = null;
        toolState.vertices = [];

        $('tool-workflow').style.display = 'none';
        updateElementList();
    }

    // Execute newline: create the line from accumulated coords
    function executeNewline() {
        if (toolState.accumulatedCoords.length < 2) {
            updateStatus('Servono almeno 2 punti per creare una linea');
            return;
        }

        const layerName = selectedLayerName;

        // Create line (no closing needed)
        const lineLatLngs = L.GeoJSON.coordsToLatLngs(toolState.accumulatedCoords, 0);
        const newLine = L.polyline(lineLatLngs, styles.line);

        // Add to layer
        drawnItems.addLayer(newLine);
        addElementToLayer(layerName, newLine, 'line');
        addElementClickHandler(newLine);

        updateStatus(`Nuova linea creata con ${toolState.accumulatedCoords.length} vertici`);

        // Clean up (with event cleanup)
        if (toolState.previewLayer) {
            map.removeLayer(toolState.previewLayer);
        }
        toolState.vertexMarkers.forEach(m => removeMarker(m));
        toolState.vertexMarkers = [];
        toolState.previewLayer = null;
        toolState.closeMarker = null;
        toolState.accumulatedCoords = [];
        toolState.segmentCount = 0;
        toolState.active = null;
        toolState.step = 0;
        toolState.sourceFeature = null;
        toolState.currentObjectType = null;
        toolState.vertices = [];

        $('tool-workflow').style.display = 'none';
        updateElementList();
    }

    // Execute snip: polygon -> line (delete one vertex, open the polygon)
    function executeSnip() {
        const element = toolState.sourceFeature;
        const [vertex] = toolState.vertices;
        const layerName = element.layerName;
        const layer = layers[layerName];

        // Get polygon coordinates (outer ring only)
        const geom = element.toGeoJSON().geometry;
        const coords = geom.coordinates[0].slice(0, -1);  // Remove closing point
        const n = coords.length;

        if (n < 4) {
            throw new Error('Poligono troppo piccolo per essere aperto');
        }

        // Delete the selected vertex and create a line starting from the next vertex
        // Example: A-B-C-D-E-F-A, delete C -> line D-E-F-A-B
        const deleteIdx = vertex.index;

        // Build line coordinates starting from vertex after deleted, going all the way around
        const lineCoords = [];
        for (let i = 1; i < n; i++) {
            const idx = (deleteIdx + i) % n;
            lineCoords.push(coords[idx]);
        }

        // Create line
        const latlngs = L.GeoJSON.coordsToLatLngs(lineCoords, 0);
        const newLine = L.polyline(latlngs, styles.line);
        newLine.elementName = element.name;

        // Remove old polygon, add new line
        drawnItems.removeLayer(element);
        layer.elements = layer.elements.filter(el => el !== element);
        drawnItems.addLayer(newLine);
        addElementToLayer(layerName, newLine, 'line');
        addElementClickHandler(newLine);

        updateStatus(`Poligono aperto in linea con ${lineCoords.length} vertici`);
    }

    // Execute close: line -> polygon
    function executeClose() {
        const element = toolState.sourceFeature;
        const [lv1, lv2] = toolState.vertices;
        const layerName = element.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const geom = element.toGeoJSON().geometry;
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
        newPolygon.elementName = element.name;

        // Remove old line
        drawnItems.removeLayer(element);
        layer.elements = layer.elements.filter(el => el !== element);

        // Add new polygon
        drawnItems.addLayer(newPolygon);
        addElementToLayer(layerName, newPolygon, 'polygon');
        addElementClickHandler(newPolygon);

        // Create leftover lines if any
        if (idx1 > 0) {
            const leftCoords = coords.slice(0, idx1 + 1);
            if (leftCoords.length >= 2) {
                const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.line);
                leftLine.elementName = element.name + ' (rimanente 1)';
                drawnItems.addLayer(leftLine);
                addElementToLayer(layerName, leftLine, 'line');
                addElementClickHandler(leftLine);
            }
        }

        if (idx2 < coords.length - 1) {
            const rightCoords = coords.slice(idx2);
            if (rightCoords.length >= 2) {
                const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.line);
                rightLine.elementName = element.name + ' (rimanente 2)';
                drawnItems.addLayer(rightLine);
                addElementToLayer(layerName, rightLine, 'line');
                addElementClickHandler(rightLine);
            }
        }

        updateStatus(`Linea chiusa in poligono con ${polygonCoords.length} vertici`);
    }

    // Execute split: line -> 2 lines
    function executeSplit() {
        const element = toolState.sourceFeature;
        const [splitVertex] = toolState.vertices;
        const layerName = element.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const geom = element.toGeoJSON().geometry;
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
        drawnItems.removeLayer(element);
        layer.elements = layer.elements.filter(el => el !== element);

        // Create left line
        const leftLine = L.polyline(L.GeoJSON.coordsToLatLngs(leftCoords, 0), styles.line);
        leftLine.elementName = element.name + ' (1)';
        drawnItems.addLayer(leftLine);
        addElementToLayer(layerName, leftLine, 'line');
        addElementClickHandler(leftLine);

        // Create right line
        const rightLine = L.polyline(L.GeoJSON.coordsToLatLngs(rightCoords, 0), styles.line);
        rightLine.elementName = element.name + ' (2)';
        drawnItems.addLayer(rightLine);
        addElementToLayer(layerName, rightLine, 'line');
        addElementClickHandler(rightLine);

        updateStatus(`Linea divisa in 2 linee`);
    }

    // Execute join: 2 lines -> line (connect at selected endpoints)
    function executeJoin() {
        const line1 = toolState.sourceFeature;
        const line2 = toolState.targetFeature;
        const [e1, e2] = toolState.vertices;
        const layerName = line1.layerName;
        const layer = layers[layerName];

        // Get line coordinates
        const coords1 = line1.toGeoJSON().geometry.coordinates;
        const coords2 = line2.toGeoJSON().geometry.coordinates;

        // Determine orientation based on which endpoints were selected
        // e1 is from line 1, e2 is from line 2
        // We want: line1 (oriented so e1 is at end) + line2 (oriented so e2 is at start)

        let segment1, segment2;

        // If e1 is at the end of line1 (last index), use line1 as-is
        // If e1 is at the start of line1 (index 0), reverse line1
        if (e1.index === 0) {
            segment1 = coords1.slice().reverse();
        } else {
            segment1 = coords1.slice();
        }

        // If e2 is at the start of line2 (index 0), use line2 as-is
        // If e2 is at the end of line2 (last index), reverse line2
        if (e2.index === 0) {
            segment2 = coords2.slice();
        } else {
            segment2 = coords2.slice().reverse();
        }

        // Join the two lines
        const joinedCoords = [...segment1, ...segment2];

        // Remove old lines
        drawnItems.removeLayer(line1);
        drawnItems.removeLayer(line2);
        layer.elements = layer.elements.filter(el => el !== line1 && el !== line2);

        // Create joined line
        const joinedLine = L.polyline(L.GeoJSON.coordsToLatLngs(joinedCoords, 0), styles.line);
        joinedLine.elementName = line1.name;
        drawnItems.addLayer(joinedLine);
        addElementToLayer(layerName, joinedLine, 'line');
        addElementClickHandler(joinedLine);

        updateStatus(`2 linee unite in una linea con ${joinedCoords.length} vertici`);
    }

    // Apply offset delta to a layer's elements
    function applyLayerOffset(layerName) {
        const layer = layers[layerName];
        if (!layer) return;

        const deltaEw = layer.offset.ew - layer.oldOffset.ew;
        const deltaNs = layer.offset.ns - layer.oldOffset.ns;
        if (deltaEw === 0 && deltaNs === 0) return;

        const { lon: deltaLon, lat: deltaLat } = getOffsetDegrees(deltaEw, deltaNs);
        const wasEditing = selectedElement?.layerName === layerName ? selectedElement : null;

        // Apply to all elements
        layer.elements.forEach(element => {
            // Skip lines if OFFSET_LINES is false
            if (element.type === 'line' && !OFFSET_LINES) return;

            const geom = element.toGeoJSON().geometry;
            const newCoords = offsetCoordinates(geom.coordinates, deltaLon, deltaLat);
            const style = element.options;

            let newLayer;
            if (element.type === 'polygon') {
                const depth = geom.type === 'Polygon' ? 1 : 2;
                const newLatLngs = L.GeoJSON.coordsToLatLngs(newCoords, depth);
                newLayer = L.polygon(newLatLngs, style);
            } else {
                const newLatLngs = L.GeoJSON.coordsToLatLngs(newCoords, 0);
                newLayer = L.polyline(newLatLngs, style);
            }

            // Copy properties
            newLayer.id = element.id;
            newLayer.name = element.name;
            newLayer.type = element.type;
            newLayer.layerName = layerName;
            newLayer.visible = element.visible;

            // Replace in array
            const idx = layer.elements.indexOf(element);
            if (idx !== -1) {
                layer.elements[idx] = newLayer;
            }

            drawnItems.removeLayer(element);
            drawnItems.addLayer(newLayer);
            addElementClickHandler(newLayer);

            // Update wasEditing reference if this was the selected element
            if (wasEditing === element) {
                selectedElement = newLayer;
            }
        });

        if (wasEditing && selectedElement) {
            const style = selectedElement.type === 'line' ? styles.lineSelected : styles.selected;
            selectedElement.setStyle(style);
            selectedElement.editing.enable();
        }

        layer.oldOffset = { ew: layer.offset.ew, ns: layer.offset.ns };
    }

    // UI updates
    function updateLayerList() {
        const list = $('layer-list');
        if (!list) return;
        list.innerHTML = '';

        const names = Object.keys(layers).sort((a, b) => a.localeCompare(b));

        names.forEach(name => {
            const layer = layers[name];
            const isSelected = name === selectedLayerName;
            const div = document.createElement('div');
            div.className = 'list-item' + (isSelected ? ' selected' : '') + (layer.visible ? '' : ' hidden');

            const visIcon = layer.visible ? '◉' : '◌';
            div.innerHTML = `
                <span class="item-name" onclick="ParcelEditor.onLayerClick('${name}')">${name}</span>
                <span class="item-actions">
                    <span class="edit-btn" onclick="ParcelEditor.renameLayer('${name}')" title="Rinomina">✎</span>
                    <span class="hide-btn" onclick="ParcelEditor.toggleLayerVisibility('${name}')" title="Mostra/nascondi">${visIcon}</span>
                    <span class="delete-btn" onclick="ParcelEditor.deleteLayerByName('${name}')" title="Elimina">✕</span>
                </span>
            `;
            list.appendChild(div);
        });
    }

    function updateElementList() {
        const list = $('element-list');
        if (!list) return;
        list.innerHTML = '';
        list.className = 'feature-list';

        const layer = getSelectedLayer();
        if (!layer) return;

        // Sort elements by name
        const elements = [...layer.elements];
        elements.sort((a, b) => a.name.localeCompare(b.name));

        elements.forEach(element => {
            const isSelected = selectedElement === element;
            const isHidden = element.visible === false;
            const div = document.createElement('div');
            div.className = 'list-item' + (isSelected ? ' selected' : '') + (isHidden ? ' hidden' : '');

            const typeIcon = element.type === 'polygon' ? '▢' : '─';
            const visIcon = isHidden ? '◌' : '◉';

            div.innerHTML = `
                <span class="item-type-icon">${typeIcon}</span>
                <span class="item-name" onclick="ParcelEditor.onElementClick(${element.id})">${element.name}</span>
                <span class="item-actions">
                    <span class="edit-btn" onclick="ParcelEditor.startRename(${element.id})" title="Rinomina">✎</span>
                    <span class="hide-btn" onclick="ParcelEditor.toggleElementVisibility(${element.id})" title="Mostra/nascondi">${visIcon}</span>
                    <span class="move-btn" onclick="ParcelEditor.moveElementToLayer(${element.id})" title="Sposta in altro strato">↗</span>
                    <span class="delete-btn" onclick="ParcelEditor.onDeleteElement(${element.id})" title="Elimina">✕</span>
                </span>
            `;
            list.appendChild(div);
        });
    }

    function findElementById(id) {
        for (const layer of Object.values(layers)) {
            const element = layer.elements.find(el => el.id === id);
            if (element) return element;
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

        let newPolygons = 0;
        let newLines = 0;
        let newLayers = 0;

        // Merge features into existing layers or create new ones
        Object.entries(featuresByLayer).forEach(([layerName, features]) => {
            const isNewLayer = !layers[layerName];
            if (isNewLayer) {
                createLayer(layerName);
                newLayers++;
            }

            features.forEach(feature => {
                if (feature.geometry?.type === 'Polygon' || feature.geometry?.type === 'MultiPolygon') {
                    // Handle polygons
                    const depth = feature.geometry.type === 'Polygon' ? 1 : 2;
                    const latlngs = L.GeoJSON.coordsToLatLngs(feature.geometry.coordinates, depth);
                    const mapLayer = L.polygon(latlngs, styles.otherLayer);
                    mapLayer.elementName = feature.properties?.name;
                    drawnItems.addLayer(mapLayer);
                    addElementToLayer(layerName, mapLayer, 'polygon');
                    addElementClickHandler(mapLayer);
                    newPolygons++;
                } else if (feature.geometry?.type === 'LineString') {
                    // Handle lines
                    const latlngs = L.GeoJSON.coordsToLatLngs(feature.geometry.coordinates, 0);
                    const mapLayer = L.polyline(latlngs, styles.lineOtherLayer);
                    mapLayer.elementName = feature.properties?.name;
                    drawnItems.addLayer(mapLayer);
                    addElementToLayer(layerName, mapLayer, 'line');
                    addElementClickHandler(mapLayer);
                    newLines++;
                }
            });
        });

        return { newPolygons, newLines, newLayers };
    }

    function initUI(loadResults) {
        // Update UI
        updateLayerList();

        // Select appropriate layer
        if (!selectedLayerName && Object.keys(layers).length > 0) {
            const firstLayer = Object.keys(layers).sort()[0];
            selectLayer(firstLayer);
        } else if (selectedLayerName && layers[selectedLayerName]) {
            selectLayer(selectedLayerName);
        }

        if (drawnItems.getBounds().isValid()) {
            map.fitBounds(drawnItems.getBounds());
        }

        // Show status
        const totalPolygons = Object.values(layers).reduce((sum, l) => sum + l.elements.filter(e => e.type === 'polygon').length, 0);
        const totalLines = Object.values(layers).reduce((sum, l) => sum + l.elements.filter(e => e.type === 'line').length, 0);

        const totalNewPolygons = loadResults.reduce((sum, r) => sum + r.newPolygons, 0);
        const totalNewLines = loadResults.reduce((sum, r) => sum + r.newLines, 0);
        const totalNewLayers = loadResults.reduce((sum, r) => sum + r.newLayers, 0);

        const fileCount = loadResults.length;
        const msg = fileCount > 1
            ? `Caricati ${fileCount} file: ${totalNewPolygons} poligoni e ${totalNewLines} linee (${totalNewLayers} nuovi strati). Totale: ${totalPolygons} poligoni, ${totalLines} linee in ${Object.keys(layers).length} strati`
            : totalNewLayers > 0
                ? `Aggiunti ${totalNewPolygons} poligoni e ${totalNewLines} linee (${totalNewLayers} nuovi strati). Totale: ${totalPolygons} poligoni, ${totalLines} linee in ${Object.keys(layers).length} strati`
                : `Aggiunti ${totalNewPolygons} poligoni e ${totalNewLines} linee agli strati esistenti. Totale: ${totalPolygons} poligoni, ${totalLines} linee`;
        updateStatus(msg);
    }

    function exportGeoJSON() {
        const features = [];

        Object.entries(layers).sort((a, b) => a[0].localeCompare(b[0])).forEach(([layerName, layer]) => {
            layer.elements.forEach(element => {
                const geojson = element.toGeoJSON();
                geojson.properties = geojson.properties || {};
                geojson.properties.name = element.name;
                geojson.properties.layer = layerName;
                geojson.properties.id = element.id;
                geojson.properties.type = element.type;  // 'polygon' or 'line'
                features.push(geojson);
            });
        });

        const geojson = { type: 'FeatureCollection', features };
        const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'elementi.geojson';
        a.click();
        URL.revokeObjectURL(url);

        updateStatus(`Esportate ${features.length} feature`);
    }

    function clearAll(confirm_needed = true) {
        if (confirm_needed && Object.keys(layers).length > 0) {
            if (!confirm('Cancellare tutti gli strati e gli elementi?')) return;
        }

        deselectElement();
        drawnItems.clearLayers();
        layers = {};
        selectedLayerName = null;
        elementCounter = 0;
        updateLayerList();
        updateElementList();
        updateStatus('Tutti gli strati cancellati');
    }

    function addElementClickHandler(element) {
        element.on('click', function(e) {
            L.DomEvent.stopPropagation(e);
            if (!element.id || element.layerName !== selectedLayerName) return;

            // Check if tool workflow is active
            if (toolState.active) {
                const isPolygon = element.type === 'polygon';
                if (!handleToolFeatureClick(element, isPolygon)) {
                    updateStatus('Questa feature non è valida per questo passo');
                }
                return;
            }

            selectElement(element);
        });
    }

    // Rename functionality
    function findElementListItem(id) {
        const list = $('element-list');
        if (!list) return null;

        const items = list.querySelectorAll('.list-item');
        for (const item of items) {
            const nameEl = item.querySelector('.item-name');
            if (!nameEl) continue;

            const onclick = nameEl.getAttribute('onclick') || '';
            if (onclick.includes(`onElementClick(${id})`)) {
                return item;
            }
        }
        return null;
    }

    function startRename(elementId) {
        const element = findElementById(elementId);
        if (!element) return;

        const item = findElementListItem(elementId);
        if (!item) return;

        const typeIcon = element.type === 'polygon' ? '▢' : '─';
        item.innerHTML = `
            <span class="item-type-icon">${typeIcon}</span>
            <input type="text" class="rename-input" value="${element.name}"
                   onkeydown="ParcelEditor.handleRenameKey(event, ${elementId})" />
            <span class="item-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishRename(${elementId})" title="Salva">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateElementList()" title="Annulla">✕</span>
            </span>
        `;
        const input = item.querySelector('.rename-input');
        input.focus();
        input.select();
    }

    function handleRenameKey(event, elementId) {
        if (event.key === 'Enter') finishRename(elementId);
        else if (event.key === 'Escape') updateElementList();
    }

    function finishRename(elementId) {
        const element = findElementById(elementId);
        if (!element) return;

        const input = $('element-list').querySelector('.rename-input');
        if (!input) return;

        element.name = input.value.trim() || 'Senza nome';
        element.bindPopup(`<b>${element.name}</b>`);
        const layer = layers[element.layerName];
        if (layer) sortElements(layer);
        updateElementList();

        const typeLabel = element.type === 'line' ? 'Linea' : 'Poligono';
        updateStatus(`${typeLabel} rinominato a "${element.name}"`);
    }

    // Layer visibility and rename
    function toggleLayerVisibility(name) {
        const layer = layers[name];
        if (!layer) return;

        layer.visible = !layer.visible;
        updateElementStyles();
        updateLayerList();
        updateStatus(`Strato "${name}" ${layer.visible ? 'visibile' : 'nascosto'}`);
    }

    function toggleAllLayers() {
        layersVisible = !layersVisible;
        Object.keys(layers).forEach(name => {
            layers[name].visible = layersVisible;
        });
        updateElementStyles();
        updateLayerList();
        updateStatus(`Tutti gli strati ${layersVisible ? 'visibili' : 'nascosti'}`);
        if (!layersVisible) {
            $('toggle-all-layers-btn').innerText = 'Mostra tutti';
        } else {
            $('toggle-all-layers-btn').innerText = 'Nascondi tutti';
        }
    }

    function renameLayer(oldName) {
        const newName = prompt('Nuovo nome dello strato:', oldName);
        if (!newName || !newName.trim() || newName.trim() === oldName) return;

        const trimmedName = newName.trim();
        if (layers[trimmedName]) {
            updateStatus('Esiste già uno strato con questo nome');
            return;
        }

        // Move layer data to new name
        layers[trimmedName] = layers[oldName];
        delete layers[oldName];

        // Update all elements to reference the new layer name
        layers[trimmedName].elements.forEach(el => {
            el.layerName = trimmedName;
        });

        // Update selection if needed
        if (selectedLayerName === oldName) {
            selectedLayerName = trimmedName;
        }

        updateLayerList();
        updateStatus(`Strato rinominato da "${oldName}" a "${trimmedName}"`);
    }

    function deleteLayerByName(name) {
        if (!confirm(`Elimina lo strato "${name}" e tutti i suoi elementi?`)) return;
        deleteLayer(name);
        updateStatus(`Eliminato strato: ${name}`);
    }

    // Element visibility and move
    function toggleElementVisibility(elementId) {
        const element = findElementById(elementId);
        if (!element) return;

        element.visible = element.visible === false ? true : false;

        // If hiding and currently selected, deselect to hide edit controls
        if (element.visible === false && selectedElement === element) {
            deselectElement();
        } else {
            updateElementStyle(element);
        }
        updateElementList();
    }

    function moveElementToLayer(elementId) {
        const element = findElementById(elementId);
        if (!element) return;

        const currentLayer = element.layerName;
        const otherLayers = Object.keys(layers).filter(n => n !== currentLayer).sort();

        if (otherLayers.length === 0) {
            updateStatus('Non ci sono altri strati');
            return;
        }

        const item = findElementListItem(elementId);
        if (!item) return;

        const options = otherLayers.map(name => `<option value="${name}">${name}</option>`).join('');
        item.innerHTML = `
            <select class="move-select" style="flex-grow: 1; margin-right: 5px;">
                ${options}
            </select>
            <span class="item-actions">
                <span class="edit-btn" onclick="ParcelEditor.finishMoveElement(${elementId})" title="Conferma">✓</span>
                <span class="delete-btn" onclick="ParcelEditor.updateElementList()" title="Annulla">✕</span>
            </span>
        `;
        const select = item.querySelector('.move-select');
        select.focus();
    }

    function finishMoveElement(elementId) {
        const element = findElementById(elementId);
        if (!element) return;

        const select = $('element-list').querySelector('.move-select');
        if (!select) return;

        const targetName = select.value;
        const currentLayerName = element.layerName;

        // Remove from current layer
        const sourceLayer = layers[currentLayerName];
        sourceLayer.elements = sourceLayer.elements.filter(el => el !== element);

        // Add to target layer
        element.layerName = targetName;
        layers[targetName].elements.push(element);
        sortElements(layers[targetName]);

        // Update style
        updateElementStyle(element);
        updateElementList();
        updateStatus(`Elemento spostato in "${targetName}"`);
    }

    // Public API
    return {
        init(filename = null) {
            // Localize Leaflet.Draw to Italian
            L.drawLocal.draw.toolbar.buttons.polygon = 'Disegna un poligono';
            L.drawLocal.draw.toolbar.buttons.polyline = 'Disegna una linea';
            L.drawLocal.draw.handlers.polygon.tooltip.start = 'Clicca per iniziare a disegnare';
            L.drawLocal.draw.handlers.polygon.tooltip.cont = 'Clicca per continuare';
            L.drawLocal.draw.handlers.polygon.tooltip.end = 'Clicca il primo punto per chiudere';
            L.drawLocal.draw.handlers.polyline.tooltip.start = 'Clicca per iniziare a disegnare';
            L.drawLocal.draw.handlers.polyline.tooltip.cont = 'Clicca per continuare';
            L.drawLocal.draw.handlers.polyline.tooltip.end = 'Clicca l\'ultimo punto per terminare';

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
                    polygon: {
                        allowIntersection: false,
                        shapeOptions: styles.default,
                        showArea: true,
                    },
                    polyline: { showLength: true },
                    rectangle: false,
                    circle: false,
                    marker: false,
                    circlemarker: false
                },
                edit: false
            });
            map.addControl(drawControl);

            // Fix tooltip positioning on draw start
            map.on(L.Draw.Event.DRAWSTART, e => {
                // Show tooltips only after first mousemove (when properly positioned)
                map.once('mousemove', () => {
                    const tooltips = document.querySelectorAll('.leaflet-draw-tooltip');
                    tooltips.forEach(t => t.classList.add('positioned'));
                });
            });

            map.on(L.Draw.Event.DRAWSTOP, e => {
                // Reset tooltip visibility for next draw
                const tooltips = document.querySelectorAll('.leaflet-draw-tooltip');
                tooltips.forEach(t => t.classList.remove('positioned'));
            });

            // New feature drawn
            map.on(L.Draw.Event.CREATED, e => {
                if (!selectedLayerName) {
                    updateStatus('Seleziona o crea uno strato prima di disegnare');
                    return;
                }

                const mapLayer = e.layer;
                drawnItems.addLayer(mapLayer);

                // Check if it's a polyline (line) or polygon
                if (e.layerType === 'polyline') {
                    const element = addElementToLayer(selectedLayerName, mapLayer, 'line');
                    addElementClickHandler(mapLayer);
                    mapLayer.bindPopup(`<b>${element.name}</b>`);
                    selectElement(element);
                    updateElementList();
                    updateStatus(`Creata linea ${element.name}`);
                } else {
                    const element = addElementToLayer(selectedLayerName, mapLayer, 'polygon');
                    addElementClickHandler(mapLayer);
                    mapLayer.bindPopup(`<b>${element.name}</b>`);
                    selectElement(element);
                    updateElementList();
                    updateStatus(`Creato poligono ${element.name}`);
                }
            });

            // Click map to deselect (feature clicks stop propagation)
            map.on('click', () => {
                deselectElement();
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', e => {
                if (e.key === 'Escape') {
                    if (toolState.active) {
                        cancelTool();
                    } else {
                        deselectElement();
                    }
                } else if (e.key === 'Enter') {
                    // Complete newline tool with Enter
                    if (toolState.active === 'newline' && toolState.step === 0 && toolState.segmentCount > 0) {
                        e.preventDefault();
                        executeNewline();
                    }
                } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    undoEdit();
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
                const files = e.target.files;
                if (!files || files.length === 0) return;

                // Load all files, then process together
                const filePromises = Array.from(files).map(file => {
                    return new Promise((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onload = evt => {
                            try {
                                const data = JSON.parse(evt.target.result);
                                resolve(data);
                            } catch (err) {
                                reject(err);
                            }
                        };
                        reader.onerror = reject;
                        reader.readAsText(file);
                    });
                });

                Promise.all(filePromises)
                    .then(datasets => {
                        const results = datasets.map(data => loadGeoJSON(data));
                        initUI(results);
                    })
                    .catch(err => {
                        updateStatus('Errore nel caricamento dei file: ' + err.message);
                    });
            });

            if (filename) {
                fetch(filename)
                    .then(r => r.ok ? r.json() : Promise.reject())
                    .then(data => {
                        const result = loadGeoJSON(data);
                        initUI([result]);
                    })
                    .catch(() => {});
            } else {
                // Default view: center of Calabria
                map.setView([39.0, 16.5], 9);
            }
            updateLayerList();
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
            const name = prompt('Nome dello strato:');
            if (!name || !name.trim()) return;

            if (layers[name.trim()]) {
                updateStatus('Lo strato esiste già');
                return;
            }

            createLayer(name.trim());
            updateLayerList();
            selectLayer(name.trim());
            updateStatus(`Creato strato: ${name.trim()}`);
        },

        exportGeoJSON,
        clearAll,
        toggleAllLayers,

        // Layer functions
        onLayerClick(name) {
            selectLayer(name);
        },

        toggleLayerVisibility,
        renameLayer,
        deleteLayerByName,

        // Element functions (unified)
        onElementClick(id) {
            const element = findElementById(id);
            if (element) selectElement(element);
        },

        onDeleteElement(id) {
            const element = findElementById(id);
            if (element) deleteElement(element);
        },

        toggleElementVisibility,
        moveElementToLayer,
        finishMoveElement,

        startRename,
        handleRenameKey,
        finishRename,

        // Tool functions
        startTool,
        cancelTool,
        finishNewline: executeNewline,
        updateElementList
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelEditor.init());
