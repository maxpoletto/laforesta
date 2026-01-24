// Initialize map - centered on Calabria region
const map = L.map('map').setView([38.65, 16.3], 12);

// Basemap layers
const basemaps = {
    osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }),
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '© Esri'
    }),
    topo: L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenTopoMap'
    })
};

let currentBasemap = basemaps.satellite.addTo(map);

function setBasemap(name) {
    map.removeLayer(currentBasemap);
    currentBasemap = basemaps[name].addTo(map);
}

// Reference data storage
let originalReferenceData = null;
let currentOffsetEW = 0;  // meters
let currentOffsetNS = 0;  // meters

// CRS presets (approximate offsets at ~38°N, 16°E in Calabria)
// These convert FROM the named CRS TO WGS84
const crsPresets = {
    'none': { ew: 0, ns: 0 },
    'wgs84': { ew: 0, ns: 0 },
    'ed50': { ew: -90, ns: -130 },        // ED50 → WGS84
    'monte-mario': { ew: -50, ns: -70 },  // Monte Mario → WGS84
    'custom': null  // Keep current values
};

// Layer visibility tracking
let visibleLayers = new Set();
let allLayers = new Set();

// Reference layer (existing parcels)
let referenceLayer = L.geoJSON(null, {
    style: {
        color: '#ff6600',
        weight: 2,
        fillOpacity: 0.2,
    },
    onEachFeature: function(feature, layer) {
        if (feature.properties) {
            const props = Object.entries(feature.properties)
                .map(([k, v]) => `<b>${k}:</b> ${v}`)
                .join('<br>');
            layer.bindPopup(props || 'No properties');
        }
    }
}).addTo(map);

// Drawn parcels layer
const drawnItems = new L.FeatureGroup().addTo(map);
let parcelCounter = 0;
let selectedLayer = null;

// Style constants
const defaultStyle = { color: '#3388ff', weight: 2, fillOpacity: 0.2 };
const loadedStyle = { color: '#ff6600', weight: 2, fillOpacity: 0.2 };
const selectedStyle = { color: '#00ff00', weight: 3, fillOpacity: 0.4 };

// Drawing controls - only draw polygon, no global edit (we do per-parcel)
const drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
        polygon: {
            allowIntersection: false,
            shapeOptions: defaultStyle
        },
        polyline: false,
        rectangle: false,
        circle: false,
        marker: false,
        circlemarker: false
    },
    edit: false
});
map.addControl(drawControl);

// Handle drawing events
map.on(L.Draw.Event.CREATED, function(e) {
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

// Click on map background deselects
map.on('click', function(e) {
    if (e.originalEvent.target === map._container || e.originalEvent.target.classList.contains('leaflet-tile')) {
        deselectParcel();
    }
});

function addLayerClickHandler(layer) {
    layer.on('click', function(e) {
        L.DomEvent.stopPropagation(e);
        selectParcel(layer._leaflet_id);
    });
}

function selectParcel(leafletId) {
    // Deselect previous
    if (selectedLayer && selectedLayer.editing) {
        selectedLayer.editing.disable();
    }
    if (selectedLayer) {
        const style = selectedLayer.isLoaded ? loadedStyle : defaultStyle;
        selectedLayer.setStyle(style);
    }

    // Find and select new
    selectedLayer = null;
    drawnItems.eachLayer(function(layer) {
        if (layer._leaflet_id === leafletId) {
            selectedLayer = layer;
            layer.setStyle(selectedStyle);
            layer.editing.enable();
            updateStatus(`Selected: ${layer.parcelName} - drag vertices to edit`);
        }
    });
    updateParcelList();
}

function deselectParcel() {
    if (selectedLayer) {
        if (selectedLayer.editing) selectedLayer.editing.disable();
        const style = selectedLayer.isLoaded ? loadedStyle : defaultStyle;
        selectedLayer.setStyle(style);
        selectedLayer = null;
        updateParcelList();
        updateStatus('Ready');
    }
}

// Show coordinates on mouse move
map.on('mousemove', function(e) {
    document.getElementById('coords').innerHTML =
        `Lat: ${e.latlng.lat.toFixed(6)}, Lng: ${e.latlng.lng.toFixed(6)}`;
});

// Offset controls
document.getElementById('offset-ew').addEventListener('input', function(e) {
    currentOffsetEW = parseInt(e.target.value);
    document.getElementById('offset-ew-value').textContent = `${currentOffsetEW}m`;
    updateReferenceWithOffset();
});

document.getElementById('offset-ns').addEventListener('input', function(e) {
    currentOffsetNS = parseInt(e.target.value);
    document.getElementById('offset-ns-value').textContent = `${currentOffsetNS}m`;
    updateReferenceWithOffset();
});

function resetOffset() {
    currentOffsetEW = 0;
    currentOffsetNS = 0;
    document.getElementById('offset-ew').value = 0;
    document.getElementById('offset-ns').value = 0;
    document.getElementById('offset-ew-value').textContent = '0m';
    document.getElementById('offset-ns-value').textContent = '0m';
    document.getElementById('crs-preset').value = 'none';
    updateReferenceWithOffset();
}

function applyCrsPreset() {
    const preset = document.getElementById('crs-preset').value;
    if (preset === 'custom' || !crsPresets[preset]) return;

    const offsets = crsPresets[preset];
    currentOffsetEW = offsets.ew;
    currentOffsetNS = offsets.ns;
    document.getElementById('offset-ew').value = currentOffsetEW;
    document.getElementById('offset-ns').value = currentOffsetNS;
    document.getElementById('offset-ew-value').textContent = `${currentOffsetEW}m`;
    document.getElementById('offset-ns-value').textContent = `${currentOffsetNS}m`;
    updateReferenceWithOffset();
}

function offsetCoordinates(coords, offsetLon, offsetLat) {
    if (typeof coords[0] === 'number') {
        // Single coordinate pair [lon, lat]
        return [coords[0] + offsetLon, coords[1] + offsetLat];
    } else {
        // Nested array
        return coords.map(c => offsetCoordinates(c, offsetLon, offsetLat));
    }
}

function updateReferenceWithOffset() {
    if (!originalReferenceData) return;

    // Convert meter offset to approximate degrees
    // At ~38°N: 1° lat ≈ 111km, 1° lon ≈ 87km
    const offsetLon = currentOffsetEW / 87000;
    const offsetLat = currentOffsetNS / 111000;

    // Deep clone and offset
    const offsetData = JSON.parse(JSON.stringify(originalReferenceData));

    for (const feature of offsetData.features || []) {
        if (feature.geometry && feature.geometry.coordinates) {
            feature.geometry.coordinates = offsetCoordinates(
                feature.geometry.coordinates, offsetLon, offsetLat
            );
        }
    }

    referenceLayer.clearLayers();
    referenceLayer.addData(offsetData);

    updateStatus(`Offset: ${currentOffsetEW}m E, ${currentOffsetNS}m N`);
}

function applyOffset() {
    if (!originalReferenceData) {
        updateStatus('No reference data loaded');
        return;
    }

    const offsetLon = currentOffsetEW / 87000;
    const offsetLat = currentOffsetNS / 111000;

    const offsetData = JSON.parse(JSON.stringify(originalReferenceData));

    for (const feature of offsetData.features || []) {
        if (feature.geometry && feature.geometry.coordinates) {
            feature.geometry.coordinates = offsetCoordinates(
                feature.geometry.coordinates, offsetLon, offsetLat
            );
        }
    }

    // Add offset info to properties
    offsetData.properties = offsetData.properties || {};
    offsetData.properties.offset_applied = {
        east_meters: currentOffsetEW,
        north_meters: currentOffsetNS
    };

    // Download
    const blob = new Blob([JSON.stringify(offsetData, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'parcels_corrected.geojson';
    a.click();
    URL.revokeObjectURL(url);

    updateStatus(`Exported with offset: ${currentOffsetEW}m E, ${currentOffsetNS}m N`);
}

// Load reference GeoJSON (preview only - use "Import" to make editable)
document.getElementById('load-reference').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            originalReferenceData = JSON.parse(e.target.result);
            resetOffset();
            referenceLayer.clearLayers();
            referenceLayer.addData(originalReferenceData);

            if (referenceLayer.getBounds().isValid()) {
                map.fitBounds(referenceLayer.getBounds());
            }

            const count = originalReferenceData.features ? originalReferenceData.features.length : 0;
            updateStatus(`Loaded ${count} reference parcels. Adjust offset, then click "Import" to edit.`);
        } catch (err) {
            updateStatus('Error loading GeoJSON: ' + err.message);
        }
    };
    reader.readAsText(file);
});

function importReferenceToEdit() {
    if (!originalReferenceData) {
        updateStatus('No reference data loaded');
        return;
    }

    const offsetLon = currentOffsetEW / 87000;
    const offsetLat = currentOffsetNS / 111000;

    // Clone and apply offset
    const offsetData = JSON.parse(JSON.stringify(originalReferenceData));
    for (const feature of offsetData.features || []) {
        if (feature.geometry && feature.geometry.coordinates) {
            feature.geometry.coordinates = offsetCoordinates(
                feature.geometry.coordinates, offsetLon, offsetLat
            );
        }
    }

    // Add to drawnItems
    let loadedCount = 0;
    allLayers.clear();
    L.geoJSON(offsetData, {
        onEachFeature: function(feature, layer) {
            if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
                parcelCounter++;
                layer.parcelId = parcelCounter;
                layer.parcelName = feature.properties?.name ||
                                    (feature.properties?.parcel_index !== undefined ? `Parcel ${feature.properties.parcel_index}` :
                                    `Imported ${parcelCounter}`);
                layer.isLoaded = true;
                layer.originalProperties = { ...feature.properties };
                layer.bindPopup(`<b>${layer.parcelName}</b>`);
                addLayerClickHandler(layer);
                drawnItems.addLayer(layer);
                loadedCount++;

                // Track layers
                if (feature.properties?.layer) {
                    allLayers.add(feature.properties.layer);
                }
            }
        },
        style: loadedStyle
    });

    // Initialize all layers as visible
    visibleLayers = new Set(allLayers);

    // Clear reference layer (no longer needed)
    referenceLayer.clearLayers();
    originalReferenceData = null;
    resetOffset();
    updateLayerFilter();
    updateParcelList();
    updateStatus(`Imported ${loadedCount} parcels for editing.`);
}

function clearReference() {
    referenceLayer.clearLayers();
    originalReferenceData = null;
    resetOffset();
    updateStatus('Reference data cleared');
}

function updateLayerFilter() {
    const container = document.getElementById('layer-filter-container');
    const filter = document.getElementById('layer-filter');

    if (allLayers.size === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    filter.innerHTML = '';

    const sortedLayers = Array.from(allLayers).sort();
    for (const layerName of sortedLayers) {
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
    if (visible) {
        visibleLayers.add(layerName);
    } else {
        visibleLayers.delete(layerName);
    }
    updateParcelVisibility();
    updateParcelList();
}

function updateParcelVisibility() {
    drawnItems.eachLayer(function(layer) {
        const layerName = layer.originalProperties?.layer || '';
        const shouldShow = visibleLayers.size === 0 || visibleLayers.has(layerName);

        if (shouldShow) {
            if (!map.hasLayer(layer)) {
                // Re-add to map (but it's in drawnItems, so just show it)
                layer.setStyle({ ...layer.options, opacity: 1, fillOpacity: 0.2 });
            }
        } else {
            // Hide by making transparent
            layer.setStyle({ ...layer.options, opacity: 0, fillOpacity: 0 });
            if (selectedLayer && selectedLayer._leaflet_id === layer._leaflet_id) {
                deselectParcel();
            }
        }
    });
}

function updateParcelList() {
    const list = document.getElementById('parcel-list');
    list.innerHTML = '';

    drawnItems.eachLayer(function(layer) {
        const layerName = layer.originalProperties?.layer || '';
        const shouldShow = visibleLayers.size === 0 || visibleLayers.has(layerName);
        if (!shouldShow) return;

        const item = document.createElement('div');
        const isSelected = selectedLayer && selectedLayer._leaflet_id === layer._leaflet_id;
        item.className = 'parcel-item' + (isSelected ? ' selected' : '');
        item.dataset.leafletId = layer._leaflet_id;
        const loadedClass = layer.isLoaded ? ' loaded' : '';
        item.innerHTML = `
            <span class="parcel-name${loadedClass}" onclick="selectParcel(${layer._leaflet_id})">${layer.parcelName || 'Unnamed'}</span>
            <span class="parcel-actions">
                <span class="edit-btn" onclick="startRename(${layer._leaflet_id})" title="Rename">✎</span>
                <span class="delete-btn" onclick="deleteParcel(${layer._leaflet_id})" title="Delete">✕</span>
            </span>
        `;
        list.appendChild(item);
    });
}

function startRename(leafletId) {
    const item = document.querySelector(`.parcel-item[data-leaflet-id="${leafletId}"]`);
    if (!item) return;

    let layer = null;
    drawnItems.eachLayer(function(l) {
        if (l._leaflet_id === leafletId) layer = l;
    });
    if (!layer) return;

    const nameSpan = item.querySelector('.parcel-name');
    const currentName = layer.parcelName || 'Unnamed';
    const loadedClass = layer.isLoaded ? ' loaded' : '';

    item.innerHTML = `
        <input type="text" class="rename-input" value="${currentName}" onkeydown="handleRenameKey(event, ${leafletId})" />
        <span class="parcel-actions">
            <span class="edit-btn" onclick="finishRename(${leafletId})" title="Save">✓</span>
            <span class="delete-btn" onclick="updateParcelList()" title="Cancel">✕</span>
        </span>
    `;
    const input = item.querySelector('.rename-input');
    input.focus();
    input.select();
}

function handleRenameKey(event, leafletId) {
    if (event.key === 'Enter') {
        finishRename(leafletId);
    } else if (event.key === 'Escape') {
        updateParcelList();
    }
}

function finishRename(leafletId) {
    const item = document.querySelector(`.parcel-item[data-leaflet-id="${leafletId}"]`);
    if (!item) return;

    const input = item.querySelector('.rename-input');
    if (!input) return;

    const newName = input.value.trim() || 'Unnamed';

    drawnItems.eachLayer(function(layer) {
        if (layer._leaflet_id === leafletId) {
            layer.parcelName = newName;
            layer.bindPopup(`<b>${newName}</b>`);
            updateStatus(`Renamed to "${newName}"`);
        }
    });

    updateParcelList();
}

function zoomToParcel(leafletId) {
    drawnItems.eachLayer(function(layer) {
        if (layer._leaflet_id === leafletId) {
            map.fitBounds(layer.getBounds(), { padding: [50, 50] });
            selectParcel(leafletId);
        }
    });
}

function deleteParcel(leafletId) {
    drawnItems.eachLayer(function(layer) {
        if (layer._leaflet_id === leafletId) {
            if (selectedLayer && selectedLayer._leaflet_id === leafletId) {
                deselectParcel();
            }
            drawnItems.removeLayer(layer);
        }
    });
    updateParcelList();
    updateStatus('Parcel deleted');
}

function clearAllParcels() {
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
}

function exportGeoJSON() {
    const features = [];

    drawnItems.eachLayer(function(layer) {
        const geojson = layer.toGeoJSON();
        // Preserve original properties for loaded parcels, add name
        if (layer.isLoaded && layer.originalProperties) {
            geojson.properties = { ...layer.originalProperties };
        }
        geojson.properties = geojson.properties || {};
        geojson.properties.name = layer.parcelName;
        geojson.properties.id = layer.parcelId;
        features.push(geojson);
    });

    const geojson = {
        type: 'FeatureCollection',
        features: features
    };

    const blob = new Blob([JSON.stringify(geojson, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'parcels_edited.geojson';
    a.click();
    URL.revokeObjectURL(url);

    updateStatus(`Exported ${features.length} parcels`);
}

function exportFilteredGeoJSON() {
    const features = [];

    drawnItems.eachLayer(function(layer) {
        const layerName = layer.originalProperties?.layer || '';
        const shouldInclude = visibleLayers.size === 0 || visibleLayers.has(layerName);
        if (!shouldInclude) return;

        const geojson = layer.toGeoJSON();
        if (layer.isLoaded && layer.originalProperties) {
            geojson.properties = { ...layer.originalProperties };
        }
        geojson.properties = geojson.properties || {};
        geojson.properties.name = layer.parcelName;
        geojson.properties.id = layer.parcelId;
        features.push(geojson);
    });

    const geojson = {
        type: 'FeatureCollection',
        features: features
    };

    // Generate filename based on visible layers
    const layerSuffix = visibleLayers.size > 0
        ? '_' + Array.from(visibleLayers).join('_').replace(/[^a-zA-Z0-9_]/g, '')
        : '';
    const filename = `parcels${layerSuffix}.geojson`;

    const blob = new Blob([JSON.stringify(geojson, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);

    updateStatus(`Exported ${features.length} filtered parcels`);
}

function updateStatus(msg) {
    document.getElementById('status').innerHTML = msg;
}

// Auto-load all_parcels.json on startup if available
fetch('all_parcels.json')
    .then(response => {
        if (!response.ok) throw new Error('Not found');
        return response.json();
    })
    .then(data => {
        originalReferenceData = data;
        referenceLayer.clearLayers();
        referenceLayer.addData(data);
        if (referenceLayer.getBounds().isValid()) {
            map.fitBounds(referenceLayer.getBounds());
        }
        const count = data.features ? data.features.length : 0;
        updateStatus(`Auto-loaded ${count} parcels from all_parcels.json. Adjust offset, then "Import to Edit".`);
    })
    .catch(() => {
        // File not found or not running on server - that's fine
    });
