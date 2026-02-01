// Sample Area Viewer - Display sample areas on a map with parcel boundaries
const SampleAreaViewer = (function() {
    'use strict';

    let map = null;
    let currentBasemap = null;
    let parcelLayer = null;
    let allMarkers = [];  // Array of {marker, compresa, parcel, cp}
    let areaLayer = null;  // LayerGroup containing all visible markers
    let parcelVisible = {};
    let parcelCounts = {};

    const $ = id => document.getElementById(id);

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

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function loadParcels() {
        return fetch('../data/particelle.geojson')
            .then(r => r.json())
            .then(data => {
                // Include all parcel features
                parcelLayer = L.geoJSON(data, {
                    style: {
                        color: '#3388ff',
                        weight: 2,
                        opacity: 0.8,
                        fillOpacity: 0.1
                    }
                }).addTo(map);

                return data.features.length;
            });
    }

    function loadSampleAreas() {
        return fetch('../data/aree-di-saggio.csv')
            .then(r => r.text())
            .then(csvText => {
                const result = Papa.parse(csvText, {
                    header: true,
                    skipEmptyLines: true
                });

                // Filter to areas with valid coordinates
                const areas = result.data.filter(row =>
                    row.Lon && row.Lat &&
                    !isNaN(parseFloat(row.Lon)) &&
                    !isNaN(parseFloat(row.Lat))
                );

                // Collect parcels with compresa names
                const parcelMap = new Map();

                areas.forEach(area => {
                    const compresa = area.Compresa || 'Sconosciuta';
                    const parcel = area.Particella || '?';
                    const cp = area.CP || `${compresa}-${parcel}`;

                    // Track parcel counts by CP
                    parcelCounts[cp] = (parcelCounts[cp] || 0) + 1;

                    // Store compresa/parcel mapping for display
                    if (!parcelMap.has(cp)) {
                        parcelMap.set(cp, { compresa, parcel });
                    }
                });

                // Create markers for each sample area
                areas.forEach(area => {
                    const compresa = area.Compresa || 'Sconosciuta';
                    const parcel = area.Particella || '?';
                    const cp = area.CP || `${compresa}-${parcel}`;
                    const lat = parseFloat(area.Lat);
                    const lon = parseFloat(area.Lon);
                    const altitude = area.Quota || '?';

                    const marker = L.circleMarker([lat, lon], {
                        radius: 6,
                        fillColor: '#ffd700',  // Yellow for visibility
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    });

                    marker.bindTooltip(
                        `<b>Area di saggio</b><br>Compresa: ${compresa}<br>Particella: ${parcel}<br>Quota: ${altitude} m`,
                        { direction: 'top', offset: [0, -5] }
                    );

                    allMarkers.push({ marker, compresa, parcel, cp });

                    // Initialize visibility state
                    parcelVisible[cp] = true;
                });

                return { total: areas.length, parcelMap };
            });
    }

    function updateAreaVisibility() {
        allMarkers.forEach(({ marker, cp }) => {
            const shouldShow = parcelVisible[cp];
            const isInLayer = areaLayer.hasLayer(marker);
            if (shouldShow && !isInLayer) {
                areaLayer.addLayer(marker);
            } else if (!shouldShow && isInLayer) {
                areaLayer.removeLayer(marker);
            }
        });
    }

    function updateParticelle(parcelMap) {
        const list = $('parcel-list');
        list.innerHTML = '';

        // Sort parcels by compresa, then by parcel ID
        const sortedParcels = Array.from(parcelMap.keys()).sort((a, b) => {
            const infoA = parcelMap.get(a);
            const infoB = parcelMap.get(b);

            // First sort by compresa
            const compresaCompare = infoA.compresa.localeCompare(infoB.compresa);
            if (compresaCompare !== 0) return compresaCompare;

            // Then sort by parcel ID
            const numA = parseInt(infoA.parcel);
            const numB = parseInt(infoB.parcel);
            if (!isNaN(numA) && !isNaN(numB)) {
                return numA - numB;
            }
            return infoA.parcel.localeCompare(infoB.parcel);
        });

        sortedParcels.forEach(cp => {
            const info = parcelMap.get(cp);
            const count = parcelCounts[cp];
            const label = `${info.compresa} / ${info.parcel}`;

            const item = document.createElement('label');
            item.className = 'checkbox-row parcel-item';
            item.innerHTML = `
                <input type="checkbox" ${parcelVisible[cp] ? 'checked' : ''}
                       onchange="SampleAreaViewer.toggleParticella('${cp}', this.checked)" />
                <span class="parcel-name">${label}</span>
                <span class="parcel-count">(${count})</span>
            `;
            list.appendChild(item);
        });
    }

    function updateStats(data) {
        const stats = $('stats');
        const parcelCount = data.parcelMap.size;

        let html = `<p>Aree di saggio: <b>${data.total}</b></p>`;
        html += `<p>Particelle: <b>${parcelCount}</b></p>`;

        stats.innerHTML = html;
    }

    return {
        init() {
            map = L.map('map', { preferCanvas: true });
            currentBasemap = basemaps.satellite().addTo(map);

            // Create layer group for area markers
            areaLayer = L.layerGroup().addTo(map);

            // Coordinate display
            map.on('mousemove', e => {
                $('coords').textContent = `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
            });

            // Load data sequentially (parcels first, then sample areas)
            (async () => {
                try {
                    const parcelCount = await loadParcels();
                    const areaData = await loadSampleAreas();

                    updateParticelle(areaData.parcelMap);
                    updateStats(areaData);

                    // Fit bounds to parcels
                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        map.fitBounds(parcelLayer.getBounds());
                    }

                    // Show areas after bounds are set
                    updateAreaVisibility();

                    updateStatus(`Caricate ${parcelCount} particelle, ${areaData.total} aree di saggio`);
                } catch (err) {
                    updateStatus('Errore nel caricamento: ' + err.message);
                    console.error(err);
                }
            })();
        },

        setBasemap(name) {
            map.removeLayer(currentBasemap);
            currentBasemap = basemaps[name]().addTo(map);
        },

        toggleParticella(cp, visible) {
            parcelVisible[cp] = visible;
            updateAreaVisibility();
        },

        showAllParcelle() {
            Object.keys(parcelVisible).forEach(cp => {
                parcelVisible[cp] = true;
            });
            updateAreaVisibility();
            // Re-render the parcel list to update checkboxes
            const parcelMap = new Map();
            allMarkers.forEach(({ compresa, parcel, cp }) => {
                if (!parcelMap.has(cp)) {
                    parcelMap.set(cp, { compresa, parcel });
                }
            });
            updateParticelle(parcelMap);
        },

        hideAllParcelle() {
            Object.keys(parcelVisible).forEach(cp => {
                parcelVisible[cp] = false;
            });
            updateAreaVisibility();
            // Re-render the parcel list to update checkboxes
            const parcelMap = new Map();
            allMarkers.forEach(({ compresa, parcel, cp }) => {
                if (!parcelMap.has(cp)) {
                    parcelMap.set(cp, { compresa, parcel });
                }
            });
            updateParticelle(parcelMap);
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => SampleAreaViewer.init());
