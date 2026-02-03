// Sample Area Viewer - Display sample areas on a map with parcel boundaries
const SampleAreaViewer = (function() {
    'use strict';

    let mapWrapper = null;
    let leafletMap = null;
    let parcelLayer = null;
    let allMarkers = [];  // Array of {marker, compresa, parcel, cp}
    let areaLayer = null;  // LayerGroup containing all visible markers
    let parcelVisible = {};
    let parcelData = {};
    let numAreas = {};

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function loadParcelsGeo() {
        return fetch('../data/particelle.geojson')
            .then(r => r.json())
            .then(data => {
                parcelLayer = L.geoJSON(data, {
                    style: {
                        color: '#3388ff',
                        weight: 2,
                        opacity: 0.8,
                        fillOpacity: 0.1
                    }
                }).addTo(leafletMap);

                return data.features.length;
            });
    }

    function loadParcels() {
        return fetch('../data/particelle.csv')
            .then(r => r.text())
            .then(csvText => {
                const result = Papa.parse(csvText, {
                    header: true,
                    skipEmptyLines: true
                });
                result.data.forEach(row => {
                    const compresa = row['Compresa'] || 'Sconosciuta';
                    const parcel = row['Particella'] || '?';
                    const cp = `${compresa}-${parcel}`;
                    parcelData[cp] = { area: row['Area (ha)'] };
                });
                return parcelData;
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
                const parcels = new Map();

                areas.forEach(area => {
                    const compresa = area['Compresa'] || 'Sconosciuta';
                    const parcel = area['Particella'] || '?';
                    const cp = `${compresa}-${parcel}`;
                    numAreas[cp] = (numAreas[cp] || 0) + 1;
                    if (!parcels.has(cp)) {
                        parcels.set(cp, { compresa, parcel });
                    }
                });

                // Create markers for each sample area
                areas.forEach(area => {
                    const compresa = area['Compresa'] || 'Sconosciuta';
                    const parcel = area['Particella'] || '?';
                    const ads = area['Area saggio'] || '?';
                    const cp = `${compresa}-${parcel}`;
                    const lat = parseFloat(area.Lat);
                    const lon = parseFloat(area.Lon);
                    const ha = parcelData[cp]?.area || '?';
                    const altitude = area['Quota'] || '?';

                    const marker = L.circleMarker([lat, lon], {
                        radius: 6,
                        fillColor: '#ffd700',
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    });

                    marker.bindTooltip(
                        `<b>Area di saggio ${ads}</b><br>Particella: ${parcel} (${ha} ha)<br>Quota: ${altitude} m`,
                        { direction: 'top', offset: [0, -5] }
                    );

                    allMarkers.push({ marker, compresa, parcel, cp });
                    parcelVisible[cp] = true;
                });

                return { total: areas.length, parcels };
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

    function updateParticelle(parcels) {
        const list = $('parcel-list');
        list.innerHTML = '';

        // Sort parcels by compresa, then by parcel ID
        const sortedParcels = Array.from(parcels.keys()).sort((a, b) => {
            const infoA = parcels.get(a);
            const infoB = parcels.get(b);

            const compresaCompare = infoA.compresa.localeCompare(infoB.compresa);
            if (compresaCompare !== 0) return compresaCompare;

            const numA = parseInt(infoA.parcel);
            const numB = parseInt(infoB.parcel);
            if (!isNaN(numA) && !isNaN(numB)) {
                return numA - numB;
            }
            return infoA.parcel.localeCompare(infoB.parcel);
        });

        sortedParcels.forEach(cp => {
            const info = parcels.get(cp);
            const count = numAreas[cp];
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
        const parcelCount = data.parcels.size;

        let html = `<p>Aree di saggio: <b>${data.total}</b></p>`;
        html += `<p>Particelle: <b>${parcelCount}</b></p>`;

        stats.innerHTML = html;
    }

    return {
        init() {
            // Create map with shared features (measure, location, coords)
            mapWrapper = MapCommon.create('map', {
                basemap: 'satellite'
            });
            leafletMap = mapWrapper.getLeafletMap();

            // Create layer group for area markers
            areaLayer = L.layerGroup().addTo(leafletMap);

            // Load data
            (async () => {
                try {
                    const parcelCount = await loadParcelsGeo();
                    await loadParcels();
                    const areaData = await loadSampleAreas();

                    updateParticelle(areaData.parcels);
                    updateStats(areaData);

                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        leafletMap.fitBounds(parcelLayer.getBounds());
                    }

                    updateAreaVisibility();
                    updateStatus(`Caricate ${parcelCount} particelle, ${areaData.total} aree di saggio`);
                } catch (err) {
                    updateStatus('Errore nel caricamento: ' + err.message);
                    console.error(err);
                }
            })();
        },

        setBasemap(name) {
            mapWrapper.setBasemap(name);
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
            const parcels = new Map();
            allMarkers.forEach(({ compresa, parcel, cp }) => {
                if (!parcels.has(cp)) {
                    parcels.set(cp, { compresa, parcel });
                }
            });
            updateParticelle(parcels);
        },

        hideAllParcelle() {
            Object.keys(parcelVisible).forEach(cp => {
                parcelVisible[cp] = false;
            });
            updateAreaVisibility();
            const parcels = new Map();
            allMarkers.forEach(({ compresa, parcel, cp }) => {
                if (!parcels.has(cp)) {
                    parcels.set(cp, { compresa, parcel });
                }
            });
            updateParticelle(parcels);
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => SampleAreaViewer.init());
