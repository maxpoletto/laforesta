// Tree Viewer - Display trees on a map with parcel boundaries
const TreeViewer = (function() {
    'use strict';

    let map = null;
    let currentBasemap = null;
    let parcelLayer = null;
    let allTreeMarkers = [];  // Array of {marker, species, parcel}
    let treeLayer = null;     // LayerGroup containing all visible markers
    let speciesColors = {};
    let speciesVisible = {};
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

    // Color palette for species
    const colorPalette = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
        '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f'
    ];

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function getSpeciesColor(species) {
        if (!speciesColors[species]) {
            const idx = Object.keys(speciesColors).length % colorPalette.length;
            speciesColors[species] = colorPalette[idx];
        }
        return speciesColors[species];
    }

    function loadParcels() {
        return fetch('data/particelle.geojson')
            .then(r => r.json())
            .then(data => {
                // Filter to only Serra features
                const serraFeatures = data.features.filter(
                    f => f.properties?.layer === 'Serra'
                );

                parcelLayer = L.geoJSON({
                    type: 'FeatureCollection',
                    features: serraFeatures
                }, {
                    style: {
                        color: '#3388ff',
                        weight: 2,
                        opacity: 0.8,
                        fillOpacity: 0.1
                    },
                    // Do not display popups until parcel names are correct.
                    // onEachFeature: (feature, layer) => {
                    //     if (feature.properties?.name) {
                    //         layer.bindPopup(`<b>${feature.properties.name}</b>`);
                    //     }
                    // }
                }).addTo(map);

                return serraFeatures.length;
            });
    }

    function loadTrees() {
        return fetch('data/piante-accrescimento-indefinito.csv')
            .then(r => r.text())
            .then(csvText => {
                const result = Papa.parse(csvText, {
                    header: true,
                    skipEmptyLines: true
                });

                // Filter to Serra trees with valid coordinates
                const trees = result.data.filter(row =>
                    row.Compresa === 'Serra' &&
                    row.Lon && row.Lat &&
                    !isNaN(parseFloat(row.Lon)) &&
                    !isNaN(parseFloat(row.Lat))
                );

                // Group by species and collect particelle
                const bySpecies = {};
                const parcelSet = new Set();

                trees.forEach(tree => {
                    const species = tree.Genere || 'Sconosciuto';
                    const parcel = tree.Particella || '?';

                    if (!bySpecies[species]) {
                        bySpecies[species] = [];
                    }
                    bySpecies[species].push(tree);

                    parcelSet.add(parcel);
                    parcelCounts[parcel] = (parcelCounts[parcel] || 0) + 1;
                });

                // Create markers for each tree
                trees.forEach(tree => {
                    const species = tree.Genere || 'Sconosciuto';
                    const parcel = tree.Particella || '?';
                    const lat = parseFloat(tree.Lat);
                    const lon = parseFloat(tree.Lon);
                    const diameter = tree.Diametro || '?';
                    const color = getSpeciesColor(species);

                    const marker = L.circleMarker([lat, lon], {
                        radius: 5,
                        fillColor: color,
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    });

                    marker.bindTooltip(
                        `<b>${species}</b><br>D: ${diameter} cm<br>Particella: ${parcel}`,
                        { direction: 'top', offset: [0, -5] }
                    );

                    allTreeMarkers.push({ marker, species, parcel });

                    // Initialize visibility state
                    speciesVisible[species] = true;
                    parcelVisible[parcel] = true;
                });

                return { total: trees.length, bySpecies, parcels: Array.from(parcelSet).sort() };
            });
    }

    function updateTreeVisibility() {
        allTreeMarkers.forEach(({ marker, species, parcel }) => {
            const shouldShow = speciesVisible[species] && parcelVisible[parcel];
            const isInLayer = treeLayer.hasLayer(marker);
            if (shouldShow && !isInLayer) {
                treeLayer.addLayer(marker);
            } else if (!shouldShow && isInLayer) {
                treeLayer.removeLayer(marker);
            }
        });
    }

    function updateSpeciesList() {
        const list = $('species-list');
        list.innerHTML = '';

        const speciesCounts = {};
        allTreeMarkers.forEach(({ species }) => {
            speciesCounts[species] = (speciesCounts[species] || 0) + 1;
        });

        const sortedSpecies = Object.keys(speciesCounts).sort();

        sortedSpecies.forEach(species => {
            const color = speciesColors[species];
            const count = speciesCounts[species];

            const item = document.createElement('label');
            item.className = 'checkbox-row species-item';
            item.innerHTML = `
                <input type="checkbox" ${speciesVisible[species] ? 'checked' : ''}
                       onchange="TreeViewer.toggleSpecies('${species}', this.checked)" />
                <span class="color-dot" style="background: ${color}"></span>
                <span class="species-name">${species}</span>
                <span class="species-count">(${count})</span>
            `;
            list.appendChild(item);
        });
    }

    function updateParticelle() {
        const list = $('parcel-list');
        list.innerHTML = '';

        const sortedParcelle = Object.keys(parcelCounts).sort((a, b) => {
            const numA = parseInt(a);
            const numB = parseInt(b);
            if (!isNaN(numA) && !isNaN(numB)) {
                return numA - numB;
            }
            return a.localeCompare(b);
        });

        sortedParcelle.forEach(parcel => {
            const count = parcelCounts[parcel];

            const item = document.createElement('label');
            item.className = 'checkbox-row species-item';
            item.innerHTML = `
                <input type="checkbox" ${parcelVisible[parcel] ? 'checked' : ''}
                       onchange="TreeViewer.toggleParticella('${parcel}', this.checked)" />
                <span class="species-name">${parcel}</span>
                <span class="species-count">(${count})</span>
            `;
            list.appendChild(item);
        });
    }

    function updateStats(treeData) {
        const stats = $('stats');
        const total = treeData.total;
        const speciesCount = Object.keys(treeData.bySpecies).length;

        let html = `<p>Totale alberi: <b>${total}</b></p>`;
        html += `<p>Specie: <b>${speciesCount}</b></p>`;

        stats.innerHTML = html;
    }

    return {
        init() {
            map = L.map('map', { preferCanvas: true });
            currentBasemap = basemaps.satellite().addTo(map);

            // Create layer group for tree markers
            treeLayer = L.layerGroup().addTo(map);

            // Coordinate display
            map.on('mousemove', e => {
                $('coords').textContent = `${e.latlng.lat.toFixed(6)}, ${e.latlng.lng.toFixed(6)}`;
            });

            // Load data sequentially (parcels first, then trees)
            (async () => {
                try {
                    const parcelCount = await loadParcels();
                    const treeData = await loadTrees();

                    updateSpeciesList();
                    updateParticelle();
                    updateStats(treeData);

                    // Fit bounds to parcels
                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        map.fitBounds(parcelLayer.getBounds());
                    }

                    // Show trees after bounds are set
                    updateTreeVisibility();

                    updateStatus(`Caricate ${parcelCount} particelle, ${treeData.total} alberi`);
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

        toggleSpecies(species, visible) {
            speciesVisible[species] = visible;
            updateTreeVisibility();
        },

        toggleParticella(parcel, visible) {
            parcelVisible[parcel] = visible;
            updateTreeVisibility();
        },

        showAllSpecies() {
            Object.keys(speciesVisible).forEach(species => {
                speciesVisible[species] = true;
            });
            updateTreeVisibility();
            updateSpeciesList();
        },

        hideAllSpecies() {
            Object.keys(speciesVisible).forEach(species => {
                speciesVisible[species] = false;
            });
            updateTreeVisibility();
            updateSpeciesList();
        },

        showAllParcelle() {
            Object.keys(parcelVisible).forEach(parcel => {
                parcelVisible[parcel] = true;
            });
            updateTreeVisibility();
            updateParticelle();
        },

        hideAllParcelle() {
            Object.keys(parcelVisible).forEach(parcel => {
                parcelVisible[parcel] = false;
            });
            updateTreeVisibility();
            updateParticelle();
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => TreeViewer.init());
