// Parcel Properties Viewer - Display forest parcels color-coded by property
const ParcelProps = (function() {
    'use strict';

    const DEFAULT_STYLE = {
        color: '#3388ff',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.1
    };

    // Color ramps
    const CONTINUOUS_COLORS = { low: [0, 100, 0], high: [154, 205, 50] }; // dark green → yellow-green
    const BINARY_COLORS = { Fustaia: '#228B22', Ceduo: '#FFD700' };

    // Property definitions: how to extract a value from a CSV row
    const PROPERTIES = {
        eta: {
            label: 'Età media',
            unit: 'anni',
            type: 'continuous',
            getValue: row => parseFloat(row['Età media'])
        },
        governo: {
            label: 'Governo',
            type: 'binary',
            categories: BINARY_COLORS,
            getValue: row => row['Governo']
        },
        altitudine: {
            label: 'Altitudine media',
            unit: 'm',
            type: 'continuous',
            getValue: row => {
                const min = parseFloat(row['Altitudine min']);
                const max = parseFloat(row['Altitudine max']);
                return (min + max) / 2;
            }
        }
    };

    let mapWrapper = null;
    let leafletMap = null;
    let parcelLayer = null;
    let parcelData = {};    // keyed by CP: { feature, csv row, leaflet layer }
    let currentProperty = '';

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function interpolateColor(t) {
        const r = Math.round(CONTINUOUS_COLORS.low[0] + t * (CONTINUOUS_COLORS.high[0] - CONTINUOUS_COLORS.low[0]));
        const g = Math.round(CONTINUOUS_COLORS.low[1] + t * (CONTINUOUS_COLORS.high[1] - CONTINUOUS_COLORS.low[1]));
        const b = Math.round(CONTINUOUS_COLORS.low[2] + t * (CONTINUOUS_COLORS.high[2] - CONTINUOUS_COLORS.low[2]));
        return `rgb(${r},${g},${b})`;
    }

    function buildTooltip(row) {
        const altMin = parseFloat(row['Altitudine min']);
        const altMax = parseFloat(row['Altitudine max']);
        const altMedia = (!isNaN(altMin) && !isNaN(altMax))
            ? ((altMin + altMax) / 2).toFixed(0)
            : '?';

        return `<b>${row.CP}</b><br>`
            + `Area: ${row['Area (ha)']} ha<br>`
            + `Età media: ${row['Età media']} anni<br>`
            + `Governo: ${row['Governo']}<br>`
            + `Altitudine media: ${altMedia} m<br>`
            + `Esposizione: ${row['Esposizione']}<br>`
            + `Pendenza: ${row['Pendenza %']}%`;
    }

    function loadData() {
        return Promise.all([
            fetch('../data/serra.geojson').then(r => r.json()),
            fetch('../data/particelle.csv').then(r => r.text())
        ]).then(([geojson, csvText]) => {
            const csv = Papa.parse(csvText, { header: true, skipEmptyLines: true });

            // Index CSV rows by CP
            const csvByCP = {};
            csv.data.forEach(row => {
                if (row.CP) csvByCP[row.CP] = row;
            });

            // Create GeoJSON layer and join with CSV
            parcelLayer = L.geoJSON(geojson, {
                style: DEFAULT_STYLE,
                onEachFeature(feature, layer) {
                    const cp = feature.properties.name;
                    const row = csvByCP[cp];

                    if (!row) {
                        console.error(`No CSV data for parcel: ${cp}`);
                        return;
                    }

                    parcelData[cp] = { feature, row, layer };
                    layer.bindTooltip(buildTooltip(row), {
                        direction: 'top',
                        offset: [0, -5]
                    });
                }
            }).addTo(leafletMap);

            return Object.keys(parcelData).length;
        });
    }

    function applyProperty(propKey) {
        currentProperty = propKey;

        if (!propKey) {
            // Reset to default style
            Object.values(parcelData).forEach(({ layer }) => layer.setStyle(DEFAULT_STYLE));
            $('legend').innerHTML = '';
            return;
        }

        const prop = PROPERTIES[propKey];

        if (prop.type === 'continuous') {
            applyContinuous(prop);
        } else {
            applyBinary(prop);
        }
    }

    function applyContinuous(prop) {
        // Collect valid values to find min/max
        const entries = [];
        Object.values(parcelData).forEach(({ row, layer }) => {
            const val = prop.getValue(row);
            if (isNaN(val)) {
                console.error(`Missing ${prop.label} for parcel: ${row.CP}`);
                layer.setStyle({ ...DEFAULT_STYLE, fillColor: '#999', fillOpacity: 0.4 });
            } else {
                entries.push({ val, layer });
            }
        });

        if (entries.length === 0) return;

        const min = Math.min(...entries.map(e => e.val));
        const max = Math.max(...entries.map(e => e.val));
        const range = max - min || 1;

        entries.forEach(({ val, layer }) => {
            const t = (val - min) / range;
            layer.setStyle({
                ...DEFAULT_STYLE,
                fillColor: interpolateColor(t),
                fillOpacity: 0.6
            });
        });

        renderContinuousLegend(prop, min, max);
    }

    function applyBinary(prop) {
        Object.values(parcelData).forEach(({ row, layer }) => {
            const val = prop.getValue(row);
            const color = prop.categories[val];
            if (!color) {
                console.error(`Unknown ${prop.label} value "${val}" for parcel: ${row.CP}`);
                layer.setStyle({ ...DEFAULT_STYLE, fillColor: '#999', fillOpacity: 0.4 });
            } else {
                layer.setStyle({
                    ...DEFAULT_STYLE,
                    fillColor: color,
                    fillOpacity: 0.6
                });
            }
        });

        renderBinaryLegend(prop);
    }

    function renderContinuousLegend(prop, min, max) {
        const legend = $('legend');
        legend.innerHTML = '';

        const container = document.createElement('div');
        container.style.marginTop = '8px';

        // Gradient bar
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;height:16px;border:1px solid #ccc;border-radius:2px;overflow:hidden';
        const GRADIENT_STEPS = 20;
        for (let i = 0; i <= GRADIENT_STEPS; i++) {
            const cell = document.createElement('div');
            cell.style.cssText = `flex:1;background:${interpolateColor(i / GRADIENT_STEPS)}`;
            bar.appendChild(cell);
        }
        container.appendChild(bar);

        // Labels
        const labels = document.createElement('div');
        labels.style.cssText = 'display:flex;justify-content:space-between;font-size:11px;margin-top:2px';
        const LABEL_STEPS = 5;
        for (let i = 0; i <= LABEL_STEPS; i++) {
            const span = document.createElement('span');
            const val = min + (max - min) * i / LABEL_STEPS;
            span.textContent = val.toFixed(0);
            labels.appendChild(span);
        }
        container.appendChild(labels);

        // Unit
        const unit = document.createElement('div');
        unit.style.cssText = 'text-align:center;font-size:11px;color:#666;margin-top:2px';
        unit.textContent = prop.unit;
        container.appendChild(unit);

        legend.appendChild(container);
    }

    function renderBinaryLegend(prop) {
        const legend = $('legend');
        legend.innerHTML = '';

        const container = document.createElement('div');
        container.style.marginTop = '8px';

        Object.entries(prop.categories).forEach(([label, color]) => {
            const row = document.createElement('div');
            row.style.cssText = 'display:flex;align-items:center;gap:8px;font-size:12px;margin:4px 0';

            const dot = document.createElement('span');
            dot.className = 'color-dot';
            dot.style.background = color;
            dot.style.display = 'inline-block';
            row.appendChild(dot);

            const text = document.createElement('span');
            text.textContent = label;
            row.appendChild(text);

            container.appendChild(row);
        });

        legend.appendChild(container);
    }

    function updateStats() {
        const count = Object.keys(parcelData).length;
        const totalArea = Object.values(parcelData)
            .reduce((sum, { row }) => sum + (parseFloat(row['Area (ha)']) || 0), 0);

        const stats = $('stats');
        stats.innerHTML = '';

        const pCount = document.createElement('p');
        pCount.innerHTML = `Particelle: <b>${count}</b>`;
        stats.appendChild(pCount);

        const pArea = document.createElement('p');
        pArea.innerHTML = `Area totale: <b>${totalArea.toFixed(1)} ha</b>`;
        stats.appendChild(pArea);
    }

    return {
        init() {
            mapWrapper = MapCommon.create('map', { basemap: 'satellite' });
            leafletMap = mapWrapper.getLeafletMap();

            (async () => {
                try {
                    const count = await loadData();
                    updateStats();

                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        leafletMap.fitBounds(parcelLayer.getBounds());
                    }

                    updateStatus(`Caricate ${count} particelle`);
                } catch (err) {
                    updateStatus('Errore nel caricamento: ' + err.message);
                    console.error(err);
                }
            })();
        },

        setBasemap(name) {
            mapWrapper.setBasemap(name);
        },

        setProperty(propKey) {
            applyProperty(propKey);
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelProps.init());
