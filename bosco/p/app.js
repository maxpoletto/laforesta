// Parcel Properties Viewer - Display forest parcels color-coded by property
const ParcelProps = (function() {
    'use strict';

    const DEFAULT_STYLE = {
        color: '#3388ff',
        weight: 2,
        opacity: 0.8,
        fillOpacity: 0.1
    };
    const NO_DATA_STYLE = { ...DEFAULT_STYLE, fillColor: '#ccc', fillOpacity: 0.3 };

    // Color ramps
    const CONTINUOUS_COLORS = { low: [0, 100, 0], high: [154, 205, 50] }; // dark green → yellow-green
    const BINARY_COLORS = { Fustaia: '#228B22', Ceduo: '#FFD700' };

    // Property definitions: getValue receives { particelle, ripresa } source rows
    const PROPERTIES = {
        eta: {
            label: 'Età media',
            unit: 'anni',
            type: 'continuous',
            getValue: s => parseFloat(s.particelle?.['Età media'])
        },
        governo: {
            label: 'Governo',
            type: 'binary',
            categories: BINARY_COLORS,
            getValue: s => s.particelle?.['Governo']
        },
        altitudine: {
            label: 'Altitudine media',
            unit: 'm',
            type: 'continuous',
            getValue: s => {
                const min = parseFloat(s.particelle?.['Altitudine min']);
                const max = parseFloat(s.particelle?.['Altitudine max']);
                return (min + max) / 2;
            }
        },
        vol_tot: {
            label: 'Volume totale',
            unit: 'm³',
            type: 'continuous',
            getValue: s => parseFloat(s.ripresa?.['Vol tot (m³)'])
        },
        vol_ha: {
            label: 'Volume/ha',
            unit: 'm³/ha',
            type: 'continuous',
            getValue: s => parseFloat(s.ripresa?.['Vol/ha (m³/ha)'])
        },
        prelievo: {
            label: 'Prelievo',
            unit: 'm³',
            type: 'continuous',
            getValue: s => parseFloat(s.ripresa?.['Prelievo (m³)'])
        }
    };

    let mapWrapper = null;
    let leafletMap = null;
    let parcelLayer = null;
    let parcelData = {};    // keyed by CP: { feature, layer, particelle, ripresa }
    let currentProperty = '';

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function makeCP(row) {
        return row.Compresa + '-' + row.Particella;
    }

    function interpolateColor(t) {
        const r = Math.round(CONTINUOUS_COLORS.low[0] + t * (CONTINUOUS_COLORS.high[0] - CONTINUOUS_COLORS.low[0]));
        const g = Math.round(CONTINUOUS_COLORS.low[1] + t * (CONTINUOUS_COLORS.high[1] - CONTINUOUS_COLORS.low[1]));
        const b = Math.round(CONTINUOUS_COLORS.low[2] + t * (CONTINUOUS_COLORS.high[2] - CONTINUOUS_COLORS.low[2]));
        return `rgb(${r},${g},${b})`;
    }

    function buildTooltip(particelle, ripresa) {
        const altMin = parseFloat(particelle['Altitudine min']);
        const altMax = parseFloat(particelle['Altitudine max']);
        const altMedia = (!isNaN(altMin) && !isNaN(altMax))
            ? ((altMin + altMax) / 2).toFixed(0)
            : '?';

        let html = `<b>${makeCP(particelle)}</b><br>`
            + `Area: ${particelle['Area (ha)']} ha<br>`
            + `Età media: ${particelle['Età media']} anni<br>`
            + `Governo: ${particelle['Governo']}<br>`
            + `Altitudine media: ${altMedia} m<br>`
            + `Esposizione: ${particelle['Esposizione']}<br>`
            + `Pendenza: ${particelle['Pendenza %']}%`;

        if (ripresa) {
            html += `<br>Vol tot: ${ripresa['Vol tot (m³)']} m³`
                + `<br>Vol/ha: ${ripresa['Vol/ha (m³/ha)']} m³/ha`
                + `<br>Prelievo: ${ripresa['Prelievo (m³)']} m³`;
        }

        return html;
    }

    function indexByCP(csvText) {
        const result = Papa.parse(csvText, { header: true, skipEmptyLines: true });
        const index = {};
        result.data.forEach(row => {
            if (row.Compresa && row.Particella) index[makeCP(row)] = row;
        });
        return index;
    }

    function loadData() {
        return Promise.all([
            fetch('../data/serra.geojson').then(r => r.json()),
            fetch('../data/particelle.csv').then(r => r.text()),
            fetch('../data/ripresa.csv').then(r => r.text())
        ]).then(([geojson, particelleCsv, ripresaCsv]) => {
            const particelleByCP = indexByCP(particelleCsv);
            const ripresaByCP = indexByCP(ripresaCsv);

            parcelLayer = L.geoJSON(geojson, {
                style: DEFAULT_STYLE,
                onEachFeature(feature, layer) {
                    const cp = feature.properties.name;
                    const particelle = particelleByCP[cp] || null;
                    const ripresa = ripresaByCP[cp] || null;

                    parcelData[cp] = { feature, layer, particelle, ripresa };

                    if (particelle) {
                        layer.bindTooltip(buildTooltip(particelle, ripresa), {
                            direction: 'top',
                            offset: [0, -5]
                        });
                    }
                }
            }).addTo(leafletMap);

            return Object.keys(parcelData).length;
        });
    }

    function applyProperty(propKey) {
        currentProperty = propKey;

        if (!propKey) {
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
        const withValue = [];
        const withoutValue = [];

        Object.values(parcelData).forEach(entry => {
            const val = prop.getValue(entry);
            if (isNaN(val)) {
                withoutValue.push(entry);
            } else {
                withValue.push({ val, layer: entry.layer });
            }
        });

        withoutValue.forEach(({ layer }) => layer.setStyle(NO_DATA_STYLE));

        if (withValue.length === 0) return;

        const min = Math.min(...withValue.map(e => e.val));
        const max = Math.max(...withValue.map(e => e.val));
        const range = max - min || 1;

        withValue.forEach(({ val, layer }) => {
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
        Object.values(parcelData).forEach(entry => {
            const val = prop.getValue(entry);
            const color = val && prop.categories[val];
            if (!color) {
                entry.layer.setStyle(NO_DATA_STYLE);
            } else {
                entry.layer.setStyle({
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
            .reduce((sum, { particelle }) =>
                sum + (parseFloat(particelle?.['Area (ha)']) || 0), 0);

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
