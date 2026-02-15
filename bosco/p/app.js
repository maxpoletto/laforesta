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

    // Vegetation index colormap: uint8 [0,255] → [r, g, b]
    // 0 (-1.0) = brown, 128 (0.0) = beige, 255 (+1.0) = dark green
    const INDEX_RAMP = [
        [0,   [139, 90, 43]],
        [128, [245, 235, 200]],
        [255, [0, 100, 0]],
    ];

    // Diverging colormap for difference display: red → white → green
    const DIFF_RAMP = [
        [0,   [180, 30, 30]],
        [128, [255, 255, 255]],
        [255, [30, 130, 30]],
    ];

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

    const SATELLITE_LAYERS = {
        ndvi: { label: 'NDVI', isIndex: true },
        ndmi: { label: 'NDMI', isIndex: true },
        evi:  { label: 'EVI',  isIndex: true },
        b08:  { label: 'B08 (NIR)',  isIndex: false },
        b04:  { label: 'B04 (Rosso)', isIndex: false },
        b02:  { label: 'B02 (Blu)',  isIndex: false },
        b11:  { label: 'B11 (SWIR)', isIndex: false },
    };

    let mapWrapper = null;
    let leafletMap = null;
    let parcelLayer = null;
    let parcelData = {};    // keyed by CP: { feature, layer, particelle, ripresa }
    let currentProperty = '';

    // Satellite state
    let satManifest = null;
    let satOverlay = null;
    let satCurrentLayer = '';  // e.g. 'ndvi'
    let satCurrentDate = '';

    // Diff state
    let diffOverlay = null;
    let diffCurrentIndex = '';
    let forestMask = null; // Uint8Array, 1 = inside forest, computed lazily

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

    // ---------------------------------------------------------------------------
    // Colormap: interpolate through a ramp of [position, [r, g, b]] stops
    // ---------------------------------------------------------------------------

    function colormapLookup(ramp, val) {
        if (val <= ramp[0][0]) return ramp[0][1];
        for (let i = 1; i < ramp.length; i++) {
            if (val <= ramp[i][0]) {
                const t = (val - ramp[i - 1][0]) / (ramp[i][0] - ramp[i - 1][0]);
                const a = ramp[i - 1][1], b = ramp[i][1];
                return [
                    Math.round(a[0] + t * (b[0] - a[0])),
                    Math.round(a[1] + t * (b[1] - a[1])),
                    Math.round(a[2] + t * (b[2] - a[2])),
                ];
            }
        }
        return ramp[ramp.length - 1][1];
    }

    // ---------------------------------------------------------------------------
    // Data loading
    // ---------------------------------------------------------------------------

    function buildTooltip(particelle, ripresa) {
        const altMin = parseFloat(particelle['Altitudine min']);
        const altMax = parseFloat(particelle['Altitudine max']);
        const altMedia = (!isNaN(altMin) && !isNaN(altMax))
            ? ((altMin + altMax) / 2).toFixed(0)
            : '?';

        let html = '<b>' + makeCP(particelle) + '</b><br>'
            + 'Area: ' + particelle['Area (ha)'] + ' ha<br>'
            + 'Età media: ' + particelle['Età media'] + ' anni<br>'
            + 'Governo: ' + particelle['Governo'] + '<br>'
            + 'Altitudine media: ' + altMedia + ' m<br>'
            + 'Esposizione: ' + particelle['Esposizione'] + '<br>'
            + 'Pendenza: ' + particelle['Pendenza %'] + '%';

        if (ripresa) {
            html += '<br>Vol tot: ' + ripresa['Vol tot (m³)'] + ' m³'
                + '<br>Vol/ha: ' + ripresa['Vol/ha (m³/ha)'] + ' m³/ha'
                + '<br>Prelievo: ' + ripresa['Prelievo (m³)'] + ' m³';
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
            fetch('../data/ripresa.csv').then(r => r.text()),
            fetch('../data/satellite/manifest.json').then(r => r.json()),
        ]).then(([geojson, particelleCsv, ripresaCsv, manifest]) => {
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

            // Initialize satellite
            satManifest = manifest;
            initSatelliteDateSelector();
            initDiffYearSelectors();

            return Object.keys(parcelData).length;
        });
    }

    // ---------------------------------------------------------------------------
    // Satellite layer display
    // ---------------------------------------------------------------------------

    function initSatelliteDateSelector() {
        const sel = $('satellite-date');
        sel.textContent = '';
        satManifest.dates.forEach(date => {
            const opt = document.createElement('option');
            opt.value = date;
            opt.textContent = date;
            sel.appendChild(opt);
        });
        // Default to most recent
        satCurrentDate = satManifest.dates[satManifest.dates.length - 1];
        sel.value = satCurrentDate;
    }

    function removeSatOverlay() {
        if (satOverlay) {
            leafletMap.removeLayer(satOverlay);
            satOverlay = null;
        }
    }

    async function showSatelliteLayer(layerName, date) {
        satCurrentLayer = layerName;
        satCurrentDate = date;
        removeSatOverlay();

        const url = '../data/satellite/' + date + '/' + layerName + '.tif';
        updateStatus('Caricamento ' + layerName.toUpperCase() + ' ' + date + '...');

        const { raster, width, height } = await loadRaster(url);

        // Render to canvas
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(width, height);

        const isIndex = SATELLITE_LAYERS[layerName].isIndex;
        for (let i = 0; i < raster.length; i++) {
            const val = raster[i];
            const px = i * 4;
            if (isIndex) {
                const [r, g, b] = colormapLookup(INDEX_RAMP, val);
                imgData.data[px] = r;
                imgData.data[px + 1] = g;
                imgData.data[px + 2] = b;
            } else {
                imgData.data[px] = val;
                imgData.data[px + 1] = val;
                imgData.data[px + 2] = val;
            }
            imgData.data[px + 3] = 200; // semi-transparent
        }
        ctx.putImageData(imgData, 0, 0);

        satOverlay = L.imageOverlay(
            canvas.toDataURL(),
            satManifest.bbox_leaflet,
            { opacity: 0.85 }
        ).addTo(leafletMap);

        // Keep parcels on top
        parcelLayer.bringToFront();

        renderSatelliteLegend(layerName, isIndex);
        updateStatus(SATELLITE_LAYERS[layerName].label + ' — ' + date);
    }

    function renderSatelliteLegend(layerName, isIndex) {
        const legend = $('legend');
        legend.textContent = '';

        const container = document.createElement('div');
        container.style.marginTop = '8px';

        const title = document.createElement('div');
        title.style.cssText = 'font-size:12px;font-weight:bold;margin-bottom:4px';
        title.textContent = SATELLITE_LAYERS[layerName].label + ' — ' + satCurrentDate;
        container.appendChild(title);

        // Gradient bar
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;height:16px;border:1px solid #ccc;border-radius:2px;overflow:hidden';
        const STEPS = 30;
        for (let i = 0; i <= STEPS; i++) {
            const cell = document.createElement('div');
            const val = Math.round(i / STEPS * 255);
            let r, g, b;
            if (isIndex) {
                [r, g, b] = colormapLookup(INDEX_RAMP, val);
            } else {
                r = g = b = val;
            }
            cell.style.cssText = 'flex:1;background:rgb(' + r + ',' + g + ',' + b + ')';
            bar.appendChild(cell);
        }
        container.appendChild(bar);

        // Labels
        const labels = document.createElement('div');
        labels.style.cssText = 'display:flex;justify-content:space-between;font-size:11px;margin-top:2px';
        const labelTexts = isIndex
            ? ['-1.0', '-0.5', '0', '+0.5', '+1.0']
            : ['0', '0.25', '0.5', '0.75', '1.0'];
        labelTexts.forEach(t => {
            const span = document.createElement('span');
            span.textContent = t;
            labels.appendChild(span);
        });
        container.appendChild(labels);

        legend.appendChild(container);
    }

    // ---------------------------------------------------------------------------
    // Shared GeoTIFF loading
    // ---------------------------------------------------------------------------

    async function loadRaster(url) {
        const resp = await fetch(url);
        const buf = await resp.arrayBuffer();
        const tiff = await GeoTIFF.fromArrayBuffer(buf);
        const image = await tiff.getImage();
        const [raster] = await image.readRasters();
        return { raster, width: image.getWidth(), height: image.getHeight() };
    }

    // Decode uint8 index value to real [-1, 1]
    function uint8ToIndex(v) {
        return v / 127.5 - 1;
    }

    // ---------------------------------------------------------------------------
    // Difference display
    // ---------------------------------------------------------------------------

    function initDiffYearSelectors() {
        const years = satManifest.dates.map(d => d.slice(0, 4));
        ['diff-year1', 'diff-year2'].forEach((id, i) => {
            const sel = $(id);
            sel.textContent = '';
            satManifest.dates.forEach((date, j) => {
                const opt = document.createElement('option');
                opt.value = date;
                opt.textContent = years[j];
                sel.appendChild(opt);
            });
            // Default: year1 = earliest, year2 = latest
            sel.value = i === 0
                ? satManifest.dates[0]
                : satManifest.dates[satManifest.dates.length - 1];
        });
    }

    function removeDiffOverlay() {
        if (diffOverlay) {
            leafletMap.removeLayer(diffOverlay);
            diffOverlay = null;
        }
    }

    function getForestMask() {
        if (forestMask) return forestMask;

        const { width, height, bbox_leaflet } = satManifest;
        const south = bbox_leaflet[0][0], west = bbox_leaflet[0][1];
        const north = bbox_leaflet[1][0], east = bbox_leaflet[1][1];

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        // Rasterize all parcel polygons
        ctx.fillStyle = '#fff';
        Object.values(parcelData).forEach(({ feature }) => {
            const geom = feature.geometry;
            const polygons = geom.type === 'Polygon'
                ? [geom.coordinates]
                : geom.coordinates;

            polygons.forEach(rings => {
                ctx.beginPath();
                rings.forEach(ring => {
                    ring.forEach(([lon, lat], k) => {
                        const x = (lon - west) / (east - west) * width;
                        const y = (north - lat) / (north - south) * height;
                        if (k === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                    ctx.closePath();
                });
                ctx.fill('evenodd');
            });
        });

        // Read back as boolean mask
        const imgData = ctx.getImageData(0, 0, width, height);
        forestMask = new Uint8Array(width * height);
        for (let i = 0; i < forestMask.length; i++) {
            forestMask[i] = imgData.data[i * 4] > 0 ? 1 : 0;
        }
        return forestMask;
    }

    const OUTSIDE_ALPHA = 60;
    const INSIDE_ALPHA = 210;

    async function showDiff(indexName, date1, date2) {
        removeDiffOverlay();

        const label = indexName.toUpperCase();
        updateStatus('Caricamento ' + label + ' ' + date1.slice(0,4) + ' e ' + date2.slice(0,4) + '...');

        const url1 = '../data/satellite/' + date1 + '/' + indexName + '.tif';
        const url2 = '../data/satellite/' + date2 + '/' + indexName + '.tif';
        const [r1, r2] = await Promise.all([loadRaster(url1), loadRaster(url2)]);

        const limitToForest = $('diff-forest-only').checked;
        const mask = limitToForest ? getForestMask() : null;

        // Compute pixel-wise difference (anno2 - anno1) in real index units
        const n = r1.raster.length;
        const diff = new Float32Array(n);
        let minDiff = Infinity, maxDiff = -Infinity;
        for (let i = 0; i < n; i++) {
            const d = uint8ToIndex(r2.raster[i]) - uint8ToIndex(r1.raster[i]);
            diff[i] = d;
            // When limiting to forest, only forest pixels affect the scale
            if (!mask || mask[i]) {
                if (d < minDiff) minDiff = d;
                if (d > maxDiff) maxDiff = d;
            }
        }
        const maxAbs = Math.max(Math.abs(minDiff), Math.abs(maxDiff)) || 0.01;

        // Render to canvas with diverging colormap
        const canvas = document.createElement('canvas');
        canvas.width = r1.width;
        canvas.height = r1.height;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(r1.width, r1.height);

        for (let i = 0; i < n; i++) {
            const normalized = Math.round(((diff[i] / maxAbs) + 1) * 127.5);
            const clamped = Math.max(0, Math.min(255, normalized));
            const [r, g, b] = colormapLookup(DIFF_RAMP, clamped);
            const px = i * 4;
            imgData.data[px] = r;
            imgData.data[px + 1] = g;
            imgData.data[px + 2] = b;
            imgData.data[px + 3] = mask ? (mask[i] ? INSIDE_ALPHA : OUTSIDE_ALPHA) : INSIDE_ALPHA;
        }
        ctx.putImageData(imgData, 0, 0);

        diffOverlay = L.imageOverlay(
            canvas.toDataURL(),
            satManifest.bbox_leaflet,
            { opacity: 0.85 }
        ).addTo(leafletMap);

        parcelLayer.bringToFront();

        renderDiffLegend(indexName, date1, date2, maxAbs);
        updateStatus(label + ' ' + date2.slice(0,4) + ' \u2212 ' + label + ' ' + date1.slice(0,4));
    }

    function renderDiffLegend(indexName, date1, date2, maxAbs) {
        const legend = $('diff-legend');
        legend.textContent = '';

        const container = document.createElement('div');
        container.style.marginTop = '8px';

        const label = indexName.toUpperCase();
        const title = document.createElement('div');
        title.style.cssText = 'font-size:12px;font-weight:bold;margin-bottom:4px';
        title.textContent = label + ' ' + date2.slice(0,4) + ' \u2212 ' + label + ' ' + date1.slice(0,4);
        container.appendChild(title);

        // Gradient bar
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;height:16px;border:1px solid #ccc;border-radius:2px;overflow:hidden';
        const STEPS = 30;
        for (let i = 0; i <= STEPS; i++) {
            const val = Math.round(i / STEPS * 255);
            const [r, g, b] = colormapLookup(DIFF_RAMP, val);
            const cell = document.createElement('div');
            cell.style.cssText = 'flex:1;background:rgb(' + r + ',' + g + ',' + b + ')';
            bar.appendChild(cell);
        }
        container.appendChild(bar);

        // Labels (symmetric around 0)
        const labels = document.createElement('div');
        labels.style.cssText = 'display:flex;justify-content:space-between;font-size:11px;margin-top:2px';
        const absStr = maxAbs.toFixed(2);
        const halfStr = (maxAbs / 2).toFixed(2);
        ['-' + absStr, '-' + halfStr, '0', '+' + halfStr, '+' + absStr].forEach(t => {
            const span = document.createElement('span');
            span.textContent = t;
            labels.appendChild(span);
        });
        container.appendChild(labels);

        legend.appendChild(container);
    }

    function applyDiff(indexName) {
        diffCurrentIndex = indexName;
        const opts = $('diff-options');

        if (!indexName) {
            opts.classList.add('hidden');
            removeDiffOverlay();
            $('diff-legend').textContent = '';
            return;
        }

        opts.classList.remove('hidden');

        // Clear the property satellite overlay so they don't stack
        removeSatOverlay();
        $('property-select').value = '';
        $('satellite-date-row').style.display = 'none';
        $('legend').textContent = '';
        Object.values(parcelData).forEach(({ layer }) =>
            layer.setStyle({ ...DEFAULT_STYLE, fillOpacity: 0 }));
        currentProperty = '';

        showDiff(indexName, $('diff-year1').value, $('diff-year2').value);
    }

    // ---------------------------------------------------------------------------
    // Property application (parcel coloring + satellite)
    // ---------------------------------------------------------------------------

    function applyProperty(propKey) {
        currentProperty = propKey;
        const dateRow = $('satellite-date-row');

        // Clear diff when a property is selected
        if (propKey) {
            removeDiffOverlay();
            $('diff-select').value = '';
            $('diff-options').classList.add('hidden');
            $('diff-legend').textContent = '';
            diffCurrentIndex = '';
        }

        if (propKey.startsWith('sat:')) {
            // Satellite layer
            const layerName = propKey.slice(4);
            dateRow.style.display = '';
            // Reset parcel fill to transparent so satellite shows through
            Object.values(parcelData).forEach(({ layer }) =>
                layer.setStyle({ ...DEFAULT_STYLE, fillOpacity: 0 }));
            showSatelliteLayer(layerName, satCurrentDate);
            return;
        }

        // Non-satellite: hide date selector, remove overlay
        dateRow.style.display = 'none';
        removeSatOverlay();

        if (!propKey) {
            Object.values(parcelData).forEach(({ layer }) => layer.setStyle(DEFAULT_STYLE));
            $('legend').textContent = '';
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

    // ---------------------------------------------------------------------------
    // Legends (parcel properties)
    // ---------------------------------------------------------------------------

    function renderContinuousLegend(prop, min, max) {
        const legend = $('legend');
        legend.textContent = '';

        const container = document.createElement('div');
        container.style.marginTop = '8px';

        // Gradient bar
        const bar = document.createElement('div');
        bar.style.cssText = 'display:flex;height:16px;border:1px solid #ccc;border-radius:2px;overflow:hidden';
        const GRADIENT_STEPS = 20;
        for (let i = 0; i <= GRADIENT_STEPS; i++) {
            const cell = document.createElement('div');
            cell.style.cssText = 'flex:1;background:' + interpolateColor(i / GRADIENT_STEPS);
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
        legend.textContent = '';

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

    // ---------------------------------------------------------------------------
    // Stats
    // ---------------------------------------------------------------------------

    function updateStats() {
        const count = Object.keys(parcelData).length;
        const totalArea = Object.values(parcelData)
            .reduce((sum, { particelle }) =>
                sum + (parseFloat(particelle?.['Area (ha)']) || 0), 0);

        const stats = $('stats');
        stats.textContent = '';

        const pCount = document.createElement('p');
        const countBold = document.createElement('b');
        countBold.textContent = count;
        pCount.textContent = 'Particelle: ';
        pCount.appendChild(countBold);
        stats.appendChild(pCount);

        const pArea = document.createElement('p');
        const areaBold = document.createElement('b');
        areaBold.textContent = totalArea.toFixed(1) + ' ha';
        pArea.textContent = 'Area totale: ';
        pArea.appendChild(areaBold);
        stats.appendChild(pArea);
    }

    // ---------------------------------------------------------------------------
    // Satellite info modal
    // ---------------------------------------------------------------------------

    function initInfoModal() {
        const modal = $('sat-info-modal');
        function open(e) { e.preventDefault(); modal.classList.remove('hidden'); }
        function close(e) { e.preventDefault(); modal.classList.add('hidden'); }
        $('info-toggle-prop').addEventListener('click', open);
        $('info-toggle-diff').addEventListener('click', open);
        $('sat-info-close').addEventListener('click', close);
        modal.addEventListener('click', function(e) {
            if (e.target === modal) close(e);
        });
    }

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    return {
        init() {
            mapWrapper = MapCommon.create('map', { basemap: 'satellite' });
            leafletMap = mapWrapper.getLeafletMap();
            initInfoModal();

            (async () => {
                try {
                    const count = await loadData();
                    updateStats();

                    if (parcelLayer && parcelLayer.getBounds().isValid()) {
                        leafletMap.fitBounds(parcelLayer.getBounds());
                    }

                    updateStatus('Caricate ' + count + ' particelle');
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
        },

        setSatelliteDate(date) {
            satCurrentDate = date;
            if (currentProperty.startsWith('sat:')) {
                const layerName = currentProperty.slice(4);
                showSatelliteLayer(layerName, date);
            }
        },

        setDiff(indexName) {
            applyDiff(indexName);
        },

        updateDiff() {
            if (diffCurrentIndex) {
                showDiff(diffCurrentIndex, $('diff-year1').value, $('diff-year2').value);
            }
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelProps.init());
