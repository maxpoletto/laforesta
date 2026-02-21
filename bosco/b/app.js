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
    const DIMMED_STYLE = { fillOpacity: 0, opacity: 0.3, color: '#999', weight: 1 };

    // Color ramps
    const CONTINUOUS_COLORS = { low: [0, 100, 0], high: [154, 205, 50] }; // dark green -> yellow-green
    const BINARY_COLORS = { Fustaia: '#228B22', Ceduo: '#FFD700' };

    // Vegetation index colormap: uint8 [0,255] -> [r, g, b]
    // 0 (-1.0) = brown, 128 (0.0) = beige, 255 (+1.0) = dark green
    const INDEX_RAMP = [
        [0,   [139, 90, 43]],
        [128, [245, 235, 200]],
        [255, [0, 100, 0]],
    ];

    // Diverging colormap for difference display: red -> white -> green
    const DIFF_RAMP = [
        [0,   [180, 30, 30]],
        [128, [255, 255, 255]],
        [255, [30, 130, 30]],
    ];

    const GRADIENT_STEPS = 30;

    // Property definitions: getValue receives { particelle, ripresa } source rows
    const PROPERTIES = {
        eta: {
            label: 'Eta media',
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
            unit: 'm\u00b3',
            type: 'continuous',
            getValue: s => parseFloat(s.ripresa?.['Vol tot (m³)'])
        },
        vol_ha: {
            label: 'Volume/ha',
            unit: 'm\u00b3/ha',
            type: 'continuous',
            getValue: s => parseFloat(s.ripresa?.['Vol/ha (m³/ha)'])
        },
        prelievo: {
            label: 'Prelievo',
            unit: 'm\u00b3',
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
    let parcelData = {};    // keyed by CP: { layers, particelle, ripresa, tooltipHtml }
    let currentProperty = '';

    // Per-compresa state
    let comprese = {};          // compresa name -> [GeoJSON features]
    let currentCompresa = null;

    // Satellite state (per-compresa, reloaded on switch)
    let satManifest = null;
    let satOverlay = null;
    let satCurrentLayer = '';  // e.g. 'ndvi'
    let satCurrentDate = '';

    // Diff state
    let diffOverlay = null;
    let diffCurrentIndex = '';

    // Precomputed data (per-compresa, reloaded on switch)
    let parcelMaskPromise = null; // resolves to { raster, width, height }
    let parcelMask = null;        // Uint8Array, 0 outside forest, parcel ID inside forest
    let timeseriesData = null;    // from timeseries.json

    // Time series chart state
    let tsChart = null;           // current Chart.js instance
    let tsParcelClickMode = false;

    function updateStatus(msg) {
        $('status').textContent = msg;
    }

    function makeCP(row) {
        return row.Compresa + '-' + row.Particella;
    }

    function rgbStr(rgb) {
        return 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    }

    /** Apply a Leaflet style to all layers of a parcel entry. */
    function setStyleAll(entry, style) {
        entry.layers.forEach(l => l.setStyle(style));
    }

    function continuousColorStr(t) {
        return rgbStr(interpolateColor(t, CONTINUOUS_COLORS.low, CONTINUOUS_COLORS.high));
    }

    /** Check if a parcel CP belongs to the currently selected compresa. */
    function isActiveParcel(cp) {
        return cp.startsWith(currentCompresa + '-');
    }

    /** Iterate only parcel entries belonging to the current compresa. */
    function forActiveEntries(fn) {
        Object.entries(parcelData).forEach(([cp, entry]) => {
            if (isActiveParcel(cp)) fn(entry, cp);
        });
    }

    // ---------------------------------------------------------------------------
    // Gradient legend builder
    // ---------------------------------------------------------------------------

    function createGradientLegend(targetEl, opts) {
        targetEl.textContent = '';

        const container = document.createElement('div');
        container.className = 'legend-container';

        if (opts.title) {
            const title = document.createElement('div');
            title.className = 'legend-title';
            title.textContent = opts.title;
            container.appendChild(title);
        }

        // Gradient bar
        const bar = document.createElement('div');
        bar.className = 'legend-bar';
        const steps = opts.steps || GRADIENT_STEPS;
        for (let i = 0; i <= steps; i++) {
            const cell = document.createElement('div');
            cell.className = 'legend-bar-cell';
            cell.style.background = rgbStr(opts.colorFn(i, steps));
            bar.appendChild(cell);
        }
        container.appendChild(bar);

        // Labels
        const labels = document.createElement('div');
        labels.className = 'legend-labels';
        opts.labelTexts.forEach(t => {
            const span = document.createElement('span');
            span.textContent = t;
            labels.appendChild(span);
        });
        container.appendChild(labels);

        if (opts.unit) {
            const unit = document.createElement('div');
            unit.className = 'legend-unit';
            unit.textContent = opts.unit;
            container.appendChild(unit);
        }

        targetEl.appendChild(container);
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
            fetch('../data/terreni.geojson').then(r => r.json()),
            fetch('../data/particelle.csv').then(r => r.text()),
            fetch('../data/ripresa.csv').then(r => r.text()),
        ]).then(([geojson, particelleCsv, ripresaCsv]) => {
            const particelleByCP = indexByCP(particelleCsv);
            const ripresaByCP = indexByCP(ripresaCsv);

            // Group features by compresa (properties.layer)
            comprese = {};
            geojson.features.forEach(f => {
                const name = f.properties.layer;
                if (!comprese[name]) comprese[name] = [];
                comprese[name].push(f);
            });

            parcelLayer = L.geoJSON(geojson, {
                style: DEFAULT_STYLE,
                onEachFeature(feature, layer) {
                    const cp = feature.properties.name;
                    const particelle = particelleByCP[cp] || null;
                    const ripresa = ripresaByCP[cp] || null;

                    // Build tooltip HTML once, store for rebinding on compresa switch
                    const tooltipHtml = particelle
                        ? buildTooltip(particelle, ripresa)
                        : null;

                    // Multi-polygon parcels: accumulate layers for same name
                    if (parcelData[cp]) {
                        parcelData[cp].layers.push(layer);
                    } else {
                        parcelData[cp] = { layers: [layer], particelle, ripresa, tooltipHtml };
                    }

                    if (tooltipHtml) {
                        layer.bindTooltip(tooltipHtml, {
                            direction: 'top',
                            offset: [0, -5]
                        });
                    }

                    layer.on('click', function() { handleParcelClick(cp); });
                }
            }).addTo(leafletMap);

            // Populate compresa selector
            const select = $('compresa-select');
            Object.keys(comprese).sort().forEach(name => {
                const count = new Set(comprese[name].map(f => f.properties.name)).size;
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name + ' (' + count + ' particelle)';
                select.appendChild(opt);
            });
            select.addEventListener('change', function() {
                switchCompresa(this.value);
            });

            return Object.keys(parcelData).length;
        });
    }

    // ---------------------------------------------------------------------------
    // Per-compresa switching
    // ---------------------------------------------------------------------------

    function satDataUrl(path) {
        return '../data/satellite/' + currentCompresa + '/' + path;
    }

    async function switchCompresa(name) {
        currentCompresa = name;

        // Restyle parcels: active vs dimmed
        Object.entries(parcelData).forEach(([cp, entry]) => {
            if (isActiveParcel(cp)) {
                setStyleAll(entry, DEFAULT_STYLE);
                // Rebind tooltip
                if (entry.tooltipHtml) {
                    entry.layers.forEach(l => {
                        l.unbindTooltip();
                        l.bindTooltip(entry.tooltipHtml, {
                            direction: 'top',
                            offset: [0, -5]
                        });
                    });
                }
            } else {
                setStyleAll(entry, DIMMED_STYLE);
                entry.layers.forEach(l => l.unbindTooltip());
            }
        });

        // Fit map to selected compresa
        const features = comprese[name];
        if (features) {
            const group = L.geoJSON({ type: 'FeatureCollection', features });
            leafletMap.fitBounds(group.getBounds());
        }

        // Clear raster state
        removeSatOverlay();
        removeDiffOverlay();
        rasterCache.clear();
        parcelMask = null;
        parcelMaskPromise = null;

        // Load satellite data for this compresa
        try {
            const [manifest, timeseries] = await Promise.all([
                fetch(satDataUrl('manifest.json')).then(r => r.json()),
                fetch(satDataUrl('timeseries.json')).then(r => r.json()),
            ]);
            satManifest = manifest;
            timeseriesData = timeseries;
            parcelMaskPromise = loadRaster(satDataUrl('parcel-mask.tif'));

            initSatelliteDateSelector();
            initDiffDateSelectors();
        } catch (err) {
            console.error('Failed to load satellite data for ' + name + ':', err);
            satManifest = null;
            timeseriesData = null;
        }

        // Re-apply current view if a property or diff is active
        if (currentProperty) {
            applyProperty(currentProperty);
        } else if (diffCurrentIndex) {
            applyDiff(diffCurrentIndex);
        } else {
            // Clear legends
            $('legend').textContent = '';
            $('diff-legend').textContent = '';
        }

        updateStats();
        updateStatus(name + ' \u2014 ' + Object.keys(parcelData).filter(isActiveParcel).length + ' particelle');
    }

    // ---------------------------------------------------------------------------
    // Satellite layer display
    // ---------------------------------------------------------------------------

    function initSatelliteDateSelector() {
        const sel = $('satellite-date');
        sel.textContent = '';
        if (!satManifest) return;
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

        if (!satManifest) return;

        const url = satDataUrl(date + '/' + layerName + '.tif');
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
            imgData.data[px + 3] = SAT_OVERLAY_ALPHA;
        }
        ctx.putImageData(imgData, 0, 0);

        satOverlay = L.imageOverlay(
            canvas.toDataURL(),
            satManifest.bbox,
            { opacity: 0.85 }
        ).addTo(leafletMap);

        // Keep parcels on top
        parcelLayer.bringToFront();

        renderSatelliteLegend(layerName, isIndex);
        updateStatus(SATELLITE_LAYERS[layerName].label + ' \u2014 ' + date);
    }

    function renderSatelliteLegend(layerName, isIndex) {
        const labelTexts = isIndex
            ? ['-1.0', '-0.5', '0', '+0.5', '+1.0']
            : ['0', '0.25', '0.5', '0.75', '1.0'];

        createGradientLegend($('legend'), {
            title: SATELLITE_LAYERS[layerName].label + ' \u2014 ' + satCurrentDate,
            colorFn(i, steps) {
                const val = Math.round(i / steps * 255);
                if (isIndex) return colormapLookup(INDEX_RAMP, val);
                return [val, val, val];
            },
            labelTexts,
        });
    }

    // ---------------------------------------------------------------------------
    // Cached GeoTIFF loading
    // ---------------------------------------------------------------------------

    const rasterCache = new Map();

    async function loadRaster(url) {
        if (rasterCache.has(url)) return rasterCache.get(url);
        const resp = await fetch(url);
        const buf = await resp.arrayBuffer();
        const tiff = await GeoTIFF.fromArrayBuffer(buf);
        const image = await tiff.getImage();
        const [raster] = await image.readRasters();
        const result = { raster, width: image.getWidth(), height: image.getHeight() };
        rasterCache.set(url, result);
        return result;
    }

    // ---------------------------------------------------------------------------
    // Difference display
    // ---------------------------------------------------------------------------

    function initDiffDateSelectors() {
        if (!satManifest) return;
        const dates = satManifest.dates.map(d => d.slice(0, 7));
        ['diff-date1', 'diff-date2'].forEach((id, i) => {
            const sel = $(id);
            sel.textContent = '';
            satManifest.dates.forEach((date, j) => {
                const opt = document.createElement('option');
                opt.value = date;
                opt.textContent = dates[j];
                sel.appendChild(opt);
            });
            // Default: date1 = earliest, date2 = latest
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

    async function getParcelMask() {
        if (parcelMask) return parcelMask;
        if (!parcelMaskPromise) return null;
        const { raster } = await parcelMaskPromise;
        parcelMask = raster;
        return parcelMask;
    }

    const SAT_OVERLAY_ALPHA = 200;
    const OUTSIDE_ALPHA = 60;
    const INSIDE_ALPHA = 210;

    async function showDiff(indexName, date1, date2) {
        removeDiffOverlay();

        if (!satManifest) return;

        const label = indexName.toUpperCase();
        updateStatus('Caricamento ' + label + ' ' + date1.slice(0,4) + ' e ' + date2.slice(0,4) + '...');

        const url1 = satDataUrl(date1 + '/' + indexName + '.tif');
        const url2 = satDataUrl(date2 + '/' + indexName + '.tif');
        const [r1, r2] = await Promise.all([loadRaster(url1), loadRaster(url2)]);

        const limitToForest = $('diff-forest-only').checked;
        const mask = limitToForest ? await getParcelMask() : null;

        const { diff, maxAbs } = computeDiff(r1.raster, r2.raster, mask);

        const canvas = document.createElement('canvas');
        canvas.width = r1.width;
        canvas.height = r1.height;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(r1.width, r1.height);
        imgData.data.set(diffToRgba(diff, maxAbs, DIFF_RAMP, mask, INSIDE_ALPHA, OUTSIDE_ALPHA));
        ctx.putImageData(imgData, 0, 0);

        diffOverlay = L.imageOverlay(
            canvas.toDataURL(),
            satManifest.bbox,
            { opacity: 0.85 }
        ).addTo(leafletMap);

        parcelLayer.bringToFront();

        renderDiffLegend(indexName, date1, date2, maxAbs);
        updateStatus(label + ' ' + date2.slice(0,4) + ' \u2212 ' + label + ' ' + date1.slice(0,4));
    }

    function renderDiffLegend(indexName, date1, date2, maxAbs) {
        const label = indexName.toUpperCase();
        const displayMax = maxAbs / 127.5; // convert from uint8 diff to index scale [-1, +1]
        const absStr = displayMax.toFixed(2);
        const halfStr = (displayMax / 2).toFixed(2);

        createGradientLegend($('diff-legend'), {
            title: label + ' ' + date2.slice(0,7) + ' \u2212 ' + label + ' ' + date1.slice(0,7),
            colorFn(i, steps) {
                return colormapLookup(DIFF_RAMP, Math.round(i / steps * 255));
            },
            labelTexts: ['-' + absStr, '-' + halfStr, '0', '+' + halfStr, '+' + absStr],
        });
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

        // Clear property/satellite state so they don't stack
        removeSatOverlay();
        $('property-select').value = '';
        $('satellite-date-row').classList.remove('satellite-date-row-visible');
        $('ts-section').classList.add('hidden');
        parcelClickModeOff();
        $('legend').textContent = '';
        // Only clear fill on active parcels; dimmed stay dimmed
        forActiveEntries(e => setStyleAll(e, { ...DEFAULT_STYLE, fillOpacity: 0 }));
        currentProperty = '';

        opts.classList.remove('hidden');
        showDiff(indexName, $('diff-date1').value, $('diff-date2').value);
    }

    // ---------------------------------------------------------------------------
    // Property application (parcel coloring + satellite)
    // ---------------------------------------------------------------------------

    function applyProperty(propKey) {
        currentProperty = propKey;

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
            $('satellite-date-row').classList.add('satellite-date-row-visible');
            $('ts-section').classList.remove('hidden');
            parcelClickModeOff();
            // Reset active parcel fill to transparent so satellite shows through
            forActiveEntries(e => setStyleAll(e, { ...DEFAULT_STYLE, fillOpacity: 0 }));
            showSatelliteLayer(layerName, satCurrentDate);
            return;
        }

        // Non-satellite: hide date selector, remove overlay
        $('satellite-date-row').classList.remove('satellite-date-row-visible');
        $('ts-section').classList.add('hidden');
        parcelClickModeOff();
        removeSatOverlay();

        if (!propKey) {
            forActiveEntries(e => setStyleAll(e, DEFAULT_STYLE));
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

        forActiveEntries(entry => {
            const val = prop.getValue(entry);
            if (isNaN(val)) {
                withoutValue.push(entry);
            } else {
                withValue.push({ val, entry });
            }
        });

        withoutValue.forEach(e => setStyleAll(e, NO_DATA_STYLE));

        if (withValue.length === 0) return;

        const min = Math.min(...withValue.map(e => e.val));
        const max = Math.max(...withValue.map(e => e.val));
        const range = max - min || 1;

        withValue.forEach(({ val, entry }) => {
            const t = (val - min) / range;
            setStyleAll(entry, {
                ...DEFAULT_STYLE,
                fillColor: continuousColorStr(t),
                fillOpacity: 0.6
            });
        });

        renderContinuousLegend(prop, min, max);
    }

    function applyBinary(prop) {
        forActiveEntries(entry => {
            const val = prop.getValue(entry);
            const color = val && prop.categories[val];
            if (!color) {
                setStyleAll(entry, NO_DATA_STYLE);
            } else {
                setStyleAll(entry, {
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
        const LABEL_STEPS = 5;
        const labelTexts = [];
        for (let i = 0; i <= LABEL_STEPS; i++) {
            labelTexts.push((min + (max - min) * i / LABEL_STEPS).toFixed(0));
        }

        createGradientLegend($('legend'), {
            colorFn(i, steps) {
                return interpolateColor(i / steps, CONTINUOUS_COLORS.low, CONTINUOUS_COLORS.high);
            },
            steps: 20,
            labelTexts,
            unit: prop.unit,
        });
    }

    function renderBinaryLegend(prop) {
        const legend = $('legend');
        legend.textContent = '';

        const container = document.createElement('div');
        container.className = 'legend-container';

        Object.entries(prop.categories).forEach(([label, color]) => {
            const row = document.createElement('div');
            row.className = 'legend-row';

            const dot = document.createElement('span');
            dot.className = 'color-dot';
            dot.style.background = color;
            row.appendChild(dot);

            const text = document.createElement('span');
            text.textContent = label;
            row.appendChild(text);

            container.appendChild(row);
        });

        legend.appendChild(container);
    }

    // ---------------------------------------------------------------------------
    // Stats (scoped to current compresa)
    // ---------------------------------------------------------------------------

    function updateStats() {
        const activeKeys = Object.keys(parcelData).filter(isActiveParcel);
        const count = activeKeys.length;
        const totalArea = activeKeys.reduce((sum, cp) => {
            const { particelle } = parcelData[cp];
            return sum + (parseFloat(particelle?.['Area (ha)']) || 0);
        }, 0);

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
    // Time series bar charts
    // ---------------------------------------------------------------------------

    const TS_BAR_COLOR = '#006400';

    function initTimeSeriesModal() {
        const modal = $('ts-modal');
        function close(e) { e.preventDefault(); modal.classList.add('hidden'); }
        $('ts-close').addEventListener('click', close);
        modal.addEventListener('click', function(e) {
            if (e.target === modal) close(e);
        });
    }

    function showTimeSeriesChart(title, values) {
        const modal = $('ts-modal');
        $('ts-title').textContent = title;
        modal.classList.remove('hidden');

        if (tsChart) { tsChart.destroy(); tsChart = null; }

        const labels = timeseriesData.dates.map(d => d.slice(0, 7));
        tsChart = new Chart($('ts-canvas'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    data: values,
                    backgroundColor: TS_BAR_COLOR,
                }],
            },
            options: {
                animation: false,
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: false } },
                    y: { beginAtZero: true },
                },
            },
        });
    }

    function currentSatelliteLayerName() {
        return currentProperty.startsWith('sat:') ? currentProperty.slice(4) : null;
    }

    function showForestTimeSeries() {
        const layer = currentSatelliteLayerName();
        if (!layer || !timeseriesData) return;
        const label = SATELLITE_LAYERS[layer].label;
        showTimeSeriesChart(
            label + ' \u2014 ' + currentCompresa,
            timeseriesData.means.forest[layer]
        );
    }

    function parcelClickModeOn() {
        tsParcelClickMode = true;
        $('ts-hint').classList.remove('hidden');
        $('ts-parcel-btn').textContent = 'Annulla';
    }

    function parcelClickModeOff() {
        tsParcelClickMode = false;
        $('ts-hint').classList.add('hidden');
        $('ts-parcel-btn').textContent = 'Per particella';
    }

    function toggleParcelTimeSeries() {
        if (tsParcelClickMode) {
            parcelClickModeOff();
        } else {
            parcelClickModeOn();
        }
    }

    function handleParcelClick(cp) {
        if (!tsParcelClickMode || !isActiveParcel(cp)) return;
        parcelClickModeOff();

        const layer = currentSatelliteLayerName();
        if (!layer || !timeseriesData) return;

        const parcelMeans = timeseriesData.means.parcels[cp];
        if (!parcelMeans) return;

        const label = SATELLITE_LAYERS[layer].label;
        showTimeSeriesChart(label + ' \u2014 ' + cp, parcelMeans[layer]);
    }

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    return {
        init() {
            mapWrapper = MapCommon.create('map', { basemap: 'satellite' });
            leafletMap = mapWrapper.getLeafletMap();
            initInfoModal();
            initTimeSeriesModal();

            (async () => {
                try {
                    const count = await loadData();
                    updateStatus('Caricate ' + count + ' particelle');

                    // Select first compresa (satellite data loaded inside switchCompresa)
                    const first = Object.keys(comprese).sort()[0];
                    if (first) {
                        $('compresa-select').value = first;
                        await switchCompresa(first);
                    }
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
                showDiff(diffCurrentIndex, $('diff-date1').value, $('diff-date2').value);
            }
        },

        showForestTimeSeries() {
            showForestTimeSeries();
        },

        toggleParcelTimeSeries() {
            toggleParcelTimeSeries();
        }
    };
})();

document.addEventListener('DOMContentLoaded', () => ParcelProps.init());
