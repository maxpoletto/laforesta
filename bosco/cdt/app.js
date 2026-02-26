// Calendario di taglio — harvest schedule Gantt chart
(function() {
    'use strict';

    const FILES = [
        { path: '../data/calendario-2011-2025-cropanese.csv', label: 'Calendario tagli 2011\u20132025 (Cropanese)' },
        { path: '../data/calendario-2011-2025-malvaso.csv', label: 'Calendario tagli 2011\u20132025 (Malvaso)' },
        { path: '../data/registro-gestione-2016-2025.csv', label: 'Registro gestione 2016\u20132025' },
        { path: '../data/calendario-mannesi.csv', label: 'Calendario tagli (dati mannesi)' },
    ];

    const GOVERNO_CLASS = {
        'Fustaia':        'gov-fustaia',
        'Ceduo':          'gov-ceduo',
        'Rimboschimento': 'gov-rimboschimento',
    };

    const GOVERNO_LEGEND = [
        { cls: 'gov-fustaia',        label: 'Fustaia' },
        { cls: 'gov-ceduo',          label: 'Ceduo' },
        { cls: 'gov-rimboschimento', label: 'Rimboschimento' },
    ];

    const CALENDAR_LEGEND = [
        { cls: 'cal-0', label: FILES[0].label },
        { cls: 'cal-1', label: FILES[1].label },
        { cls: 'cal-2', label: FILES[2].label },
        { cls: 'cal-3', label: FILES[3].label },
    ];

    // Reverse alphabetical: Serra, Fabrizia, Capistrano
    const COMPRESA_ORDER = ['Serra', 'Fabrizia', 'Capistrano'];

    let validParcels = new Set(); // "Compresa-Particella" strings

    // Current dataset (set by loadFile, read by renderGrid)
    let currentData = null; // { harvests, byCompresa, allYears, unmatched }

    // Compare mode state
    let compareMode = false;
    let compareData = null; // array of processData results, one per FILE

    // Cache: path → processData result (never fetch/parse a file twice)
    const dataCache = new Map();

    const $ = id => document.getElementById(id);

    function naturalSort(a, b) {
        return a.localeCompare(b, undefined, { numeric: true });
    }

    function setStatus(msg) {
        $('status').textContent = msg;
    }

    function clearChildren(el) {
        while (el.firstChild) el.removeChild(el.firstChild);
    }

    function parseCsv(text) {
        return Papa.parse(text, { header: true, skipEmptyLines: true }).data;
    }

    function loadParticelle() {
        return fetch('../data/particelle.csv')
            .then(r => r.text())
            .then(text => {
                parseCsv(text).forEach(row => {
                    const compresa = (row['Compresa'] || '').trim();
                    const particella = (row['Particella'] || '').trim();
                    if (compresa && particella) {
                        validParcels.add(compresa + '-' + particella);
                    }
                });
            });
    }

    function populateDropdown() {
        const sel = $('file-select');
        FILES.forEach((f, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = f.label;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => loadFile(FILES[sel.value].path));
    }

    function buildLegend(items) {
        const container = $('legend');
        clearChildren(container);
        items.forEach(({ cls, label }) => {
            const item = document.createElement('span');
            item.className = 'legend-item';
            const swatch = document.createElement('span');
            swatch.className = 'legend-swatch ' + cls;
            item.appendChild(swatch);
            item.appendChild(document.createTextNode(label));
            container.appendChild(item);
        });
    }

    // --- Year range slider (shared module) ---

    let yearSlider = null; // initialized in init()

    // --- Data loading ---

    function fetchAndProcess(path) {
        if (dataCache.has(path)) return Promise.resolve(dataCache.get(path));
        return fetch(path)
            .then(r => r.text())
            .then(text => {
                const data = processData(parseCsv(text));
                if (data) dataCache.set(path, data);
                return data;
            });
    }

    function loadFile(path) {
        setStatus('Caricamento...');
        clearChildren($('grid'));
        $('unmatched').textContent = '';

        fetchAndProcess(path)
            .then(data => {
                currentData = data;
                if (!currentData) return;
                yearSlider.setRange(currentData.allYears);
                renderGrid();
                setStatus('');
            })
            .catch(err => setStatus('Errore: ' + err.message));
    }

    function processData(rows) {
        const harvests = new Map();
        const years = new Set();
        const unmatched = new Set();

        rows.forEach(row => {
            const compresa = (row['Compresa'] || '').trim();
            const particella = (row['Particella'] || '').trim();
            const anno = parseInt(row['Anno'], 10);
            const governo = (row['Governo'] || '').trim();
            if (!compresa || !particella || isNaN(anno)) return;

            const key = compresa + '-' + particella;
            if (!validParcels.has(key)) {
                unmatched.add(key);
            }

            years.add(anno);
            if (!harvests.has(key)) {
                harvests.set(key, { compresa, particella, yearGov: new Map() });
            }
            harvests.get(key).yearGov.set(anno, governo);
        });

        // Group and sort parcels by compresa
        const byCompresa = new Map();
        for (const [, info] of harvests) {
            if (!byCompresa.has(info.compresa)) byCompresa.set(info.compresa, new Set());
            byCompresa.get(info.compresa).add(info.particella);
        }
        for (const [comp, parcels] of byCompresa) {
            byCompresa.set(comp, [...parcels].sort(naturalSort));
        }

        const sortedYears = [...years].sort((a, b) => a - b);
        if (sortedYears.length === 0) {
            setStatus('Nessun dato trovato.');
            return null;
        }
        const allYears = [];
        for (let y = sortedYears[0]; y <= sortedYears[sortedYears.length - 1]; y++) {
            allYears.push(y);
        }

        return { harvests, byCompresa, allYears, unmatched };
    }

    function enterCompareMode() {
        compareMode = true;
        $('file-select').disabled = true;
        setStatus('Caricamento...');
        clearChildren($('grid'));

        Promise.all(FILES.map(f => fetchAndProcess(f.path)))
            .then(results => {
                compareData = results; // some may be null
                // Build union of years across all datasets
                const allYearsSet = new Set();
                const unionByCompresa = new Map();
                const unionUnmatched = new Set();
                compareData.forEach(data => {
                    if (!data) return;
                    data.allYears.forEach(y => allYearsSet.add(y));
                    for (const [comp, parcels] of data.byCompresa) {
                        if (!unionByCompresa.has(comp)) unionByCompresa.set(comp, new Set());
                        parcels.forEach(p => unionByCompresa.get(comp).add(p));
                    }
                    data.unmatched.forEach(k => unionUnmatched.add(k));
                });
                const sorted = [...allYearsSet].sort((a, b) => a - b);
                if (sorted.length === 0) { setStatus('Nessun dato trovato.'); return; }
                const allYears = [];
                for (let y = sorted[0]; y <= sorted[sorted.length - 1]; y++) allYears.push(y);
                // Sort parcels within each compresa
                for (const [comp, parcels] of unionByCompresa) {
                    unionByCompresa.set(comp, [...parcels].sort(naturalSort));
                }
                // Store merged metadata for rendering
                currentData = { harvests: null, byCompresa: unionByCompresa, allYears, unmatched: unionUnmatched };
                buildLegend(CALENDAR_LEGEND);
                yearSlider.setRange(allYears);
                renderGrid();
                setStatus('');
            })
            .catch(err => setStatus('Errore: ' + err.message));
    }

    function exitCompareMode() {
        compareMode = false;
        compareData = null;
        $('file-select').disabled = false;
        buildLegend(GOVERNO_LEGEND);
        loadFile(FILES[$('file-select').value].path);
    }

    function renderGrid() {
        if (!currentData) return;
        const { byCompresa, unmatched } = currentData;
        const [yearFrom, yearTo] = yearSlider.getRange();
        const visibleYears = currentData.allYears.filter(y => y >= yearFrom && y <= yearTo);

        const table = $('grid');
        clearChildren(table);

        // Header: corner + year labels
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        const corner = document.createElement('th');
        corner.className = 'corner';
        headerRow.appendChild(corner);
        visibleYears.forEach(year => {
            const th = document.createElement('th');
            th.textContent = year;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Body: compresa separator rows + parcel rows
        const compresaList = [...COMPRESA_ORDER];
        for (const comp of byCompresa.keys()) {
            if (!compresaList.includes(comp)) compresaList.push(comp);
        }
        const tbody = document.createElement('tbody');
        for (const comp of compresaList) {
            const parcels = byCompresa.get(comp);
            if (!parcels || parcels.length === 0) continue;

            const sepRow = document.createElement('tr');
            sepRow.className = 'compresa-row';
            const sepTd = document.createElement('td');
            sepTd.colSpan = visibleYears.length + 1;
            sepTd.textContent = comp;
            sepRow.appendChild(sepTd);
            tbody.appendChild(sepRow);

            parcels.forEach(particella => {
                const key = comp + '-' + particella;
                const isUnmatched = unmatched.has(key);
                const tr = document.createElement('tr');

                const labelTd = document.createElement('td');
                labelTd.className = 'parcel-cell';
                if (isUnmatched) labelTd.classList.add('unmatched-parcel');
                labelTd.textContent = particella + (isUnmatched ? '*' : '');
                tr.appendChild(labelTd);

                if (compareMode) {
                    visibleYears.forEach(year => {
                        const td = document.createElement('td');
                        td.className = 'cell';
                        const container = document.createElement('div');
                        container.className = 'cell-compare';
                        const tooltipParts = [];
                        compareData.forEach((data, i) => {
                            const sub = document.createElement('span');
                            const info = data && data.harvests.get(key);
                            const governo = info && info.yearGov.get(year);
                            if (governo) {
                                sub.className = CALENDAR_LEGEND[i].cls;
                                tooltipParts.push(CALENDAR_LEGEND[i].label + ': ' + governo);
                            }
                            container.appendChild(sub);
                        });
                        if (tooltipParts.length > 0) {
                            td.title = comp + ' ' + particella + ' \u2014 ' + year + '\n' + tooltipParts.join('\n');
                        }
                        td.appendChild(container);
                        tr.appendChild(td);
                    });
                } else {
                    const info = currentData.harvests.get(key);
                    visibleYears.forEach(year => {
                        const td = document.createElement('td');
                        td.className = 'cell';
                        const governo = info && info.yearGov.get(year);
                        if (governo) {
                            const cls = GOVERNO_CLASS[governo];
                            if (cls) td.classList.add(cls);
                            td.title = comp + ' ' + particella + ' \u2014 ' + year + ' \u2014 ' + governo;
                        }
                        tr.appendChild(td);
                    });
                }
                tbody.appendChild(tr);
            });
        }
        table.appendChild(tbody);

        // Unmatched
        const unmatchedEl = $('unmatched');
        if (unmatched.size > 0) {
            const sorted = [...unmatched].sort(naturalSort);
            unmatchedEl.textContent = '* Particelle sconosciute: ' +
                sorted.map(k => k.replace('-', ' ')).join(', ');
        } else {
            unmatchedEl.textContent = '';
        }
    }

    function init() {
        populateDropdown();
        buildLegend(GOVERNO_LEGEND);
        yearSlider = createRangeSlider($('year-min'), $('year-max'), $('year-label'), renderGrid);
        $('compare-all').checked = false;
        $('compare-all').addEventListener('change', function() {
            if (this.checked) enterCompareMode();
            else exitCompareMode();
        });
        loadParticelle()
            .then(() => loadFile(FILES[0].path))
            .catch(err => setStatus('Errore inizializzazione: ' + err.message));
    }

    document.addEventListener('DOMContentLoaded', init);
})();
