// Calendario di taglio — harvest schedule Gantt chart
(function() {
    'use strict';

    const FILES = [
        { path: '../data/calendario-tagli-mannesi.csv', label: 'Calendario tagli (dati mannesi)' },
        { path: '../data/calendario-tagli-2011-2025.csv', label: 'Calendario tagli 2011\u20132025' },
        { path: '../data/calendario-tagli-2026-2040.csv', label: 'Calendario tagli 2026\u20132040' },
        { path: '../data/registro-gestione-2016-2025.csv', label: 'Registro gestione 2016\u20132025' },
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

    // Reverse alphabetical: Serra, Fabrizia, Capistrano
    const COMPRESA_ORDER = ['Serra', 'Fabrizia', 'Capistrano'];

    let validParcels = new Set(); // "Compresa-Particella" strings

    // Current dataset (set by loadFile, read by renderGrid)
    let currentData = null; // { harvests, byCompresa, allYears, unmatched }

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

    function buildLegend() {
        const container = $('legend');
        GOVERNO_LEGEND.forEach(({ cls, label }) => {
            const item = document.createElement('span');
            item.className = 'legend-item';
            const swatch = document.createElement('span');
            swatch.className = 'legend-swatch ' + cls;
            item.appendChild(swatch);
            item.appendChild(document.createTextNode(label));
            container.appendChild(item);
        });
    }

    // --- Year range slider ---

    function setupSlider() {
        $('year-min').addEventListener('input', onSliderInput);
        $('year-max').addEventListener('input', onSliderInput);
    }

    function onSliderInput() {
        const lo = $('year-min');
        const hi = $('year-max');
        // Clamp: min thumb can't exceed max, and vice versa
        if (parseInt(lo.value, 10) > parseInt(hi.value, 10)) {
            if (this === lo) lo.value = hi.value;
            else hi.value = lo.value;
        }
        updateYearLabel();
        renderGrid();
    }

    function updateSlider(allYears) {
        const lo = $('year-min');
        const hi = $('year-max');
        const min = allYears[0];
        const max = allYears[allYears.length - 1];
        lo.min = hi.min = min;
        lo.max = hi.max = max;
        lo.value = min;
        hi.value = max;
        updateYearLabel();
    }

    function updateYearLabel() {
        const a = $('year-min').value;
        const b = $('year-max').value;
        $('year-label').textContent = a === b ? a : a + '\u2013' + b;
    }

    function getYearRange() {
        return [parseInt($('year-min').value, 10), parseInt($('year-max').value, 10)];
    }

    // --- Data loading ---

    function loadFile(path) {
        setStatus('Caricamento...');
        clearChildren($('grid'));
        $('unmatched').textContent = '';

        fetch(path)
            .then(r => r.text())
            .then(text => {
                currentData = processData(parseCsv(text));
                if (!currentData) return;
                updateSlider(currentData.allYears);
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

    // --- Grid rendering ---

    function renderGrid() {
        if (!currentData) return;
        const { harvests, byCompresa, unmatched } = currentData;
        const [yearFrom, yearTo] = getYearRange();

        // Filter years to slider range
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
        // Include any comprese not in COMPRESA_ORDER (e.g. from unmatched parcels)
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
                const info = harvests.get(key);
                const isUnmatched = unmatched.has(key);
                const tr = document.createElement('tr');

                const labelTd = document.createElement('td');
                labelTd.className = 'parcel-cell';
                if (isUnmatched) labelTd.classList.add('unmatched-parcel');
                labelTd.textContent = particella + (isUnmatched ? '*' : '');
                tr.appendChild(labelTd);

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
                tbody.appendChild(tr);
            });
        }
        table.appendChild(tbody);

        // Unmatched
        const unmatchedEl = $('unmatched');
        if (unmatched.size > 0) {
            const sorted = [...unmatched].sort(naturalSort);
            unmatchedEl.textContent = '* Non in anagrafica: ' +
                sorted.map(k => k.replace('-', ' ')).join(', ');
        } else {
            unmatchedEl.textContent = '';
        }
    }

    function init() {
        populateDropdown();
        buildLegend();
        setupSlider();
        loadParticelle()
            .then(() => loadFile(FILES[0].path))
            .catch(err => setStatus('Errore inizializzazione: ' + err.message));
    }

    document.addEventListener('DOMContentLoaded', init);
})();
