// Calendario di taglio — harvest schedule Gantt chart
(function() {
    'use strict';

    const FILES = [
        { path: '../data/calendario-tagli-2011-2025.csv', label: 'Calendario tagli 2011\u20132025' },
        { path: '../data/calendario-tagli-2026-2040.csv', label: 'Calendario tagli 2026\u20132040' },
        { path: '../data/registro-gestione-2016-2025.csv', label: 'Registro gestione 2016\u20132025' },
    ];

    const GOVERNO_COLORS = {
        'Fustaia':        { color: '#006400', label: 'Fustaia' },
        'Ceduo':          { color: '#DAA520', label: 'Ceduo' },
        'Rimboschimento': { color: '#8B4513', label: 'Rimboschimento' },
    };

    const GOVERNO_CLASS = {
        'Fustaia':        'gov-fustaia',
        'Ceduo':          'gov-ceduo',
        'Rimboschimento': 'gov-rimboschimento',
    };

    // Reverse alphabetical: Serra, Fabrizia, Capistrano
    const COMPRESA_ORDER = ['Serra', 'Fabrizia', 'Capistrano'];

    let validParcels = new Set(); // set of "Compresa-Particella" strings

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
                const rows = parseCsv(text);
                rows.forEach(row => {
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
        Object.values(GOVERNO_COLORS).forEach(({ color, label }) => {
            const item = document.createElement('span');
            item.className = 'legend-item';
            const swatch = document.createElement('span');
            swatch.className = 'legend-swatch';
            swatch.style.backgroundColor = color;
            item.appendChild(swatch);
            item.appendChild(document.createTextNode(label));
            container.appendChild(item);
        });
    }

    function loadFile(path) {
        setStatus('Caricamento...');
        clearChildren($('grid'));
        $('unmatched').textContent = '';

        fetch(path)
            .then(r => r.text())
            .then(text => {
                const rows = parseCsv(text);
                render(rows);
                setStatus('');
            })
            .catch(err => setStatus('Errore: ' + err.message));
    }

    function render(rows) {
        // Collect harvest data: key = "Compresa-Particella", value = Map(year -> governo)
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
                return;
            }

            years.add(anno);
            if (!harvests.has(key)) {
                harvests.set(key, { compresa, particella, yearGov: new Map() });
            }
            // If a parcel appears multiple times in the same year, keep the last entry
            harvests.get(key).yearGov.set(anno, governo);
        });

        // Build sorted column list grouped by compresa
        const byCompresa = new Map();
        for (const [key, info] of harvests) {
            if (!byCompresa.has(info.compresa)) byCompresa.set(info.compresa, []);
            byCompresa.get(info.compresa).push(info.particella);
        }
        // Deduplicate parcels within each compresa
        for (const [comp, parcels] of byCompresa) {
            byCompresa.set(comp, [...new Set(parcels)].sort(naturalSort));
        }

        // Ordered columns: compresa in COMPRESA_ORDER, parcels sorted naturally
        const columns = []; // array of { compresa, particella, key }
        const compresaSpans = []; // array of { compresa, count }
        for (const comp of COMPRESA_ORDER) {
            const parcels = byCompresa.get(comp);
            if (!parcels || parcels.length === 0) continue;
            compresaSpans.push({ compresa: comp, count: parcels.length });
            parcels.forEach(p => columns.push({
                compresa: comp,
                particella: p,
                key: comp + '-' + p,
            }));
        }

        // Year range
        const sortedYears = [...years].sort((a, b) => a - b);
        if (sortedYears.length === 0) {
            setStatus('Nessun dato trovato.');
            return;
        }
        const minYear = sortedYears[0];
        const maxYear = sortedYears[sortedYears.length - 1];
        const allYears = [];
        for (let y = minYear; y <= maxYear; y++) allYears.push(y);

        // Build table
        const table = $('grid');
        clearChildren(table);

        // Header
        const thead = document.createElement('thead');

        // Row 1: compresa spans
        const tr1 = document.createElement('tr');
        const thCorner1 = document.createElement('th');
        thCorner1.className = 'year-cell compresa-header';
        thCorner1.rowSpan = 2;
        tr1.appendChild(thCorner1);
        compresaSpans.forEach(({ compresa, count }) => {
            const th = document.createElement('th');
            th.className = 'compresa-header';
            th.colSpan = count;
            th.textContent = compresa;
            tr1.appendChild(th);
        });
        thead.appendChild(tr1);

        // Row 2: parcel names
        const tr2 = document.createElement('tr');
        columns.forEach(col => {
            const th = document.createElement('th');
            th.className = 'parcel-header';
            th.textContent = col.particella;
            th.title = col.compresa + ' ' + col.particella;
            tr2.appendChild(th);
        });
        thead.appendChild(tr2);
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        allYears.forEach(year => {
            const tr = document.createElement('tr');
            const tdYear = document.createElement('td');
            tdYear.className = 'year-cell';
            tdYear.textContent = year;
            tr.appendChild(tdYear);

            columns.forEach(col => {
                const td = document.createElement('td');
                td.className = 'cell';
                const info = harvests.get(col.key);
                const governo = info && info.yearGov.get(year);
                if (governo) {
                    const cls = GOVERNO_CLASS[governo];
                    if (cls) td.classList.add(cls);
                    td.title = col.compresa + ' ' + col.particella + ' \u2014 ' + year + ' \u2014 ' + governo;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);

        // Unmatched
        const unmatchedEl = $('unmatched');
        if (unmatched.size > 0) {
            const sorted = [...unmatched].sort(naturalSort);
            unmatchedEl.textContent = 'Particelle non trovate in anagrafica: ' +
                sorted.map(k => k.replace('-', ' ')).join(', ');
        } else {
            unmatchedEl.textContent = '';
        }
    }

    function init() {
        populateDropdown();
        buildLegend();
        loadParticelle()
            .then(() => loadFile(FILES[0].path))
            .catch(err => setStatus('Errore inizializzazione: ' + err.message));
    }

    document.addEventListener('DOMContentLoaded', init);
})();
