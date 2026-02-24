import * as duckdb from 'https://esm.sh/@duckdb/duckdb-wasm@1.29.0';

const $ = id => document.getElementById(id);

// Query examples for the help modal
const QUERY_EXAMPLES = [
    {
        description: 'Classi diametriche per genere',
        query: 'SELECT ROUND(("D(cm)"-1) / 5 +1)*5 AS Diametro, Genere, COUNT(*) as "N. piante" FROM alberi_calcolati GROUP BY Genere, Diametro ORDER BY Genere, Diametro ASC;'
    },
    {
        description: 'Particelle in ordine decrescente di prelievo',
        query: 'SELECT Compresa, Particella, "Vol tot (m³)", "Vol/ha (m³/ha)", "Prelievo (m³)" FROM ripresa ORDER BY "Prelievo (m³)" DESC;'
    },
    {
        description: 'Particelle in ordine decrescente di superficie campionata',
        query: 'SELECT Compresa, Particella, ANY_VALUE(p."Area (ha)") AS "Area (ha)", COUNT(*) as "N. aree saggio", ROUND(COUNT(*) * 12.5 / ANY_VALUE(p."Area (ha)"),2) as "% campionato",  FROM particelle p JOIN aree_di_saggio a USING (Compresa, Particella) GROUP BY Compresa, Particella ORDER BY "% campionato" DESC;'
    },
    {
        description: 'Produttività totale per particella e anno',
        query: 'SELECT Anno,Compresa,Particella,ROUND(SUM(abete+pino+douglas+faggio+castagno+ontano+altro)) AS "Q.li" FROM mannesi GROUP BY Compresa,Particella,Anno ORDER BY Anno ASC,Compresa DESC,Particella ASC;'
    },
    {
        description: 'Produttività trattori per anno',
        query: 'SELECT anno as Anno, ROUND(SUM("Equus")) AS "Equus", ROUND(SUM("Fiat 110-90")) AS "Fiat 110-90", ROUND(SUM("Fiat 80-66")) AS "Fiat 80-66", ROUND(SUM("Landini 135")) AS "Landini 135", ROUND(SUM("New Holland T5050")) AS "New Holland T5050" FROM mannesi WHERE anno > 2021 GROUP BY anno;'
    },
];

const CSV_FILES = [
    ['particelle', 'Comprese e particelle forestali'],
    ['aree-di-saggio', 'Aree di saggio per il piano di gestione'],
    ['alberi', 'Tutti gli alberi delle aree di saggio'],
    ['alberi-calcolati', 'Come "alberi", ma con altezze calcolate tramite equazioni interpolanti'],
    ['altezze', 'Sottoinsieme di alberi con misure di altezza tramite ipsometro laser'],
    ['alberi-modello', 'Diametri e altezze degli alberi modello'],
    ['mannesi', 'Archivio interventi boschivi'],
    ['piante-accrescimento-indefinito', 'Piante ad accrescimento indefinito'],
    ['ripresa', 'Provvigione per particella']
];

const statusEl = $('status');
const errorEl = $('error');
const resultEl = $('result-container');
const queryEl = $('query');
const runBtn = $('run');
const exportBtn = $('export');

let conn;
let lastColumns = [];
let lastRows = [];

function populateExamples() {
    const list = $('queries-list');
    QUERY_EXAMPLES.forEach(example => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#';
        a.textContent = example.description;
        a.addEventListener('click', e => {
            e.preventDefault();
            queryEl.value = example.query;
            runQuery();
        });
        li.appendChild(a);
        list.appendChild(li);
    });
}

function populateTables() {
    const tablesInfo = CSV_FILES.map(([name, desc]) =>
        `<li><b>${name.replace(/-/g, '_')}</b>: ${desc}</li>`
    ).join('\n');
    $('tables-list').innerHTML = tablesInfo;
}

async function initDB() {
    const BUNDLES = duckdb.getJsDelivrBundles();
    const bundle = await duckdb.selectBundle(BUNDLES);
    const workerUrl = URL.createObjectURL(
        new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
    );
    const worker = new Worker(workerUrl);
    const logger = new duckdb.ConsoleLogger();
    const db = new duckdb.AsyncDuckDB(logger, worker);
    await db.instantiate(bundle.mainModule);
    conn = await db.connect();

    for (const [name, description] of CSV_FILES) {
        const tableName = name.replace(/-/g, '_');
        const url = new URL(`../data/${name}.csv`, window.location.href).href;
        const resp = await fetch(url);
        const buf = new Uint8Array(await resp.arrayBuffer());
        await db.registerFileBuffer(`${name}.csv`, buf);
        await conn.query(`CREATE TABLE ${tableName} AS SELECT * FROM read_csv('${name}.csv', header=true, delim=',', quote='"')`);
    }
}

async function executeQuery() {
    const sql = queryEl.value.trim();
    if (!sql) return;
    errorEl.textContent = '';
    resultEl.innerHTML = '';
    try {
        const result = await conn.query(sql);
        const NUMERIC_TYPE_IDS = new Set([2, 3, 7]); // Arrow: Int, Float, Decimal
        const columns = result.schema.fields.map(f => f.name);
        const numericCols = new Set(
            result.schema.fields.filter(f => NUMERIC_TYPE_IDS.has(f.type.typeId)).map(f => f.name)
        );
        const rows = result.toArray().map(row => {
            const obj = {};
            for (const col of columns) obj[col] = row[col];
            return obj;
        });

        lastColumns = columns;
        lastRows = rows;
        exportBtn.disabled = rows.length === 0;

        if (rows.length === 0) {
            resultEl.innerHTML = '<p style="color:#555">Nessun risultato.</p>';
            return;
        }

        let html = '<table><thead><tr>';
        for (const col of columns) {
            const cls = numericCols.has(col) ? ' class="num"' : '';
            html += `<th${cls}>${col}</th>`;
        }
        html += '</tr></thead><tbody>';
        for (const row of rows) {
            html += '<tr>';
            for (const col of columns) {
                const v = row[col];
                const cls = numericCols.has(col) ? ' class="num"' : '';
                html += `<td${cls}>${v === null || v === undefined ? '' : v}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        resultEl.innerHTML = html;
    } catch (e) {
        errorEl.textContent = e.message;
    }
}

function runQuery() {
    const sql = queryEl.value.trim();
    if (!sql) return;
    const url = new URL(window.location);
    url.searchParams.set('q', sql);
    history.pushState(null, '', url);
    executeQuery();
}

function loadQueryFromURL() {
    const q = new URLSearchParams(window.location.search).get('q');
    if (q) {
        queryEl.value = q;
        executeQuery();
    }
}

function csvField(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    if (s.includes(',') || s.includes('"') || s.includes('\n')) return '"' + s.replace(/"/g, '""') + '"';
    return s;
}

function exportCSV() {
    const lines = [lastColumns.map(csvField).join(',')];
    for (const row of lastRows) lines.push(lastColumns.map(col => csvField(row[col])).join(','));
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'dati_forestali.csv';
    a.click();
    URL.revokeObjectURL(a.href);
}

function initHandlers() {
    runBtn.addEventListener('click', runQuery);
    exportBtn.addEventListener('click', exportCSV);
    queryEl.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runQuery();
        }
    });
    window.addEventListener('popstate', loadQueryFromURL);
}

async function init() {
    try {
        populateTables();
        populateExamples();
        initHandlers();
        await initDB();

        statusEl.style.visibility = 'hidden';
        runBtn.disabled = false;
        loadQueryFromURL();
    } catch (e) {
        statusEl.textContent = 'Errore durante il caricamento.';
        errorEl.textContent = e.message;
    }
}

init();
