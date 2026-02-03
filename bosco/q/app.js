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
        query: 'SELECT Compresa, Particella, ANY_VALUE(p."Area (ha)") AS "Area (ha)", COUNT(*) as "N. aree saggio", ROUND(COUNT(*) * 12.5 / ANY_VALUE(p."Area (ha)"),2) as "% campionato",  FROM particelle p NATURAL JOIN aree_di_saggio a GROUP BY p.Compresa, P.Particella ORDER BY "% campionato" DESC;'
    }
];

// Populate query examples modal
function populateExamples() {
    const list = $('query-examples-list');
    list.innerHTML = '';
    QUERY_EXAMPLES.forEach(example => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `?q=${encodeURIComponent(example.query)}`;
        a.target = '_blank';
        a.textContent = example.description;
        li.appendChild(a);
        list.appendChild(li);
    });
}

// Modal controls
window.showExamples = function() {
    $('examples-modal').style.display = 'block';
};

window.hideExamples = function() {
    $('examples-modal').style.display = 'none';
};

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = $('examples-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
};

const CSV_FILES = [
    ['particelle', 'Comprese e particelle forestali'],
    ['comparti', 'Comparti forestali'],
    ['aree-di-saggio', 'Aree di saggio per il piano di gestione'],
    ['alberi', 'Tutti gli alberi delle aree di saggio'],
    ['alberi-calcolati', 'Come "alberi", ma con altezze calcolate tramite equazioni interpolanti'],
    ['altezze', 'Sottoinsieme di alberi con misure di altezza tramite ipsometro laser'],
    ['alberi-modello', 'Diametri e altezze degli alberi modello'],
    ['ripresa', 'Provvigione per particella']
];

const statusEl = $('status');
const errorEl = $('error');
const resultEl = $('result-container');
const queryEl = $('query');
const runBtn = $('run');

let conn;

async function init() {
    try {
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
            await db.registerFileURL(`${name}.csv`, url, duckdb.DuckDBDataProtocol.HTTP, true);
            await conn.query(`CREATE TABLE ${tableName} AS SELECT * FROM read_csv_auto('${name}.csv')`);
        }

        const tablesInfo = CSV_FILES.map(([name, desc]) =>
            `<b>${name.replace(/-/g, '_')}</b>: ${desc}`
        ).join('<br>');
        $('tables-list').innerHTML = `Tabelle disponibili:<br>${tablesInfo}`;
        statusEl.style.visibility = 'hidden';
        runBtn.disabled = false;
        const params = new URLSearchParams(window.location.search);
        if (params.has('q')) {
            queryEl.value = params.get('q');
            window.runQuery();
        }
    } catch (e) {
        statusEl.textContent = 'Errore durante il caricamento.';
        errorEl.textContent = e.message;
    }
}

window.runQuery = async function() {
    const sql = queryEl.value.trim();
    if (!sql) return;
    const url = new URL(window.location);
    url.searchParams.set('q', sql);
    history.replaceState(null, '', url);
    errorEl.textContent = '';
    resultEl.innerHTML = '';
    try {
        const result = await conn.query(sql);
        const columns = result.schema.fields.map(f => f.name);
        const rows = result.toArray().map(row => {
            const obj = {};
            for (const col of columns) obj[col] = row[col];
            return obj;
        });

        if (rows.length === 0) {
            resultEl.innerHTML = '<p style="color:#555">Nessun risultato.</p>';
            return;
        }

        let html = '<table><thead><tr>';
        for (const col of columns) html += `<th>${col}</th>`;
        html += '</tr></thead><tbody>';
        for (const row of rows) {
            html += '<tr>';
            for (const col of columns) {
                const v = row[col];
                html += `<td>${v === null || v === undefined ? '' : v}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        resultEl.innerHTML = html;
    } catch (e) {
        errorEl.textContent = e.message;
    }
};

queryEl.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        window.runQuery();
    }
});

populateExamples();
init();
