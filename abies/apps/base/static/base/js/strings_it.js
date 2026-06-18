/**
 * Italian UI string constants.
 *
 * To switch language, change the re-export in strings.js.
 */

// Shell / chrome
export const LOCALE = 'it';
export const LOADING = 'Caricamento...';
export const DISMISS = 'Chiudi';
export const SAVE = 'Salva';
export const DELETE_CONFIRM = 'I dati cancellati non potranno essere recuperati. Confermi?';
export const CANCEL = 'Annulla';
export const CONFIRM = 'Conferma';

// Tables
export const FILTER_LABEL = 'Filtra';
export const SEARCH_PLACEHOLDER = 'Cerca...';
export const EXPORT_CSV = 'Esporta CSV';
export const IMPORT_CSV = 'Importa CSV';
export const NO_RESULTS = 'Nessun risultato.';
export const ACTION_EDIT = 'Modifica';
export const ACTION_DELETE = 'Elimina';

// TableWrapper localization: labels bundle and CSV format for Italian.
// Passed as `labels` / `csvFormat` options when constructing a TableWrapper.
// `add` / `boolYes` / `boolNo` / `pageInfo` are inlined here because they have
// no other callers; if a caller appears elsewhere, lift them back out as
// named exports.
export const TABLE_LABELS = {
  search: FILTER_LABEL,
  searchPlaceholder: SEARCH_PLACEHOLDER,
  exportCSV: EXPORT_CSV,
  add: 'Aggiungi',
  empty: NO_RESULTS,
  actionEdit: ACTION_EDIT,
  actionDelete: ACTION_DELETE,
  boolYes: 'Sì',
  boolNo: 'No',
  pageInfo: 'Pagina {current} di {total}',
};

export const TABLE_CSV_FORMAT = {
  separator: ';',
  decimal: ',',
  dateFormat: 'YYYY-MM-DD',  // CSV dates are ISO8601 regardless of locale
};

// Errors
export const ERROR_NETWORK = 'Errore di rete. Riprovare.';
export const ERROR_CONFLICT = 'Il record è stato modificato da un altro utente.';
export const ERROR_GENERIC = 'Errore imprevisto.';

// Ipso import inbox.
export const IPSO_INBOX_TITLE = 'Importazione';
export const IPSO_INBOX_CSV = 'ipso-importazione.csv';
export const IPSO_COL_RECEIVED = 'Ricevuto';
export const IPSO_COL_MODE = 'Modalità';
export const IPSO_COL_OPERATOR = 'Operatore';
export const IPSO_COL_RECORDS = 'Righe';
export const IPSO_COL_STATE = 'Stato';
export const IPSO_COL_WORK_PACKAGE = 'Pacchetto';
export const IPSO_COL_REFERENCE = 'Riferimento';
export const IPSO_COL_TARGET = 'Destinazione';
export const IPSO_COL_ERROR = 'Errore';
export const IPSO_ACTION_OPEN = 'Apri';
export const IPSO_ACTION_OPEN_ICON = '\u{1F50D}\u{FE0E}';
export const IPSO_ACTION_REJECT = 'Rifiuta';
export const IPSO_LOADING_DETAIL = 'Caricamento dettaglio...';
export const IPSO_REJECT_CONFIRM = 'Rifiutare questo caricamento Ipso?';
export const IPSO_SESSION_TITLE = (id) => `Sessione ${id}`;
export const IPSO_PREVIEW_TITLE = (n) => `Anteprima record (${n})`;
export const IPSO_EMPTY_RECORDS = 'Nessun record.';
export const IPSO_SUMMARY_EMPTY = 'Nessun caricamento Ipso.';
export const IPSO_SUMMARY = (n, pending) => `${n} caricamenti, ${pending} da importare.`;
export const IPSO_TARGET_PLAN_LABEL = 'Piano di taglio';
export const IPSO_TARGET_SURVEY_LABEL = 'Rilevamento';
export const IPSO_TARGET_SELECT = 'Seleziona destinazione';
export const IPSO_IMPORT_CONFIRM = 'Importare questo caricamento nel piano selezionato?';
export const IPSO_IMPORT_SAMPLES_CONFIRM = 'Importare questo caricamento nel rilevamento selezionato?';
export const IPSO_IMPORT_PAI_CONFIRM = 'Importare questo caricamento come PAI?';
export const IPSO_COL_SEQ = '#';
export const IPSO_COL_ACCURACY = 'Acc.';
export const IPSO_EMPTY_VALUE = '-';

// Validation (client-side)
export const ERR_DATE_FUTURE = 'La data non può essere nel futuro.';
export const ERR_SPECIES_PCT_SUM = 'Le percentuali delle specie devono sommare a 100.';
export const ERR_TRACTOR_PCT_SUM = 'Le percentuali dei trattori devono sommare a 100.';
export const ERR_CREW_REQUIRED = 'Squadra obbligatoria.';
export const ERR_HOURS_POSITIVE = 'Le ore devono essere maggiori di zero.';
export const ERR_CREDITS_POSITIVE = 'I quintali devono essere maggiori di zero.';
export const ERR_SLIP_COUNT_MULTIPLE = 'Il numero di verbali deve essere un multiplo di 4.';
export const ERR_SLIP_COUNT_POSITIVE = 'Il numero di verbali deve essere maggiore di zero.';
export const ERR_LICENSE_PLATE_REQUIRED = 'Targa autocarro obbligatoria.';
export const ERR_RECEIPTS_EMPTY = 'Nessuna produzione nel mese selezionato.';

// Charts
export const CHART_TOTAL = 'Totale';
export const CHART_OTHER = 'Altro';

// Prelievi columns
export const COL_DATE = 'Data';
export const COL_SURVEY_DATE = 'Data di rilevamento';
export const COL_PARCEL = 'Particella';
export const COL_CREW = 'Squadra';
export const COL_VDP = 'VDP';
export const COL_QUINTALS = 'Q.li';
export const COL_HOURS = 'Ore';
export const COL_CREDITS_Q = 'Quintali';
export const COL_NOTE = 'Note';
export const COL_EXTRA_NOTE = 'Altre note';
export const COL_VOLUME_M3 = 'Volume (m³)';
export const COL_ACTIVE = 'Attivo';
export const COL_MINOR = 'Minore';

// Settings
export const SETTINGS_PASSWORD = 'Cambio password';
export const SETTINGS_CREWS = 'Squadre';
export const SETTINGS_TRACTORS = 'Trattori';
export const SETTINGS_SPECIES = 'Specie';
export const SETTINGS_USERS = 'Utenti';
export const SETTINGS_HYPSO = 'Parametri ipsometrici';
export const ONLY_ACTIVE = 'Solo attivi';

// Hypsometric-parameters section.  (Accept/Reject/title are literal in the
// candidate <template>, per docs/ui-design-patterns.md.)
export const HYPSO_DESC_NONE = 'Nessun parametro ipsometrico attivo.';
export const HYPSO_SOURCE_COMPUTED_LABEL = 'Calcolato';
export const HYPSO_SOURCE_IMPORTED_LABEL = 'Importato da CSV';
export const HYPSO_DESC_MIN_N = 'N minimo';
export const HYPSO_DESC_SURVEYS = 'Rilevamenti';
export const HYPSO_DESC_HEIGHT_PLOTS = 'Altezze: rilevamenti ipsometrici';
export const HYPSO_IMPORT_CONFIRM =
  "L'importazione sostituisce i parametri attivi. Continuare?";
export const HYPSO_CLEAR_CONFIRM =
  'I parametri attivi verranno eliminati. Continuare?';
export const PASSWORD_NEW = 'Nuova password';
export const PASSWORD_REPEAT = 'Ripeti password';
export const PASSWORD_MISMATCH = 'Le password non coincidono.';

// Audit
export const COL_TIMESTAMP = 'Data/Ora';
export const COL_USER = 'Utente';
export const COL_TABLE = 'Tabella';
export const COL_ACTION = 'Azione';
export const COL_OLD_VALUE = 'Valore precedente';
export const COL_NEW_VALUE = 'Valore successivo';

// Form labels
export const LABEL_DATE = 'Data';
export const LABEL_NAME = 'Nome';
export const LABEL_EMAIL = 'Email';
export const LABEL_LAST_NAME = 'Cognome';
export const LABEL_LOGIN_METHOD = 'Metodo di accesso';
export const LABEL_LICENSE_PLATE = 'Targa autocarro';
export const LABEL_START_NUMBER = 'Numero iniziale';
export const LABEL_SLIP_COUNT = 'Numero di verbali';
export const LABEL_MONTH = 'Mese';

// Mannesi PDFs
export const MANNESI_VDP_NUMBER = 'N.';
export const MANNESI_VDP_LICENSE_PLATE = 'Targa';
export const MANNESI_VDP_SPECIES = 'Essenza';
export const MANNESI_VDP_GROSS_WEIGHT_Q = 'Peso lordo ql';
export const MANNESI_VDP_TARE_Q = 'Tara ql';
export const MANNESI_VDP_NET_WEIGHT_Q = 'Peso netto ql';
export const MANNESI_VDP_SIGNATURE = 'Firma';
export const MANNESI_RECEIPT_HOURS = 'Ore lavorate';
export const MANNESI_RECEIPT_PRODUCTION = 'Produzione';
export const MANNESI_RECEIPT_TOTAL_PRODUCTION = 'Totale produzione';
export const MANNESI_RECEIPT_CREDITS = 'Acconti';
export const MANNESI_RECEIPT_TOTAL = 'Totale';
export const MANNESI_RECEIPT_DETAIL = 'Dettaglio produzione';
export const LABEL_PERCENT = '%';

// Export filenames
export const CSV_PRELIEVI = 'prelievi.csv';
export const CSV_MANNESI_HOURS = 'ore-mannesi.csv';
export const CSV_MANNESI_CREDITS = 'acconti-mannesi.csv';
export const PDF_MANNESI_VDP = 'vdp.pdf';
export const PDF_MANNESI_RECEIPTS = (month) => `ricevute-mannesi-${month}.pdf`;
export const CSV_CREWS = 'squadre.csv';
export const CSV_TRACTORS = 'trattori.csv';
export const CSV_SPECIES = 'specie.csv';
export const CSV_USERS = 'utenti.csv';
export const CSV_AUDIT = 'controllo.csv';
export const CSV_SAMPLED_TREES = 'alberi-campionati.csv';

// Samples
export const SAMPLES_PICK_SURVEY_FIRST =
  'Seleziona prima un rilevamento.';
export const SAMPLES_PICK_AREA_FIRST =
  'Seleziona prima un\'area di saggio sulla mappa.';
export const SAMPLES_INSERT_AREA_HERE =
  'Inserire una nuova area qui?';
export const AREA_IN_USE_TOOLTIP =
  'Area di saggio con campioni: non può essere eliminata.';
export const SAMPLES_TREE_COUNT = (n) =>
  `${n} ${n === 1 ? 'albero' : 'alberi'}`;
export const SAMPLES_TREES_HEADER_ALL = (n) =>
  `(Tutte le aree di campionamento / ${SAMPLES_TREE_COUNT(n)})`;
export const SAMPLES_TREES_HEADER_AREA = (area, n) =>
  `(${area.compresa} ${area.particella} / ` +
  `area di campionamento ${area.numero} / ${SAMPLES_TREE_COUNT(n)})`;
export const SAMPLES_TREES_HEADER_COUNT = (n) =>
  `(${SAMPLES_TREE_COUNT(n)})`;

// Shared lat/lng input
export const USE_CURRENT_LOCATION = 'Usa GPS';

// Cascade-delete warning (Section 1/2 garbage on populated resources)
export const CASCADE_CONFIRM_TITLE = 'Conferma eliminazione';
export const CASCADE_WARN_SURVEY =
  'Questa operazione cancellerà {n_samples} campioni e {n_trees} ' +
  'misure di alberi che non possono essere recuperati.';
export const CASCADE_EXPORT_REQUIRED =
  'Per sicurezza, esporta i dati prima di procedere all\'eliminazione.';
export const RENAME_TITLE_GRID = 'Modifica griglia';
export const RENAME_TITLE_SURVEY = 'Modifica rilevamento';
export const EDIT_GRID_TAB_IMPORT  = 'Importa aree da CSV';
export const EDIT_SURVEY_TAB_IMPORT = 'Importa alberi da CSV';
export const GRID_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Area saggio, Lon, Lat, Quota. ' +
  'Raggio è opzionale; se assente usa 12 m. Sono accettate anche le ' +
  'intestazioni Alt. (m) e Raggio (m).';
export const SURVEY_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Area saggio, Albero, Pollone, ' +
  'Matricina, D_cm, H_m, L10_mm, Genere, Fustaia. Opzionali: Data, PAI.';

// CSV import status (large files can take many seconds)
export const CSV_IMPORT_IN_PROGRESS =
  'Importazione in corso, attendere…  ' +
  'Non chiudere la finestra né cliccare di nuovo "Importa".';
export const CSV_EXTRA_ERRORS = (n) => `… +${n} altri errori`;

// Coppice (per-shoot) entry
export const REMOVE_POLLONE = 'Rimuovi';

// CSV column headers for round-trip import/export (mirror config/strings_it.py).
export const CSV_COL_REGION        = 'Compresa';
export const CSV_COL_PARTICELLA    = 'Particella';
export const CSV_COL_AREA_SAGGIO   = 'Area saggio';
export const CSV_COL_LON           = 'Lon';
export const CSV_COL_LAT           = 'Lat';
export const CSV_COL_ALT           = 'Quota';
export const CSV_COL_RADIUS        = 'Raggio';
export const CSV_COL_ALBERO        = 'Albero';
export const CSV_COL_COPPICE_SHOOT = 'Pollone';
export const CSV_COL_COPPICE_STD   = 'Matricina';
export const CSV_COL_D_CM          = 'D_cm';
export const CSV_COL_H_M           = 'H_m';
export const CSV_COL_L10_MM        = 'L10_mm';
export const CSV_COL_PRESSLER       = 'Pressler';
export const CSV_COL_GENERE        = 'Genere';
export const CSV_COL_HIGHFOREST    = 'Fustaia';
export const CSV_COL_DATA          = 'Data';
export const CSV_COL_PRESERVED     = 'PAI';

// CSV export filenames for the symmetric "Esporta CSV" buttons on
// the Griglie + Rilevamenti pulldown rows (mirror the import column shape).
export const CSV_GRID_AREAS = 'aree-saggio.csv';
export const CSV_SURVEY_TREES = 'alberi-rilevamento.csv';

// Grid-planner / delete-grid (campionamenti.js, grid-planner.js)
export const LABEL_DESCRIPTION_OPTIONAL = 'Descrizione (opzionale)';
export const LABEL_REGIONS = 'Comprese';
export const LABEL_RADIUS_M = 'Raggio (m)';
export const LABEL_COVERAGE_PCT = 'Copertura (%)';
export const ACTION_PLAN = 'Pianifica';
export const ACTION_CREATE = 'Crea';
export const ERR_GRID_HAS_SURVEYS =
  'La griglia è usata da uno o più rilevamenti: eliminarli prima.';
export const ERR_GRID_NAME_REQUIRED = 'Occorre specificare un nome.';
export const ERR_SELECT_REGION = 'Seleziona almeno una compresa.';
export const ERR_RADIUS_POSITIVE = 'Raggio (m) deve essere > 0.';
export const ERR_D_POSITIVE = 'Il diametro deve essere positivo.';
export const ERR_H_POSITIVE = 'L\'altezza deve essere positiva.';
export const ERR_DENSITY_POSITIVE = 'La densità deve essere un numero positivo.';
export const ERR_PRESSLER_POSITIVE = 'Il coefficiente Pressler deve essere positivo.';
export const ERR_COVERAGE_RANGE = 'Copertura deve essere > 0% e ≤ 100%.';
export const ERR_PARAMS_ZERO_POINTS = 'Parametri danno 0 punti.';
export const ERR_PLAN_FIRST = 'Esegui prima "Pianifica".';
export const STATUS_SAVING = 'Salvataggio in corso...';
export const STATUS_PLAN_COMPLETE = 'Pianificazione completata: {n} punti.';
export const STATS_POINTS = 'Punti: {n} (obiettivo: {target})';
export const STATS_TOTAL_AREA_HA = 'Superficie totale: {ha} ha';
export const STATS_AREA_PER_POINT_M2 = 'Area singola adc: {area} m²';
export const TOOLTIP_ADC = 'adc {n} · {compresa} {particella}';
// Sample-area marker tooltips on the Griglie / Rilevamenti maps.  Flat
// templates (not concatenated fragments) so word order stays translatable.
export const TOOLTIP_SAMPLE_AREA = '{compresa} {particella} / adc {numero}';
export const TOOLTIP_SAMPLE_AREA_VISITED =
  '{compresa} {particella} / adc {numero} / {alberi}';

// Bosco placeholder page
export const BOSCO_PLACEHOLDER_MESSAGE =
  'La visualizzazione del bosco sarà disponibile in una prossima versione.';
export const BOSCO_OPEN_BOSCOSCOPIO = 'Apri Boscoscopio';
export const BOSCO_NO_DATA = 'n.d.';
export const BOSCO_NO_DATA_AVAILABLE = 'Nessun dato disponibile.';
export const BOSCO_NO_DATE = 'Nessuna data';
export const BOSCO_NO_REGION = 'Nessuna compresa';
export const BOSCO_NO_GEOMETRY = (region) => `${region} — nessuna geometria`;
export const BOSCO_PARCELS = (region, n) => `${region} — ${n} particelle`;
export const BOSCO_REGION_SUMMARY = (region) => `${region} — riepilogo compresa`;
export const BOSCO_LOADING_HARVESTS = 'Caricamento prelievi...';
export const BOSCO_HARVESTS_UNAVAILABLE = 'Prelievi non disponibili.';
export const BOSCO_LOADING_SATELLITE = 'Caricamento dati satellitari...';
export const BOSCO_SATELLITE_UNAVAILABLE = 'Dati satellitari non disponibili.';
export const BOSCO_NO_SATELLITE = 'Nessun dato satellitare disponibile.';
export const BOSCO_LOADING_RASTER = 'Caricamento raster...';
export const BOSCO_RASTER_UNAVAILABLE = 'Raster non disponibile.';
export const BOSCO_REGION_PARCELS = 'Particelle';
export const BOSCO_DENDROMETRY_UNAVAILABLE = 'Dendrometria non disponibile.';
export const BOSCO_NO_DENDROMETRY = 'Nessun dato dendrometrico.';
export const BOSCO_TREE_COUNT = 'Numero alberi';
export const BOSCO_TREE_COUNT_PER_HA = 'Numero alberi/ha';
export const BOSCO_TREES = (n) => `${n} alberi`;
export const BOSCO_VOLUME_PER_HA = 'Volume (m³/ha)';
export const BOSCO_BASAL_AREA_PER_HA = 'Area bas. (m²/ha)';
export const BOSCO_HISTORICAL_PRODUCTION_UNAVAILABLE =
  'Produzione storica non disponibile.';
export const BOSCO_NO_HISTORICAL_HARVEST = 'Nessun prelievo storico.';
export const BOSCO_INTERVENTIONS = (n) => `${n} interventi`;
export const BOSCO_LAT_LON_REQUIRED = 'Lat e Lon obbligatorie.';
export const BOSCO_INSERT_PAI_TREE_HERE = 'Inserire un nuovo albero qui?';
export const BOSCO_PAI_TREE_META = (parcel, number) => `${parcel} · n. ${number}`;
export const BOSCO_NO_PAI_TREES = 'Nessuna pianta.';
export const BOSCO_METRIC_AGE = 'Età';
export const BOSCO_METRIC_TYPE = 'Governo';
export const BOSCO_METRIC_ALTITUDE = 'Altitudine';
export const BOSCO_METRIC_HISTORICAL_HARVEST = 'Prelievo storico';
export const BOSCO_METRIC_FUTURE_HARVEST = 'Prelievo previsto';
export const BOSCO_HARVEST_METRIC = 'Prelievo';
export const BOSCO_QUINTALS_PER_HA = 'Q.li/ha';
export const BOSCO_REGRESSION = (name, r2, n) => `${name} regressione (R² ${r2}, n ${n})`;

// Digest column headers (mirror config/strings_it.py COL_* additions).
// These are looked up against the JSON digest `columns` array and used
// as keys / labels in `TREES_COLS` for the sortable-table column config.
export const COL_NAME            = 'Nome';
export const COL_REGION          = 'Compresa';
export const COL_DESCRIPTION     = 'Descrizione';
export const COL_N_AREAS         = 'N. aree';
export const COL_REGIONS         = 'Comprese';
export const COL_N_SURVEYS       = 'N. rilevamenti';
export const COL_LAST_UPDATE     = 'Ultimo aggiornamento';
export const COL_GRID            = 'Griglia';
export const COL_HARVEST_PLAN    = 'Piano di taglio';
export const COL_N_AREAS_VISITED = 'N. aree visitate';
export const COL_N_AREAS_TOTAL   = 'N. aree totali';
export const COL_DATE_FIRST      = 'Data primo';
export const COL_DATE_LAST       = 'Data ultimo';
export const COL_LAT             = 'Lat';
export const COL_LON             = 'Lon';
export const COL_AREA_HA         = 'Area (ha)';
export const COL_AREA_CAD_HA     = 'Area cat. (ha)';
export const COL_AVE_AGE         = 'Età media (a)';
export const COL_LOCATION        = 'Località';
export const COL_ALT_MIN         = 'Alt. min. (m)';
export const COL_ALT_MAX         = 'Alt. max. (m)';
export const COL_ASPECT          = 'Esposizione';
export const COL_GRADE_PCT       = 'Pendenza (%)';
export const COL_DESC_VEG        = 'Soprassuolo';
export const COL_DESC_GEO        = 'Stazione';
export const COL_DIAM_CLASS_CM   = 'Classe diam. (cm)';
export const COL_BASAL_AREA_M2   = 'Area bas. (m²)';
export const COL_AVG_H_M         = 'Altezza media (m)';
export const COL_INCREMENT_PCT   = 'Incremento %';
export const COL_ALT             = 'Alt. (m)';
export const COL_RADIUS          = 'Raggio (m)';
export const COL_SURVEY          = 'Rilevamento';
export const COL_SAMPLE_AREA     = 'Area di saggio';
export const COL_SAMPLE_AREA_HA  = 'Area saggi (ha)';
export const COL_N_TREES         = 'N. alberi';
export const COL_SAMPLE_DATE     = 'Data campione';
export const COL_AREA_NUM        = 'N. area';
export const COL_TREE_NUM        = 'N. albero';
export const COL_SPECIES         = 'Specie';
export const COL_COPPICE_SHOOT   = 'Pollone';
export const COL_COPPICE_STD     = 'Matricina';
export const COL_D_CM            = 'D (cm)';
export const COL_H_M             = 'h (m)';
export const COL_L10_MM          = 'L10 (mm)';
export const COL_PRESSLER        = 'Pressler';
export const COL_V_M3            = 'V (m³)';
export const COL_MASS_Q          = 'm (q)';
export const COL_PRESERVED       = 'PAI';

// Abbreviated column labels used only in the TREES_COLS table header to
// save horizontal space; not in any digest.
export const COL_TREE_NUM_SHORT      = 'N. alb.';
export const COL_COPPICE_SHOOT_SHORT = 'Poll.';
export const COL_COPPICE_STD_SHORT   = 'Mat.';

// Status message fragments
export const STATUS_NO_SAMPLES = 'nessun campione';

// Basemap-switcher labels (Leaflet control rendered by MapCommon).
export const BASEMAP_OSM  = 'OSM';
export const BASEMAP_TOPO = 'Topo';
export const BASEMAP_SAT  = 'Satellite';

// Map control + tool labels (zoom in the MapCommon shim; measure/location/
// sidebar in map-tools).  Distances localize via format.js — {d} is already a
// locale-formatted number, so these stay separator-agnostic.
export const MAP_ZOOM_IN = 'Ingrandisci';
export const MAP_ZOOM_OUT = 'Rimpicciolisci';
export const MAP_MEASURE_TITLE = 'Misura distanza';
export const MAP_LOCATION_TITLE = 'Mostra posizione';
export const MAP_SIDEBAR_TITLE = 'Mostra/nascondi pannello';
export const MAP_LOCATION_CURRENT = 'Posizione attuale';
export const MAP_LOCATION_ACCURACY = 'Precisione: ±{m} m';
export const MAP_LOCATION_ERROR = 'Impossibile determinare la posizione: {msg}';
export const MAP_DISTANCE_M = '{d} m';
export const MAP_DISTANCE_KM = '{d} km';

// ---------------------------------------------------------------------------
// Piano di taglio (mirrors of config/strings_it.py additions).
// ---------------------------------------------------------------------------

// Harvest-plan-item state machine display labels (mirror STATE_* in
// config/strings_it.py).  Keyed by integer values from HarvestPlanItemState.
export const STATE_PLANNED    = 'pianificato';
export const STATE_MARKED     = 'martellato';
export const STATE_OPEN       = 'cantiere aperto';
export const STATE_HARVESTING = 'in prelievo';
export const STATE_CLOSED     = 'cantiere chiuso';

// Plan-item / harvest boolean flag labels.  At most two of the three
// co-occur in practice; rendered as a comma-joined string in the
// calendar's Note column.
export const FLAG_DAMAGED     = 'Catastrofato';
export const FLAG_UNHEALTHY   = 'Fitosanitario';
export const FLAG_PSR         = 'PSR';

// Type-of-intervention labels for the COL_TYPE column.
export const TYPE_HIGHFOREST = 'Fustaia';
export const TYPE_COPPICE    = 'Ceduo';

// Plan-selector header (top of the Piano di taglio page).
export const LABEL_HARVEST_PLAN = 'Piano di taglio';

// Nuovo piano modal (creation) + Modifica piano modal (pencil, tabbed).
export const NEW_PLAN_TITLE           = 'Nuovo piano';
export const LABEL_PLAN_NAME          = 'Nome piano';
export const LABEL_PLAN_DESCRIPTION   = 'Descrizione';
export const ERR_PLAN_NAME_REQUIRED   = 'Nome piano obbligatorio.';
export const ERR_PLAN_YEAR_RANGE      =
  'Anno fine deve essere maggiore o uguale ad anno inizio.';
export const ERR_CSV_FILE_REQUIRED    = 'File CSV obbligatorio.';
export const EDIT_PLAN_TITLE          = 'Modifica piano';
export const TAB_DETAILS    = 'Dettagli';
export const EDIT_PLAN_TAB_CALENDAR   = 'Importa calendario da CSV';
export const EDIT_PLAN_CHECKBOX_COPPICE = 'File contiene il calendario ceduo';
export const LABEL_CSV_FILE           = 'File CSV';
export const IMPORT_LABEL             = 'Importa';

// Dangerous-delete (plan + per-item).
export const DELETE_PLAN_TITLE   = 'Elimina piano';
export const DELETE_PLAN_WARNING =
  'Il piano "{name}" e tutte le sue voci saranno ' +
  'eliminati definitivamente.';
export const ERR_PLAN_HAS_ACTIVE_ITEMS =
  'Il piano contiene voci non in stato "pianificato". ' +
  'Elimina prima le voci dipendenti.';
export const ERR_PLAN_ITEM_STATE_NOT_PLANNED =
  'La voce non è in stato "pianificato"; eliminazione non consentita.';
export const DELETE_ITEM_TITLE = 'Elimina intervento';
export const DELETE_ITEM_WARNING =
  'L\'intervento {year} {region} {parcel} sarà eliminato definitivamente.';

// Piano di taglio digest column headers (mirror config/strings_it.py).
export const COL_YEAR_START           = 'Anno inizio';
export const COL_YEAR_END             = 'Anno fine';
export const COL_YEAR_PLANNED         = 'Anno previsto';
export const COL_YEAR_ACTUAL          = 'Anno effettivo';
export const COL_ESTIMATED_BIRTH_YEAR = 'Anno di nascita stimato';
export const COL_TYPE                 = 'Tipo';
export const COL_STATE                = 'Stato';
export const COL_VOLUME_PLANNED       = 'Volume previsto (m³)';
export const COL_VOLUME_MARKED        = 'Volume martellato (m³)';
export const COL_VOLUME_ACTUAL        = 'Volume effettivo (m³)';
export const COL_INTERVENTION_AREA_HA = 'Superficie intervento (ha)';
export const COL_PARCEL_AREA_HA       = 'Superficie totale (ha)';
export const COL_PERIOD_Y             = 'Turno (a)';
export const COL_OPERATOR             = 'Operatore';
export const COL_NUMBER               = 'Numero';
export const COL_H_MEASURED           = 'h misurata';
export const COL_FUNCTION             = 'funzione';
export const COL_A                    = 'a';
export const COL_B                    = 'b';
export const COL_R2                   = 'r²';
export const COL_N_REGRESSION         = 'n';
export const COL_DENSITY              = 'Densità (q/m³)';

// View/edit-item page.
export const VIEW_ITEM_TITLE       = 'Intervento';
export const LABEL_OPEN_WORKSITE   = 'Apri cantiere';
export const LABEL_CLOSE_WORKSITE  = 'Chiudi cantiere';
export const LABEL_WORKSITE_OPENED = 'Apertura cantiere';
export const LABEL_WORKSITE_CLOSED = 'Chiusura cantiere';
export const ERR_DATE_REQUIRED     = 'Data obbligatoria.';
export const SECTION_HARVESTS      = 'Prelievi';
export const SECTION_MARK          = 'Martellate';
export const LABEL_VOLUME_TOTAL    = 'Volume totale';
export const LABEL_MASS_TOTAL      = 'Massa totale';
export const COL_WORKSITE          = 'Cantiere';

// Marks (tree-mark CRUD + CSV import in the per-item modal).
export const NEW_MARK_LABEL       = '+ Nuovo albero';
export const IMPORT_MARKS_LABEL   = '+ Importa martellata';
export const MARK_CLOSED_BANNER   =
  'Il cantiere è chiuso, non si possono aggiungere martellate.';
export const NEW_MARK_TITLE       = 'Nuovo albero martellato';
export const EDIT_MARK_TITLE      = 'Modifica albero martellato';
export const IMPORT_MARKS_TITLE   = 'Importa martellata da CSV';
export const LABEL_D_CM           = 'D (cm)';
export const LABEL_H_M            = 'h (m)';
export const LABEL_OPERATOR       = 'Operatore';
export const LABEL_H_MEASURED     = 'h misurata';
export const ERR_D_CM_REQUIRED    = 'D obbligatorio.';
export const ERR_H_M_REQUIRED     = 'h obbligatoria.';
export const ERR_SPECIES_REQUIRED = 'Specie obbligatoria.';
export const ERR_OPERATOR_REQUIRED = 'Operatore obbligatorio.';
export const MARK_NULL_VOLUME_NOTE =
  'volume e massa non calcolati per alcune specie rare';

// Per-item exports use the item id; constants here are filename prefixes
// (the JS code appends `<id>.csv`).
export const CSV_MARKS_PREFIX    = 'martellate_';
export const CSV_HARVESTS_PREFIX = 'prelievi_';
