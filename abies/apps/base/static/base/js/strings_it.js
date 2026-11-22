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
export const DELETE_REFERENCE_CONFIRM = name => `Vuoi eliminare «${name}»?`;
export const REFERENCE_ALREADY_DELETED = 'Questa voce è già stata eliminata.';
export const CANCEL = 'Annulla';
export const DELETE_SELECTED_CONFIRM = count =>
  `Eliminare definitivamente queste ${count} righe? I dati cancellati non potranno essere recuperati.`;
export const DELETE_SELECTED_ROW = (id, description) =>
  description ? `${description} (ID ${id})` : `ID ${id}`;
export const CONFIRM = 'Conferma';

// Tables
export const FILTER_LABEL = 'Filtra';
export const SEARCH_PLACEHOLDER = 'Cerca...';
export const EXPORT = 'Esporta';
export const EXPORT_ALL = 'Esporta tutto';
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
  exportCSV: EXPORT,
  add: 'Aggiungi',
  empty: NO_RESULTS,
  actionEdit: ACTION_EDIT,
  actionDelete: ACTION_DELETE,
  selectForDelete: 'Seleziona elementi da eliminare',
  cancelSelection: 'Annulla selezione',
  selectRow: 'Seleziona riga {id}',
  selectedCount: '{count} selezionati',
  deleteSelected: 'Elimina selezionati ({count})',
  selectionCleared: 'Selezione azzerata: i filtri o i dati sono cambiati.',
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
export const IPSO_MODE_MARTELLATE_LABEL = 'Martellate';
export const IPSO_MODE_SAMPLES_LABEL = 'Rilevamenti predefiniti';
export const IPSO_MODE_FREE_SURVEY_LABEL = 'Rilevamenti liberi';
export const IPSO_MODE_POINTS_LABEL = 'Rilevamento punti';
export const IPSO_POINT_LABEL = 'Punto';
export const IPSO_TRACE_LABEL = 'Traccia';
export const IPSO_MAX_ACCURACY = 'Errore GPS max (m)';
export const IPSO_FIX_COUNT = 'Posizioni';
export const IPSO_START_TIME = 'Inizio';
export const IPSO_END_TIME = 'Fine';
export const IPSO_MODE_OBSERVATIONS_LABEL = 'Osservazioni';
export const IPSO_REFERENCE_LEGACY_CONVERTED = 'Convertito da CSV storico';
export const IPSO_COL_STATE = 'Stato';
export const IPSO_COL_WORK_PACKAGE = 'Contesto';
export const IPSO_COL_REFERENCE = 'Riferimento';
export const IPSO_COL_TARGET = 'Destinazione';
export const IPSO_COL_ERROR = 'Errore';
export const IPSO_ACTION_OPEN = 'Apri';
export const IPSO_ACTION_OPEN_ICON = '\u{1F50D}\u{FE0E}';
export const IPSO_ACTION_REJECT = 'Rifiuta';
export const IPSO_LOADING_DETAIL = 'Caricamento dettaglio...';
export const IPSO_REJECT_CONFIRM = 'Rifiutare questo caricamento Ipso?';
export const IPSO_DELETE_TITLE = 'Elimina sessione Ipso';
export const IPSO_DELETE_WARNING =
  'Questa operazione elimina i file ricevuti da Ipso. I dati già importati non verranno eliminati.';
export const IPSO_DELETE_EXPORT_REQUIRED =
  'Per sicurezza, scarica una copia della sessione prima di procedere.';
export const IPSO_MODE_SAVE_ERROR_IMPORTED =
  'Una sessione già importata non può cambiare modalità.';
export const IPSO_SESSION_TITLE = (id) => `Sessione ${id}`;
export const IPSO_PREVIEW_TITLE = (n) => `Anteprima record (${n})`;
export const IPSO_EMPTY_RECORDS = 'Nessun record.';
export const IPSO_SUMMARY_EMPTY = 'Nessun dato Ipso.';
export const IPSO_SUMMARY = (n, pending) => `${n} caricamenti, ${pending} da importare.`;
export const IPSO_INCLUDE_IMPORTED = 'Anche dati già importati';
export const IPSO_TARGET_PLAN_LABEL = 'Piano di taglio';
export const IPSO_TARGET_SURVEY_LABEL = 'Rilevamento';
export const IPSO_TARGET_SELECT = 'Seleziona destinazione';
export const IPSO_IMPORT_CONFIRM = 'Importare i dati nel piano selezionato?';
export const IPSO_IMPORT_SAMPLES_CONFIRM = 'Importare i dati nel rilevamento selezionato?';
export const IPSO_IMPORT_FREE_SURVEY_CONFIRM = 'Importare i dati nel rilevamento libero selezionato?';
export const IPSO_IMPORT_OBSERVATIONS_CONFIRM = 'Importare le osservazioni?';
export const IPSO_COL_SEQ = '#';
export const IPSO_EMPTY_VALUE = '-';

// Validation (client-side)
export const ERR_DATE_FUTURE = 'La data non può essere nel futuro.';
export const ERR_SPECIES_PCT_SUM = 'Le percentuali delle specie devono sommare a 100.';
export const ERR_TRACTOR_PCT_SUM = 'Le percentuali dei trattori devono sommare a 100.';
export const ERR_CREW_REQUIRED = 'Squadra obbligatoria.';
export const ERR_HOURS_POSITIVE = 'Le ore devono essere maggiori di zero.';
export const ERR_CREDITS_POSITIVE = 'I quintali devono essere maggiori di zero.';
export const ERR_NAME_REQUIRED = 'Nome obbligatorio.';
export const ERR_REPORTS_EMPTY = 'Nessuna produzione nel mese selezionato.';

// Charts
export const CHART_TOTAL = 'Totale';
export const CHART_OTHER = 'Altro';
export const PRELIEVI_SECTION_SUMMARY = 'Riassunto';
export const PRELIEVI_SECTION_CALENDAR = 'Calendario';
export const PRELIEVI_SECTION_HARVESTS = 'Interventi';
export const PRELIEVI_CHART_BY_YEAR = 'Per Anno';
export const PRELIEVI_CHART_BY_PARCEL = 'Per Particella';
export const PRELIEVI_CHART_MONTHS = 'Per mese';
export const PRELIEVI_RESET_FILTERS = 'Azzera filtri';
export const PRELIEVI_CALENDAR_EMPTY = 'Nessun prelievo nel filtro corrente.';

// Prelievi columns
export const COL_DATE = 'Data';
export const COL_SURVEY_DATE = 'Data di rilevamento';
export const COL_PARCEL = 'Particella';
export const COL_CREW = 'Squadra';
export const COL_VDP = 'VDP';
export const COL_QUINTALS = 'Q.li';
export const COL_HOURS = 'Ore';
export const COL_CREDITS_Q = 'Quintali';
export const COL_ID = 'ID';
export const COL_NOTE = 'Note';
export const COL_TEXT = 'Testo';
export const COL_EXTRA_NOTE = 'Altre note';
export const COL_VOLUME_M3 = 'Volume (m³)';
export const COL_ACTIVE = 'Attivo';
export const CSV_COL_SORT_ORDER = 'Ordine';
export const COL_MINOR = 'Minore';
export const COL_CONIFER = 'Conifera';
export const COL_BROADLEAF = 'Latifoglia';
export const SPECIES_TYPE_UNCLASSIFIED = '—';
export const COL_SORT_ORDER = 'Sort order';
export const SPECIES_OTHER = 'Altro';
export const SPECIES_OTHER_CONIFER = 'Altra conifera';
export const SPECIES_OTHER_BROADLEAF = 'Altra latifoglia';

// Height computation tools in Species. (Accept/Reject/title are literal in the
// candidate <template>, per docs/ui-design-patterns.md.)
export const HYPSO_DESC_NONE = 'Nessun parametro ipsometrico attivo.';
export const HYPSO_SOURCE_COMPUTED_LABEL = 'Calcolato';
export const HYPSO_SOURCE_IMPORTED_LABEL = 'Importato da CSV';
export const HYPSO_DESC_MIN_N = 'N minimo';
export const HYPSO_DESC_SURVEYS = 'Rilevamenti';
export const HYPSO_DESC_HEIGHT_PLOTS = 'Altezze: rilevamenti ipsometrici';
export const PASSWORD_CURRENT = 'Password attuale';
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
export const LABEL_MONTH = 'Mese';

// Squadre PDFs
export const SQUADRE_REPORT_HOURS = 'Ore lavorate';
export const SQUADRE_REPORT_PRODUCTION = 'Produzione';
export const SQUADRE_REPORT_TOTAL_PRODUCTION = 'Totale produzione';
export const SQUADRE_REPORT_CREDITS = 'Acconti';
export const SQUADRE_REPORT_TOTAL = 'Totale';
export const SQUADRE_REPORT_DETAIL = 'Dettaglio produzione';
export const LABEL_PERCENT = '%';

// Export filenames
export const CSV_PRELIEVI = 'prelievi.csv';
export const CSV_SQUADRE_HOURS = 'ore-squadre.csv';
export const CSV_SQUADRE_CREDITS = 'acconti-squadre.csv';
export const PDF_SQUADRE_REPORTS = (month) => `rendiconti-squadre-${month}.pdf`;
export const CSV_CREWS = 'squadre.csv';
export const CSV_VEHICLES = 'mezzi.csv';
export const NEW_VEHICLE_LABEL = 'Nuovo mezzo';
export const NEW_OBSERVATION_CATEGORY_LABEL = 'Nuova categoria';
export const NEW_ROAD_TYPE_LABEL = 'Nuovo tipo di viabilità';
export const NEW_ECLASS_LABEL = 'Nuova classe economica';
export const CSV_SPECIES = 'specie.csv';
export const CSV_USERS = 'utenti.csv';
export const CSV_OBSERVATION_CATEGORIES = 'categorie-osservazioni.csv';
export const CSV_ROAD_TYPES = 'tipi-di-viabilita.csv';
export const CSV_ECLASSES = 'classi-economiche.csv';
export const CSV_AUDIT = 'controllo.csv';
export const CSV_SAMPLED_TREES = 'alberi-campionati.csv';

// Samples
export const SAMPLES_MAP_TREES = 'Mappa degli alberi';
export const SAMPLES_MAP_AREAS = 'Mappa delle aree di saggio';
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
export const SAMPLES_FREE_SURVEYS_GROUP = '-- Rilevamenti liberi --';
export const SAMPLES_STRUCTURED_SURVEYS_GROUP = '-- Rilevamenti predefiniti --';
export const SAMPLES_FREE_SURVEY_OPTION = (name, nTrees) =>
  `${name} (${SAMPLES_TREE_COUNT(nTrees)})`;
export const SAMPLES_SURVEY_OPTION = (name, visited, total) =>
  `${name} (${visited}/${total} aree)`;
export const SAMPLES_GRID_SUMMARY = (areas, regions, surveys, updatedAt) =>
  `${areas} aree · ${regions} · ${surveys} rilevamenti · aggiornata ${updatedAt}`;
export const SAMPLES_SURVEY_OVERVIEW = (first, last, count) =>
  `${first ? `Dal ${first} al ${last}.` : 'Nessun campione.'} ${count} ` +
  `${count === 1 ? 'albero campionato' : 'alberi campionati'}.`;
export const DELETE_GRID_AREAS_WARNING = (n) =>
  `${n} aree saranno eliminate. ${DELETE_CONFIRM}`;

// Shared lat/lng input
export const USE_CURRENT_LOCATION = 'Usa GPS';

// Cascade-delete warning (Section 1/2 garbage on populated resources)
export const CASCADE_CONFIRM_TITLE = 'Conferma eliminazione';
export const CASCADE_WARN_SURVEY = (nSamples, nTrees) =>
  `Questa operazione cancellerà ${nSamples} campioni e ${nTrees} ` +
  'misure di alberi che non possono essere recuperati.';
export const CASCADE_EXPORT_REQUIRED =
  'Per sicurezza, esporta i dati prima di procedere all\'eliminazione.';
export const RENAME_TITLE_GRID = 'Modifica griglia';
export const RENAME_TITLE_SURVEY = 'Modifica rilevamento';
export const GRID_IMPORT_TITLE  = 'Importa aree da CSV';
export const SURVEY_IMPORT_TITLE = 'Importa alberi da CSV';
export const GRID_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Area saggio, Lon, Lat, Quota, Raggio. ' +
  'Quota può essere vuota. Sono accettate anche le intestazioni Alt. (m) e Raggio (m).';
export const SURVEY_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Albero, Pollone, Matricina, ' +
  'D_cm, H_m, L10_mm, Pressler, Genere, Fustaia. Per rilevamenti ' +
  'strutturati serve anche Area saggio. Opzionali: Data, Campione, PAI, Numero PAI, H_measured, ' +
  'Lat, Lon, Acc_m.';

// CSV import status (large files can take many seconds)
export const CSV_IMPORT_IN_PROGRESS =
  'Importazione in corso, attendere…  ' +
  'Non chiudere la finestra né cliccare di nuovo "Importa".';
export const CSV_EXTRA_ERRORS = (n) => `… +${n} altri errori`;
export const IMPORT_WARNINGS_CONFIRM = 'Procedi';

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

// CSV export filenames for the symmetric "Esporta" buttons on
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
export const GRID_PLANNER_REGION_OPTION = (name, nParcels) =>
  `${name} (${nParcels} particelle)`;
export const STATUS_PLAN_COMPLETE = n => `Pianificazione completata: ${n} punti.`;
export const STATS_POINTS = (n, target) => `Punti: ${n} (obiettivo: ${target})`;
export const STATS_TOTAL_AREA_HA = ha => `Superficie totale: ${ha} ha`;
export const STATS_AREA_PER_POINT_M2 = area => `Area singola adc: ${area} m²`;
export const TOOLTIP_ADC = (n, regionName, parcelName) =>
  `adc ${n} · ${regionName} ${parcelName}`;
// Sample-area marker tooltips on the Griglie / Rilevamenti maps.  Flat
// templates (not concatenated fragments) so word order stays translatable.
export const TOOLTIP_SAMPLE_AREA = area =>
  `${area.regionName} ${area.parcelName} / adc ${area.number}`;
export const TOOLTIP_SAMPLE_AREA_VISITED = (area, treesLabel) =>
  `${area.regionName} ${area.parcelName} / adc ${area.number} / ${treesLabel}`;

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
export const BOSCO_LOADING_HILLSHADE = 'Caricamento del rilievo ombreggiato...';
export const BOSCO_HILLSHADE_UNAVAILABLE = 'Rilievo ombreggiato non disponibile per questa compresa.';
export const BOSCO_SATELLITE_UNAVAILABLE = 'Dati satellitari non disponibili.';
export const BOSCO_NO_SATELLITE = 'Nessun dato satellitare disponibile.';
export const BOSCO_LOADING_RASTER = 'Caricamento raster...';
export const BOSCO_RASTER_UNAVAILABLE = 'Raster non disponibile.';
export const BOSCO_REGION_PARCELS = 'Particelle';
export const BOSCO_DENDROMETRY_UNAVAILABLE = 'Dendrometria non disponibile.';
export const BOSCO_NO_DENDROMETRY = 'Nessun dato dendrometrico.';
export const BOSCO_TREE_COUNT = 'Numero alberi';
export const BOSCO_TREE_COUNT_PER_HA = 'Numero alberi/ha';
export const BOSCO_TOTAL_TREES = (n) => `Alberi totali: ${n}`;
export const BOSCO_TREES_PER_HA = (n) => `Alberi per ettaro: ${n}`;
export const BOSCO_VOLUME_PER_HA = 'Volume (m³/ha)';
export const BOSCO_BASAL_AREA_PER_HA = 'Area bas. (m²/ha)';
export const BOSCO_BASAL_AREA_VALUE = value => `${value} m²`;
export const BOSCO_BASAL_AREA_PER_HA_VALUE = value => `${value} m²/ha`;
export const BOSCO_TOTAL_VOLUME = value => `Volume totale: ${value}`;
export const BOSCO_VOLUME_PER_HA_SUMMARY = value => `Volume per ettaro: ${value}`;
export const BOSCO_TOTAL_BASAL_AREA = value => `Area totale: ${value}`;
export const BOSCO_BASAL_AREA_PER_HA_SUMMARY = value =>
  `Area basimetrica per ettaro: ${value}`;
export const BOSCO_AVG_DIAMETER = (mean, sigma) =>
  `Diametro medio: ${mean}, σ=${sigma}`;
export const BOSCO_HISTORICAL_PRODUCTION_UNAVAILABLE =
  'Produzione storica non disponibile.';
export const BOSCO_NO_HISTORICAL_HARVEST = 'Nessun prelievo storico.';
export const BOSCO_INTERVENTIONS = (n) => `${n} interventi`;
export const BOSCO_LAT_LON_REQUIRED = 'Lat e Lon obbligatorie.';
export const BOSCO_POSITION = 'Posizione';
export const BOSCO_INSERT_PAI_TREE_HERE = 'Inserire un nuovo albero qui?';
export const BOSCO_PAI_TREE_META = (parcel, number) => `${parcel} · n. ${number}`;
export const BOSCO_NO_PAI_TREES = 'Nessuna pianta.';
export const BOSCO_LOADING_OBSERVATIONS = 'Caricamento osservazioni...';
export const BOSCO_OBSERVATIONS_UNAVAILABLE = 'Osservazioni non disponibili.';
export const BOSCO_NO_OBSERVATIONS = 'Nessuna osservazione.';
export const BOSCO_NO_OBSERVATION_CATEGORIES = 'Nessuna categoria.';
export const BOSCO_OBSERVATIONS_SHOWN = (shown, total) =>
  `${shown} di ${total} osservazioni visualizzate.`;
export const BOSCO_OBSERVATION_DETAIL_TITLE = 'Osservazione';
export const BOSCO_OBSERVATION_CHOOSER_TITLE = 'Osservazioni sovrapposte';
export const BOSCO_OBSERVATION_GROUP_TITLE = count => `${count} osservazioni`;
export const BOSCO_OBSERVATION_CATEGORY = 'Categoria';
export const BOSCO_OBSERVATION_NEW_TITLE = 'Nuova osservazione';
export const BOSCO_OBSERVATION_EDIT_TITLE = 'Modifica osservazione';
export const BOSCO_OBSERVATION_PHOTO_ADD = '+ Aggiungi';
export const BOSCO_OBSERVATION_PHOTO_REMOVE = 'Rimuovi foto';
export const BOSCO_OBSERVATION_PHOTO_MAP = 'Posizioni foto';
export const BOSCO_OBSERVATION_PDF = 'Genera PDF';
export const BOSCO_OBSERVATION_REPORT_TITLE = category => `Rapporto ${category}`;
export const BOSCO_OBSERVATION_PDF_ERROR = 'Impossibile generare il PDF.';
export const BOSCO_METRIC_AGE = 'Età media';
export const BOSCO_METRIC_BOUNDARIES = 'Solo confini';
export const BOSCO_METRIC_TYPE = 'Governo';
export const COL_GOVERNANCE = 'Governo';
export const BOSCO_METRIC_ALTITUDE = 'Altitudine media';
export const BOSCO_METRIC_AVG_GRADE = 'Pendenza media';
export const BOSCO_METRIC_HISTORICAL_HARVEST = 'Prelievo storico';
export const BOSCO_METRIC_FUTURE_HARVEST = 'Prelievo previsto';
export const BOSCO_PLAN_SOURCE_ERROR = 'Impossibile aggiornare il piano di taglio. Riprova.';
export const BOSCO_NO_HARVEST_PLANS = 'Nessun piano di taglio disponibile.';
export const BOSCO_METRIC_CUTTING_CALENDAR = 'Calendario di taglio';
export const BOSCO_PLANNED_CUT_YEARS = 'Anni di taglio previsti';
export const BOSCO_NO_PLANNED_CUTS = 'Nessun taglio previsto';
export const BOSCO_PRINT_GEOMETRY_ERROR = 'Impossibile preparare la mappa: controlla i confini delle particelle.';
export const BOSCO_PRINT_LABEL_ERROR = 'Impossibile disporre tutte le etichette in modo leggibile. Prova una scala con denominatore minore.';
export const BOSCO_PRINT_SCALE_ERROR = 'Inserisci un denominatore di scala intero maggiore di zero.';
export const BOSCO_PRINT_CONTEXT_LOADING = 'Caricamento degli strati cartografici…';
export const BOSCO_PRINT_CONTEXT_MISSING = 'Strati cartografici non ancora disponibili per questa compresa.';
export const BOSCO_PRINT_CONTEXT_STALE = 'Gli strati cartografici devono essere aggiornati per i confini attuali.';
export const BOSCO_PRINT_CONTEXT_ERROR = 'Impossibile caricare gli strati cartografici. Puoi scaricare la mappa delle particelle.';
export const BOSCO_PRINT_CONTEXT_CONTOURS = interval => `Curve ogni ${interval} m; curve principali ogni 100 m.`;
export const BOSCO_PRINT_DSM_NOTE = 'Curve ottenute da Copernicus GLO-30 (DSM, circa 30 m): includono la vegetazione e non rappresentano il terreno nudo.';
export const BOSCO_PRINT_OSM_CREDIT = '© OpenStreetMap contributors · ODbL · www.openstreetmap.org/copyright';
export const BOSCO_PRINT_DSM_DESCRIPTION = "Curve da Copernicus GLO-30 (DSM). Le organizzazioni responsabili del programma Copernicus non assumono responsabilità per l'uso del Copernicus WorldDEM-30.";
export const BOSCO_PRINT_DSM_CREDIT = 'Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.';
export const BOSCO_PRINT_ELEVATION_CREDIT = 'Curve derivate dal modello di elevazione locale.';
export const BOSCO_PRINT_ELEVATION = elevation => `${elevation} m`;
export const BOSCO_PRINT_SIZE_ERROR = (width, height) =>
  `La mappa misura ${width} × ${height} cm e non entra in un foglio A0 con margini, intestazione ed etichette. Prova una scala con denominatore maggiore.`;
export const BOSCO_PRINT_SCALE = denominator => `Scala 1 : ${denominator}`;
export const BOSCO_PRINT_DIMENSIONS = (width, height) => `Dimensioni della mappa: ${width} × ${height} cm.`;
export const BOSCO_PRINT_PAPER = (paper, width, height, landscape) =>
  `Formato consigliato: ${paper} ${landscape ? 'orizzontale' : 'verticale'} (${width} × ${height} cm).`;
export const BOSCO_PRINT_PARCEL_MISMATCH = 'I confini e l’elenco delle particelle non corrispondono. Controlla i dati della compresa prima di esportare.';
export const BOSCO_PRINT_SELECT_PLAN = 'Seleziona un piano di taglio e attendi il caricamento dei dati.';
export const BOSCO_PRINT_DOWNLOAD_ERROR = 'Impossibile generare il PDF. Riprova.';
export const BOSCO_CALENDAR_MAP_TITLE = region => `Calendario di taglio - ${region}`;
export const BOSCO_CALENDAR_MAP_PLAN = (plan, ranges) =>
  `${plan.name}${ranges ? ` (${ranges})` : ''}`;
export const BOSCO_CALENDAR_HIGH_FOREST_YEARS = years => `fustaia ${years}`;
export const BOSCO_CALENDAR_COPPICE_YEARS = years => `ceduo ${years}`;
export const BOSCO_PRINT_PARCEL_NAME = (parcel, coppice) => `${parcel} (${coppice ? 'C' : 'F'})`;
export const BOSCO_CALENDAR_MAP_LEGEND = '(C) = Ceduo, (F) = Fustaia, (-) = Nessun taglio previsto.';
export const BOSCO_PRINT_MAP_FOOTER = date => `Esportato il ${date}`;
export const BOSCO_PRINT_UNSUPPORTED = 'La caratteristica selezionata non supporta la stampa.';
export const BOSCO_PRINT_METRIC_TITLE = (metric, region) => `${metric} - ${region}`;
export const BOSCO_PRINT_BOUNDARIES_TITLE = region => `Confini - ${region}`;
export const BOSCO_PRINT_DIALOG_TITLE = metric => `Mappa stampabile - ${metric}`;
export const BOSCO_PRINT_METRIC_LEGEND = '(C) = Ceduo, (F) = Fustaia, (-) = Dato non disponibile.';
export const BOSCO_PRINT_BOUNDARIES_LEGEND = '(C) = Ceduo, (F) = Fustaia.';
export const BOSCO_PRINT_BOUNDARIES_FILE = 'confini';
export const BOSCO_PRINT_AGE_FILE = 'eta_media';
export const BOSCO_PRINT_TYPE_FILE = 'governo';
export const BOSCO_PRINT_ALTITUDE_FILE = 'altitudine_media';
export const BOSCO_PRINT_SLOPE_FILE = 'pendenza_media';
export const BOSCO_PRINT_HISTORY_FILE = 'prelievo_storico';
export const BOSCO_PRINT_FORECAST_FILE = 'prelievo_previsto';
export const BOSCO_PRINT_PER_HA_NOTE = cadastral =>
  `Valori per ettaro calcolati sulle aree ${cadastral ? 'catastali' : 'geometriche'}.`;
export const BOSCO_PRINT_METRIC_FILENAME = (field, region, { perHa = false, plan } = {}) =>
  `${field}_${region.replace(/[\\/:*?"<>|]/g, '_')}${plan ? `_${plan.year_start}-${plan.year_end}` : ''}${perHa ? '_per_ettaro' : ''}.pdf`;
export const BOSCO_CALENDAR_PDF_FILENAME = (region, plan) =>
  `calendario_taglio_${region.replace(/[\\/:*?"<>|]/g, '_')}_${plan.year_start}-${plan.year_end}.pdf`;
export const BOSCO_HARVEST_METRIC = 'Prelievo';
export const BOSCO_QUINTALS_PER_HA = 'Q.li/ha';
export const BOSCO_QUINTALS_PER_HA_VALUE = value => `${value} q/ha`;
export const BOSCO_VOLUME_PER_HA_VALUE = value => `${value} m³/ha`;
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
export const COL_ACC_M           = 'Acc. (m)';
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
export const COL_CUTTING_PLAN    = 'Piano dei tagli';
export const COL_HARVEST_MECHANISM = 'Sistema di esbosco';
export const COL_STUMPS_PER_HA    = 'Ceppaie / ha';
export const COL_INTERVENTION_INTERVAL = 'Intervallo interventi';
export const COL_STANDARDS_PER_HA = 'Matricine / ha';
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
export const COL_OBSERVATION_CATEGORIES = 'Categorie';
export const COL_PHOTO_COUNT = 'Foto';

// Table display labels; digest keys and CSV headers are independent.
const TREE_MEASUREMENT_LABELS = {
  diameter: 'Diam (cm)',
  height: 'Alt (m)',
  volume: 'Vol (m3)',
  mass: 'Massa (q)',
  accuracy: 'Acc (m)',
};
export const SAMPLED_TREE_COLUMN_LABELS = {
  area: 'Area',
  number: 'Albero',
  shoot: 'Poll',
  standard: 'Mat',
  ...TREE_MEASUREMENT_LABELS,
};
export const MARKED_TREE_COLUMN_LABELS = {
  ...TREE_MEASUREMENT_LABELS,
  heightMeasured: 'Misura alt',
};

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
export const MAP_LOCATION_ACCURACY_LABEL = (m) => `Precisione: ±${m} m`;
export const MAP_LOCATION_ERROR = 'Impossibile determinare la posizione: {msg}';
export const MAP_LOCATION_ERROR_MESSAGE = (msg) => `Impossibile determinare la posizione: ${msg}`;
export const MAP_DISTANCE_M = '{d} m';
export const MAP_DISTANCE_KM = '{d} km';
export const MAP_DISTANCE_M_LABEL = (d) => `${d} m`;
export const MAP_DISTANCE_KM_LABEL = (d) => `${d} km`;

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
export const DELETE_PLAN_WARNING = name =>
  `Il piano "${name}" e tutte le sue voci saranno eliminati definitivamente.`;
export const DELETE_ITEM_TITLE = 'Elimina intervento';
export const DELETE_ITEM_WARNING = (year, region, parcel) =>
  `L'intervento ${year} ${region}${parcel ? ` ${parcel}` : ''} sarà eliminato definitivamente.`;

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
export const VIEW_ITEM_HEADING = (planName, year, location) =>
  `${VIEW_ITEM_TITLE} del ${planName}, anno ${year}, ${location}`;
export const LABEL_OPEN_WORKSITE   = 'Apri cantiere';
export const LABEL_REOPEN_WORKSITE = 'Riapri cantiere';
export const LABEL_CLOSE_WORKSITE  = 'Chiudi cantiere';
export const LABEL_WORKSITE_OPENED = 'Apertura cantiere';
export const LABEL_WORKSITE_CLOSED = 'Chiusura cantiere';
export const ERR_DATE_REQUIRED     = 'Data obbligatoria.';
export const ERR_BOSCO_OBSERVATION_TEXT_REQUIRED = 'Descrizione obbligatoria.';
export const SECTION_HARVESTS      = 'Prelievi';
export const SECTION_MARK          = 'Martellate';
export const SECTION_DENDROMETRIC_SUMMARY = 'Riassunto dendrometrico';
export const LABEL_VOLUME_TOTAL    = 'Volume totale';
export const LABEL_MASS_TOTAL      = 'Massa totale';
export const COL_WORKSITE          = 'Cantiere';

// Marks (tree-mark CRUD + CSV import in the per-item modal).
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
export const DENDROMETRY_VOLUME = 'Volume';
export const DENDROMETRY_BASAL_AREA = 'Area basimetrica';
export const TREE_DENDROMETRY_EMPTY = 'Nessun dato dendrometrico nel filtro corrente.';

// Per-item exports use the item id; constants here are filename prefixes
// (the JS code appends `<id>.csv`).
export const CSV_MARKS_PREFIX    = 'martellate_';
export const CSV_HARVESTS_PREFIX = 'prelievi_';
export const VOLUME_UNAVAILABLE = 'Volume non calcolabile: equazione di volume mancante.';
export const HEIGHT_UNAVAILABLE = 'Equazione di altezza mancante per questa compresa: inserisci h.';
export const COL_HEIGHT_EQUATIONS = 'Eq. altezza';
export const COL_VOLUME_EQUATION = 'Eq. volume';
export const EQUATION_COVERAGE = count => `✓ · ${count}`;
export const HYPSO_SOURCE_MANUAL_LABEL = 'Manuale';
export const HYPSO_SOURCE_MIXED_LABEL = 'Misto';

// Review of staged tree imports and recovery of excluded rows.
export const IPSO_IMPORT_REVIEW_SUMMARY = (valid, excluded) =>
  `Righe totali: ${valid + excluded}\nRighe escluse a causa di errore: ${excluded}\nRighe importabili: ${valid}`;
export const IPSO_IMPORT_VALID_ROWS = count => `Importa righe valide (${count})`;
export const IPSO_IMPORT_EXPORT_REQUIRED = 'Per sicurezza, esporta le righe escluse prima di importare quelle valide.';
export const IPSO_EXPORT_EXCLUDED = 'Esporta righe escluse';
export const IPSO_IMPORT_RESULT = (imported, excluded) => `Righe importate: ${imported}. Righe escluse: ${excluded}.`;

export const CSV_IMPORT_REFRESH_FAILED = 'Importazione completata. Impossibile aggiornare la vista; ricarica la pagina.';

export const NEW_TREE_LABEL = 'Nuovo albero';
export const NEW_INTERVENTION_LABEL = 'Nuovo intervento';
export const NEW_HARVEST_LABEL = 'Nuovo prelievo';
export const CALENDAR_IMPORT_TITLE = coppice =>
  `Importa calendario ${coppice ? 'ceduo' : 'fustaia'} da CSV`;

export const CARTOGRAPHY_NEW_REGION = 'Nuova compresa';
export const CARTOGRAPHY_EDIT_REGION = 'Modifica compresa';
export const CARTOGRAPHY_DELETE_REGION = (name, counts) =>
  `Eliminare la compresa «${name}» e tutto il suo contenuto cartografico? `
  + `Particelle: ${counts.parcels}. Contorni: ${counts.outlines}. Strade: ${counts.roads}. Disegni: ${counts.drawings}.`;
export const CARTOGRAPHY_IMPORT_COUNT = n => `Disegni aggiunti: ${n}.`;
export const CARTOGRAPHY_IMPORT_FILE_TYPE = 'Scegli un file GeoJSON, GPX o DXF.';
export const CARTOGRAPHY_IMPORT_FILE_ERROR = (name, error) => `${name} — ${error}`;
export const CARTOGRAPHY_IMPORT_FILE_PROGRESS = (done, total) => `File importati: ${done} di ${total}.`;
export const CARTOGRAPHY_IMPORT_NO_SURVEYS = 'Nessun rilevamento con posizioni in questa compresa.';
export const CARTOGRAPHY_IMPORT_SURVEY_OPTION = ({name, structured, count}) =>
  `${name} · ${structured ? 'Predefinito' : 'Libero'} · ${count} ${count === 1 ? 'posizione' : 'posizioni'}`;
export const CARTOGRAPHY_CLEAR = (region, n) => `Eliminare tutti i disegni di ${region} (${n})?`;
export const CARTOGRAPHY_FORGET = name => `Eliminare il rilievo originale «${name}»?`;
export const CARTOGRAPHY_DRAG = 'Trascina i vertici e salva le modifiche.';
export const CARTOGRAPHY_EMPTY_EDIT = 'Non ci sono più punti in questa modifica.';
export const SATELLITE_SECRET_CONFIGURED = 'Segreto salvato';
export const SATELLITE_REMOVE_CONFIRM = 'Eliminare le credenziali Copernicus salvate?';
export const TERRAIN_REFRESH_NO_PARCELS = 'La compresa deve avere almeno una particella con confini pubblicati.';
export const CARTOGRAPHY_ROAD_NEEDS_LINE = 'Per salvare la strada, la geometria deve essere una sola linea.';
export const CARTOGRAPHY_PROMOTE_GEOMETRY = 'Il disegno deve contenere una sola linea oppure solo poligoni.';
export const CARTOGRAPHY_PROMOTE_MULTIPLE_LINES = 'Unisci i tratti o elimina quelli in più per creare una strada.';
export const CARTOGRAPHY_PROMOTE_ROAD_TYPE = 'Aggiungi un tipo di viabilità attivo per creare una strada.';
export const CARTOGRAPHY_OPEN = 'Importa in Cartografia';
export const CARTOGRAPHY_NO_DESTINATION = 'Crea una compresa in Cartografia per importare il rilievo.';

export const CARTOGRAPHY_CHOOSE_TOOL = 'Scegli Linea o Poligono sulla mappa.';
export const CARTOGRAPHY_PLACE = 'Tocca la mappa per aggiungere i vertici, poi premi Termina.';
export const CARTOGRAPHY_CANCEL = 'Abbandonare le modifiche al disegno?';
export const CARTOGRAPHY_DISCARD = 'Abbandona modifiche';

export const CARTOGRAPHY_VISIBILITY = name => `Mostra/nascondi «${name}»`;
export const CARTOGRAPHY_REMOVE_ITEM = name => `Elimina «${name}»`;
export const CARTOGRAPHY_PARCEL_IN_USE = 'Questa particella è in uso';
export const CARTOGRAPHY_PARCEL_UNLINKED = 'Questa particella è priva di scheda catastale';

export const CARTOGRAPHY_ROAD_PROPERTIES = 'Proprietà strada';
export const CARTOGRAPHY_CANCEL_BOUNDARIES = 'Abbandonare le modifiche ai confini?';
export const CARTOGRAPHY_CANCEL_POINTS = 'Abbandonare le modifiche ai punti della strada?';
export const CARTOGRAPHY_INSERT_VERTEX = 'Inserisci vertice';
export const CARTOGRAPHY_SELECT_VERTEX = 'Seleziona vertice';
export const CARTOGRAPHY_SELECT_LINE = i => `Seleziona linea ${i + 1}`;
export const CARTOGRAPHY_BOUNDARY_TARGET = names => `Vertice sul confine di ${names}`;
export const CARTOGRAPHY_FOLLOW_FROM = 'Segui il confine da questo vertice';
export const CARTOGRAPHY_FOLLOW_TARGET = 'Scegli un altro vertice sullo stesso contorno.';
export const CARTOGRAPHY_FOLLOW_UNAVAILABLE = 'Il percorso verso questo vertice non è utilizzabile. Scegline un altro.';
export const CARTOGRAPHY_REMOVE_VERTEX = 'Rimuovi vertice';
export const CARTOGRAPHY_REMOVE_LINE = 'Elimina linea';
export const CARTOGRAPHY_REMOVE_POLYGON = 'Elimina poligono';
export const CARTOGRAPHY_REDUCE_VERTICES = spacing => `Riduci vertici (${spacing} m)`;
export const CARTOGRAPHY_REDUCTION_SPACING = spacing => `Distanza minima tra i vertici: ${spacing} m.`;
export const CARTOGRAPHY_JOIN_FIRST = 'Seleziona un punto o un estremo di una linea.';
export const CARTOGRAPHY_JOIN_SECOND = 'Seleziona il secondo punto o estremo da unire.';
export const CARTOGRAPHY_JOIN_POINT = i => `Punto ${i + 1}`;
export const CARTOGRAPHY_JOIN_ENDPOINT = (i, start) => `Linea ${i + 1}: ${start ? 'inizio' : 'fine'}`;
export const CARTOGRAPHY_JOIN_NEEDS_VERTICES = 'Servono almeno tre vertici distinti per chiudere un poligono.';
export const CARTOGRAPHY_JOIN_ROAD = 'Per unire i tratti della strada, seleziona gli estremi di due linee diverse.';
export const CARTOGRAPHY_SAVE_OPEN_BOUNDARIES = 'Solo i poligoni verranno salvati. Continuare?';
export const CARTOGRAPHY_VERTEX_REQUIRED = 'Questo vertice serve a mantenere un contorno valido o un confine condiviso.';

export const CARTOGRAPHY_PLACE_POLYGON = 'Aggiungi almeno tre vertici, poi tocca il primo per chiudere il poligono.';
export const CARTOGRAPHY_FIRST_VERTEX = 'Primo vertice';
export const CARTOGRAPHY_CLOSE_SKETCH = 'Chiudi il poligono';
