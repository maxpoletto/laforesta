/**
 * Italian UI string constants.
 *
 * To switch language, change the re-export in strings.js.
 */

// Shell / chrome
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
export const ADD_LABEL = 'Aggiungi';
export const NO_RESULTS = 'Nessun risultato.';
export const BOOL_YES = 'Sì';
export const BOOL_NO = 'No';
export const ACTION_EDIT = 'Modifica';
export const ACTION_DELETE = 'Elimina';

// TableWrapper localization: labels bundle and CSV format for Italian.
// Passed as `labels` / `csvFormat` options when constructing a TableWrapper.
export const TABLE_LABELS = {
  search: FILTER_LABEL,
  searchPlaceholder: SEARCH_PLACEHOLDER,
  exportCSV: EXPORT_CSV,
  add: ADD_LABEL,
  empty: NO_RESULTS,
  actionEdit: ACTION_EDIT,
  actionDelete: ACTION_DELETE,
  boolYes: BOOL_YES,
  boolNo: BOOL_NO,
};

export const TABLE_CSV_FORMAT = {
  separator: ';',
  decimal: ',',
  dateFormat: 'DD/MM/YYYY',
};

// Errors
export const ERROR_NETWORK = 'Errore di rete. Riprovare.';
export const ERROR_CONFLICT = 'Il record è stato modificato da un altro utente.';
export const ERROR_GENERIC = 'Errore imprevisto.';
export const ERROR_DELETED = 'Il record è stato eliminato da un altro utente.';
export const ERROR_RATE_LIMIT = 'Troppe richieste. Riprovare tra un minuto.';

// Validation (client-side)
export const ERR_DATE_FUTURE = 'La data non può essere nel futuro.';
export const ERR_SPECIES_PCT_SUM = 'Le percentuali delle specie devono sommare a 100.';
export const ERR_TRACTOR_PCT_SUM = 'Le percentuali dei trattori devono sommare a 100.';

// Prelievi
export const LABEL_YEARS = 'Anni';
export const RESET_FILTERS = 'Azzera filtri';

// Charts
export const CHART_PRODUCTION = 'Produzione';
export const CHART_SPECIES_BY_PARCEL = 'Specie per particella';
export const CHART_TOTAL = 'Totale';
export const CHART_BY_MONTHS = 'Mesi';
export const CHART_OTHER = 'Altro';

// Page sections
export const SECTION_INTERVENTI = 'Interventi';

// Prelievi columns
export const COL_DATE = 'Data';
export const COL_REGION = 'Compresa';
export const COL_PARCEL = 'Particella';
export const COL_CREW = 'Squadra';
export const COL_VDP = 'VDP';
export const COL_QUINTALS = 'Q.li';
export const COL_NOTE = 'Note';
export const COL_EXTRA_NOTE = 'Altre note';
export const COL_PRODUCT = 'Tipo';
export const COL_VOLUME_M3 = 'Volume (m³)';
export const COL_PROT = 'Prot';
export const COL_ACTIVE = 'Attivo';
export const COL_TRACTOR = 'Trattore';

// Settings
export const SETTINGS_PASSWORD = 'Cambio password';
export const SETTINGS_CREWS = 'Squadre';
export const SETTINGS_TRACTORS = 'Trattori';
export const SETTINGS_SPECIES = 'Specie';
export const SETTINGS_USERS = 'Utenti';
export const ONLY_ACTIVE = 'Solo attivi';
export const PASSWORD_NEW = 'Nuova password';
export const PASSWORD_REPEAT = 'Ripeti password';
export const PASSWORD_MISMATCH = 'Le password non coincidono.';
export const PASSWORD_CHANGED = 'Password modificata.';

// Audit
export const COL_TIMESTAMP = 'Data/Ora';
export const COL_USER = 'Utente';
export const COL_TABLE = 'Tabella';
export const COL_ACTION = 'Azione';
export const COL_OLD_VALUE = 'Valore precedente';
export const COL_NEW_VALUE = 'Valore successivo';

// Form labels
export const LABEL_DATE = 'Data';
export const LABEL_REGION = 'Compresa';
export const LABEL_PARCEL = 'Particella';
export const LABEL_CREW = 'Squadra';
export const LABEL_PRODUCT = 'Tipo';
export const LABEL_DENSITY = 'Densità (q/m³)';
export const LABEL_NOTE = 'Note';
export const LABEL_VDP = 'VDP';
export const LABEL_EXTRA_NOTE = 'Altre note';
export const LABEL_QUINTALS = 'Q.li';
export const LABEL_SPECIES = 'Specie';
export const LABEL_TRACTORS = 'Trattori';
export const LABEL_MANUFACTURER = 'Marca';
export const LABEL_MODEL = 'Modello';
export const LABEL_YEAR = 'Anno';
export const LABEL_NAME = 'Nome';
export const LABEL_LATIN_NAME = 'Nome latino';
export const LABEL_NOTES = 'Note';
export const LABEL_USERNAME = 'Nome utente';
export const LABEL_EMAIL = 'Email';
export const LABEL_FIRST_NAME = 'Nome';
export const LABEL_LAST_NAME = 'Cognome';
export const LABEL_ROLE = 'Ruolo';
export const LABEL_LOGIN_METHOD = 'Metodo di accesso';
export const LABEL_CREATED_AT = 'Creato il';

// CSV export filenames
export const CSV_PRELIEVI = 'prelievi.csv';
export const CSV_CREWS = 'squadre.csv';
export const CSV_TRACTORS = 'trattori.csv';
export const CSV_SPECIES = 'specie.csv';
export const CSV_USERS = 'utenti.csv';
export const CSV_AUDIT = 'controllo.csv';
export const CSV_SAMPLED_TREES = 'alberi-campionati.csv';

// Campionamenti
export const SURVEY_LABEL = 'Rilevamento';
export const GRID_LABEL = 'Griglia';
export const NEW_GRID_LABEL = '+ Nuova griglia';
export const NEW_SURVEY_LABEL = '+ Nuovo rilevamento';
export const SECTION_GRIGLIE = 'Griglie di campionamento';
export const SECTION_RILEVAMENTI = 'Rilevamenti';
export const SECTION_ALBERI_CAMPIONATI = 'Alberi campionati';
export const CAMPIONAMENTI_EMPTY =
  'Seleziona un rilevamento per visualizzare gli alberi campionati.';
export const CAMPIONAMENTI_PICK_SURVEY_FIRST =
  'Seleziona prima un rilevamento.';
export const CAMPIONAMENTI_PICK_AREA_FIRST =
  'Seleziona prima un\'area di saggio sulla mappa.';
export const CAMPIONAMENTI_INSERT_AREA_HERE =
  'Inserire una nuova area qui?';
export const CAMPIONAMENTI_NO_AREAS_HINT =
  'Clicca sulla mappa per aggiungere un\'area, o su un\'area per modificarla.';
export const CAMPIONAMENTI_NO_GRIDS =
  'Nessuna griglia.  Premi "Nuova griglia" per crearne una.';
export const ADD_AREA_LABEL = '+ Aggiungi area';
export const AREA_IN_USE_TOOLTIP =
  'Area di saggio con campioni: non può essere eliminata.';

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

// CSV import status (large files can take many seconds)
export const CSV_IMPORT_IN_PROGRESS =
  'Importazione in corso, attendere…  ' +
  'Non chiudere la finestra né cliccare di nuovo "Importa".';

// Coppice (per-shoot) entry
export const REMOVE_POLLONE = 'Rimuovi';

// CSV export filenames for the symmetric "Esporta CSV" buttons on
// the Griglie + Rilevamenti pulldown rows (mirror the import column shape).
export const CSV_GRID_AREAS = 'aree-saggio.csv';
export const CSV_SURVEY_TREES = 'alberi-rilevamento.csv';

// Grid-planner / delete-grid (campionamenti.js, grid-planner.js)
export const LABEL_DESCRIPTION = 'Descrizione';
export const LABEL_DESCRIPTION_OPTIONAL = 'Descrizione (opzionale)';
export const LABEL_REGIONS = 'Comprese';
export const LABEL_RADIUS_M = 'Raggio (m)';
export const LABEL_COVERAGE_PCT = 'Copertura (%)';
export const ACTION_PLAN = 'Pianifica';
export const ACTION_CREATE = 'Crea';
export const ERR_GRID_HAS_SURVEYS =
  'La griglia è usata da uno o più rilevamenti: eliminarli prima.';
export const ERR_GRID_NAME_REQUIRED = 'Nome richiesto.';
export const ERR_SELECT_REGION = 'Seleziona almeno una compresa.';
export const ERR_RADIUS_POSITIVE = 'Raggio deve essere > 0.';
export const ERR_COVERAGE_RANGE = 'Copertura deve essere tra 0 e 100%.';
export const ERR_PARAMS_ZERO_POINTS = 'Parametri danno 0 punti.';
export const ERR_PLAN_FIRST = 'Esegui prima "Pianifica".';
export const STATUS_SAVING = 'Salvataggio in corso...';
export const STATUS_PLAN_COMPLETE = 'Pianificazione completata: {n} punti.';
export const STATS_POINTS = 'Punti: {n} (obiettivo: {target})';
export const STATS_TOTAL_AREA_HA = 'Superficie totale: {ha} ha';
export const STATS_AREA_PER_POINT_M2 = 'Area singola adc: {area} m²';
export const TOOLTIP_ADC = 'adc {n} · {compresa} {particella}';

// Bosco placeholder page
export const BOSCO_PLACEHOLDER_MESSAGE =
  'La visualizzazione del bosco sarà disponibile in una prossima versione.';
export const BOSCO_OPEN_BOSCOSCOPIO = 'Apri Boscoscopio';

// Digest column headers (mirror config/strings_it.py COL_* additions).
// These are looked up against the JSON digest `columns` array and used
// as keys / labels in `TREES_COLS` for the sortable-table column config.
export const COL_NAME            = 'Nome';
export const COL_COMPRESA        = 'Compresa';
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
export const COL_NUMBER          = 'Numero';
export const COL_LAT             = 'Lat';
export const COL_LON             = 'Lon';
export const COL_QUOTA           = 'Quota';
export const COL_RAGGIO          = 'Raggio';
export const COL_SURVEY          = 'Rilevamento';
export const COL_SAMPLE_AREA     = 'Area di saggio';
export const COL_N_TREES         = 'N. alberi';
export const COL_SAMPLE_DATE     = 'Data campione';
export const COL_AREA_NUM        = 'N. area';
export const COL_TREE_NUM        = 'N. albero';
export const COL_SPECIES         = 'Specie';
export const COL_POLLONE         = 'Pollone';
export const COL_MATRICINA       = 'Matricina';
export const COL_D_CM            = 'D (cm)';
export const COL_H_M             = 'h (m)';
export const COL_L10_MM          = 'L10 (mm)';
export const COL_V_M3            = 'V (m³)';
export const COL_MASS_Q          = 'm (q)';
export const COL_PAI             = 'PAI';

// Abbreviated column labels used only in the TREES_COLS table header to
// save horizontal space; not in any digest.
export const COL_TREE_NUM_SHORT  = 'N. alb.';
export const COL_POLLONE_SHORT   = 'Poll.';
export const COL_MATRICINA_SHORT = 'Mat.';

// Status message fragments
export const STATUS_NO_SAMPLES = 'nessun campione';

