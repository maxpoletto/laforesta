/**
 * Italian UI string constants.
 *
 * To switch language, change the re-export in strings.js.
 */

// Shell / chrome
export const LOADING = 'Caricamento...';
export const DISMISS = 'Chiudi';
export const SAVE = 'Salva';
export const SAVE_AND_CONTINUE = 'Salva e continua';
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
export const EDIT_GRID_TAB_IMPORT  = 'Importa aree da CSV';
export const EDIT_SURVEY_TAB_IMPORT = 'Importa alberi da CSV';
export const GRID_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Area saggio, Lon, Lat, Quota, Raggio.';
export const SURVEY_IMPORT_HELP =
  'Colonne necessarie: Compresa, Particella, Area saggio, Albero, Pollone, ' +
  'Matricina, D_cm, H_m, L10_mm, Genere, Fustaia. Opzionali: Data, PAI.';

// CSV import status (large files can take many seconds)
export const CSV_IMPORT_IN_PROGRESS =
  'Importazione in corso, attendere…  ' +
  'Non chiudere la finestra né cliccare di nuovo "Importa".';

// Coppice (per-shoot) entry
export const REMOVE_POLLONE = 'Rimuovi';

// CSV column headers for round-trip import/export (mirror config/strings_it.py).
export const CSV_COL_COMPRESA    = 'Compresa';
export const CSV_COL_PARTICELLA  = 'Particella';
export const CSV_COL_AREA_SAGGIO = 'Area saggio';
export const CSV_COL_LON         = 'Lon';
export const CSV_COL_LAT         = 'Lat';
export const CSV_COL_QUOTA       = 'Quota';
export const CSV_COL_RAGGIO      = 'Raggio';
export const CSV_COL_ALBERO      = 'Albero';
export const CSV_COL_POLLONE     = 'Pollone';
export const CSV_COL_MATRICINA   = 'Matricina';
export const CSV_COL_D_CM        = 'D_cm';
export const CSV_COL_H_M         = 'H_m';
export const CSV_COL_L10_MM      = 'L10_mm';
export const CSV_COL_GENERE      = 'Genere';
export const CSV_COL_FUSTAIA     = 'Fustaia';
export const CSV_COL_DATA        = 'Data';
export const CSV_COL_PAI         = 'PAI';

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

// Basemap-switcher labels (Leaflet control rendered by MapCommon).
export const BASEMAP_OSM  = 'OSM';
export const BASEMAP_TOPO = 'Topo';
export const BASEMAP_SAT  = 'Satellite';

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
export const TYPE_FUSTAIA = 'fustaia';
export const TYPE_CEDUO   = 'ceduo';

// Plan-selector header (top of the Piano di taglio page).
export const LABEL_HARVEST_PLAN = 'Piano di taglio';
export const NEW_PLAN_LABEL     = '+ Nuovo piano';
export const PDT_NO_PLANS       = 'Nessun piano. Premi "+ Nuovo piano" per crearne uno.';
export const PDT_SECTION_EMPTY              = 'Nessun intervento.';
export const PDT_EMPTY_STATE_ADD_MANUAL     = '+ Aggiungi manualmente';

// Calendar section titles.
export const SECTION_INTERVENTI_FUSTAIA = 'Interventi fustaia';
export const SECTION_INTERVENTI_CEDUO   = 'Interventi ceduo';
export const ADD_ITEM_LABEL             = '+ Aggiungi';

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
export const EDIT_PLAN_TAB_REGRESSION = 'Importa equazioni da CSV';
export const EDIT_PLAN_CHECKBOX_CEDUO = 'File contiene il calendario ceduo';
export const LABEL_CSV_FILE           = 'File CSV';
export const IMPORT_LABEL             = 'Importa';

// Dangerous-delete (plan + per-item).
export const DELETE_PLAN_TITLE   = 'Elimina piano';
export const DELETE_PLAN_WARNING =
  'Il piano "{name}" e tutte le sue voci e regressioni saranno ' +
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
export const COL_TYPE                 = 'Tipo';
export const COL_STATE                = 'Stato';
export const COL_VOLUME_PLANNED       = 'Volume previsto';
export const COL_VOLUME_MARKED        = 'Volume martellato';
export const COL_VOLUME_ACTUAL        = 'Volume effettivo';
export const COL_INTERVENTION_AREA_HA = 'Superficie intervento (ha)';
export const COL_PARCEL_AREA_HA       = 'Superficie totale (ha)';
export const COL_TURNO_A              = 'Turno (a)';
export const COL_OPERATOR             = 'Operatore';
export const COL_NUMERO               = 'Numero';
export const COL_H_MEASURED           = 'h misurata';
export const COL_FUNCTION             = 'funzione';
export const COL_A                    = 'a';
export const COL_B                    = 'b';

// View/edit-item page.
export const VIEW_ITEM_TITLE     = 'Intervento';
export const LABEL_OPEN_CANTIERE    = 'Apri cantiere';
export const LABEL_CLOSE_CANTIERE   = 'Chiudi cantiere';
export const LABEL_CANTIERE_OPENED   = 'Apertura cantiere';
export const LABEL_CANTIERE_CLOSED   = 'Chiusura cantiere';
export const ERR_DATE_REQUIRED    = 'Data obbligatoria.';
export const SECTION_PRELIEVI     = 'Prelievi';
export const SECTION_MARTELLATA   = 'Martellate';
export const LABEL_VOLUME_TOTAL   = 'Volume totale';
export const LABEL_MASS_TOTAL     = 'Massa totale';
export const COL_CANTIERE         = 'Cantiere';

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

// CSV export filenames (per-plan zip + per-item zip; see piano-di-taglio.md).
export const CSV_PIANO       = 'piano.csv';
export const CSV_CEDUO       = 'ceduo.csv';
export const CSV_EQUAZIONI   = 'equazioni_ipsometro.csv';
// Per-item exports use the item id; constants here are filename prefixes
// (the JS code appends `<id>.csv`).
export const CSV_MARTELLATE_PREFIX = 'martellate_';
export const CSV_PRELIEVI_PREFIX   = 'prelievi_';

