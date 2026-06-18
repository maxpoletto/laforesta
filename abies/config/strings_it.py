"""Italian UI strings for Abies."""

# ---------------------------------------------------------------------------
# Button / control labels (used by templates via the `S` context processor
# and by Python views).  Names mirror the JS-side constants in
# apps/base/static/base/js/strings_it.js so the same logical label has the
# same name on both sides of the membrane.
# ---------------------------------------------------------------------------

SAVE = 'Salva'
CANCEL = 'Annulla'
CONFIRM = 'Conferma'
DISMISS = 'Chiudi'
ACTION_EDIT = 'Modifica'
ACTION_DELETE = 'Elimina'
EXPORT_CSV = 'Esporta CSV'
IMPORT_LABEL = 'Importa'
FILTER_LABEL = 'Filtra'
SEARCH_PLACEHOLDER = 'Cerca…'
TAB_DETAILS = 'Dettagli'
LABEL_CSV_FILE = 'File CSV'

# ---------------------------------------------------------------------------
# Shell / chrome
# ---------------------------------------------------------------------------

DELETE_CONFIRM = 'I dati cancellati non potranno essere recuperati. Confermi?'
ERROR_CONFLICT = 'Il record è stato modificato da un altro utente.'
ERROR_GENERIC = 'Errore imprevisto.'
ERROR_RATE_LIMIT = 'Troppe richieste. Riprovare tra un minuto.'
ERR_JSON_INVALID = 'Richiesta JSON non valida.'

# Ipso import inbox.
IPSO_INBOX_TITLE = 'Importazione'
IPSO_COL_RECEIVED = 'Ricevuto'
IPSO_COL_MODE = 'Modalità'
IPSO_COL_OPERATOR = 'Operatore'
IPSO_COL_RECORDS = 'Righe'
IPSO_COL_STATE = 'Stato'
IPSO_COL_WORK_PACKAGE = 'Pacchetto'
IPSO_COL_REFERENCE = 'Riferimento'
IPSO_COL_TARGET = 'Destinazione'
IPSO_COL_ERROR = 'Errore'
IPSO_STATE_RECEIVED = 'Da importare'
IPSO_STATE_IMPORTED = 'Importato'
IPSO_STATE_REJECTED = 'Rifiutato'
IPSO_STATE_CONFLICT = 'Conflitto'
IPSO_ERR_IMPORTED_CANNOT_REJECT = 'Un caricamento importato non può essere rifiutato.'
IPSO_ERR_MODE_UNSUPPORTED = 'Modalità non supportata.'
IPSO_ERR_UPLOAD_NOT_RECEIVED = 'Solo i caricamenti da importare possono essere importati.'
IPSO_ERR_INVALID_MARTELLATE_TARGET = 'Destinazione non valida per Martellate.'
IPSO_ERR_INVALID_SAMPLES_TARGET = 'Destinazione non valida per Campionamenti.'
IPSO_TARGET_PAI_LABEL = 'PAI'
IPSO_REJECT_DEFAULT_REASON = 'Rifiutato in revisione Abies.'
IPSO_ERR_DUPLICATE_SESSION_CONTENT = 'ID sessione duplicato con contenuto diverso.'
IPSO_ERR_UPLOAD_JSON_MISSING = 'File upload.json non trovato.'
IPSO_ERR_UPLOAD_JSON_INVALID = 'File upload.json non valido.'
IPSO_ERR_IMPORT_RECORDS_ARRAY = 'Il campo records deve essere un array.'
IPSO_ERR_IMPORT_RECORD_INVALID = 'Riga {}: formato non valido.'
IPSO_ERR_IMPORT_RECORD_PARCEL_NOT_FOUND = 'Riga {}: particella non trovata.'
IPSO_ERR_IMPORT_RECORD_SPECIES_NOT_FOUND = 'Riga {}: specie non trovata.'
IPSO_ERR_IMPORT_RECORD_DH_DATE_INVALID = 'Riga {}: D/H/data non validi.'
IPSO_ERR_IMPORT_RECORD_COPPICE_MARTELLATE = 'Riga {}: particella cedua non valida per Martellate.'
IPSO_ERR_IMPORT_RECORD_AREA_NOT_FOUND = 'Riga {}: area di saggio non trovata.'
IPSO_ERR_IMPORT_RECORD_AREA_OUT_OF_SURVEY = 'Riga {}: area di saggio fuori dal rilevamento selezionato.'
IPSO_ERR_IMPORT_RECORD_AREA_PARCEL_MISMATCH = 'Riga {}: area di saggio e particella non corrispondono.'
IPSO_ERR_IMPORT_RECORD_SAMPLE_FIELDS_INVALID = 'Riga {}: dati campionamento non validi.'
IPSO_ERR_IMPORT_RECORD_COORDS_REQUIRED = 'Riga {}: Lat e Lon obbligatorie.'
IPSO_ERR_IMPORT_RECORD_PAI_NUMBER_DUPLICATE = 'Riga {}: numero PAI già presente nella particella.'
IPSO_ERR_JSON_MALFORMED = 'JSON non valido.'
IPSO_ERR_PAYLOAD_OBJECT = 'Il payload deve essere un oggetto.'
IPSO_ERR_CSV_TEXT_STRING = 'csv_text deve essere una stringa quando presente.'
IPSO_ERR_HEADER_SESSION_MISMATCH = (
    'X-Ipso-Session-Id non corrisponde a session.session_id.'
)
IPSO_ERR_SESSION_ID_UUID = 'session_id deve essere un UUID.'
IPSO_ERR_SCHEMA_VERSION_UNSUPPORTED = 'schema_version non supportata.'
IPSO_ERR_RECORD_OBJECT = 'Riga {}: deve essere un oggetto.'
IPSO_ERR_RECORD_DATE_INVALID = 'Riga {}: date deve essere YYYY-MM-DD.'
IPSO_ERR_RECORD_D_CM_POSITIVE = 'Riga {}: d_cm deve essere positivo.'
IPSO_ERR_RECORD_H_M_POSITIVE = 'Riga {}: h_m deve essere positivo.'
IPSO_ERR_RECORD_NUMBER_POSITIVE = 'Riga {}: numero deve essere positivo.'
IPSO_ERR_RECORD_NUMBER_REQUIRED = 'Riga {}: numero obbligatorio.'
IPSO_ERR_RECORD_SAMPLE_AREA_REQUIRED = 'Riga {}: area di saggio obbligatoria.'
IPSO_ERR_RECORD_SAMPLE_FIELDS_INVALID = 'Riga {}: campi campionamento non validi.'
IPSO_ERR_RECORD_HYPSO_ONLY_MARTELLATE = (
    'Riga {}: hypso_param_set_id è valido solo per martellate.'
)
IPSO_ERR_UNKNOWN_SPECIES_ID = 'species_id sconosciuto: {}.'
IPSO_ERR_UNKNOWN_PARCEL_ID = 'parcel_id sconosciuto: {}.'
IPSO_ERR_UNKNOWN_SAMPLE_AREA_ID = 'Area di saggio sconosciuta: {}.'
IPSO_ERR_UNKNOWN_HYPSO_PARAM_SET_ID = 'hypso_param_set_id sconosciuto: {}.'
IPSO_ERR_RECORD_PARCEL_REGION = 'Riga {}: parcel_id non appartiene a region_id.'
IPSO_ERR_RECORD_SAMPLE_AREA_PARCEL = 'Riga {}: area di saggio non appartiene alla particella.'
IPSO_ERR_FIELD_OBJECT = '{} deve essere un oggetto.'
IPSO_ERR_FIELD_ARRAY = '{} deve essere un array.'
IPSO_ERR_FIELD_REQUIRED = '{} è obbligatorio.'
IPSO_ERR_FIELD_STRING = '{} deve essere una stringa.'
IPSO_ERR_FIELD_INTEGER = '{} deve essere un intero.'
IPSO_ERR_FIELD_INTEGER_NULL = '{} deve essere un intero o null.'
IPSO_ERR_FIELD_BOOLEAN = '{} deve essere un booleano.'
IPSO_ERR_FIELD_NUMBER_NULL = '{} deve essere un numero o null.'
IPSO_ERR_FIELD_FINITE = '{} deve essere finito.'
IPSO_ERR_FIELD_DECIMAL = '{} deve essere un valore decimale.'
IPSO_ERR_FIELD_DECIMAL_FINITE = '{} deve essere un valore decimale finito.'

ERR_LOGIN_INVALID = 'Nome utente o password non validi.'

# Validation
ERR_DATE_REQUIRED = 'Data obbligatoria.'
ERR_DATE_INVALID = 'Data non valida.'
ERR_QUINTALS_POSITIVE = 'I quintali devono essere positivi.'
ERR_SPECIES_PCT_SUM = 'Le percentuali delle specie devono sommare a 100.'
ERR_TRACTOR_PCT_SUM = 'Le percentuali dei trattori devono sommare a 100.'
ERR_DATE_FUTURE = 'La data non può essere nel futuro.'
ERR_VDP_DUPLICATE = 'VDP {} è già utilizzato.'
ERR_NOT_FOUND = 'Record non trovato.'
ERR_FORBIDDEN = 'Non autorizzato.'
ERR_CREW_REQUIRED = 'Squadra obbligatoria.'
ERR_HOURS_POSITIVE = 'Le ore devono essere maggiori di zero.'
ERR_CREDITS_POSITIVE = 'I quintali devono essere maggiori di zero.'
ERR_SLIP_COUNT_MULTIPLE = 'Il numero di verbali deve essere un multiplo di 4.'
ERR_SLIP_COUNT_POSITIVE = 'Il numero di verbali deve essere maggiore di zero.'
ERR_LICENSE_PLATE_REQUIRED = 'Targa autocarro obbligatoria.'

# Column / field labels (used in audit digest and shared with JS)
COL_DATE = 'Data'
COL_SURVEY_DATE = 'Data di rilevamento'
COL_PARCEL = 'Particella'
COL_CREW = 'Squadra'
COL_PRODUCT = 'Tipo'
COL_QUINTALS = 'Q.li'
COL_HOURS = 'Ore'
COL_CREDITS_Q = 'Quintali'
COL_VOLUME_M3 = 'Volume (m³)'
COL_VDP = 'VDP'
COL_PROT = 'Prot'
COL_NOTE = 'Note'
COL_EXTRA_NOTE = 'Altre note'
COL_ACTIVE = 'Attivo'
COL_MINOR = 'Minore'
LABEL_USERNAME = 'Nome utente'
LABEL_EMAIL = 'Email'
LABEL_ROLE = 'Ruolo'
# Display labels for the Role pulldown; distinct from the JS wire-format
# ROLE_* constants in `apps/base/static/base/js/constants.js`.
LABEL_ROLE_ADMIN = 'Amministratore'
LABEL_ROLE_WRITER = 'Redattore'
LABEL_ROLE_READER = 'Membro'
LABEL_NAME = 'Nome'
LABEL_NOTES = 'Note'
LABEL_MANUFACTURER = 'Marca'
LABEL_MODEL = 'Modello'
LABEL_TRACTOR_NAME = 'Trattore'
LABEL_DENSITY = 'Densità (q/m³)'
LABEL_FIRST_NAME = 'Nome'
LABEL_LAST_NAME = 'Cognome'
LABEL_LOGIN_METHOD = 'Metodo di accesso'
LABEL_CREATED_AT = 'Creato il'
LABEL_LICENSE_PLATE = 'Targa autocarro'
LABEL_START_NUMBER = 'Numero iniziale'
LABEL_SLIP_COUNT = 'Numero di verbali'
LABEL_MONTH = 'Mese'

# Audit
COL_TIMESTAMP = 'Data/Ora'
COL_USER = 'Utente'
COL_TABLE = 'Tabella'
COL_ACTION = 'Azione'
COL_OLD_VALUE = 'Valore precedente'
COL_NEW_VALUE = 'Valore successivo'

TABLE_HARVEST = 'Prelievo'
TABLE_USER = 'Utente'
TABLE_CREW = 'Squadra'
TABLE_TRACTOR = 'Trattore'
TABLE_SPECIES = 'Specie'
TABLE_HARVEST_PLAN = 'Piano di gestione'
TABLE_HARVEST_PLAN_ITEM = 'Voce di piano'
TABLE_PARCEL = 'Particella'
TABLE_SAMPLE_GRID = 'Griglia di campionamento'
TABLE_SAMPLE_AREA = 'Area di saggio'
TABLE_SURVEY = 'Rilevamento'
TABLE_SAMPLE = 'Campione'
TABLE_HYPSO_PARAM_SET = 'Parametri ipsometrici'
TABLE_HYPSO_PARAM = 'Parametro ipsometrico'
TABLE_MANNESI_LICENSE_PLATE = 'Targa autocarro'
TABLE_MANNESI_LICENSE_PLATES = 'Targhe autocarri'
TABLE_MANNESI_HOURS = 'Ore mannesi'
TABLE_MANNESI_CREDIT = 'Acconto mannesi'
TABLE_MANNESI_CREDITS = 'Acconti mannesi'

AUDIT_INSERT = 'Inserimento'
AUDIT_UPDATE = 'Modifica'
AUDIT_DELETE = 'Eliminazione'

# Settings
PASSWORD_MISMATCH = 'Le password non coincidono.'
PASSWORD_CHANGED = 'Password modificata.'

HYPSO_SAVED = 'Parametri ipsometrici aggiornati.'
HYPSO_CLEARED = 'Parametri ipsometrici eliminati.'
FUTURE_PRODUCTION_SAVED = 'Produzione futura aggiornata.'
DENDROMETRY_SAVED = 'Parametri dendrometrici aggiornati.'
ERR_FUTURE_PLAN_REQUIRED = 'Selezionare un piano.'
ERR_DENDROMETRY_SURVEYS_REQUIRED = 'Seleziona almeno un rilevamento.'
ERR_PASSWORD_REQUIRED = 'Password obbligatoria.'
ERR_NAME_REQUIRED = 'Nome obbligatorio.'
ERR_USERNAME_REQUIRED = 'Nome utente obbligatorio.'
ERR_EMAIL_REQUIRED = 'Email obbligatoria.'
ERR_DENSITY_INVALID = 'La densità deve essere un numero positivo.'
ERR_PRESSLER_POSITIVE = 'Il coefficiente Pressler deve essere positivo.'
ERR_OTHER_NOT_MINOR = 'La specie "{}" non può essere contrassegnata come minore.'
ERR_D_POSITIVE = 'Il diametro deve essere positivo.'
ERR_H_POSITIVE = 'L\'altezza deve essere positiva.'
ERR_TREE_NUMBER_REQUIRED = 'Numero albero obbligatorio.'
ERR_TREE_NUMBER_DUPLICATE = 'Numero albero {} già utilizzato in questo campione.'
ERR_TREE_ALREADY_IN_SAMPLE = (
    'Albero n.{} già presente in questo campione. Modifica la riga esistente.'
)
ERR_TRACTOR_NAME_DUPLICATE = 'Esiste già un trattore con questo nome.'
ERR_GRID_NAME_REQUIRED = 'Nome griglia obbligatorio.'
ERR_GRID_NAME_DUPLICATE = 'Esiste già una griglia con questo nome.'
ERR_SURVEY_NAME_REQUIRED = 'Nome rilevamento obbligatorio.'
ERR_SURVEY_NAME_DUPLICATE = 'Esiste già un rilevamento con questo nome.'
ERR_SURVEY_GRID_REQUIRED = 'Selezionare una griglia.'
ERR_AREA_OUT_OF_SURVEY = (
    'L\'area di saggio non appartiene alla griglia del rilevamento.'
)
ERR_COPPICE_NO_SHOOTS = (
    'Inserire almeno un pollone per un albero ceduo.'
)
ERR_COPPICE_SHOOT_DUPLICATE = (
    'Pollone {} già rilevato in questo campione.'
)
ERR_GRID_AUTO_NO_POINTS = (
    'Nessun punto generato.  Esegui prima "Pianifica".'
)
ERR_GRID_AUTO_PARCEL_UNRESOLVED = (
    'Particella non trovata: {} / {}.'
)
ERR_AREA_NUMBER_REQUIRED = 'Numero area obbligatorio.'
ERR_AREA_NUMBER_DUPLICATE = 'Numero area già presente in questa compresa.'
ERR_PLAN_NAME_REQUIRED = 'Nome piano obbligatorio.'
ERR_PLAN_NAME_DUPLICATE = 'Esiste già un piano con questo nome.'
ERR_PLAN_YEAR_RANGE = 'Anno fine deve essere maggiore o uguale ad anno inizio.'
ERR_PLAN_HAS_ACTIVE_ITEMS = (
    'Il piano contiene voci non in stato "pianificato". '
    'Elimina prima le voci dipendenti.'
)
ERR_PLAN_NOT_FOUND = 'Piano non trovato.'
ERR_CSV_WHOLE_REGION_REQUIRES_FLAG = (
    'Riga {}: voce a livello di compresa (Particella = "{}"): la colonna '
    'Note deve contenere "Catastrofato" o "Fitosanitario".'
)
ERR_PLAN_ITEM_COMPRESA_REQUIRED = (
    'Compresa obbligatoria.'
)
ERR_PLAN_ITEM_REGION_REQUIRES_FLAG = (
    'Le voci a livello di compresa (senza particella) richiedono il '
    'flag "catastrofato" o "fitosanitario".'
)
ERR_CSV_PLAN_ITEM_REGION_REQUIRES_FLAG = (
    'Riga {}: una voce a livello di compresa richiede il flag '
    'Danneggiato o Fitosanitario.'
)
ERR_PLAN_ITEM_NOT_FOUND = 'Voce di piano non trovata.'
ERR_PLAN_ITEM_STATE_NOT_PLANNED = (
    'La voce non è in stato "pianificato"; eliminazione non consentita.'
)
ERR_PLAN_ITEM_HAS_DEPS = (
    'La voce ha martellate, prelievi o transizioni di cantiere associati; '
    'eliminale prima.'
)
ERR_PLAN_ITEM_VOLUME_NEGATIVE = 'Il volume previsto (m³) deve essere positivo.'
ERR_PLAN_ITEM_AREA_NEGATIVE = 'La superficie deve essere positiva.'
ERR_TRANSITION_INVALID_STATE = (
    'Transizione non consentita dallo stato attuale.'
)
ERR_TRANSITION_DATE_REQUIRED = 'Data della transizione obbligatoria.'
ERR_CSV_NO_FILES = (
    'Nessun file CSV caricato. Allega almeno un file (fustaia o ceduo).'
)
ERR_CSV_PARCEL_NOT_FOUND = 'Riga {}: particella non trovata ({} / {}).'
ERR_CSV_PLAN_NOT_FOUND   = 'Riga {}: piano non trovato ({}).'
ERR_CSV_REGION_NOT_FOUND = 'Riga {}: compresa non trovata ({}).'
ERR_CSV_SPECIES_NOT_FOUND = 'Riga {}: specie non trovata ({}).'
ERR_CSV_FUNCTION_INVALID = (
    'Riga {}: funzione di regressione non supportata ({}).'
)
ERR_CSV_VALUE_PARSE = 'Riga {}: valore non valido nella colonna {} ({}).'
ERR_CSV_VALUE_RANGE = 'Riga {}: valore fuori intervallo nella colonna {} ({}).'
ERR_CSV_DUPLICATE_PARAM = 'Riga {}: coppia compresa/genere duplicata ({} / {}).'
ERR_REGION_XOR_PARCEL = 'Indicare esattamente una particella o una compresa.'
ERR_HARVEST_REGION_XOR_PARCEL = ERR_REGION_XOR_PARCEL
ERR_STATE_REGRESSION = 'Lo stato non può regredire: {} → {}.'
ERR_CANTIERE_REQUIRED = (
    'Selezionare un cantiere aperto.'
)
ERR_CANTIERE_STATE_INVALID = (
    'Il cantiere selezionato non è in stato "aperto" o "in prelievo".'
)
ERR_PARCEL_REQUIRED_FOR_REGION_WIDE = (
    'Cantiere a livello di compresa: specificare la particella.'
)
ERR_PARCEL_NOT_IN_REGION = (
    'La particella selezionata non appartiene alla compresa del cantiere.'
)
ERR_AREA_IN_USE = (
    'Area di saggio già usata in un campione: non può essere eliminata.'
)
ERR_GRID_IN_USE = (
    'La griglia è usata da uno o più rilevamenti: eliminarli prima.'
)
ERR_CSV_FILE_REQUIRED = 'File CSV obbligatorio.'
ERR_CSV_NOT_UTF8 = 'Il file deve essere codificato in UTF-8.'
ERR_CSV_EMPTY = 'Il file CSV è vuoto.'
ERR_CSV_MISSING_COLS = 'Colonne CSV mancanti: {}.'
ERR_CSV_ROW_AREA = 'Riga {}: area di saggio non trovata ({} / {} / {}).'
ERR_CSV_ROW_SPECIES = 'Riga {}: specie sconosciuta: {}.'
ERR_CSV_ROW_PARSE = 'Riga {}: errore di parsing ({}).'
ERR_CSV_VALUE_REQUIRED = 'Riga {}: valore obbligatorio mancante nella colonna {}.'
ERR_CSV_DUPLICATE_KEY  = 'Riga {}: chiave duplicata nella colonna {} ({}).'
ERR_CSV_ECLASS_NOT_FOUND = 'Riga {}: comparto non trovato ({}).'
ERR_CSV_GRID_NOT_FOUND   = 'Riga {}: griglia non trovata ({}).'
ERR_BOOTSTRAP_REQUIRED_FILE  = 'File obbligatorio mancante: {}.'
ERR_BOOTSTRAP_NOT_EMPTY      = ('Istanza non vuota (popolati: {}); il bootstrap '
                               'carica solo in un’istanza vuota.')
ERR_BOOTSTRAP_UNKNOWN_GRID   = 'File {}: griglia non trovata ({}).'
ERR_BOOTSTRAP_UNKNOWN_SURVEY = 'File {}: rilevamento non trovato ({}).'
ERR_BOOTSTRAP_FAILED         = '{} errore/i durante il bootstrap; nulla è stato caricato.'
BOOTSTRAP_OPTIONAL_SKIPPED   = 'assente (opzionale, saltato)'
BOOTSTRAP_DEFAULT_SEEDED     = 'assente (caricati valori predefiniti)'
BOOTSTRAP_CHECK_NOTICE       = '--check: nessuna modifica è stata salvata.'
BOOTSTRAP_DONE               = 'Bootstrap completato.'
BOOTSTRAP_LOADED             = '{} caricati'    # report line: rows loaded
BOOTSTRAP_ABSENT             = '—'              # report line: file not present
BOOTSTRAP_INTERNAL           = '(bootstrap)'    # synthetic report row for an unexpected error
ERR_CSV_SURVEY_REQUIRED = 'Seleziona prima un rilevamento.'
ERR_CSV_GRID_REQUIRED = 'Seleziona prima una griglia di destinazione.'
ERR_MIN_N_INVALID = 'N minimo deve essere un intero positivo.'
ERR_HYPSO_SURVEYS_REQUIRED = 'Seleziona almeno un rilevamento.'
ERR_CSV_ROW_AREA_DUPLICATE = (
    'Riga {}: area ({} / {} / {}) già presente nella griglia.'
)
ERR_CSV_ROW_SAMPLE_DATE_CONFLICT = (
    'Riga {}: data diversa per area di saggio già presente nel rilevamento '
    '({} / {} / {}; data già registrata: {}).'
)
ERR_CSV_DATE_REQUIRED = (
    'Il file CSV non ha una colonna "Data": indicare una data predefinita.'
)
ERR_MARK_SPECIES_REQUIRED = 'Specie obbligatoria.'
ERR_MARK_D_REQUIRED = 'D obbligatorio (intero > 0).'
ERR_MARK_H_REQUIRED = 'h obbligatoria (> 0).'
ERR_MARK_OPERATOR_REQUIRED = 'Operatore obbligatorio.'
ERR_MARK_ITEM_CLOSED = 'Il cantiere è chiuso: aggiunta non consentita.'
ERR_MARK_PARCEL_REQUIRED = 'Particella obbligatoria per interventi a livello di compresa.'
ERR_MARK_PARCEL_NOT_IN_REGION = (
    'La particella non appartiene alla compresa dell\'intervento.'
)
ERR_BOSCO_AREA_REQUIRED = 'Superficie obbligatoria.'
ERR_BOSCO_INTEGER_REQUIRED = '{} deve essere un numero intero.'
ERR_BOSCO_TEXT_TOO_LONG = '{} troppo lunga.'
ERR_BOSCO_ALTITUDE_RANGE = 'Altitudine minima maggiore della massima.'
ERR_BOSCO_SPECIES_REQUIRED = 'Specie obbligatoria.'
ERR_BOSCO_PARCEL_REQUIRED = 'Particella obbligatoria.'
ERR_BOSCO_YEAR_REQUIRED = 'Anno obbligatorio.'
ERR_BOSCO_NUMBER_REQUIRED = 'Numero obbligatorio.'
ERR_BOSCO_PAI_NUMBER_DUPLICATE = 'Numero già presente per questa particella.'
ERR_BOSCO_POSITIVE_INTEGER_REQUIRED = '{} deve essere un numero intero > 0.'
ERR_BOSCO_LAT_LON_REQUIRED = 'Lat e Lon obbligatorie.'
LABEL_BOSCO_AVE_AGE = 'Età media'
LABEL_BOSCO_VEG_DESC = 'Descrizione vegetazione'
LABEL_BOSCO_GEO_DESC = 'Descrizione geologia'
LABEL_BOSCO_ALTITUDE_MIN = 'Altitudine minima'
LABEL_BOSCO_ALTITUDE_MAX = 'Altitudine massima'
LABEL_BOSCO_GRADE = 'Pendenza'

# ---------------------------------------------------------------------------
# CSV column headers.  These are simultaneously the wire-format identifiers
# the importer matches against AND the user-facing copy in the import help
# text.  Both sides reference these constants so they cannot drift.
# ---------------------------------------------------------------------------

CSV_COL_REGION        = 'Compresa'
CSV_COL_PARCEL        = 'Particella'
CSV_COL_SAMPLE_AREA   = 'Area saggio'
CSV_COL_LON           = 'Lon'
CSV_COL_LAT           = 'Lat'
CSV_COL_ALT           = 'Quota'
CSV_COL_RADIUS        = 'Raggio'
CSV_COL_TREE          = 'Albero'
CSV_COL_COPPICE_SHOOT = 'Pollone'
CSV_COL_COPPICE_STD   = 'Matricina'
CSV_COL_D_CM          = 'D_cm'
CSV_COL_H_M           = 'H_m'
CSV_COL_L10_MM        = 'L10_mm'
CSV_COL_PRESSLER       = 'Pressler'
CSV_COL_SPECIES       = 'Genere'
CSV_COL_HIGHFOREST       = 'Fustaia'
CSV_COL_DATA          = 'Data'
CSV_COL_ESTIMATED_BIRTH_YEAR = 'Anno di nascita stimato'
CSV_COL_PRESERVED     = 'PAI'

# --- Reference-table CSV headers (bootstrap: regions/eclasses/crews/
# species/products).  Paired with COL_* display labels where one exists;
# the CSV_COL_* form is the canonical import/export header.
CSV_COL_COPPICE     = 'Ceduo'               # Eclass.coppice (bool)
CSV_COL_MIN_VOLUME  = 'Volume minimo (m³)'  # Eclass.min_harvest_volume
CSV_COL_ACTIVE      = 'Attivo'              # Crew/Species.active (bool); paired with COL_ACTIVE
CSV_COL_LATIN       = 'Nome latino'         # Species.latin_name; paired with COL_LATIN_NAME
CSV_COL_DENSITY     = 'Densità (q/m³)'      # Species.density; paired with COL_DENSITY
CSV_COL_MINOR       = 'Minore'              # Species.minor (bool); paired with COL_MINOR
CSV_COL_SORT_ORDER  = 'Ordine'             # Species.sort_order

# --- Named-container CSV headers (bootstrap: sample_grids/surveys/harvest_plans).
CSV_COL_GRID         = 'Griglia'       # SampleGrid.name / Survey grid FK; paired with COL_GRID
CSV_COL_SURVEY       = 'Rilevamento'   # Survey.name; paired with COL_SURVEY
CSV_COL_DESCRIPTION  = 'Descrizione'   # description; paired with COL_DESCRIPTION
CSV_COL_PLAN         = 'Piano'         # HarvestPlan.name
CSV_COL_YEAR_START   = 'Anno inizio'   # HarvestPlan.year_start; paired with COL_YEAR_START
CSV_COL_YEAR_END     = 'Anno fine'     # HarvestPlan.year_end; paired with COL_YEAR_END

# CSV column headers used by bootstrap, imports, and exports.  Where a
# column also appears in a digest, the names are paired (`CSV_COL_X` /
# `COL_X`); the values may match (same Italian token used in both
# contexts) or differ (e.g. `Altitudine min` in the CSV vs
# `Alt. min. (m)` in the digest column).
CSV_COL_CLASS          = 'Comparto'
CSV_COL_CREW           = 'Squadra'           # paired with COL_CREW
CSV_COL_PRODUCT        = 'Tipo'              # paired with COL_PRODUCT
CSV_COL_NOTE           = 'Note'              # paired with COL_NOTE
CSV_COL_QUINTALS       = 'Q.li'              # paired with COL_QUINTALS
CSV_COL_VDP            = 'VDP'               # paired with COL_VDP
CSV_COL_PROT           = 'Prot.'             # paired with COL_PROT (note '.')
CSV_COL_EXTRA_NOTE     = 'Altre note'        # paired with COL_EXTRA_NOTE
CSV_COL_AREA_HA        = 'Area (ha)'         # paired with COL_AREA_HA
CSV_COL_AVE_AGE        = 'Età media'         # paired with COL_AVE_AGE
CSV_COL_LOCATION       = 'Località'          # paired with COL_LOCATION
CSV_COL_ASPECT         = 'Esposizione'       # paired with COL_ASPECT
CSV_COL_GRADE_PCT      = 'Pendenza %'        # paired with COL_GRADE_PCT
CSV_COL_VEG_DESC       = 'Soprassuolo'       # vegetation description
CSV_COL_GEO_DESC       = 'Stazione'          # geological station
CSV_COL_ALT_MIN        = 'Altitudine min'    # paired with COL_ALT_MIN
CSV_COL_ALT_MAX        = 'Altitudine max'    # paired with COL_ALT_MAX
CSV_COL_D_CM_LEGACY    = 'D(cm)'
CSV_COL_H_M_LEGACY     = 'h(m)'
CSV_COL_L10_MM_LEGACY  = 'L10(mm)'
CSV_COL_N_LEGACY       = 'n'

# Piano di taglio CSV column headers.  These match the output of
# `pdg-2026/pdg.py --formato csv` for the three calendar files
# (`piano.csv`, `ceduo.csv`, `equazioni_ipsometro.csv`).
# The regression CSV uses lowercase `compresa` / `genere`; case
# normalisation is the parser's responsibility, so we don't define
# duplicate lowercase variants of the capitalized headers.
CSV_COL_YEAR         = 'Anno'                        # scheduled-cut year
CSV_COL_HARVEST_M3   = 'Prelievo (m³)'               # fustaia volume_planned_m3
CSV_COL_SURFACE_HA   = 'Superficie intervento (ha)'  # ceduo intervention_area_ha
CSV_COL_PERIOD_Y     = 'Turno (a)'                   # coppice rotation, years
CSV_COL_FUNCTION     = 'funzione'                    # regression function (`ln`)
CSV_COL_A            = 'a'                           # regression coefficient
CSV_COL_B            = 'b'                           # regression coefficient
CSV_COL_R2           = 'r2'                          # coefficient of determination
CSV_COL_N_REGRESSION = 'n'                           # sample count for regression fit

# Ipso CSV column headers (tree-mark import; format produced by
# laforesta/ipso — see ipso/CLAUDE.md "CSV format").
CSV_COL_DAMAGED    = 'Catastrofata'     # session-level flag (ignored row-wise)
CSV_COL_NUMBER     = 'Numero'           # operator-assigned tree number

# --- Harvest CSV column headers (harvests.csv import / export).
CSV_COL_HARVEST_DAMAGED   = 'Danneggiato'    # Harvest.damaged flag
CSV_COL_HARVEST_UNHEALTHY = 'Fitosanitario'  # Harvest.unhealthy flag
CSV_COL_HARVEST_PSR       = 'PSR'            # Harvest.psr flag
CSV_COL_SPECIES_PREFIX    = 'Specie:'        # dynamic species-percentage column prefix
CSV_COL_TRACTOR_PREFIX    = 'Trattore:'      # dynamic tractor-percentage column prefix

# --- Tractor CSV column headers (tractors.csv bootstrap).
CSV_COL_TRACTOR_NAME  = 'Trattore'    # Tractor.name
CSV_COL_MANUFACTURER  = 'Produttore'  # Tractor.manufacturer
CSV_COL_MODEL         = 'Modello'     # Tractor.model
# CSV_COL_YEAR ('Anno') already defined above — reused for Tractor.year.

# --- Ipso / tree-mark CSV column headers (tree-mark import extension).
CSV_COL_H_MEASURED = 'H_measured'       # 1 if operator typed h, 0 if auto-h
CSV_COL_ACC_M      = 'Acc_m'            # GPS accuracy in metres
CSV_COL_OPERATOR   = 'Operatore'        # operator name

# --- Harvest-CSV-specific error messages.
ERR_CSV_UNKNOWN_SPECIES_COL  = 'Intestazione CSV: specie sconosciuta nella colonna dinamica ({}).'
ERR_CSV_UNKNOWN_TRACTOR_COL  = 'Intestazione CSV: trattore sconosciuto nella colonna dinamica ({}).'
ERR_CSV_DUPLICATE_DYN_COL    = 'Intestazione CSV: colonna dinamica duplicata ({}).'
ERR_CSV_SPECIES_PCT_SUM      = 'Riga {}: la somma delle percentuali delle specie deve essere 100 (trovato {}).'
ERR_CSV_TRACTOR_PCT_SUM      = 'Riga {}: la somma delle percentuali dei trattori deve essere 0 o 100 (trovato {}).'
ERR_CSV_HARVEST_LOCATION     = 'Riga {}: compresa o particella non trovata ({}).'
ERR_CSV_UNKNOWN_CREW         = 'Riga {}: squadra sconosciuta ({}).'
ERR_CSV_UNKNOWN_PRODUCT      = 'Riga {}: tipo di prodotto sconosciuto ({}).'

# ---------------------------------------------------------------------------
# Digest column headers (display labels in JSON `columns` arrays).
# Where a column has a CSV counterpart with a different display form
# (e.g. CSV `D_cm` vs digest `D (cm)`) the names are paired
# (`CSV_COL_D_CM` / `COL_D_CM`).  Where the display value equals the
# CSV value (e.g. `Lat`, `Pollone`), both constants exist with the same
# value for symmetric usage at call sites.
# ---------------------------------------------------------------------------

COL_REGION          = 'Compresa'       # domain term for a forest region; paired with CSV_COL_REGION
COL_REGIONS         = 'Comprese'       # plural/aggregate label, distinct from one row's COL_REGION
COL_ALT             = 'Alt. (m)'       # display label; CSV accepts legacy CSV_COL_ALT
COL_RADIUS          = 'Raggio (m)'     # display label; CSV accepts legacy CSV_COL_RADIUS
COL_NAME            = 'Nome'           # paired with LABEL_NAME (HTML form)
COL_LATIN_NAME      = 'Nome latino'
COL_DENSITY         = 'Densità (q/m³)'
COL_CLASS           = 'Classe'           # parcel eclass
COL_AREA_HA         = 'Area (ha)'
COL_AREA_CAD_HA     = 'Area cat. (ha)'
COL_AVE_AGE         = 'Età media (a)'
COL_LOCATION        = 'Località'
COL_ALT_MIN         = 'Alt. min. (m)'
COL_ALT_MAX         = 'Alt. max. (m)'
COL_ASPECT          = 'Esposizione'
COL_GRADE_PCT       = 'Pendenza (%)'
COL_DESC_VEG        = 'Soprassuolo'
COL_DESC_GEO        = 'Stazione'
COL_SORT_ORDER      = 'Sort order'       # internal English; not localized
COL_DIAM_CLASS_CM   = 'Classe diam. (cm)'
COL_BASAL_AREA_M2   = 'Area bas. (m²)'
COL_AVG_H_M         = 'Altezza media (m)'
COL_INCREMENT_PCT   = 'Incremento %'
COL_YEAR            = 'Anno'
COL_ESTIMATED_BIRTH_YEAR = 'Anno di nascita stimato'
COL_DESCRIPTION     = 'Descrizione'
COL_N_AREAS         = 'N. aree'
COL_N_SURVEYS       = 'N. rilevamenti'
COL_LAST_UPDATE     = 'Ultimo aggiornamento'
COL_GRID            = 'Griglia'
COL_HARVEST_PLAN    = 'Piano di taglio'
COL_N_AREAS_VISITED = 'N. aree visitate'
COL_N_AREAS_TOTAL   = 'N. aree totali'
COL_DATE_FIRST      = 'Data primo'
COL_DATE_LAST       = 'Data ultimo'
COL_NUMBER          = 'Numero'
COL_LAT             = 'Lat'              # paired with CSV_COL_LAT
COL_LON             = 'Lon'              # paired with CSV_COL_LON
COL_SURVEY          = 'Rilevamento'
COL_SAMPLE_AREA     = 'Area di saggio'
COL_SAMPLE_AREA_HA  = 'Area saggi (ha)'
COL_N_TREES         = 'N. alberi'
COL_SAMPLE_DATE     = 'Data campione'
COL_AREA_NUM        = 'N. area'
COL_TREE_NUM        = 'N. albero'
COL_SPECIES         = 'Specie'
COL_COPPICE_SHOOT   = 'Pollone'          # domain/CSV term for coppice shoot number
COL_COPPICE_STD     = 'Matricina'        # domain/CSV term for coppice standard flag
COL_PRESERVED       = 'PAI'              # paired with CSV_COL_PRESERVED
COL_D_CM            = 'D (cm)'           # paired with CSV_COL_D_CM
COL_H_M             = 'h (m)'            # paired with CSV_COL_H_M
COL_L10_MM          = 'L10 (mm)'         # paired with CSV_COL_L10_MM
COL_PRESSLER        = 'Pressler'         # paired with CSV_COL_PRESSLER
COL_V_M3            = 'V (m³)'           # short form (sampled-trees digest)
COL_MASS_Q          = 'm (q)'

# Piano di taglio digest columns (calendar + martellate tables).
COL_YEAR_PLANNED         = 'Anno previsto'
COL_YEAR_ACTUAL          = 'Anno effettivo'
COL_TYPE                 = 'Tipo'                       # alto fusto / ceduo
COL_STATE                = 'Stato'
COL_VOLUME_PLANNED       = 'Volume previsto (m³)'
COL_VOLUME_MARKED        = 'Volume martellato (m³)'
COL_VOLUME_ACTUAL        = 'Volume effettivo (m³)'
COL_INTERVENTION_AREA_HA = 'Superficie intervento (ha)'  # paired with CSV_COL_SURFACE_HA
COL_PARCEL_AREA_HA       = 'Superficie totale (ha)'      # paired with CSV_COL_SURFACE_TOT_HA
COL_PERIOD_Y             = 'Turno (a)'                   # paired with CSV_COL_PERIOD_Y
COL_OPERATOR             = 'Operatore'                   # paired with CSV_COL_OPERATOR
COL_H_MEASURED           = 'h misurata'                  # display label; bool rendered yes/no

# Type-of-intervention display labels for COL_TYPE values
TYPE_HIGHFOREST = 'Fustaia'
TYPE_COPPICE   = 'Ceduo'

# Stand-in displayed in the Particella column for region-wide
# HarvestPlanItem rows (region set, parcel NULL).
LABEL_ALL_PARCELS = '(tutti)'  # form pulldown placeholder for region-wide
# Canonical Particella value for region-wide items in CSV import/export
# (round-trip marker) and in the calendar table display.  Single short
# character chosen for narrow columns; localizable.
PARCEL_WHOLE_REGION_MARK = 'X'

# Zip-archive filenames for the plan-level Esporta CSV.  Italian by
# default; localizable so a future language gets distinct names.
CSV_FILE_HIGHFOREST = 'fustaia.csv'
CSV_FILE_COPPICE    = 'ceduo.csv'
CSV_FILE_REGRESSION = 'equazioni_ipsometro.csv'

# Canonical bootstrap data-dir filenames (reference/container files derive their
# name from the RefTable stem; these are the bespoke + deferred files).
CSV_FILE_PARCELS            = 'particelle.csv'
CSV_FILE_SURVEYS            = 'surveys.csv'
CSV_FILE_SAMPLE_AREAS       = 'sample_areas.csv'
CSV_FILE_SAMPLED_TREES      = 'sampled-trees.csv'
CSV_FILE_HYPSO              = 'hypso_params.csv'
CSV_FILE_PRESERVED_TREES    = 'preserved-trees.csv'
CSV_FILE_HARVEST_PLAN_ITEMS = 'harvest_plan_items.csv'
CSV_FILE_HARVESTS           = 'harvests.csv'

# Harvest-plan digest columns.
COL_YEAR_START = 'Anno inizio'
COL_YEAR_END   = 'Anno fine'

# Tree-height-regression digest columns.  Single-letter coefficient
# names match the CSV header form.
COL_FUNCTION     = 'funzione'
COL_A            = 'a'
COL_B            = 'b'
COL_R2           = 'r²'
COL_N_REGRESSION = 'n'

# Prelievi: link to the HarvestPlanItem this harvest is part of.
COL_WORKSITE = 'Cantiere'

# ---------------------------------------------------------------------------
# App verbose names
# ---------------------------------------------------------------------------

APP_BASE = 'Base'
APP_PRELIEVI = 'Prelievi'
APP_BOSCO = 'Bosco'
APP_CAMPIONAMENTI = 'Campionamenti'
APP_PIANO_DI_TAGLIO = 'Piano di taglio'
APP_CONTROLLO = 'Controllo'
APP_IMPOSTAZIONI = 'Impostazioni'

# ---------------------------------------------------------------------------
# Model verbose names (singular, plural)
# ---------------------------------------------------------------------------

USER = 'utente'
USERS = 'utenti'

REGION = 'compresa'
REGIONS = 'comprese'

ECLASS = 'classe economica'
ECLASSES = 'classi economiche'

CREW = 'squadra'
CREWS = 'squadre'

TRACTOR = 'trattore'
TRACTORS = 'trattori'

SPECIES = 'specie'
SPECIES_PLURAL = 'specie'

# Default species names for new-tree entry, matched against
# Species.common_name (case-insensitive).  Picked on the fustaia /
# ceduo parcel type so the operator's most likely tree is preselected.
SPECIES_DEFAULT_HIGHFOREST = 'abete'
SPECIES_DEFAULT_COPPICE = 'castagno'

PRODUCT = 'tipo di prodotto'
PRODUCTS = 'tipi di prodotto'

HARVEST_PLAN = 'piano di gestione'
HARVEST_PLANS = 'piani di gestione'

HARVEST_DETAIL = 'prescrizione di taglio'
HARVEST_DETAILS = 'prescrizioni di taglio'

PARCEL = 'particella'
PARCELS = 'particelle'

HARVEST_PLAN_ITEM = 'voce piano di gestione'
HARVEST_PLAN_ITEMS = 'voci piano di gestione'

PARCEL_PLAN_DETAIL = 'prescrizione particella'
PARCEL_PLAN_DETAILS = 'prescrizioni particelle'

SAMPLE_GRID = 'griglia di campionamento'
SAMPLE_GRIDS = 'griglie di campionamento'

SAMPLE_AREA = 'area di saggio'
SAMPLE_AREAS = 'aree di saggio'

SURVEY = 'rilevamento'
SURVEYS = 'rilevamenti'

SAMPLE = 'campione'
SAMPLES = 'campioni'

TREE = 'albero'
TREES = 'alberi'
TREE_PRESERVED = 'pianta ad accrescimento indefinito'
TREE_PRESERVEDS = 'piante ad accrescimento indefinito'

TREE_SAMPLE = 'misurazione albero'
TREE_SAMPLES = 'misurazioni alberi'

TREE_MARK = 'albero martellato'
TREE_MARKS = 'alberi martellati'

HARVEST_TRANSITION = 'transizione cantiere'
HARVEST_TRANSITIONS = 'transizioni cantiere'

HYPSO_PARAM_SET = 'set di parametri ipsometrici'
HYPSO_PARAM_SETS = 'set di parametri ipsometrici'
HYPSO_PARAM = 'parametro ipsometrico'
HYPSO_PARAMS = 'parametri ipsometrici'
HYPSO_SOURCE_COMPUTED = 'calcolato'
HYPSO_SOURCE_IMPORTED = 'importato'

# Hypsometric-parameter-set audit field labels.
COL_HYPSO_SOURCE          = 'Origine'
COL_MIN_N                 = 'N minimo'
COL_USE_FOR_HEIGHT_PLOTS  = 'Usa per grafici altezza/diametro'
COL_SUPERSEDED_AT         = 'Sostituito il'

# Harvest-plan-item state machine display labels.  Integer values are
# encoded as HarvestPlanItemState in apps/base/models.py.
STATE_PLANNED    = 'pianificato'
STATE_MARKED     = 'martellato'
STATE_OPEN       = 'cantiere aperto'
STATE_HARVESTING = 'in prelievo'
STATE_CLOSED     = 'cantiere chiuso'

# Plan-item / harvest boolean flag labels (rendered in calendar Note
# column and in the harvest table Note column).  At most two of the
# three co-occur in practice; rendered as a comma-joined string.
FLAG_DAMAGED      = 'Catastrofato'
FLAG_UNHEALTHY    = 'Fitosanitario'
FLAG_PSR          = 'PSR'

DIGEST_STATUS = 'stato digest'
DIGEST_STATUSES = 'stati digest'

USED_NONCE = 'nonce utilizzato'
USED_NONCES = 'nonce utilizzati'

SPECIES_OTHER = 'Altro'

HARVEST = 'operazione di prelievo'
HARVESTS = 'operazioni di prelievo'

HARVEST_SPECIES = 'specie prelievo'
HARVEST_SPECIES_PLURAL = 'specie prelievo'

HARVEST_TRACTOR = 'trattore prelievo'
HARVEST_TRACTORS = 'trattori prelievo'
