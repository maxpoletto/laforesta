"""Italian UI strings for Abies."""

# ---------------------------------------------------------------------------
# Shell / chrome
# ---------------------------------------------------------------------------

COMPANY_NAME = 'La Foresta'
LOGOUT = 'Esci'
LOADING = 'Caricamento...'
DISMISS = 'Chiudi'
SAVE = 'Salva'
SAVE_AND_CONTINUE = 'Salva e continua'
DELETE_CONFIRM = 'I dati cancellati non potranno essere recuperati. Confermi?'
SEARCH_PLACEHOLDER = 'Cerca...'
EXPORT_CSV = 'Esporta CSV'
ERROR_NETWORK = 'Errore di rete. Riprovare.'
ERROR_CONFLICT = 'Il record è stato modificato da un altro utente.'
ERROR_GENERIC = 'Errore imprevisto.'
ERROR_RATE_LIMIT = 'Troppe richieste. Riprovare tra un minuto.'

# Validation
ERR_DATE_REQUIRED = 'Data obbligatoria.'
ERR_QUINTALS_POSITIVE = 'I quintali devono essere positivi.'
ERR_SPECIES_PCT_SUM = 'Le percentuali delle specie devono sommare a 100.'
ERR_TRACTOR_PCT_SUM = 'Le percentuali dei trattori devono sommare a 100.'
ERR_DATE_FUTURE = 'La data non può essere nel futuro.'
ERR_VDP_DUPLICATE = 'VDP {} è già utilizzato.'
ERR_NOT_FOUND = 'Record non trovato.'
ERR_FORBIDDEN = 'Non autorizzato.'

# Tab names (same as app verbose names but title-cased for display)
TAB_BOSCO = 'Bosco'
TAB_PRELIEVI = 'Prelievi'
TAB_CONTROLLO = 'Controllo'
TAB_IMPOSTAZIONI = 'Impostazioni'

# Column / field labels (used in audit digest and shared with JS)
COL_DATE = 'Data'
COL_PARCEL = 'Particella'
COL_CREW = 'Squadra'
COL_PRODUCT = 'Tipo'
COL_QUINTALS = 'Q.li'
COL_VOLUME_M3 = 'Volume (m³)'
COL_VDP = 'VDP'
COL_PROT = 'Prot'
COL_NOTE = 'Note'
COL_EXTRA_NOTE = 'Altre note'
COL_ACTIVE = 'Attivo'
LABEL_USERNAME = 'Nome utente'
LABEL_EMAIL = 'Email'
LABEL_ROLE = 'Ruolo'
ROLE_ADMIN = 'Amministratore'
ROLE_WRITER = 'Redattore'
ROLE_READER = 'Membro'
LABEL_NAME = 'Nome'
LABEL_NOTES = 'Note'
LABEL_MANUFACTURER = 'Marca'
LABEL_MODEL = 'Modello'
LABEL_YEAR = 'Anno'
LABEL_LATIN_NAME = 'Nome latino'
LABEL_DENSITY = 'Densità (q/m³)'
LABEL_FIRST_NAME = 'Nome'
LABEL_LAST_NAME = 'Cognome'
LABEL_LOGIN_METHOD = 'Metodo di accesso'
LABEL_CREATED_AT = 'Creato il'

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

ACTION_INSERT = 'Inserimento'
ACTION_UPDATE = 'Modifica'
ACTION_DELETE = 'Eliminazione'

# Settings
PASSWORD_MISMATCH = 'Le password non coincidono.'
PASSWORD_CHANGED = 'Password modificata.'
ERR_PASSWORD_REQUIRED = 'Password obbligatoria.'
ERR_NAME_REQUIRED = 'Nome obbligatorio.'
ERR_USERNAME_REQUIRED = 'Nome utente obbligatorio.'
ERR_EMAIL_REQUIRED = 'Email obbligatoria.'
ERR_DENSITY_INVALID = 'La densità deve essere un numero positivo.'
ERR_D_POSITIVE = 'Il diametro deve essere positivo.'
ERR_H_POSITIVE = 'L\'altezza deve essere positiva.'
ERR_TREE_NUMBER_REQUIRED = 'Numero albero obbligatorio.'
ERR_TREE_NUMBER_DUPLICATE = 'Numero albero {} già utilizzato in questo campione.'
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
ERR_AREA_IN_USE = (
    'Area di saggio già usata in un campione: non può essere eliminata.'
)
ERR_GRID_IN_USE = (
    'La griglia è usata da uno o più rilevamenti: eliminarli prima.'
)
CASCADE_WARN_SURVEY = (
    'Questa operazione cancellerà {n_samples} campioni e {n_trees} '
    'misure di alberi che non possono essere recuperati.'
)
CASCADE_EXPORT_REQUIRED = (
    'Per sicurezza, esporta i dati prima di procedere all\'eliminazione.'
)
ERR_CSV_FILE_REQUIRED = 'File CSV obbligatorio.'
ERR_CSV_NOT_UTF8 = 'Il file deve essere codificato in UTF-8.'
ERR_CSV_EMPTY = 'Il file CSV è vuoto.'
ERR_CSV_MISSING_COLS = 'Colonne CSV mancanti: {}.'
ERR_CSV_ROW_PARCEL = 'Riga {}: particella non trovata ({} / {}).'
ERR_CSV_ROW_AREA = 'Riga {}: area di saggio non trovata ({} / {} / {}).'
ERR_CSV_ROW_SPECIES = 'Riga {}: specie sconosciuta: {}.'
ERR_CSV_ROW_PARSE = 'Riga {}: errore di parsing ({}).'
ERR_CSV_SURVEY_REQUIRED = 'Seleziona prima un rilevamento.'
ERR_CSV_DATE_REQUIRED = (
    'Il file CSV non ha una colonna "Data": indicare una data predefinita.'
)

# ---------------------------------------------------------------------------
# App verbose names
# ---------------------------------------------------------------------------

APP_BASE = 'Base'
APP_PRELIEVI = 'Prelievi'
APP_BOSCO = 'Bosco'
APP_CAMPIONAMENTI = 'Campionamenti'
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

PRODUCT = 'tipo di prodotto'
PRODUCTS = 'tipi di prodotto'

NOTE = 'nota'
NOTES = 'note'

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

TREE_SAMPLE = 'misurazione albero'
TREE_SAMPLES = 'misurazioni alberi'

DIGEST_STATUS = 'stato digest'
DIGEST_STATUSES = 'stati digest'

USED_NONCE = 'nonce utilizzato'
USED_NONCES = 'nonce utilizzati'

HARVEST = 'operazione di prelievo'
HARVESTS = 'operazioni di prelievo'

HARVEST_SPECIES = 'specie prelievo'
HARVEST_SPECIES_PLURAL = 'specie prelievo'

HARVEST_TRACTOR = 'trattore prelievo'
HARVEST_TRACTORS = 'trattori prelievo'
