"""Italian UI strings for Abies."""

# ---------------------------------------------------------------------------
# Shell / chrome
# ---------------------------------------------------------------------------

COMPANY_NAME = 'La Foresta'
LOGOUT = 'Esci'
LOADING = 'Caricamento...'
DISMISS = 'Chiudi'
SAVE = 'Salva'
SAVE_AND_ADD = 'Salva e aggiungi'
DELETE_CONFIRM = 'Questa azione non può essere annullata. Confermi?'
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
COL_OPTYPE = 'Tipo'
COL_QUINTALS = 'Q.li'
COL_VDP = 'VDP'
COL_PROT = 'Prot'
COL_NOTE = 'Note'
COL_EXTRA_NOTE = 'Altre note'
COL_ACTIVE = 'Attivo'
LABEL_USERNAME = 'Nome utente'
LABEL_ROLE = 'Ruolo'
LABEL_NAME = 'Nome'
LABEL_NOTES = 'Note'
LABEL_MANUFACTURER = 'Marca'
LABEL_MODEL = 'Modello'
LABEL_YEAR = 'Anno'
LABEL_LATIN_NAME = 'Nome latino'
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

TABLE_HARVEST_OP = 'Prelievo'
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

# ---------------------------------------------------------------------------
# App verbose names
# ---------------------------------------------------------------------------

APP_BASE = 'Base'
APP_PRELIEVI = 'Prelievi'
APP_BOSCO = 'Bosco'
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

OPTYPE = 'tipo di operazione'
OPTYPES = 'tipi di operazione'

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

SAMPLE_AREA = 'area di saggio'
SAMPLE_AREAS = 'aree di saggio'

PRESERVED_TREE = 'pianta ad accrescimento indefinito'
PRESERVED_TREES = 'piante ad accrescimento indefinito'

DIGEST_STATUS = 'stato digest'
DIGEST_STATUSES = 'stati digest'

USED_NONCE = 'nonce utilizzato'
USED_NONCES = 'nonce utilizzati'

HARVEST_OP = 'operazione di prelievo'
HARVEST_OPS = 'operazioni di prelievo'

HARVEST_SPECIES = 'specie prelievo'
HARVEST_SPECIES_PLURAL = 'specie prelievo'

HARVEST_TRACTOR = 'trattore prelievo'
HARVEST_TRACTORS = 'trattori prelievo'
