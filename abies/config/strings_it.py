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
# Display labels for the Role pulldown; distinct from the JS wire-format
# ROLE_* constants in `apps/base/static/base/js/constants.js`.
LABEL_ROLE_ADMIN = 'Amministratore'
LABEL_ROLE_WRITER = 'Redattore'
LABEL_ROLE_READER = 'Membro'
LABEL_NAME = 'Nome'
LABEL_NOTES = 'Note'
LABEL_MANUFACTURER = 'Marca'
LABEL_MODEL = 'Modello'
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

AUDIT_INSERT = 'Inserimento'
AUDIT_UPDATE = 'Modifica'
AUDIT_DELETE = 'Eliminazione'

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
ERR_PLAN_ITEM_NOT_FOUND = 'Voce di piano non trovata.'
ERR_PLAN_ITEM_STATE_NOT_PLANNED = (
    'La voce non è in stato "pianificato"; eliminazione non consentita.'
)
ERR_PLAN_ITEM_HAS_DEPS = (
    'La voce ha martellate, prelievi o transizioni di cantiere associati; '
    'eliminale prima.'
)
ERR_PLAN_ITEM_VOLUME_NEGATIVE = 'Il volume previsto deve essere positivo.'
ERR_PLAN_ITEM_AREA_NEGATIVE = 'La superficie deve essere positiva.'
ERR_TRANSITION_INVALID_STATE = (
    'Transizione non consentita dallo stato attuale.'
)
ERR_TRANSITION_DATE_REQUIRED = 'Data della transizione obbligatoria.'
ERR_CSV_NO_FILES = (
    'Nessun file CSV caricato. Allega almeno un file '
    '(fustaia, ceduo o equazioni).'
)
ERR_CSV_PARCEL_NOT_FOUND = 'Riga {}: particella non trovata ({} / {}).'
ERR_CSV_REGION_NOT_FOUND = 'Riga {}: compresa non trovata ({}).'
ERR_CSV_SPECIES_NOT_FOUND = 'Riga {}: specie non trovata ({}).'
ERR_CSV_FUNCTION_INVALID = (
    'Riga {}: funzione di regressione non supportata ({}).'
)
ERR_CSV_VALUE_PARSE = 'Riga {}: valore non valido nella colonna {} ({}).'
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
ERR_CSV_SURVEY_REQUIRED = 'Seleziona prima un rilevamento.'
ERR_CSV_GRID_REQUIRED = 'Seleziona prima una griglia di destinazione.'
ERR_CSV_ROW_AREA_DUPLICATE = (
    'Riga {}: area ({} / {} / {}) già presente nella griglia.'
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

# ---------------------------------------------------------------------------
# CSV column headers.  These are simultaneously the wire-format identifiers
# the importer matches against AND the user-facing copy in the import help
# text.  Both sides reference these constants so they cannot drift.
# ---------------------------------------------------------------------------

CSV_COL_COMPRESA    = 'Compresa'
CSV_COL_PARTICELLA  = 'Particella'
CSV_COL_AREA_SAGGIO = 'Area saggio'
CSV_COL_LON         = 'Lon'
CSV_COL_LAT         = 'Lat'
CSV_COL_QUOTA       = 'Quota'
CSV_COL_RAGGIO      = 'Raggio'
CSV_COL_ALBERO      = 'Albero'
CSV_COL_POLLONE     = 'Pollone'
CSV_COL_MATRICINA   = 'Matricina'
CSV_COL_D_CM        = 'D_cm'
CSV_COL_H_M         = 'H_m'
CSV_COL_L10_MM      = 'L10_mm'
CSV_COL_GENERE      = 'Genere'
CSV_COL_FUSTAIA     = 'Fustaia'
CSV_COL_DATA        = 'Data'
CSV_COL_PAI         = 'PAI'

# CSV column headers used by the legacy bosco/data `import_*` management
# commands.  Where a column also appears in a digest, the names are
# paired (`CSV_COL_X` / `COL_X`); the values may match (same Italian
# token used in both contexts) or differ (e.g. `Altitudine min` in the
# CSV vs `Alt. min` in the digest column).
CSV_COL_COMPARTO       = 'Comparto'
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
CSV_COL_ANNO              = 'Anno'                        # scheduled-cut year
CSV_COL_PRELIEVO_M3       = 'Prelievo (m³)'               # fustaia volume_planned_m3
CSV_COL_SUPERFICIE_HA     = 'Superficie intervento (ha)'  # ceduo intervention_area_ha
CSV_COL_TURNO_A           = 'Turno (a)'                   # coppice rotation, years
CSV_COL_FUNZIONE          = 'funzione'                    # regression function (`ln`)
CSV_COL_A                 = 'a'                           # regression coefficient
CSV_COL_B                 = 'b'                           # regression coefficient
CSV_COL_R2                = 'r2'                          # coefficient of determination
CSV_COL_N_REGRESSION      = 'n'                           # sample count for regression fit

# Ipso CSV column headers (tree-mark import; format produced by
# laforesta/ipso — see ipso/CLAUDE.md "CSV format").
CSV_COL_CATASTROFATA      = 'Catastrofata'                # session-level flag (ignored row-wise)
CSV_COL_NUMERO            = 'Numero'                      # operator-assigned tree number
CSV_COL_H_MEASURED        = 'H_measured'                  # 1 if operator typed h, 0 if auto-h
CSV_COL_ACC_M             = 'Acc_m'                       # GPS accuracy in metres
CSV_COL_OPERATORE         = 'Operatore'                   # operator name

# ---------------------------------------------------------------------------
# Digest column headers (display labels in JSON `columns` arrays).
# Where a column has a CSV counterpart with a different display form
# (e.g. CSV `D_cm` vs digest `D (cm)`) the names are paired
# (`CSV_COL_D_CM` / `COL_D_CM`).  Where the display value equals the
# CSV value (e.g. `Lat`, `Pollone`), both constants exist with the same
# value for symmetric usage at call sites.
# ---------------------------------------------------------------------------

COL_COMPRESA           = 'Compresa'         # paired with CSV_COL_COMPRESA
COL_QUOTA              = 'Quota'            # paired with CSV_COL_QUOTA
COL_RAGGIO             = 'Raggio'           # paired with CSV_COL_RAGGIO
COL_NAME               = 'Nome'             # paired with LABEL_NAME (HTML form)
COL_LATIN_NAME         = 'Nome latino'
COL_DENSITY            = 'Densità (q/m³)'
COL_CLASS              = 'Classe'           # parcel eclass
COL_AREA_HA            = 'Area (ha)'
COL_AVE_AGE            = 'Età media'
COL_LOCATION           = 'Località'
COL_ALT_MIN            = 'Alt. min'
COL_ALT_MAX            = 'Alt. max'
COL_ASPECT             = 'Esposizione'
COL_GRADE_PCT          = 'Pendenza %'
COL_SORT_ORDER         = 'Sort order'       # internal English; not localized
COL_YEAR               = 'Anno'
COL_DESCRIPTION        = 'Descrizione'
COL_N_AREAS            = 'N. aree'
COL_REGIONS            = 'Comprese'
COL_N_SURVEYS          = 'N. rilevamenti'
COL_LAST_UPDATE        = 'Ultimo aggiornamento'
COL_GRID               = 'Griglia'
COL_HARVEST_PLAN       = 'Piano di taglio'
COL_N_AREAS_VISITED    = 'N. aree visitate'
COL_N_AREAS_TOTAL      = 'N. aree totali'
COL_DATE_FIRST         = 'Data primo'
COL_DATE_LAST          = 'Data ultimo'
COL_NUMBER             = 'Numero'
COL_LAT                = 'Lat'              # paired with CSV_COL_LAT
COL_LON                = 'Lon'              # paired with CSV_COL_LON
COL_SURVEY             = 'Rilevamento'
COL_SAMPLE_AREA        = 'Area di saggio'
COL_N_TREES            = 'N. alberi'
COL_SAMPLE_DATE        = 'Data campione'
COL_AREA_NUM           = 'N. area'
COL_TREE_NUM           = 'N. albero'
COL_SPECIES            = 'Specie'
COL_POLLONE            = 'Pollone'          # paired with CSV_COL_POLLONE
COL_MATRICINA          = 'Matricina'        # paired with CSV_COL_MATRICINA
COL_PAI                = 'PAI'              # paired with CSV_COL_PAI
COL_D_CM               = 'D (cm)'           # paired with CSV_COL_D_CM
COL_H_M                = 'h (m)'            # paired with CSV_COL_H_M
COL_L10_MM             = 'L10 (mm)'         # paired with CSV_COL_L10_MM
COL_V_M3               = 'V (m³)'           # short form (sampled-trees digest)
COL_MASS_Q             = 'm (q)'

# Piano di taglio digest columns (calendar + martellate tables).
COL_YEAR_PLANNED         = 'Anno previsto'
COL_YEAR_ACTUAL          = 'Anno effettivo'
COL_TYPE                 = 'Tipo'                       # alto fusto / ceduo
COL_STATE                = 'Stato'
COL_VOLUME_PLANNED       = 'Volume previsto'
COL_VOLUME_MARKED        = 'Volume martellato'
COL_VOLUME_ACTUAL        = 'Volume effettivo'
COL_INTERVENTION_AREA_HA = 'Superficie intervento (ha)'  # paired with CSV_COL_SUPERFICIE_HA
COL_PARCEL_AREA_HA       = 'Superficie totale (ha)'      # paired with CSV_COL_SUPERFICIE_TOT_HA
COL_TURNO_A              = 'Turno (a)'                   # paired with CSV_COL_TURNO_A
COL_OPERATOR             = 'Operatore'                   # paired with CSV_COL_OPERATORE
COL_NUMERO               = 'Numero'                      # paired with CSV_COL_NUMERO
COL_H_MEASURED           = 'h misurata'                  # display label; bool rendered yes/no

# Type-of-intervention display labels for COL_TYPE values
TYPE_FUSTAIA = 'fustaia'
TYPE_CEDUO   = 'ceduo'

# Stand-in displayed in the Particella column for region-wide
# HarvestPlanItem rows (region set, parcel NULL).
LABEL_ALL_PARCELS = '(tutti)'  # form pulldown placeholder for region-wide
# Canonical Particella value for region-wide items in CSV import/export
# (round-trip marker) and in the calendar table display.  Single short
# character chosen for narrow columns; localizable.
PARCEL_WHOLE_REGION_MARK = 'X'

# Zip-archive filenames for the plan-level Esporta CSV.  Italian by
# default; localizable so a future language gets distinct names.
CSV_FILE_FUSTAIA    = 'fustaia.csv'
CSV_FILE_CEDUO      = 'ceduo.csv'
CSV_FILE_REGRESSION = 'equazioni_ipsometro.csv'

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
COL_CANTIERE = 'Cantiere'

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
SPECIES_DEFAULT_FUSTAIA = 'abete'
SPECIES_DEFAULT_CEDUO = 'castagno'

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

TREE_SAMPLE = 'misurazione albero'
TREE_SAMPLES = 'misurazioni alberi'

TREE_MARK = 'albero martellato'
TREE_MARKS = 'alberi martellati'

HARVEST_TRANSITION = 'transizione cantiere'
HARVEST_TRANSITIONS = 'transizioni cantiere'

TREE_HEIGHT_REGRESSION = 'regressione altezza albero'
TREE_HEIGHT_REGRESSIONS = 'regressioni altezza albero'

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
