/**
 * Italian UI string constants.
 *
 * To switch language, change the re-export in strings.js.
 */

// Shell / chrome
export const COMPANY_NAME = 'La Foresta';
export const LOADING = 'Caricamento...';
export const DISMISS = 'Chiudi';
export const SAVE = 'Salva';
export const SAVE_AND_ADD = 'Salva e aggiungi';
export const DELETE_CONFIRM = 'Questa azione non può essere annullata. Confermi?';
export const CANCEL = 'Annulla';

// Tables
export const FILTER_LABEL = 'Filtra';
export const SEARCH_PLACEHOLDER = 'Cerca...';
export const EXPORT_CSV = 'Esporta CSV';
export const ADD_LABEL = 'Aggiungi';
export const NO_RESULTS = 'Nessun risultato.';
export const BOOL_YES = 'Sì';
export const BOOL_NO = 'No';

// Errors
export const ERROR_NETWORK = 'Errore di rete. Riprovare.';
export const ERROR_CONFLICT = 'Il record è stato modificato da un altro utente.';
export const ERROR_GENERIC = 'Errore imprevisto.';
export const ERROR_DELETED = 'Il record è stato eliminato da un altro utente.';
export const ERROR_RATE_LIMIT = 'Troppe richieste. Riprovare tra un minuto.';

// Tab names
export const TAB_BOSCO = 'Bosco';
export const TAB_PRELIEVI = 'Prelievi';
export const TAB_CONTROLLO = 'Controllo';
export const TAB_IMPOSTAZIONI = 'Impostazioni';

// Prelievi
export const LABEL_YEARS = 'Anni';
export const RESET_FILTERS = 'Azzera filtri';

// Prelievi columns
export const COL_DATE = 'Data';
export const COL_REGION = 'Compresa';
export const COL_PARCEL = 'Particella';
export const COL_CREW = 'Squadra';
export const COL_VDP = 'VDP';
export const COL_QUINTALS = 'Q.li';
export const COL_NOTE = 'Note';
export const COL_EXTRA_NOTE = 'Altre note';
export const COL_OPTYPE = 'Tipo';
export const COL_PROT = 'Prot';
export const COL_ACTIVE = 'Attivo';

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
export const LABEL_OPTYPE = 'Tipo';
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

// Roles
export const ROLE_ADMIN = 'admin';
export const ROLE_WRITER = 'writer';
export const ROLE_READER = 'reader';
