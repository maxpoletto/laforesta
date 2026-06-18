// All user-facing Italian strings, in one file so a future English variant
// is a single-file replacement.
'use strict';

const S = {
  APP_TITLE: 'Ipso',

  // Mode select screen
  MODE_TITLE: 'Ipso',
  MODE_MARTELLATE: 'Martellate',
  MODE_SAMPLES: 'Campionamenti',
  MODE_PAI: 'PAI',
  MODE_BACK: 'Indietro',

  // Pre-session screen
  PRE_NEW_SESSION: 'Nuova martellata',
  PRE_NEW_SAMPLES: 'Nuovo campionamento',
  PRE_NEW_PAI: 'Nuovo PAI',
  PRE_OPERATOR: 'Operatore',
  PRE_DATA: 'Data',
  PRE_COMPRESA: 'Compresa',
  PRE_PARTICELLA: 'Particella',
  PRE_SURVEY: 'Rilevamento',
  PRE_PICK_SURVEY: '— scegli un rilevamento —',
  PRE_CATASTROFATA: 'Piante catastrofate',
  PRE_PICK_COMPRESA: '— scegli una compresa —',
  PRE_START: 'Inizia',
  CATASTROFATE: 'catastrofate',
  TYPE_HIGHFOREST: 'Fustaia',
  TYPE_COPPICE: 'Ceduo',

  // Recording screen
  REC_GPS_WAITING: 'GPS in attesa…',
  REC_OUT_OF_BOUNDS: 'Fuori dai confini',
  REC_PARTICELLA_AUTO: '(automatica)',
  REC_PARTICELLA_PLACEHOLDER: '—',
  REC_SAMPLE_AREA: 'Area',
  REC_SAMPLE_AREA_AUTO: '(automatica)',
  REC_SAMPLE_AREA_PLACEHOLDER: '—',
  REC_SAMPLE_AREA_OUT_OF_BOUNDS: 'Fuori dalle aree di saggio',
  REC_SPECIE: 'Specie',
  REC_PICK_SPECIE: '— scegli —',
  REC_NUMBER: 'Numero',
  REC_GRUPPO: 'Gruppo',
  REC_D: 'D (cm)',
  REC_H: 'h (m)',
  REC_AUTO_H_MISSING:
    'Nessuna regressione per questa compresa e specie: inserisci h manualmente.',
  REC_SAVE: 'Salva e prossimo',
  REC_MAP: 'Mappa',
  REC_VIEW_DATA: 'Dati',
  REC_END: 'Fine',
  REC_LAST_PREFIX: 'ultimo:',
  REC_NO_LAST: 'nessun albero registrato',
  REC_EDIT_LAST: 'Modifica',
  REC_DELETE_LAST: 'Elimina',
  REC_CANCEL: 'Annulla',
  REC_TREE_NUMBER: 'albero n.',
  REC_SAMPLE_AREA_NUMBER: 'Area',

  // Visualizza dati raccolti screen
  DATA_TITLE: 'Dati raccolti',
  DATA_GROUPS: 'Gruppi',
  DATA_TREES: 'Alberi',
  DATA_COUNT: 'Conteggio',
  DATA_COL_NUMBER: 'N.',
  DATA_COL_SPECIE: 'Specie',
  DATA_COL_PARTICELLA: 'Part.',
  DATA_COL_GRUPPO: 'Gruppo',
  DATA_COL_D: 'D',
  DATA_COL_H: 'h',
  DATA_CLOSE: 'Chiudi',
  DATA_EMPTY: 'Nessun albero registrato.',
  MAP_TITLE: 'Mappa',
  MAP_BACK: 'Indietro',
  MAP_CENTER: 'Centra',
  MAP_WAITING: 'GPS in attesa...',
  MAP_UNAVAILABLE: 'Mappa non disponibile',
  MAP_NO_PARCELS: 'Nessuna particella disponibile.',
  MAP_PAI_TOGGLE: 'Mostra/nascondi PAI esistenti',
  MAP_ERROR_LEAFLET_MISSING: 'Leaflet non disponibile',
  DATA_NO_GROUPS: 'Nessun gruppo assegnato.',

  // Resume modal
  RESUME_TITLE: 'Sessioni non chiuse',
  RESUME_BODY:
    'Hai una o più sessioni iniziate ma non esportate. Cosa vuoi farne?',
  RESUME_RESUME: 'Riprendi',
  RESUME_EXPORT: 'Esporta CSV',
  RESUME_DISCARD: 'Scarta',

  // Confirm-end modal
  END_TITLE: 'Termina sessione',
  END_BODY: (n) =>
    `La sessione contiene ${n} alber${n === 1 ? 'o' : 'i'}. ` +
    `Esportare il CSV e chiudere?`,
  END_CONFIRM: 'Esporta e chiudi',

  // Done screen
  DONE_TITLE: 'Sessione esportata',
  DONE_BODY: (n) =>
    `${n} alber${n === 1 ? 'o' : 'i'} salvat${n === 1 ? 'o' : 'i'} su CSV.`,
  DONE_EMPTY_TITLE: 'Sessione chiusa',
  DONE_EMPTY_BODY: 'Nessun albero registrato.',
  DONE_NEW: 'Nuova sessione',

  // Persistent-storage banner
  STORAGE_OK: '',
  STORAGE_WARNING:
    'Memoria non protetta — completa ed esporta la sessione oggi.',

  // Toasts and errors
  GPS_DENIED:
    'GPS non disponibile: gli alberi verranno registrati senza coordinate.',
  GPS_PERMISSION_BANNER:
    'Permesso GPS non concesso. Per registrare le coordinate, abilita la ' +
    'posizione per ipso.laforesta.it nelle impostazioni del browser.',
  BACKUP_SAVED: (n) => `Backup CSV salvato (${n} alberi).`,
  CSV_HEADER: [
    'Data', 'Compresa', 'Particella', 'Catastrofata',
    'Numero', 'Specie', 'D_cm', 'H_m', 'H_measured',
    'Lat', 'Lon', 'Acc_m', 'Operatore',
  ],
  CSV_FILENAME_CATASTROFATE: 'catastrofate',
  CSV_ERROR_DATE_FORMAT: (value) => `Data CSV non valida: ${value}`,
  STORE_ERROR_DB_BLOCKED:
    'Database Ipso bloccato da una scheda precedente.',
  STORE_ERROR_TX_ABORTED: 'Transazione Ipso annullata.',
  STORE_ERROR_SESSION_NOT_FOUND: (id) => `Sessione Ipso non trovata: ${id}`,
  STORE_ERROR_TREE_NOT_FOUND: (id) => `Albero Ipso non trovato: ${id}`,
  STORE_ERROR_TREE_SESSION_MISMATCH:
    'Albero Ipso non appartenente alla sessione.',
  ERROR_GEO_UNAVAILABLE: 'geo.js non disponibile',
  ERROR_HTTP_STATUS: (status) => `HTTP ${status}`,
  ERROR_BOOTSTRAP_INVALID: 'Bootstrap Ipso non valido',
  ERROR_TOKEN_MISSING: 'Token Ipso mancante',
  ERROR_GEOJSON_INVALID: 'GeoJSON non valido',
  TOAST_REFERENCE_LOAD_ERROR: (detail) =>
    `Errore caricamento reference.json: ${detail}`,
  TOAST_TERRENI_LOAD_ERROR: (detail) =>
    `Errore caricamento terreni.geojson: ${detail}`,
  TOAST_DB_OPEN_ERROR: (detail) => `Errore apertura database: ${detail}`,
  TOAST_SESSION_START_ERROR: (detail) =>
    `Errore avvio sessione: ${detail}`,
  TOAST_SAVE_ERROR: (detail) => `Errore salvataggio: ${detail}`,
  TOAST_DELETE_ERROR: (detail) => `Errore eliminazione: ${detail}`,
  TOAST_EXPORT_ERROR: (detail) => `Errore esportazione: ${detail}`,
  TOAST_UPLOAD_STATE_ERROR: (detail) =>
    `Errore salvataggio stato upload: ${detail}`,
  TOAST_STATE_SAVE_ERROR: (detail) => `Errore salvataggio stato: ${detail}`,
  TOAST_MAP_POINTS_LOAD_ERROR: (detail) =>
    `Errore caricamento punti mappa: ${detail}`,
  TOAST_DATA_LOAD_ERROR: (detail) => `Errore caricamento dati: ${detail}`,
  TOAST_BOOT_ERROR: (detail) => `Errore avvio: ${detail}`,

  // Upload screen
  UPLOAD_TITLE: 'Caricamento in corso',
  UPLOAD_ATTEMPT: (n) => `Tentativo ${n}`,
  UPLOAD_BAIL: 'Annulla caricamento e salva solo sul telefono',
  UPLOAD_SUCCESS_TOAST: 'Caricamento completato',
  UPLOAD_LOCAL_ONLY_TOAST: 'Salvato solo sul telefono',
  UPLOAD_ERROR_AUTH:
    'Errore di autenticazione. Contatta lo sviluppatore.',
  UPLOAD_ERROR_CONFLICT:
    'La sessione risulta già caricata con contenuto diverso. ' +
    'Contatta l\'ufficio.',
  UPLOAD_ERROR_INVALID:
    'Il server ha rifiutato il file. Contatta lo sviluppatore.',
  UPLOAD_ERROR_TOO_LARGE:
    'File troppo grande per il server. Contatta lo sviluppatore.',
  UPLOAD_ERROR_NETWORK: 'Errore di rete. Riprovo…',
  UPLOAD_ERROR_SERVER: 'Errore del server. Riprovo…',
  UPLOAD_ERROR_RATE_LIMITED: 'Server occupato. Riprovo…',
  UPLOAD_NEXT_RETRY_IN: (s) => `Prossimo tentativo fra ${s} s`,
  UPLOAD_ERROR_CONTEXT_MISSING: 'Contesto caricamento mancante.',
  UPLOAD_ERROR_SPECIES_ID_MISSING: (name) =>
    `Specie senza ID Abies: ${name || ''}`,
  UPLOAD_ERROR_REGION_ID_MISSING: (compresa) =>
    `Compresa senza ID Abies: ${compresa || ''}`,
  UPLOAD_ERROR_PARCEL_ID_MISSING: (compresa, particella) =>
    `Particella senza ID Abies: ${compresa || ''}/${particella || ''}`,

  // Resume modal — upload variant
  UPLOAD_RESUME_TITLE: 'Sessioni in attesa di caricamento',
  UPLOAD_RESUME_DO_NOW: 'Carica ora',
  UPLOAD_RESUME_KEEP_LOCAL: 'Mantieni solo locale',
  UPLOAD_DONE_BODY: (n) =>
    `${n} alber${n === 1 ? 'o' : 'i'} caricat${n === 1 ? 'o' : 'i'} sul server.`,

  // Pill formatter. Prepends "n. <N> · " when the operator assigned a
  // number to the tree (Number.isInteger), otherwise omits the slot
  // entirely — for trees auto-blanked by the D ≤ 17 rule there's no
  // visible number to show.
  pill(rec) {
    if (!rec) return S.REC_NO_LAST;
    const d = rec.d_cm != null ? `D=${rec.d_cm}` : 'D=—';
    const h = rec.h_m != null ? `h=${rec.h_m}` : 'h=—';
    const numPart = Number.isInteger(rec.numero) ? `n. ${rec.numero} · ` : '';
    return `${S.REC_LAST_PREFIX} ${numPart}${rec.specie}, ${d}, ${h}`;
  },

  // "Where" formatter used in the recording-screen header (initial
  // paint, before GPS commits the first parcel), the app-bar
  // sub-status, and the resume modal. Particella is per-tree post-v4,
  // so we no longer have a single representative value at the session
  // level — the header shows just the compresa (suffixed for
  // catastrofate sessions). The recording-screen header is then
  // overwritten dynamically by the parcel-locator subscriber.
  where(session) {
    if (!session) return '';
    if (session.catastrofata) {
      return `${session.compresa} · ${S.CATASTROFATE}`;
    }
    return session.compresa;
  },
};

if (typeof module !== 'undefined') module.exports = { S };
