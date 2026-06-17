// All user-facing Italian strings, in one file so a future English variant
// is a single-file replacement.
'use strict';

const S = {
  APP_TITLE: 'Ipso — martellata',

  // Pre-session screen
  PRE_NEW_SESSION: 'Nuova martellata',
  PRE_OPERATOR: 'Operatore',
  PRE_DATA: 'Data',
  PRE_COMPRESA: 'Compresa',
  PRE_PARTICELLA: 'Particella',
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
