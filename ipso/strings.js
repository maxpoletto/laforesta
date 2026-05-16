// All user-facing Italian strings, in one file so a future English variant
// is a single-file replacement.
'use strict';

const S = {
  APP_TITLE: 'Ipso — martellata',

  // Pre-session screen
  PRE_NEW_SESSION: 'Nuova martellata',
  PRE_OPERATORE: 'Operatore',
  PRE_DATA: 'Data',
  PRE_COMPRESA: 'Compresa',
  PRE_PARTICELLA: 'Particella',
  PRE_CATASTROFATA: 'Piante catastrofate',
  PRE_PICK_COMPRESA: '— scegli una compresa —',
  PRE_PICK_PARTICELLA: '— scegli una particella —',
  PRE_START: 'Inizia',
  CATASTROFATE: 'catastrofate',

  // Recording screen
  REC_GPS_WAITING: 'GPS in attesa…',
  REC_OUT_OF_BOUNDS: 'Fuori dai confini',
  REC_PARTICELLA_AUTO: '(automatica)',
  REC_PARTICELLA_PLACEHOLDER: '—',
  REC_SPECIE: 'Specie',
  REC_PICK_SPECIE: '— scegli —',
  REC_NUMERO: 'Numero',
  REC_GRUPPO: 'Gruppo',
  REC_D: 'D (cm)',
  REC_H: 'h (m)',
  REC_AUTO_H_MISSING:
    'Nessuna regressione per questa compresa e specie: inserisci h manualmente.',
  REC_SAVE: 'Salva e prossimo',
  REC_VIEW_DATA: 'Visualizza dati raccolti',
  REC_END: 'Termina e esporta CSV',
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
  DATA_COL_NUMERO: 'N.',
  DATA_COL_SPECIE: 'Specie',
  DATA_COL_GRUPPO: 'Gruppo',
  DATA_COL_D: 'D',
  DATA_COL_H: 'h',
  DATA_CLOSE: 'Chiudi',
  DATA_EMPTY: 'Nessun albero registrato.',
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

  // Pill formatter. Prepends "n. <numero> · " when the operator assigned
  // a numero to the tree (Number.isInteger), otherwise omits the slot
  // entirely — for trees auto-blanked by the D ≤ 17 rule there's no
  // visible number to show.
  pill(rec) {
    if (!rec) return S.REC_NO_LAST;
    const d = rec.d_cm != null ? `D=${rec.d_cm}` : 'D=—';
    const h = rec.h_m != null ? `h=${rec.h_m}` : 'h=—';
    const numPart = Number.isInteger(rec.numero) ? `n. ${rec.numero} · ` : '';
    return `${S.REC_LAST_PREFIX} ${numPart}${rec.specie}, ${d}, ${h}`;
  },

  // "Where" formatter used in the recording-screen header, the app-bar
  // sub-status, and the resume modal. Two shapes depending on session
  // type: "Serra / 1" for a parcel-bound mark, "Serra · catastrofate"
  // for a roaming catastrofate session (where Particella is intentionally
  // unset and the server infers parcel from GPS).
  where(session) {
    if (!session) return '';
    if (session.catastrofata) {
      return `${session.compresa} · ${S.CATASTROFATE}`;
    }
    return `${session.compresa} / ${session.particella}`;
  },
};

if (typeof module !== 'undefined') module.exports = { S };
