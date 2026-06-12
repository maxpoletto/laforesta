import { COL_COPPICE } from './constants.js';
import * as S from './strings.js';

export function recordIsCoppice(record, columns) {
  if (!record || !columns) return false;
  const coppiceCol = columns.indexOf(COL_COPPICE);
  if (coppiceCol >= 0) return record[coppiceCol] === true;
  const typeCol = columns.indexOf(S.COL_TYPE);
  if (typeCol < 0) return false;
  return String(record[typeCol] || '').toLocaleLowerCase('it') ===
    S.TYPE_COPPICE.toLocaleLowerCase('it');
}
