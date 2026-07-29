/**
 * Natural string comparators for labels that mix numbers and text.
 */

import * as S from './strings.js';

export function compareNaturalLabels(a, b) {
  const aParts = naturalSortParts(a);
  const bParts = naturalSortParts(b);
  const max = Math.max(aParts.length, bParts.length);
  for (let i = 0; i < max; i++) {
    if (aParts[i] == null) return -1;
    if (bParts[i] == null) return 1;
    if (aParts[i] === bParts[i]) continue;
    if (typeof aParts[i] === 'number' && typeof bParts[i] === 'number') {
      return aParts[i] - bParts[i];
    }
    return String(aParts[i]).localeCompare(String(bParts[i]), S.LOCALE);
  }
  return String(a || '').localeCompare(String(b || ''), S.LOCALE);
}

export function compareParcelNames(a, b) {
  return compareNaturalLabels(a, b);
}

function naturalSortParts(value) {
  return String(value || '').split(/(\d+)/)
    .filter(part => part !== '')
    .map(part => /^\d+$/.test(part) ? Number(part) : part.toLocaleLowerCase(S.LOCALE));
}
