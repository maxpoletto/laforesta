import * as S from '../../base/js/strings.js';
import { fmtCoord, fmtInt } from '../../base/js/format.js';

export function formatPosition(lat, lon, accM = null) {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return '';
  const coords = `(${fmtCoord(lat)}, ${fmtCoord(lon)})`;
  if (accM == null || accM === '') return coords;
  return `${coords} ± ${fmtInt(accM)} m`;
}

export function positionLabelValue(lat, lon, accM = null) {
  return [S.BOSCO_POSITION, formatPosition(lat, lon, accM)];
}
