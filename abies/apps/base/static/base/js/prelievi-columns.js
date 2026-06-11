/**
 * Prelievi table column definitions.
 *
 * Shared by the Prelievi page (`prelievi.js`) and the Piano-di-taglio item
 * view's Prelievi sub-table (`piano-di-taglio.js`) so both render the harvest
 * digest with identical formatting (decimals, blank-zero, widths).  Lives in
 * `base` (per the shared-JS convention) and is side-effect free.
 */
import { fmtInt, fmtDecimal1, fmtDecimal1BlankZero, fmtDecimal2 }
  from './format.js';
import * as S from './strings.js';
import { COL_PARCEL_ID, COL_REGION_ID, ROW_ID, VERSION } from './constants.js';

/** Column definitions for the fixed digest columns. */
export const STATIC_COLS = {
  [COL_REGION_ID]:   { label: COL_REGION_ID, hidden: true },
  [COL_PARCEL_ID]:   { label: COL_PARCEL_ID, hidden: true },
  [S.COL_DATE]:        { label: S.COL_DATE, type: 'date', width: '90px' },
  [S.COL_REGION]:    { label: S.COL_REGION, width: '80px' },
  [S.COL_PARCEL]:      { label: S.COL_PARCEL, width: '70px' },
  [S.COL_CREW]:        { label: S.COL_CREW, width: '108px' },
  [S.COL_TYPE]:        { label: S.COL_TYPE, width: '120px' },
  [S.COL_VDP]:         { label: S.COL_VDP, type: 'number', width: '55px', formatter: fmtInt },
  [S.COL_QUINTALS]:    { label: S.COL_QUINTALS, type: 'number', width: '55px', formatter: fmtDecimal1 },
  [S.COL_VOLUME_M3]:   { label: S.COL_VOLUME_M3, type: 'number', width: '70px', formatter: fmtDecimal2 },
  [S.COL_NOTE]:        { label: S.COL_NOTE, width: '110px' },
  [S.COL_EXTRA_NOTE]:  { label: S.COL_EXTRA_NOTE, width: '90px' },
  [S.COL_WORKSITE]:    { label: S.COL_WORKSITE, hidden: true },
  [VERSION]:           { label: VERSION, hidden: true },
};

/**
 * Build columnDefs from prelievi digest columns.
 * Known columns get labels/formatters from STATIC_COLS.
 * Columns ending in " %" are hidden (used for form pre-population only).
 * Dynamic species/tractor quintal columns get a blank-zero 1-decimal format.
 */
export function buildPrelieviColumnDefs(columns) {
  const defs = {};
  for (const name of columns) {
    if (name === ROW_ID) continue;
    if (name.endsWith(' %')) {
      defs[name] = { label: name, hidden: true };
      continue;
    }
    if (STATIC_COLS[name]) {
      defs[name] = STATIC_COLS[name];
      continue;
    }
    // Dynamic quintal column: species names are single words, tractor labels
    // contain a space (manufacturer + model).
    const isTractor = name.includes(' ');
    defs[name] = {
      label: name, type: 'number',
      width: isTractor ? '100px' : '90px',
      className: 'col-wrap-header',
      formatter: fmtDecimal1BlankZero,
    };
  }
  return defs;
}
