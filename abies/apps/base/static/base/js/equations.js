import { FIELD_REGION_ID, FIELD_SPECIES_ID } from './constants.js';
import * as S from './strings.js';

/** Look up a height equation by stable IDs, independently of display labels. */
export function heightEquationFor(digest, regionId, speciesId) {
  if (!digest?.columns || !digest?.rows) return null;
  const c = digest.columns;
  const row = digest.rows.find(r => r[c.indexOf(FIELD_REGION_ID)] === Number(regionId)
    && r[c.indexOf(FIELD_SPECIES_ID)] === Number(speciesId));
  return row ? { a: row[c.indexOf(S.COL_A)], b: row[c.indexOf(S.COL_B)],
    func: row[c.indexOf(S.COL_FUNCTION)] } : null;
}
