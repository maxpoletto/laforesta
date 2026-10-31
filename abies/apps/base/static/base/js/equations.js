import { FIELD_REGION_ID, FIELD_SPECIES_ID } from './constants.js';
import * as S from './strings.js';
import { fmtDecimal2 } from './format.js';

/** Look up a height equation by stable IDs, independently of display labels. */
export function heightEquationFor(digest, regionId, speciesId) {
  if (!digest?.columns || !digest?.rows) return null;
  const c = digest.columns;
  const row = digest.rows.find(r => r[c.indexOf(FIELD_REGION_ID)] === Number(regionId)
    && r[c.indexOf(FIELD_SPECIES_ID)] === Number(speciesId));
  return row ? { a: row[c.indexOf(S.COL_A)], b: row[c.indexOf(S.COL_B)],
    func: row[c.indexOf(S.COL_FUNCTION)] } : null;
}

export function wireHeightAutofill({ diameter, height, species, measured, notice }, digest, regionId) {
  let edited = measured?.value === '1';
  let automatic = false;
  function update() {
    if (edited) return;
    const d = Number(diameter.value);
    const equation = heightEquationFor(digest, regionId, species.value);
    const value = equation?.func === 'ln' && d > 0 ? equation.a * Math.log(d) + equation.b : null;
    automatic = true;
    height.value = value != null && Number.isFinite(value) && value > 0 ? fmtDecimal2(value) : '';
    height.dispatchEvent(new Event('input'));
    automatic = false;
    if (measured) measured.value = '0';
    if (notice) {
      notice.hidden = !!height.value || !(d > 0);
      notice.textContent = S.HEIGHT_UNAVAILABLE;
    }
  }
  height.addEventListener('input', () => {
    if (automatic) return;
    edited = String(height.value || '').trim() !== '';
    if (measured) measured.value = edited ? '1' : '0';
    if (notice && edited) notice.hidden = true;
  });
  diameter.addEventListener('input', update);
  species.addEventListener('change', () => { edited = false; update(); });
  update();
}
