/**
 * Shared tree-form wiring.
 *
 * Used by both Campionamenti (albero campione) and Piano di taglio
 * (albero martellato).  The form HTML comes from a Django template;
 * this module wires up the live V/m preview that both forms share.
 *
 * Element IDs are prefixed `tf-` and are shared between templates and
 * this JS so grep finds all references.
 */

import { tabacchiVolumeM3, massQ } from './volume.js';
import { parseDecimal, fmtDecimal } from './format.js';
import { PRESSLER_DEFAULT } from './constants.js';

// Canonical element IDs — templates and JS both reference these.
export const ID_D_CM      = 'tf-d-cm';
export const ID_H_M       = 'tf-h-m';
export const ID_SPECIES   = 'tf-species';
export const ID_COPPICE     = 'tf-ceduo';
export const ID_PREVIEW   = 'tf-vm-preview';
export const ID_PRESSLER  = 'tf-pressler-coeff';
export const ID_VOLUME    = 'tf-volume-m3';
export const ID_MASS      = 'tf-mass-q';
export const ID_LAT       = 'tf-lat';
export const ID_LON       = 'tf-lon';
export const ID_DATE      = 'tf-date';
export const ID_OPERATOR  = 'tf-operator';

/**
 * Wire the live V/m preview on a server-rendered tree form.
 *
 * Reads D, h, species from the form, computes volume + mass via
 * Tabacchi, and updates the preview line + hidden fields.
 *
 * Options:
 *   ceduoEl — checkbox element; when checked, preview is suppressed.
 */
export function wireVMPreview(form, opts = {}) {
  const d = form.querySelector(`#${ID_D_CM}`);
  const h = form.querySelector(`#${ID_H_M}`);
  const sp = form.querySelector(`#${ID_SPECIES}`);
  const preview = form.querySelector(`#${ID_PREVIEW}`);
  const pressler = form.querySelector(`#${ID_PRESSLER}`);
  const vHidden = form.querySelector(`#${ID_VOLUME}`);
  const mHidden = form.querySelector(`#${ID_MASS}`);
  if (!d || !h || !sp || !preview || !vHidden || !mHidden) return;

  const ceduoEl = opts.ceduoEl || null;
  let lastPresslerDefault = null;

  function syncPresslerDefault() {
    if (!pressler) return;
    const opt = sp.tagName === 'SELECT' ? sp.options[sp.selectedIndex] : sp;
    const nextDefault = opt?.dataset.presslerDefault || PRESSLER_DEFAULT;
    const current = String(pressler.value || '').replace(',', '.');
    if (!pressler.value || current === lastPresslerDefault) {
      pressler.value = nextDefault;
    }
    lastPresslerDefault = nextDefault;
  }

  function clearPreview() {
    preview.hidden = true;
    preview.textContent = '';
    vHidden.value = '';
    mHidden.value = '';
  }

  function update() {
    syncPresslerDefault();
    if (ceduoEl?.checked) { clearPreview(); return; }
    const dCm = parseInt(d.value, 10);
    const hM = parseDecimal(h.value);
    if (!(dCm > 0 && hM > 0)) { clearPreview(); return; }
    const opt = sp.tagName === 'SELECT' ? sp.options[sp.selectedIndex] : sp;
    const speciesName = opt?.dataset.name;
    const density = parseDecimal(opt?.dataset.density);
    const v = tabacchiVolumeM3(dCm, hM, speciesName);
    if (v == null) { clearPreview(); return; }
    // Derive mass from the rounded volume actually stored (4 dp), so a tree
    // entered here matches the CSV-import path (tabacchi.py + tree_mass_q),
    // which computes mass from the stored volume. A ≤1-ULP residual remains
    // (JS float vs Python Decimal); test/test_tabacchi.py locks the bound.
    const m = massQ(Number(v.toFixed(4)), density);
    preview.hidden = false;
    preview.textContent =
      `V = ${fmtDecimal(v, 3)} m³  ·  m = ${m == null ? '—' : fmtDecimal(m, 2)} q`;
    vHidden.value = v.toFixed(4);
    mHidden.value = m == null ? '' : m.toFixed(3);
  }

  d.addEventListener('input', update);
  h.addEventListener('input', update);
  sp.addEventListener('change', update);
  ceduoEl?.addEventListener('change', update);
  update();

  return update;
}
