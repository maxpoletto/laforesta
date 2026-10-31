/** Keep equation form collection separate from shared Settings modal behavior. */
import { fmtDecimal } from '../../base/js/format.js';
export function collectSpeciesEquations(form) {
  const value = name => form.querySelector(`[name="${name}"]`)?.value || '';
  return {
    volume_equation: {
      mode: value('volume_mode'), proxy_species_id: value('volume_proxy_species_id'),
      b0: value('volume_b0'), b1: value('volume_b1'), b2: value('volume_b2'),
      source: value('volume_source'),
    },
    height_equations: [...form.querySelectorAll('[data-region-id]')].map(row => {
      const field = key => row.querySelector(`[data-field="${key}"]`);
      return { region_id: Number(row.dataset.regionId), enabled: field('enabled').checked,
        a: field('a').value, b: field('b').value, n: field('n').value, r2: field('r2').value };
    }),
  };
}

export function wireSpeciesEquations(form) {
  if (!form.querySelector('input[name="equation_state"]')) return false;
  const mode = form.querySelector('[name="volume_mode"]');
  const reference = form.querySelector('[name="volume_proxy_species_id"]');
  const referenceValues = form.querySelector('[data-role="volume-reference-coefficients"]');
  function showReference() {
    let values = null;
    try { values = JSON.parse(reference.selectedOptions[0]?.dataset.coefficients || 'null'); } catch { /* unavailable */ }
    referenceValues.textContent = values?.map((value, index) => `b${index} = ${fmtDecimal(value, 10)}`).join(' · ') || '—';
  }
  reference.addEventListener('change', showReference);
  showReference();
  function toggleVolume() {
    for (const block of form.querySelectorAll('[data-volume-mode]')) {
      block.hidden = block.dataset.volumeMode !== mode.value;
    }
  }
  mode.addEventListener('change', toggleVolume);
  toggleVolume();
  for (const row of form.querySelectorAll('[data-region-id]')) {
    const enabled = row.querySelector('[data-field="enabled"]');
    const fields = row.querySelector('[data-role="height-coefficients"]');
    function toggleHeight() {
      fields.hidden = !enabled.checked;
      for (const input of fields.querySelectorAll('input')) input.disabled = !enabled.checked;
    }
    enabled.addEventListener('change', toggleHeight);
    toggleHeight();
  }
  return true;
}
