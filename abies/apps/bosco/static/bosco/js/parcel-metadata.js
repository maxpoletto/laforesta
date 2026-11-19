export function wireParcelMetadataGovernance(form) {
  const select = form.querySelector('[name="eclass_id"]');
  const coppiceFields = form.querySelector('[data-target="coppice-metadata-fields"]');
  if (!select || !coppiceFields) return;
  const requiredFields = coppiceFields.querySelectorAll('input, select, textarea');
  const sync = () => {
    const selected = select.selectedOptions[0];
    const isCoppice = selected?.dataset.coppice === '1';
    coppiceFields.hidden = !isCoppice;
    requiredFields.forEach(field => { field.required = isCoppice; });
  };
  select.addEventListener('change', sync);
  sync();
}
