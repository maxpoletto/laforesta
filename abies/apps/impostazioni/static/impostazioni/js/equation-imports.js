/** CSV import previews retain the file and explicit mappings until confirmation. */
import { fileToBase64, postJSON } from '../../base/js/api.js';
import * as cache from '../../base/js/cache.js';
import { downloadFromURL } from '../../base/js/csv-export.js';
import { interceptSubmit, renderModalForm, showFormError } from '../../base/js/forms.js';
import { dismiss, onDismiss, showError } from '../../base/js/modals.js';
import { TableWrapper } from '../../base/js/table.js';
import { wireActions, wireCancelButtons } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';

const API = '/api/impostazioni/';

export function importMappings(form, previous = {}) {
  const mappings = structuredClone(previous);
  for (const row of form.querySelectorAll('[data-import-row]')) {
    for (const select of row.querySelectorAll('[data-mapping]')) {
      const field = select.dataset.mapping;
      mappings[field] ??= {};
      if (select.value) mappings[field][row.dataset.importRow] = Number(select.value);
      else delete mappings[field][row.dataset.importRow];
    }
  }
  return mappings;
}

export function wireEquationImports(root, onSaved) {
  const lifetime = { active: true, generation: 0, table: null };
  const fileInput = root.querySelector('[data-role="equation-import-file"]');
  let kind = 'height';
  wireActions(root.querySelector('[data-role="equation-toolbar"]'), {
    'import-height': () => { kind = 'height'; fileInput.click(); },
    'import-volume': () => { kind = 'volume'; fileInput.click(); },
    'export-height': () => downloadFromURL(`${API}hypso-params/export/`),
    'export-volume': () => downloadFromURL(`${API}equations/volume/export/`),
  });
  fileInput.addEventListener('change', async () => {
    const selection = ++lifetime.generation;
    const selectedKind = kind;
    const file = fileInput.files[0];
    fileInput.value = '';
    if (!file) return;
    let content;
    try { content = await fileToBase64(file); }
    catch { if (lifetime.active && selection === lifetime.generation) showError(S.ERROR_GENERIC); return; }
    if (!lifetime.active || selection !== lifetime.generation) return;
    await previewFile(selectedKind, content, {}, onSaved, lifetime);
  });
  return () => {
    lifetime.active = false;
    lifetime.generation++;
    lifetime.table?.destroy();
  };
}

async function previewFile(kind, file, mappings, onSaved, lifetime, message = '') {
  const generation = ++lifetime.generation;
  let response;
  try { response = await postJSON(`${API}equations/import/preview/`, { kind, file, ...mappings }); }
  catch {
    if (lifetime.active && generation === lifetime.generation) showError(S.ERROR_NETWORK);
    return;
  }
  if (!lifetime.active || generation !== lifetime.generation) return;
  if (response.status !== 200) { showError(response.data.message || S.ERROR_GENERIC); return; }
  lifetime.table?.destroy();
  const form = renderModalForm(response.data.html);
  if (!form) return;
  const table = new TableWrapper({ container: form.querySelector('[data-target="equation-preview-table"]'),
    digest: response.data, canModify: false, inlineToolbar: false,
    labels: S.TABLE_LABELS, csvFormat: S.TABLE_CSV_FORMAT });
  lifetime.table = table;
  // Modal replacement is not a dismissal; explicitly destroy the old table on
  // each replacement, while keeping cancellation/escape cleanup as well.
  const close = () => {
    table.destroy();
    if (lifetime.table === table) lifetime.table = null;
  };
  onDismiss(() => { close(); lifetime.generation++; });
  wireCancelButtons(form, dismiss);
  form.querySelector('[data-action="recheck"]')?.addEventListener('click', () => {
    const updated = importMappings(form, mappings);
    close();
    previewFile(kind, file, updated, onSaved, lifetime);
  });
  if (message) showFormError(form, message);
  interceptSubmit(form, `${API}equations/import/confirm/`, {
    onSuccess: async data => {
      cache.applyResponseChanges(data);
      if (!lifetime.active) return;
      dismiss();
      await onSaved();
    },
    onConflict: async data => {
      if (!lifetime.active) return;
      close();
      await previewFile(kind, file, mappings, onSaved, lifetime, data.message || S.ERROR_CONFLICT);
    },
  });
}
