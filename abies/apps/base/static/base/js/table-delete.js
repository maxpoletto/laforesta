/** Shared confirmation for TableWrapper's explicit, versioned selections. */
import { deleteRowsWithVersions } from './forms.js';
import { cloneTemplate } from './templates.js';
import { show, dismiss, onDismiss } from './modals.js';
import { ROW_ID, VERSION } from './constants.js';
import * as S from './strings.js';

export function confirmDeleteSelection({
  dataId, rows, columns, url, body = {}, descriptionColumns = [],
  onSuccess, onConflict,
}) {
  if (!rows.length) return Promise.resolve(false);
  // Capture both identities and versions before opening the confirmation.
  const idColumn = columns.indexOf(ROW_ID);
  const versionColumn = columns.indexOf(VERSION);
  const selection = rows.map(row => ({
    [ROW_ID]: row[idColumn], [VERSION]: row[versionColumn] ?? 0,
  }));
  const fragment = cloneTemplate('tmpl-delete-selected-modal');
  fragment.querySelector('[data-field="message"]').textContent =
    S.DELETE_SELECTED_CONFIRM(rows.length);
  const list = fragment.querySelector('[data-target="records"]');
  for (const row of rows) {
    const item = document.createElement('li');
    const description = descriptionColumns.map(key => {
      const value = row[columns.indexOf(key)];
      if (value == null || value === '') return null;
      return `${key}: ${value}`;
    }).filter(Boolean).join(' · ');
    item.textContent = S.DELETE_SELECTED_ROW(row[idColumn], description);
    list.appendChild(item);
  }
  const confirm = fragment.querySelector('[data-action="confirm"]');
  confirm.textContent = S.TABLE_LABELS.deleteSelected.replace('{count}', rows.length);
  return new Promise(resolve => {
    let confirmed = false;
    fragment.querySelector('[data-action="cancel"]').addEventListener('click', dismiss);
    confirm.addEventListener('click', () => {
      if (confirmed) return;
      confirmed = true;
      dismiss();
      resolve(true);
    });
    onDismiss(() => { if (!confirmed) resolve(false); });
    show(fragment);
  }).then(confirmed => confirmed && deleteRowsWithVersions(dataId, selection, url, {
    body, onSuccess, onConflict,
  }));
}
