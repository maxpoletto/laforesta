/** Botanical classification in one sortable Tipo column. */

import * as S from '../../base/js/strings.js';
import { ROW_ID, SPECIES_TYPE_CONIFER, SPECIES_TYPE_BROADLEAF } from '../../base/js/constants.js';

export const PROTECTED_SPECIES_NAMES = new Set([
  S.SPECIES_OTHER, S.SPECIES_OTHER_CONIFER, S.SPECIES_OTHER_BROADLEAF,
]);

export function speciesTypeColumns(columns, onChange, pendingIds = new Set()) {
  return { [S.COL_TYPE]: {
    label: S.COL_TYPE, type: 'string', width: '140px', cellClassName: 'species-type-cell',
    formatter: value => value || S.SPECIES_TYPE_UNCLASSIFIED,
    renderCell(cell, value, row) {
      const id = row[columns.indexOf(ROW_ID)];
      const name = row[columns.indexOf(S.COL_NAME)];
      if (PROTECTED_SPECIES_NAMES.has(name)) {
        cell.replaceChildren();
        cell.textContent = value || S.SPECIES_TYPE_UNCLASSIFIED;
        return;
      }
      const select = document.createElement('select');
      select.name = `species-type-${id}`;
      select.disabled = pendingIds.has(id);
      select.setAttribute('aria-label', `${name}: ${S.COL_TYPE}`);
      if (!value) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = S.SPECIES_TYPE_UNCLASSIFIED;
        placeholder.disabled = true;
        placeholder.selected = true;
        select.appendChild(placeholder);
      }
      for (const [kind, label] of [
        [SPECIES_TYPE_CONIFER, S.COL_CONIFER],
        [SPECIES_TYPE_BROADLEAF, S.COL_BROADLEAF],
      ]) {
        const option = document.createElement('option');
        option.value = kind;
        option.textContent = label;
        option.selected = value === label;
        select.appendChild(option);
      }
      select.addEventListener('change', () => onChange(row, select.value));
      cell.replaceChildren(select);
    },
  } };
}
