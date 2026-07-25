/** Shared import-warning helpers for CSV and Ipso import flows. */

import {
  FIELD_WARNINGS, FIELD_WARNINGS_CONFIRMED, STATUS, STATUS_WARNING,
} from './constants.js';
import * as S from './strings.js';
import { showConfirmModal } from './ui-widgets.js';

export function importWarningLines(data) {
  const warnings = data?.[FIELD_WARNINGS];
  if (!Array.isArray(warnings)) return [];
  return warnings.map(warning => String(warning || '').trim()).filter(Boolean);
}

export function isImportWarningResponse(data) {
  return data?.[STATUS] === STATUS_WARNING && importWarningLines(data).length > 0;
}

export function withImportWarningsConfirmed(body) {
  return { ...body, [FIELD_WARNINGS_CONFIRMED]: true };
}

export function showImportWarningModal(warnings, onConfirm) {
  showConfirmModal(warnings.join('\n'), onConfirm, {
    confirmLabel: S.IMPORT_WARNINGS_CONFIRM,
    intent: 'confirm',
  });
}
