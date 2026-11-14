/** Export-gated review of invalid rows in a staged tree upload. */
import { postBlob } from '../../base/js/api.js';
import { downloadBlob } from '../../base/js/csv-export.js';
import { dismiss, show } from '../../base/js/modals.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { wireExportBeforeConfirm } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';

export const excludedRowsURL = id => `/api/ipso/uploads/${id}/excluded/export/`;

export function showTreeImportReview({ uploadId, review, onConfirm }) {
  const fragment = cloneTemplate('tmpl-ipso-import-review-modal');
  const field = name => fragment.querySelector(`[data-field="${name}"]`);
  const button = name => fragment.querySelector(`[data-action="${name}"]`);
  const valid = Number.isInteger(review.valid_count) ? review.valid_count : 0;
  const excluded = Number.isInteger(review.excluded_count) ? review.excluded_count : 0;
  field('summary').textContent = S.IPSO_IMPORT_REVIEW_SUMMARY(valid, excluded);
  field('message').textContent = review.message || '';
  field('message').hidden = !review.message || review.message === S.IPSO_IMPORT_EXPORT_REQUIRED;
  for (const name of ['errors', 'warnings']) {
    const lines = Array.isArray(review[name]) ? review[name] : [];
    field(name).textContent = lines.join('\n');
    fragment.querySelector(`[data-section="${name}"]`).hidden = !lines.length;
  }
  const explanation = field('export-required');
  const exportButton = button('export');
  const confirmButton = button('confirm');
  explanation.textContent = S.IPSO_IMPORT_EXPORT_REQUIRED;
  exportButton.textContent = S.IPSO_EXPORT_EXCLUDED;
  confirmButton.textContent = S.IPSO_IMPORT_VALID_ROWS(valid);
  let exportToken = null;
  if (excluded) {
    wireExportBeforeConfirm({ exportButton, confirmButton, message: explanation,
      canConfirm: valid > 0,
      onExport: async () => {
        try {
          const result = await postBlob(excludedRowsURL(uploadId), { review_token: review.review_token });
          const receipt = result.headers.get('X-Import-Export-Token');
          if (!receipt) return { error: S.ERROR_GENERIC };
          downloadBlob(result.blob, result.filename);
          exportToken = receipt;
          return { ok: true };
        } catch (error) {
          if (error.data?.status === 'import_review') {
            showTreeImportReview({ uploadId, review: error.data, onConfirm });
          }
          return { error: error.data?.message || S.ERROR_NETWORK };
        }
      },
    });
  } else {
    explanation.remove();
    exportButton.remove();
    confirmButton.disabled = valid < 1;
  }
  button('cancel').addEventListener('click', dismiss);
  confirmButton.addEventListener('click', async () => {
    if (confirmButton.disabled) return;
    dismiss();
    await onConfirm({ review_token: review.review_token, export_token: exportToken,
      warnings_confirmed: true });
  });
  show(fragment);
}
