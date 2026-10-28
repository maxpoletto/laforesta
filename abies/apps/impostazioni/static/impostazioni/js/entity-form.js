/** Modal editing for the Settings entity tables, which use local state. */
import { interceptSubmit, renderModalForm, showFormError } from '../../base/js/forms.js';
import { dismiss } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/ui-widgets.js';
import { parseDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import {
  DATA_ID, FIELD_PRESSLER_DEFAULT, HTML, LOGIN_METHOD_PASSWORD,
  MESSAGE, PATCHES, RECORD,
} from '../../base/js/constants.js';

function updateRecord(state, record) {
  if (!state.digest || !record) return;
  const rows = state.digest.rows;
  const index = rows.findIndex(row => row[0] === record[0]);
  if (index >= 0) rows[index] = record;
  else rows.push(record);
  state.table?.setData(state.digest);
}

export function wireSettingsForm(form, cfg, state) {
  wirePasswordToggle(form);
  wireCancelButtons(form, dismiss);
  interceptSubmit(form, cfg.saveUrl, {
    validate(body) {
      if (body.density !== undefined && !(parseDecimal(body.density) > 0)) {
        return S.ERR_DENSITY_POSITIVE;
      }
      if (body[FIELD_PRESSLER_DEFAULT] !== undefined && !(parseDecimal(body[FIELD_PRESSLER_DEFAULT]) > 0)) {
        return S.ERR_PRESSLER_POSITIVE;
      }
      return null;
    },
    onSuccess(data) {
      dismiss();
      const patch = data[PATCHES]?.find(p => p[DATA_ID] === cfg.key);
      updateRecord(state, patch?.[RECORD]);
    },
    onConflict(data) {
      updateRecord(state, data[RECORD]);
      if (!data[HTML]) return;
      const currentForm = renderModalForm(data[HTML]);
      if (!currentForm) return;
      wireSettingsForm(currentForm, cfg, state);
      showFormError(currentForm, data[MESSAGE] || S.ERROR_CONFLICT);
    },
  });
}

/** OAuth uses email as username; only password users need password controls. */
function wirePasswordToggle(form) {
  const radios = form.querySelectorAll('input[name="login_method"]');
  if (!radios.length) return;
  const pwFields = form.querySelectorAll('.password-login-only');
  function toggle() {
    const method = form.querySelector('input[name="login_method"]:checked')?.value;
    for (const el of pwFields) el.style.display = method === LOGIN_METHOD_PASSWORD ? '' : 'none';
  }
  for (const radio of radios) radio.addEventListener('change', toggle);
  toggle();
}
