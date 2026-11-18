/** Modal editing for the Settings entity tables, which use local state. */
import { interceptSubmit, renderModalForm, showFormError } from '../../base/js/forms.js';
import { dismiss, showError } from '../../base/js/modals.js';
import { postJSON } from '../../base/js/api.js';
import { showConfirmModal, wireCancelButtons } from '../../base/js/ui-widgets.js';
import { parseDecimal } from '../../base/js/format.js';
import * as cache from '../../base/js/cache.js';
import { collectSpeciesEquations, wireSpeciesEquations } from './species-equations.js';
import * as S from '../../base/js/strings.js';
import {
  DATA_ID, FIELD_NONCE, FIELD_PRESSLER_DEFAULT, HTML, LOGIN_METHOD_PASSWORD,
  MESSAGE, PATCHES, RECORD, ROW_ID, STATUS, STATUS_NOT_FOUND, VERSION,
} from '../../base/js/constants.js';

function updateRecord(state, record) {
  if (!state.digest || !record?.length) return;
  const rows = state.digest.rows;
  const index = rows.findIndex(row => row[0] === record[0]);
  if (index >= 0) rows[index] = record;
  else rows.push(record);
  state.table?.setData(state.digest);
}

export function confirmReferenceDelete(cfg, state, row) {
  const { columns } = state.digest;
  const id = row[columns.indexOf(ROW_ID)];
  const version = row[columns.indexOf(VERSION)];
  showConfirmModal(S.DELETE_REFERENCE_CONFIRM(row[columns.indexOf(S.COL_NAME)]), async () => {
    let response;
    try {
      response = await postJSON(cfg.deleteUrl, {
        [ROW_ID]: id, [VERSION]: version, [FIELD_NONCE]: crypto.randomUUID(),
      });
    } catch {
      showError(S.ERROR_NETWORK);
      return;
    }
    const { data, status } = response;
    cache.applyResponseChanges(data);
    if (status === 200 || data[STATUS] === STATUS_NOT_FOUND) {
      state.digest.rows = state.digest.rows.filter(r => r[0] !== id);
      state.table?.setData(state.digest);
      if (status !== 200) showError(S.REFERENCE_ALREADY_DELETED);
    } else {
      updateRecord(state, data[RECORD]);
      showError(data[MESSAGE] || S.ERROR_GENERIC);
    }
  });
}

export function wireSettingsForm(form, cfg, state) {
  wirePasswordToggle(form);
  const speciesForm = wireSpeciesEquations(form);
  wireCancelButtons(form, dismiss);
  interceptSubmit(form, cfg.saveUrl, {
    validate(body) {
      if (speciesForm) Object.assign(body, collectSpeciesEquations(form));
      if (body.density !== undefined && !(parseDecimal(body.density) > 0)) {
        return S.ERR_DENSITY_POSITIVE;
      }
      if (body[FIELD_PRESSLER_DEFAULT] !== undefined && !(parseDecimal(body[FIELD_PRESSLER_DEFAULT]) > 0)) {
        return S.ERR_PRESSLER_POSITIVE;
      }
      return null;
    },
    onSuccess(data) {
      cache.applyResponseChanges(data);
      dismiss();
      const patch = data[PATCHES]?.find(p => p[DATA_ID] === cfg.key);
      updateRecord(state, patch?.[RECORD]);
    },
    onConflict(data) {
      cache.applyResponseChanges(data);
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
