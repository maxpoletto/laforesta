import { fetchJSON, postJSON } from '../../base/js/api.js';
import { applyResponseChanges } from '../../base/js/cache.js';
import { showFormError } from '../../base/js/forms.js';
import * as modals from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { showConfirmModal } from '../../base/js/ui-widgets.js';

export async function showRegionForm(api, id, isCurrent, onChange) {
  const base = api + 'regions/';
  let region;
  try {
    if (id) region = (await fetchJSON(base + id + '/')).data;
  } catch {
    if (isCurrent()) modals.showError(S.ERROR_NETWORK);
    return;
  }
  if (!isCurrent()) return;
  const card = cloneTemplate('tmpl-cartography-region');
  card.querySelector('h2').textContent = region ? S.CARTOGRAPHY_EDIT_REGION : S.CARTOGRAPHY_NEW_REGION;
  const form = card.querySelector('form');
  const remove = form.querySelector('[data-action=delete]');
  const submit = form.querySelector('[type=submit]');
  form.elements.name.value = region?.name || '';
  form.elements.description.value = region?.description || '';
  form.querySelector('[data-action=cancel]').onclick = modals.dismiss;
  if (region?.deletion) {
    remove.onclick = () => showConfirmModal(S.CARTOGRAPHY_DELETE_REGION(region.name, region.deletion),
      () => write('delete', {version: region.version, review: region.deletion.review}));
  } else {
    remove.remove();
  }
  form.onsubmit = event => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form));
    if (region) values.version = region.version;
    write('save', values);
  };

  async function write(action, body) {
    if (submit.disabled || !isCurrent()) return;
    submit.disabled = remove.disabled = true;
    try {
      const path = id ? base + id + '/' + action + '/' : base;
      const {data, status} = await postJSON(path, {...body, nonce: crypto.randomUUID()});
      if (status < 400) {
        applyResponseChanges(data);
        if (isCurrent()) {
          if (form.isConnected) modals.dismiss();
          await onChange(data.region_id);
        }
      } else if (isCurrent()) {
        if (form.isConnected && data.status !== 'conflict') showFormError(form, data.message || S.ERROR_GENERIC);
        else if (form.isConnected && data.status === 'conflict') {
          await showRegionForm(api, id, isCurrent, onChange);
          const current = document.getElementById('cartography-region-name')?.form;
          if (current) showFormError(current, data.message);
        } else modals.showError(data.message || S.ERROR_GENERIC);
      }
    } catch {
      if (isCurrent()) {
        if (form.isConnected) showFormError(form, S.ERROR_NETWORK);
        else modals.showError(S.ERROR_NETWORK);
      }
    } finally {
      submit.disabled = remove.disabled = false;
    }
  }
  modals.show(card);
  form.elements.name.focus();
}
