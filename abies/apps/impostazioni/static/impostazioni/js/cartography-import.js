import { showFormError } from '../../base/js/forms.js';
import { wireTabbedModal } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';

function wireFileImport(form, current, saveFiles, dismiss) {
  const {file, name, epsg} = form.elements;
  const nameRow = form.querySelector('[data-role=file-name]');
  const crs = form.querySelector('[data-role=dxf-crs]');
  const submit = form.querySelector('[type=submit]');
  const status = form.querySelector('[role=status]');
  let entries = [];
  function sync(busy = false) {
    nameRow.hidden = entries.length !== 1;
    crs.hidden = !entries.some(entry => !entry.drawingId && entry.format === 'dxf');
    file.disabled = busy;
    name.required = !nameRow.hidden;
    name.disabled = busy || nameRow.hidden;
    epsg.required = !crs.hidden;
    epsg.disabled = busy || crs.hidden;
    submit.disabled = busy || !entries.length;
  }
  file.onchange = () => {
    entries = [...file.files].map(blob => {
      const extension = blob.name.split('.').at(-1).toLowerCase();
      return {blob, format: extension === 'json' ? 'geojson' : extension, nonce: crypto.randomUUID()};
    });
    name.value = entries.length === 1 ? entries[0].blob.name.slice(0, name.maxLength) : '';
    status.textContent = '';
    form.querySelector('.form-error')?.remove();
    sync();
  };
  form.onsubmit = async event => {
    event.preventDefault();
    if (submit.disabled || !current()) return;
    for (const entry of entries) {
      entry.name = entries.length === 1 ? name.value : entry.blob.name;
      entry.epsg = entry.format === 'dxf' ? epsg.value : '';
    }
    const report = () => {
      status.textContent = S.CARTOGRAPHY_IMPORT_FILE_PROGRESS(entries.filter(entry => entry.drawingId).length, entries.length);
    };
    sync(true);
    form.querySelector('.form-error')?.remove();
    report();
    try {
      const errors = await saveFiles(entries, current, report);
      if (!current()) return;
      if (errors.length) showFormError(form, errors.join(' '));
      else if (entries.every(entry => entry.drawingId)) dismiss();
    } catch {
      if (current()) showFormError(form, S.ERROR_NETWORK);
    } finally {
      sync();
    }
  };
}

function wireSurveyImport(form, current, save, load) {
  const select = form.elements.survey_id;
  const status = form.querySelector('[role=status]');
  const submit = form.querySelector('[type=submit]');
  let loading = false, loaded = false;
  form.onsubmit = async event => {
    event.preventDefault();
    if (submit.disabled || !select.value || !current()) return;
    submit.disabled = true;
    try {
      await save('import-survey', {survey_id: Number(select.value)}, form);
    } finally {
      submit.disabled = false;
    }
  };
  return async () => {
    if (loading || loaded) return;
    loading = true;
    status.textContent = S.LOADING;
    try {
      const rows = await load();
      if (!current()) return;
      select.replaceChildren(...rows.map(row => new Option(S.CARTOGRAPHY_IMPORT_SURVEY_OPTION(row), row.id)));
      submit.disabled = !rows.length;
      status.textContent = rows.length ? '' : S.CARTOGRAPHY_IMPORT_NO_SURVEYS;
      loaded = true;
    } catch {
      if (current()) status.textContent = S.ERROR_NETWORK;
    } finally {
      loading = false;
    }
  };
}

export function wireDrawingImportTabs(modal, {current, save, saveFiles, dismiss, loadSurveys}) {
  wireFileImport(modal.querySelector('form[data-path=file]'), current, saveFiles, dismiss);
  const showSurveys = wireSurveyImport(modal.querySelector('form[data-path=survey]'), current, save, loadSurveys);
  wireTabbedModal(modal, {initialTab: 'ipso', onSwitch: tab => {
    for (const button of modal.querySelectorAll('.modal-tab')) {
      button.setAttribute('aria-pressed', String(button.dataset.path === tab));
    }
    if (tab === 'survey') showSurveys();
  }});
}
