import { fileToBase64 } from '../../base/js/api.js';
import { showFormError } from '../../base/js/forms.js';
import { wireTabbedModal } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';

function wireFileImport(form, current, save) {
  const {file, name, epsg} = form.elements;
  const crs = form.querySelector('[data-role=dxf-crs]');
  const submit = form.querySelector('[type=submit]');
  let suggested = '';
  file.onchange = () => {
    const filename = file.files[0]?.name ?? '';
    const canSuggest = !name.value || name.value === suggested;
    suggested = filename.replace(/\.[^.]+$/, '').slice(0, name.maxLength);
    if (canSuggest) name.value = suggested;
    crs.hidden = !/\.dxf$/i.test(filename);
    epsg.disabled = crs.hidden;
    epsg.required = !crs.hidden;
  };
  form.onsubmit = async event => {
    event.preventDefault();
    const selected = file.files[0];
    if (!selected || submit.disabled) return;
    const extension = selected.name.split('.').at(-1).toLowerCase();
    const format = extension === 'json' ? 'geojson' : extension;
    if (!['geojson', 'gpx', 'dxf'].includes(format)) {
      showFormError(form, S.CARTOGRAPHY_IMPORT_FILE_TYPE);
      return;
    }
    const data = {name: name.value, format, epsg: epsg.disabled ? '' : epsg.value};
    submit.disabled = true;
    try {
      data.file = await fileToBase64(selected);
      if (current()) await save('import-file', data, form);
    } catch {
      if (current()) showFormError(form, S.ERROR_GENERIC);
    } finally {
      submit.disabled = false;
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

export function wireDrawingImportTabs(modal, {current, save, loadSurveys}) {
  wireFileImport(modal.querySelector('form[data-path=file]'), current, save);
  const showSurveys = wireSurveyImport(modal.querySelector('form[data-path=survey]'), current, save, loadSurveys);
  wireTabbedModal(modal, {initialTab: 'ipso', onSwitch: tab => {
    for (const button of modal.querySelectorAll('.modal-tab')) {
      button.setAttribute('aria-pressed', String(button.dataset.path === tab));
    }
    if (tab === 'survey') showSurveys();
  }});
}
