import assert from 'node:assert/strict';
import { Element, installDOM } from '../../../../../test/js-dom.mjs';
import { csvImportTemplate } from '../../../../../test/js-csv-modal.mjs';
import * as forms from './forms.js';

const body = installDOM();
const modal = new Element();
modal.id = 'modal-container';
body.appendChild(modal);
const template = new Element('template');
template.id = 'tmpl-csv-import-modal';
template.content = csvImportTemplate(tag => new Element(tag));
body.appendChild(template);
assert.equal(typeof forms.showCsvImportModal, 'function', 'CSV imports have a shared standalone modal');
let attempts = 0, succeeded = 0;
let outcome = { errors: ['Riga 2: Altezza mancante.'] };
const form = forms.showCsvImportModal({ title: 'Importa alberi da CSV',
  help: 'Required columns', context: 'Survey name', showDate: true,
  onSubmit: async current => { assert.equal(current, form); attempts++; return outcome; },
  onSuccess: async () => {
    assert.equal(modal.classList.contains('open'), false, 'Dismiss before refreshing the workspace');
    succeeded++;
  },
});
assert.equal(modal.querySelector('[data-field="title"]').textContent, 'Importa alberi da CSV');
assert.equal(modal.querySelector('[data-field="context"]').textContent, 'Survey name');
assert.equal(modal.querySelector('[data-field="date-row"]').hidden, false);
assert.equal(modal.querySelector('.modal-tabs'), null);
const file = form.querySelector('[name="file"]');
file.files = [new Blob(['csv'])];
await form.dispatch('submit');
assert.equal(attempts, 1);
assert.equal(modal.classList.contains('open'), true, 'Validation leaves the modal open');
assert.match(form.querySelector('.csv-import-errors').textContent, /Altezza mancante/);
assert.equal(form.querySelector('[name="file"]'), file, 'Keep the selected file after validation');
assert.equal(form.querySelector('button[type="submit"]').disabled, false);
outcome = { ok: true };
await form.dispatch('submit');
assert.equal(succeeded, 1);
forms.showCsvImportModal({title:'Importa calendario', onSubmit:async()=>{attempts++;}});
assert.equal(modal.querySelector('[data-field="date-row"]').hidden, true);
assert.equal(modal.querySelector('[data-field="context"]').hidden, true);
await modal.querySelector('[data-action="cancel"]').dispatch();
assert.equal(attempts, 2, 'Cancel performs no import');
let accepted = 0;
const refreshFailureForm = forms.showCsvImportModal({title:'Importa CSV',
  onSubmit:async()=>{accepted++;return {ok:true};},
  onSuccess:async()=>{throw new Error('Refresh disconnected');},
});
await refreshFailureForm.dispatch('submit');
assert.equal(accepted,1);
assert.match(modal.querySelector('.modal-error').textContent,/Importazione completata/);
assert.equal(modal.querySelector('input[type="file"]'),null,'An accepted import is not offered again after refresh failure');
console.log('Shared CSV modal: context, optional dates, validation retry, success and cancellation passed');
