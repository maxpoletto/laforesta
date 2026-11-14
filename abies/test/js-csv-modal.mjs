/** Shared DOM fixture for the production CSV modal's public controls. */
export function csvImportTemplate(createElement) {
  const make = (tag, { field, name, type, className } = {}) => {
    const node = createElement(tag);
    if (field) node.dataset.field = field;
    if (name) node.setAttribute('name', name);
    if (type) node.setAttribute('type', type);
    if (className) node.className = className;
    return node;
  };
  const root = make('div', { className: 'form-card' });
  root.appendChild(make('h2', { field: 'title' }));
  root.appendChild(make('p', { field: 'context' }));
  const form = make('form');
  form.dataset.role = 'import-form';
  form.appendChild(make('input', { name: 'file', type: 'file' }));
  const dateRow = make('div', { field: 'date-row' });
  dateRow.appendChild(make('input', { name: 'default_date', type: 'date' }));
  form.appendChild(dateRow);
  form.appendChild(make('p', { field: 'help' }));
  form.appendChild(make('p', { className: 'csv-import-status' }));
  form.appendChild(make('div', { className: 'csv-import-errors' }));
  const actions = make('div', { className: 'form-actions' });
  const cancel = make('button', { type: 'button' });
  cancel.dataset.action = 'cancel';
  actions.appendChild(cancel);
  actions.appendChild(make('button', { type: 'submit' }));
  form.appendChild(actions);
  root.appendChild(form);
  return root;
}
