/** Real-browser geometry regression fixture. Serve the Abies directory over HTTP. */
import { TableWrapper } from '../../apps/base/static/base/js/table.js';
import * as S from '../../apps/base/static/base/js/strings.js';
import { ROW_ID, VERSION } from '../../apps/base/static/base/js/constants.js';

const query = new URLSearchParams(location.search);
const view = query.get('view') || 'marks';
const geometryOnly = query.get('check') === 'geometry';
const reader = query.get('role') === 'reader';
const empty = query.get('empty') === '1';
const styles = {
  marks: 'piano_di_taglio/css/piano-di-taglio.css',
  plan: 'piano_di_taglio/css/piano-di-taglio.css',
  samples: 'campionamenti/css/campionamenti.css',
  harvests: 'prelievi/css/prelievi.css',
};
const checks = [];
const failures = [];
const rect = element => element?.getBoundingClientRect();
const centerX = element => { const box = rect(element); return box.x + box.width / 2; };
const near = (actual, expected, label) => {
  const ok = Math.abs(actual - expected) <= 0.25;
  checks.push({ label, ok, actual, expected });
  if (!ok) failures.push(`${label}: expected ${expected.toFixed(3)}, got ${actual.toFixed(3)}`);
};
const check = (ok, label) => {
  checks.push({ label, ok });
  if (!ok) failures.push(label);
};
const paint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

async function run() {
  if (!styles[view]) throw new Error(`Unknown view: ${view}`);
  const style = document.createElement('link');
  style.rel = 'stylesheet';
  style.href = `../../apps/${view === 'samples' ? 'campionamenti' : view === 'harvests' ? 'prelievi' : 'piano_di_taglio'}/static/${styles[view]}`;
  await new Promise((resolve, reject) => {
    style.onload = resolve;
    style.onerror = () => reject(new Error(`Cannot load ${style.href}`));
    document.head.appendChild(style);
  });
  const fixture = document.getElementById('fixture');
  const scope = document.createElement('section');
  if (view === 'marks') scope.className = 'pdt-item-card';
  fixture.appendChild(scope);
  const workspaceHeader = document.createElement('div');
  workspaceHeader.className = 'fixture-workspace-header';
  const workspaceButton = document.createElement('button');
  workspaceButton.type = 'button';
  workspaceButton.className = 'btn btn-export';
  workspaceButton.textContent = S.EXPORT;
  workspaceHeader.appendChild(workspaceButton);
  scope.appendChild(workspaceHeader);
  if (view === 'harvests') {
    const filter = document.createElement('div');
    filter.className = 'prelievi-filter-bar fixture-filter';
    const search = document.createElement('input');
    search.className = 'table-search';
    search.type = 'text';
    search.placeholder = S.SEARCH_PLACEHOLDER;
    search.setAttribute('aria-label', S.FILTER_LABEL);
    filter.appendChild(search);
    scope.appendChild(filter);
  }
  const host = document.createElement('div');
  host.className = view === 'marks' ? 'subsection-host' : view === 'plan' ? 'pdt-table-host' : '';
  scope.appendChild(host);
  const digest = {
    columns: [ROW_ID, VERSION, S.COL_DATE, S.COL_PARCEL, S.COL_NUMBER, S.COL_SPECIES],
    rows: empty ? [] : Array.from({ length: 32 }, (_, index) => [
      index + 1, 1, '2026-11-13', String(index % 3 + 1), index + 1, 'Abete bianco',
    ]),
  };
  const table = new TableWrapper({
    container: host, digest, canModify: !reader,
    columnDefs: { [VERSION]: { hidden: true },
      [S.COL_DATE]: { type: 'date', width: '110px' },
      [S.COL_PARCEL]: { type: 'natural', width: '100px' },
      [S.COL_NUMBER]: { type: 'number', width: '100px' },
      [S.COL_SPECIES]: { width: '180px' } },
    inlineToolbar: true,
    rowsPerPage: view === 'samples' ? 20 : undefined,
    actions: reader ? {} : { onAdd() {}, onEdit() {}, onDelete() {},
      onDeleteSelected: async () => false },
    toolbar: { search: view !== 'harvests',
      actions: reader ? [] : [{ label: S.IMPORT_CSV, className: 'btn btn-import' }] },
    labels: { ...S.TABLE_LABELS, add: view === 'plan' ? S.NEW_INTERVENTION_LABEL
      : view === 'harvests' ? S.NEW_HARVEST_LABEL : S.NEW_TREE_LABEL },
  });
  if (view === 'harvests') table.wireSearchInput(scope.querySelector('.table-search'));
  await document.fonts.ready;
  await paint();
  const toolbar = host.querySelector('.table-toolbar-actions');
  const actionLabels = [...toolbar.children]
    .filter(button => button.classList.contains('btn')).map(button => button.textContent);
  const newLabel = view === 'plan' ? S.NEW_INTERVENTION_LABEL
    : view === 'harvests' ? S.NEW_HARVEST_LABEL : S.NEW_TREE_LABEL;
  check(JSON.stringify(actionLabels) === JSON.stringify(reader ? [S.EXPORT]
    : [S.EXPORT, S.IMPORT_CSV, `+ ${newLabel}`]),
    'Workspace actions use export/import/new order, including empty tables');
  const controlRow = host.querySelector('.sortable-table-controls');
  const controlStyle = getComputedStyle(controlRow);
  near(parseFloat(controlStyle.paddingLeft), 0, 'Controls have no left padding');
  near(parseFloat(controlStyle.paddingRight), 0, 'Controls have no right padding');
  for (const button of scope.querySelectorAll('.btn')) {
    near(parseFloat(getComputedStyle(button).fontSize), 14,
      'Workspace buttons have a uniform 14px font inside and outside the table');
  }
  if (innerWidth >= 1400) {
    const lastAction = [...toolbar.children].filter(el => el.classList.contains('btn')).at(-1);
    near(rect(lastAction).right, rect(workspaceButton).right,
      'Table actions align with the workspace action above');
  }
  const toggle = host.querySelector('.table-delete-toggle');
  if (reader) {
    check(!toggle, 'Readers have no selection toggle');
    check(!host.querySelector('.table-delete-bar'), 'Readers have no destructive toolbar');
    return;
  }
  const heading = host.querySelector('.sortable-table-head');
  const grid = host.querySelector('.sortable-table');
  const firstRow = () => host.querySelector('.sortable-table-row');
  const lastRow = () => [...host.querySelectorAll('.sortable-table-row')].at(-1);
  const normal = { headingHeight: rect(heading).height, tableTop: rect(grid).top,
    tableWidth: rect(grid).width, toggleHeight: rect(toggle).height,
    rowHeight: rect(firstRow())?.height, lastRowOffset: firstRow() ? rect(lastRow()).top - rect(firstRow()).top : null };
  if (view === 'marks' || view === 'plan') {
    const dataHeader = heading.querySelector('.sortable-table-header:not(.col-actions)');
    check(getComputedStyle(dataHeader).textAlign === 'center', 'Data-header centering is retained');
    check(getComputedStyle(dataHeader).whiteSpace === 'normal', 'Data-header wrapping is retained');
  }
  const rowTrash = firstRow()?.querySelector('.action-delete');
  if (rowTrash) near(centerX(toggle), centerX(rowTrash), 'Header trash aligns with row trash');
  toggle.click();
  await paint();
  near(rect(heading).height, normal.headingHeight, 'Header height stays fixed in selection mode');
  near(rect(toggle).height, normal.toggleHeight, 'Cancel and trash controls have equal heights');
  near(rect(grid).width, normal.tableWidth, 'Table width stays fixed in selection mode');
  if (firstRow()) {
    near(rect(firstRow()).height, normal.rowHeight, 'Data row height stays fixed in selection mode');
    near(rect(lastRow()).top - rect(firstRow()).top, normal.lastRowOffset,
      'Row-height changes do not accumulate down the table');
    near(centerX(toggle), centerX(firstRow().querySelector('.table-delete-choice')),
      'Header cancel aligns with row checkbox');
    firstRow().click();
    await paint();
    near(rect(firstRow()).height, normal.rowHeight, 'Selecting a row keeps its height');
  }
  // At phone widths controls may wrap; wide layouts must not acquire another row.
  if (!geometryOnly && innerWidth >= 1400) {
    near(rect(grid).top, normal.tableTop, 'Entering selection mode keeps the table top fixed');
    const bar = host.querySelector('.table-delete-bar');
    const controlRow = host.querySelector('.sortable-table-controls');
    const pagination = controlRow.querySelector('.sortable-table-pagination');
    const deleteButton = bar.querySelector('.table-delete-selected');
    const deleteCenter = rect(deleteButton).y + rect(deleteButton).height / 2;
    for (const button of controlRow.querySelectorAll('.table-toolbar-actions > .btn')) {
      near(rect(button).y + rect(button).height / 2, deleteCenter, 'Toolbar action buttons align vertically');
    }
    const search = controlRow.querySelector('.table-search');
    if (search) check(rect(search).right <= rect(pagination).left, 'Search does not overlap pagination');
    check(rect(pagination).right <= rect(bar).left, 'Pagination does not overlap selection actions');
    check(rect(controlRow).left <= rect(pagination).left
      && rect(bar).right <= rect(controlRow).right, 'Controls fit the available width');
    check(Boolean(bar.closest('.sortable-table-controls')),
      'Selection controls use the existing pagination/action row');
  }
  toggle.click();
  await paint();
  near(rect(heading).height, normal.headingHeight, 'Cancel restores the original header height');
  near(rect(grid).top, normal.tableTop, 'Cancel restores the original table position');
}

try { await run(); }
catch (error) { failures.push(error.stack || String(error)); }
window.layoutFixtureResult = { view, width: innerWidth, reader, empty,
  checks, failures, passed: failures.length === 0 };
const output = document.getElementById('results');
output.className = failures.length ? 'fail' : 'pass';
output.textContent = `${view}: ${checks.length - checks.filter(item => !item.ok).length}/${checks.length} checks passed\n`
  + (failures.length ? failures.join('\n') : 'PASS');
