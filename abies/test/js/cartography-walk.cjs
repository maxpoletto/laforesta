// Run via test_cartography_browser.py, which provisions an isolated site.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.ABIES_PLAYWRIGHT_MODULE);

async function walk(browser) {
  const url = process.env.TEST_URL;
  const width = Number(process.env.TEST_WIDTH);
  const context = await browser.newContext({
    viewport: {width, height: 850},
    permissions: ['geolocation'], geolocation: { latitude: 38, longitude: 16, accuracy: 2 },
    serviceWorkers: 'block',
  });
  await context.addCookies([{ name: 'sessionid', value: process.env.TEST_SESSION, url }]);
  await context.route(/^https?:\/\//, route => new URL(route.request().url()).origin === url
    ? route.continue() : route.abort());
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => {
    errors.push(e.message);
    console.error(e.stack);
  });
  page.on('response', response => {
    const path = new URL(response.url());
    if (path.origin === url && response.status() >= 400 && path.pathname.startsWith('/api/')) {
      console.error(response.status(), path.pathname);
    }
  });
  page.setDefaultTimeout(7000);
  const loc = selector => page.locator(selector);
  const action = name => loc(`#settings-cartografia [data-action="${name}"]`);
  const vertices = loc('.cartography-vertex');
  async function removeFirstVertex() {
    await vertices.first().press('Enter');
    await action('remove-vertex').click();
  }
  const editor = url + '/impostazioni?scheda=cartografia';
  await page.goto(editor);
  const modal = loc('#modal-container');
  async function checkNameField() {
    const field = modal.getByLabel('Nome', {exact: true});
    assert.equal(await field.getAttribute('type'), 'text');
    assert.ok(await field.evaluate(input => {
      const group = input.closest('.form-row > .form-group');
      return group && input.offsetWidth === group.clientWidth && parseFloat(getComputedStyle(input).paddingLeft) >= 8;
    }));
  }
  const sidebar = loc('#cartography-sidebar');
  const sidebarToggle = loc('#cartography-map [title="Mostra/nascondi pannello"]');
  assert.equal(await sidebar.locator('[data-action=save], [data-action=cancel], [data-action=finish]').count(), 0);
  assert.equal(await loc('#cartography-map [data-action=save] svg').count(), 1);
  assert.equal(await loc('#cartography-map [data-action=cancel] svg').count(), 1);
  const drawings = loc('#cartography-drawings li');
  const visibleDrawing = loc('#cartography-drawings [data-action=toggle-item]');
  const drawingSummary = loc('#cartography-drawings-label');
  assert.deepEqual(await sidebar.locator(':scope > *').evaluateAll(sections => sections.map(section =>
    [section.tagName, section.querySelector('label, summary, h2, [data-action=refresh-terrain]').textContent.trim()])),
    [['SECTION', 'Compresa'], ['SECTION', 'Particelle'], ['SECTION', 'Strade'], ['DETAILS', 'Disegni'],
      ['DIV', 'Aggiorna dati dei terreni']]);
  const credits = loc('#settings-generali footer a[href*="flaticon.com"]');
  assert.equal(await credits.getAttribute('href'), 'https://www.flaticon.com/authors/th-studio');
  assert.equal(await credits.textContent(), 'th studio');
  assert.equal(await loc('#settings-cartografia a[href*="flaticon.com"]').count(), 0);
  await drawingSummary.click();
  assert.ok(await action('open-import').isHidden());
  assert.ok(await sidebar.getByText('Nessuna particella').isVisible());
  assert.ok(await sidebar.getByText('Nessuna strada').isVisible());
  await drawingSummary.click();
  const bulkButtons = sidebar.locator('.cartography-list-header button:visible');
  assert.equal(await bulkButtons.count(), 0);
  const selected = loc('#cartography-region option:checked');
  async function saveRegion(name, description) {
    await modal.getByLabel('Nome', {exact: true}).fill(name);
    await modal.getByLabel('Descrizione').fill(description);
    await modal.getByRole('button', {name: 'Salva', exact: true}).click();
    await page.waitForFunction(label => document.querySelector('#cartography-region')?.selectedOptions[0]?.text === label, name);
  }

  assert.ok(await action('edit-region').isDisabled());
  assert.ok(await action('edit-region').evaluate(button => {
    const style = getComputedStyle(button);
    return style.opacity === '1' && style.boxShadow.includes('inset')
      && style.backgroundColor !== getComputedStyle(button.closest('aside')).backgroundColor;
  }));
  await action('new-region').click();
  await saveRegion('Temporanea', '');
  assert.ok(await action('line').isHidden());
  assert.ok(await action('polygon').isHidden());
  assert.ok(await action('join-link').isHidden());
  await action('new-drawing').waitFor();
  await action('edit-region').click();
  await modal.getByRole('button', {name: 'Elimina', exact: true}).click();
  await modal.getByRole('button', {name: 'Elimina', exact: true}).click();
  await page.waitForFunction(() => !document.querySelector('#cartography-region').options.length);
  assert.ok(await action('edit-region').isDisabled());
  await action('new-region').click();
  await saveRegion('Costa', 'Versante est');
  await action('new-region').click();
  await saveRegion('Abetaia', '');
  assert.deepEqual(await loc('#cartography-region option').allTextContents(), ['Abetaia', 'Costa']);
  await loc('#cartography-region').selectOption({label: 'Costa'});
  await page.waitForURL(/&c=\d+/);
  await action('edit-region').click();
  assert.equal(await modal.getByLabel('Descrizione').inputValue(), 'Versante est');
  await saveRegion('Costa nord', 'Versante nord');
  await page.goto(url + '/impostazioni?scheda=dati-di-base');
  await page.getByRole('link', {name: 'Cartografia', exact: true}).click();
  await action('edit-region').click();
  assert.equal(await modal.getByLabel('Nome', {exact: true}).inputValue(), 'Costa nord');
  await saveRegion('Costa', 'Versante nord');
  await loc('#cartography-region').selectOption({label: 'Abetaia'});
  await action('edit-region').click();
  await modal.getByRole('button', {name: 'Elimina', exact: true}).click();
  await modal.getByRole('button', {name: 'Annulla'}).click();
  assert.equal(await selected.textContent(), 'Abetaia');
  await action('edit-region').click();
  await modal.getByRole('button', {name: 'Elimina', exact: true}).click();
  await modal.getByRole('button', {name: 'Elimina', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('#cartography-region')?.options.length === 1);
  assert.equal(await selected.textContent(), 'Costa');
  await action('open-import').click();
  const [surveyCatalog] = await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/surveys/')),
    modal.getByRole('button', {name: 'Da rilevamento', exact: true}).click(),
  ]);
  assert.ok(surveyCatalog.ok());
  assert.deepEqual((await surveyCatalog.json()).surveys, []);
  await modal.getByText('Nessun rilevamento con posizioni in questa compresa.').waitFor();
  assert.ok(await modal.locator('form[data-path=survey] [type=submit]').isDisabled());
  await modal.getByRole('button', {name: 'Annulla', exact: true}).click();
  assert.deepEqual(await sidebar.locator('.cartography-sidebar-actions').evaluateAll(rows => rows.map(row =>
    [...row.children].map(button => button.textContent.trim()))), [
    ['Proprietà', 'Nuova'], ['Proprietà', 'Confini'], ['Proprietà', 'Punti'],
    ['Nuovo', 'Importa…'], ['Modifica', 'Promuovi'],
  ]);
  const sidebarSizes = await sidebar.locator('.cartography-sidebar-actions button').evaluateAll(buttons =>
    buttons.map(button => [button.offsetWidth, button.offsetHeight].join('×')));
  assert.equal(new Set(sidebarSizes).size, 1);
  const drawingList = loc('#cartography-drawings');
  assert.ok(await drawingList.evaluate(list => list.previousElementSibling.previousElementSibling
    .querySelector('[data-action=new-drawing]') && list.nextElementSibling.querySelector('[data-action=edit]')));
  assert.equal(await action('new-drawing').evaluate(button => getComputedStyle(button.parentElement).marginBottom), '6px');

  await page.goto(url + '/ipso/#secret=cartography-browser-test');
  await loc('#btn-mode-points').click();
  await loc('#in-operatore').fill('Ada');
  await loc('#in-compresa').selectOption({ label: 'Costa' });
  await loc('#in-point-name').fill('Traccia nord');
  await loc('#btn-start').click();
  await loc('#trace-toggle').click();
  // Wait for distinct GPS windows, just as the receiver reports fixes on a walk.
  for (const lat of [38.0001, 38.0002, 38.0003]) {
    await context.setGeolocation({ latitude: lat, longitude: 16, accuracy: 2 });
    await page.waitForTimeout(1150);
  }
  await loc('#point-add').click();
  await loc('#point-note').fill('Cancello');
  await loc('#point-save').click();
  async function record(latitudes, lon = 16) {
    for (const lat of latitudes) {
      await context.setGeolocation({latitude: lat, longitude: lon, accuracy: 2});
      await page.waitForTimeout(1150);
    }
  }
  await record([38.0005, 38.0006, 38.0007]);
  await loc('#trace-toggle').click();
  await loc('#trace-toggle').click();
  await record([38.0008, 38.0009], 16.0002);
  await loc('#btn-end').click();
  await loc('#end-confirm').click();
  await loc('#screen-done').waitFor();

  if (width >= 800) {
    await page.goto(url + '/importazione');
    assert.ok(await loc('[data-ipso-pending-dot]:not([hidden])').count());
    await loc('.ipso-inbox-page [title="Apri"]').click();
    await page.getByRole('button', {name: 'Importa in Cartografia'}).click();
  } else {
    await page.goto(editor);
    await action('open-import').click();
    await modal.getByRole('button', {name: 'Importa', exact: true}).click();
  }
  await drawings.first().waitFor({state: 'attached'});
  assert.equal(await loc('[data-ipso-pending-dot]:not([hidden])').count(), 0);
  await action('edit-region').click();
  await modal.getByLabel('Nome', {exact: true}).waitFor();
  assert.equal(await modal.getByRole('button', {name: 'Elimina', exact: true}).count(), 0);
  await modal.getByRole('button', {name: 'Annulla'}).click();
  assert.equal(await drawings.count(), 1);
  assert.ok(await action('promote').isDisabled());
  assert.equal(await action('promote').getAttribute('title'), 'Il disegno deve contenere una sola linea oppure solo poligoni.');
  const layout = await drawings.first().evaluate(row => {
    const list = row.parentElement;
    const header = list.closest('details').querySelector('[data-action=toggle-all]');
    const eye = row.querySelector('[data-action=toggle-item]');
    const alignment = () => header.getBoundingClientRect().left - eye.getBoundingClientRect().left;
    const shortAlignment = alignment();
    const before = list.clientHeight;
    const rowHeight = row.offsetHeight;
    for (let i = 0; i < 6; i++) list.append(row.cloneNode(true));
    const capped = list.clientHeight;
    const scrolls = list.scrollHeight > capped;
    const scrollbar = list.offsetWidth - list.clientWidth - 2;
    const longAlignment = alignment();
    while (list.lastElementChild !== row) list.lastElementChild.remove();
    return {before, rowHeight, capped, scrolls, scrollbar, shortAlignment, longAlignment};
  });
  assert.equal(layout.rowHeight, 40);
  assert.equal(layout.before, 40);
  assert.equal(layout.capped, 240);
  assert.ok(layout.scrolls);
  assert.ok(layout.scrollbar > 0);
  assert.equal(layout.shortAlignment, 0);
  assert.equal(layout.longAlignment, 0);
  const eyeImage = () => visibleDrawing.evaluate(eye => getComputedStyle(eye, '::before').backgroundImage);
  assert.match(await eyeImage(), /show\.webp/);
  await visibleDrawing.click();
  assert.match(await eyeImage(), /hide\.webp/);
  assert.equal(await visibleDrawing.getAttribute('aria-pressed'), 'false');
  await sidebar.locator('[data-kind=drawing] [data-action=toggle-all]').click();
  assert.equal(await visibleDrawing.getAttribute('aria-pressed'), 'true');
  await drawings.first().locator('[data-action=select-item]').click();
  await loc('#cartography-roads-label').click();
  assert.equal(await sidebar.locator('[data-action=select-item][aria-pressed=true]').count(), 0);
  assert.ok(await action('edit').isDisabled());
  await drawings.first().locator('[data-action=select-item]').click();
  await action('edit').click();
  const editParts = loc('.cartography-draft-part');
  assert.equal(await loc('#cartography-parts').count(), 0);
  assert.equal(await editParts.count(), 4);
  if (width < 800) await sidebarToggle.click();
  await editParts.nth(3).click();
  await removeFirstVertex();
  await removeFirstVertex();
  await action('join-link').click();
  await loc('.cartography-endpoint[title="Punto 2"]').press('Enter');
  await loc('.cartography-endpoint[title="Linea 1: fine"]').press('Enter');
  assert.equal(await editParts.count(), 2);
  await action('undo').click();
  await editParts.nth(1).click();
  await removeFirstVertex();
  await editParts.first().click();
  assert.ok(await action('promote').isDisabled());
  const joinWalk = async () => {
    await action('join-link').click();
    await loc('.cartography-endpoint[title="Linea 2: inizio"]').click();
    await loc('.cartography-endpoint[title="Linea 1: fine"]').click();
  };
  await joinWalk();
  await action('undo').click();
  if (width < 800) await sidebarToggle.click();
  assert.ok(await action('promote').isDisabled());
  if (width < 800) await sidebarToggle.click();
  await joinWalk();
  assert.equal(await action('promote').getAttribute('title'), '');
  await vertices.first().press('Enter');
  assert.ok(await action('break-link').isDisabled());
  assert.ok(await action('remove-vertex').isEnabled());
  await vertices.nth(1).click();
  assert.ok(await action('remove-vertex').isEnabled());
  await action('break-link').click();
  if (width < 800) await sidebarToggle.click();
  assert.equal(await editParts.count(), 2);
  if (width < 800) await sidebarToggle.click();
  await action('undo').click();
  if (width < 800) await sidebarToggle.click();
  await page.getByRole('link', {name: 'Dati di base', exact: true}).click();
  await page.getByRole('link', {name: 'Cartografia', exact: true}).click();
  await action('save').waitFor();
  assert.equal(await editParts.count(), 1);
  if (width < 800) await sidebarToggle.click();
  const handle = loc('.cartography-vertex').first();
  const box = await handle.boundingBox();
  assert.ok(box);
  const x = box.x + box.width / 2, y = box.y + box.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 28, y + 12, { steps: 5 });
  await page.mouse.up();
  if (width < 800) await sidebarToggle.click();
  await action('promote').click();
  await modal.getByRole('button', {name: 'Annulla'}).click();
  assert.equal(await editParts.count(), 1);
  await action('promote').click();
  await checkNameField();
  await modal.locator('input[name=name]').fill('Strada nord');
  await modal.locator('select').selectOption({label: 'Pista di prova'});
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/road/') && r.request().method() === 'POST' && r.ok()),
    modal.locator('button[type=submit]').click(),
  ]);
  assert.equal((await (await page.request.get(url + '/api/geo/roads.geojson')).json()).features.length, 1);
  await action('road-properties').click();
  assert.equal(await modal.locator('[data-action=delete]').count(), 0);
  await modal.getByLabel('Nome').fill('Via nord');
  await modal.getByRole('button', {name: 'Salva', exact: true}).click();
  await loc('#cartography-roads').getByRole('button', {name: 'Via nord', exact: true}).waitFor();
  await action('points').click();
  await action('save').waitFor();
  if (width < 800) await sidebarToggle.click();
  assert.ok(await action('join-link').isDisabled());
  await loc('.cartography-insert-vertex').first().click();
  const roadVertices = loc('.cartography-vertex');
  const insertedCount = await roadVertices.count();
  await loc('.cartography-insert-vertex').nth(2).click();
  await loc('.cartography-insert-vertex').last().click();
  assert.equal(await roadVertices.count(), insertedCount + 2);
  await action('undo').click();
  await action('undo').click();
  assert.equal(await roadVertices.count(), insertedCount);
  assert.ok(await action('remove-vertex').isDisabled());
  await roadVertices.nth(1).click();
  assert.equal(await roadVertices.nth(1).getAttribute('aria-pressed'), 'true');
  assert.equal(await roadVertices.count(), insertedCount);
  await action('remove-vertex').click();
  assert.equal(await roadVertices.count(), insertedCount - 1);
  await action('undo').click();
  assert.equal(await roadVertices.count(), insertedCount);
  await roadVertices.nth(1).click();
  await action('break-link').click();
  assert.ok(await action('join-link').isEnabled());
  assert.ok(await action('save').isDisabled());
  await removeFirstVertex();
  assert.equal(await roadVertices.count(), 1);
  assert.ok(await action('save').isDisabled());
  await removeFirstVertex();
  assert.ok(await action('save').isEnabled());
  await action('undo').click();
  assert.equal(await roadVertices.count(), 1);
  await action('undo').click();
  assert.equal(await roadVertices.count(), 2);
  await action('undo').click();
  assert.equal(await roadVertices.count(), insertedCount);
  if (width < 800) await sidebarToggle.click();
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/road-points/') && r.ok()),
    action('save').click(),
  ]);
  await action('save').waitFor({state: 'hidden'});
  await page.goto(url + '/bosco');
  const filter = page.getByRole('checkbox', {name: 'Pista di prova'});
  await filter.waitFor();
  assert.ok(await filter.isChecked());
  assert.equal(await loc('path.bosco-road').count(), 1);
  await filter.uncheck();
  await loc('path.bosco-road').waitFor({state: 'detached'});
  await page.reload();
  await filter.waitFor();
  assert.equal(await filter.isChecked(), false);
  assert.equal(await loc('path.bosco-road').count(), 0);
  await filter.check();
  await loc('path.bosco-road').waitFor({state: 'attached'});

  await page.goto(editor);
  await action('clear').click();
  await modal.locator('[data-action=confirm]').click();
  await page.waitForFunction(() => !document.querySelector('#cartography-drawings').children.length);
  assert.ok(await action('clear').isHidden());
  await action('open-import').click();
  await modal.locator('[data-action=restore]').click();
  await drawings.first().waitFor({state: 'attached'});
  await action('new-region').click();
  await saveRegion('Costa alta', '');
  await action('new-drawing').click();
  if (width < 800) await sidebarToggle.click();
  // At world zoom the viewport can include repeated copies of a longitude.
  for (let i = 0; i < 4; i++) await loc('#cartography-map .leaflet-control-zoom-in').click();
  const lineIcon = await action('line').boundingBox();
  const polygonIcon = await action('polygon').boundingBox();
  assert.equal(lineIcon.x, polygonIcon.x);
  assert.ok(lineIcon.y + lineIcon.height <= polygonIcon.y);
  const map = loc('#cartography-map .leaflet-container');
  await action('line').click();
  await map.click({position: {x: 100, y: 295}});
  await map.click({position: {x: 165, y: 295}});
  await action('finish').click();
  await checkNameField();
  await modal.getByLabel('Nome').fill('Segmento');
  await modal.locator('[type=submit]').click();
  await drawings.first().waitFor({state: 'attached'});
  if (width < 800) await sidebarToggle.click();
  await action('edit').click();
  if (width < 800) await sidebarToggle.click();
  await removeFirstVertex();
  assert.equal(await vertices.count(), 1);
  assert.ok(await action('join-link').isDisabled());
  if (width < 800) await sidebarToggle.click();
  assert.equal(await editParts.count(), 1);
  await action('save').click();
  await action('save').waitFor({state: 'hidden'});
  await action('edit').click();
  if (width < 800) await sidebarToggle.click();
  await removeFirstVertex();
  assert.equal(await vertices.count(), 0);
  await action('undo').click();
  assert.equal(await vertices.count(), 1);
  await removeFirstVertex();
  if (width < 800) await sidebarToggle.click();
  await action('save').click();
  assert.match(await modal.textContent(), /Segmento/);
  await modal.locator('[data-action=confirm]').click();
  await drawings.first().waitFor({state: 'detached'});
  assert.ok(await action('save').isHidden());

  await action('new-drawing').click();
  if (width < 800) await sidebarToggle.click();
  await action('polygon').click();
  assert.ok(await action('line').isHidden());
  assert.ok(await action('finish').isHidden());
  const startVertex = loc('.cartography-sketch-start button');
  await map.click({position: {x: 100, y: 295}});
  await map.click({position: {x: 165, y: 295}});
  assert.ok(await startVertex.isDisabled());
  assert.ok(await modal.isHidden());
  await map.click({position: {x: 145, y: 360}});
  await map.click({position: {x: 100, y: 340}});
  assert.ok(await modal.isHidden());
  await startVertex.press('Enter');
  await modal.getByLabel('Nome').waitFor();
  assert.ok(await startVertex.isHidden());
  await modal.getByRole('button', {name: 'Annulla', exact: true}).click();
  await startVertex.waitFor();
  await startVertex.press('Enter');
  await modal.getByLabel('Nome').fill('Confine');
  await modal.locator('[type=submit]').click();
  if (width < 800) await sidebarToggle.click();
  await action('promote').click();
  assert.match(await modal.locator('[name=eclass_id] option:checked').textContent(), /A —/);
  assert.equal(await modal.getByLabel('Area catastale (ha)').inputValue(), '');
  await modal.getByLabel('Nome', {exact: true}).fill('2a');
  const [promoted] = await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/parcel/') && r.request().method() === 'POST' && r.ok()),
    modal.locator('[type=submit]').click(),
  ]);
  const {parcel_id: parcelId} = await promoted.json();
  const selectedParcel = loc('#cartography-parcels [data-action=select-item][aria-pressed=true]');
  await selectedParcel.waitFor();
  assert.equal((await selectedParcel.textContent()).trim(), '2a');
  assert.equal(await drawings.count(), 0);
  assert.equal(await bulkButtons.count(), 1);
  await page.getByText('Aggiornamento in attesa.', {exact: true}).waitFor();
  assert.ok(await action('refresh-terrain').isDisabled());
  const terrainUrl = `/api/impostazioni/cartography/${await loc('#cartography-region').inputValue()}/terrain/`;
  const failure = 'Impossibile scaricare le immagini da Copernicus (HTTP 503).';
  const monitor = await sidebar.evaluateHandle(element => {
    const changed = [];
    const observer = new MutationObserver(records => changed.push(...records));
    observer.observe(element, {attributes: true, childList: true, subtree: true});
    return {observer, changed};
  });
  await page.route(url + terrainUrl, route => route.fulfill({json: {
    state: 'failed', available: true, message: failure,
  }}));
  await sidebar.getByText(failure, {exact: true}).waitFor();
  await page.unroute(url + terrainUrl);
  assert.ok(await action('refresh-terrain').isEnabled());
  assert.equal(await monitor.evaluate(({observer, changed}) => {
    observer.disconnect();
    return changed.filter(record => !record.target.closest('.cartography-terrain')).length;
  }), 0);
  const parcelTrash = loc('#cartography-parcels [data-action=remove-item]');
  await parcelTrash.click();
  assert.match(await modal.textContent(), /«2a»/);
  await modal.getByRole('button', {name: 'Annulla', exact: true}).click();

  await action('parcel-properties').click();
  assert.equal(await modal.locator('[data-action=delete]').count(), 0);
  await modal.getByLabel('Nome', {exact: true}).fill('2b');
  const rename = modal.locator('[name=confirm_rename]');
  assert.equal(await rename.getAttribute('required'), '');
  await rename.check();
  await modal.getByRole('button', {name: 'Salva', exact: true}).click();
  await loc('#cartography-parcels').getByRole('button', {name: '2b', exact: true}).waitFor();

  await action('new-drawing').click();
  if (width < 800) await sidebarToggle.click();
  await action('line').click();
  await map.click({position: {x: 100, y: 295}});
  await action('cancel').click();
  await modal.locator('[data-action=confirm]').click();
  if (width < 800) await sidebarToggle.click();
  assert.equal((await selectedParcel.textContent()).trim(), '2b');
  assert.ok(await action('polygon').isHidden());

  await action('boundaries').click();
  await action('save').waitFor();
  if (width < 800) await sidebarToggle.click();
  assert.ok(await action('line').isHidden());
  assert.ok(await action('join-link').isDisabled());
  const parcelVertices = loc('.cartography-vertex');
  await parcelVertices.first().press('Enter');
  assert.equal(await parcelVertices.first().getAttribute('aria-pressed'), 'true');
  await action('remove-vertex').click();
  assert.equal(await parcelVertices.count(), 3);
  await parcelVertices.first().click();
  assert.ok(await action('remove-vertex').isDisabled());
  await action('polygon').click();
  const size = await map.boundingBox();
  const corner = [size.width - 95, size.height - 105];
  for (const [dx, dy] of [[0, 0], [55, 0], [40, 42]]) {
    await map.click({position: {x: corner[0] + dx, y: corner[1] + dy}});
  }
  await startVertex.click();
  assert.ok(await modal.isHidden());
  if (width < 800) await sidebarToggle.click();
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/parcel-boundaries/') && r.ok()),
    action('save').click(),
  ]);
  await action('save').waitFor({state: 'hidden'});

  const regionId = await loc('#cartography-region').inputValue();
  const headers = {'X-CSRFToken': await page.locator('body').getAttribute('data-csrf')};
  const outline = await (await page.request.get(url + '/api/geo/map.geojson')).json();
  assert.equal(outline.features.length, 2);
  assert.ok(outline.features.every(feature => feature.properties.name === 'Costa alta-2b'));
  // Removing the first vertex leaves a closed triangle.
  const editedRing = outline.features[0].geometry.coordinates[0];
  assert.equal(editedRing.length, 4);
  assert.deepEqual(editedRing[0], editedRing.at(-1));

  await action('boundaries').click();
  await action('save').waitFor();
  if (width < 800) await sidebarToggle.click();
  await parcelVertices.first().click();
  assert.ok(await action('remove-vertex').isDisabled());
  await action('break-link').click();
  assert.equal(await parcelVertices.count(), 2);
  assert.ok(await action('join-link').isDisabled());
  await removeFirstVertex();
  assert.equal(await parcelVertices.count(), 1);
  await removeFirstVertex();
  assert.equal(await parcelVertices.count(), 3);
  await action('undo').click();
  assert.equal(await parcelVertices.count(), 1);
  if (width < 800) await sidebarToggle.click();
  await action('save').click();
  assert.equal(await modal.locator('[data-field=message]').textContent(), 'Solo i poligoni verranno salvati. Continuare?');
  assert.ok(await modal.getByRole('button', {name: 'Annulla', exact: true}).isVisible());
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/parcel-boundaries/') && r.ok()),
    modal.getByRole('button', {name: 'Salva', exact: true}).click(),
  ]);
  await action('save').waitFor({state: 'hidden'});
  const remaining = await (await page.request.get(url + '/api/geo/map.geojson')).json();
  assert.deepEqual(remaining.features, [outline.features[1]]);
  const [lon, lat] = remaining.features[0].geometry.coordinates[0][0];
  const tree = await page.request.post(url + '/api/bosco/pai/save/', {
    headers, data: {parcel_id: parcelId, species_id: Number(process.env.TEST_SPECIES),
      number: 1, d_cm: 42, h_m: 18.5, date: '2026-11-19', lat, lon, nonce: 'list-pai'},
  });
  assert.ok(tree.ok(), await tree.text());
  const observation = await page.request.post(url + '/api/bosco/observations/save/', {
    headers, form: {region_id: regionId, date: '2026-11-19', text: 'Verifica elenco',
      lat, lon, category_ids: process.env.TEST_CATEGORY, nonce: 'list-observation'},
  });
  assert.ok(observation.ok(), await observation.text());
  await page.goto(editor + `&c=${regionId}`);
  await parcelTrash.waitFor({state: 'attached'});
  assert.ok(await parcelTrash.isDisabled());
  assert.equal(await parcelTrash.getAttribute('title'), 'Questa particella è in uso');

  await action('open-import').click();
  await modal.getByRole('button', {name: 'Da rilevamento', exact: true}).click();
  await modal.locator('#cartography-surveys option').first().waitFor({state: 'attached'});
  assert.deepEqual(await modal.locator('#cartography-surveys option').allTextContents(),
    ['Alberi da preservare · Libero · 1 posizione']);
  assert.deepEqual(await modal.locator('form[data-path=survey] .form-actions button').allTextContents(),
    ['Annulla', 'Importa']);
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/import-survey/') && r.ok()),
    modal.getByRole('button', {name: 'Importa', exact: true}).click(),
  ]);
  await drawings.getByRole('button', {name: 'Alberi da preservare', exact: true}).waitFor();

  await action('open-import').click();
  await modal.getByRole('button', {name: 'Da file', exact: true}).click();
  const file = modal.locator('input[type=file]');
  assert.deepEqual(await modal.locator('form[data-path=file] .form-actions button').allTextContents(),
    ['Annulla', 'Importa']);
  await file.setInputFiles({name: 'confini.dxf', mimeType: 'application/dxf', buffer: Buffer.from('')});
  assert.ok(await modal.getByLabel('Codice EPSG del DXF').isVisible());
  await file.setInputFiles({name: 'Traccia.geojson', mimeType: 'application/geo+json',
    buffer: Buffer.from(JSON.stringify({type: 'LineString', coordinates: [[16, 38], [16, 38.1]]}))});
  assert.ok(await modal.getByLabel('Codice EPSG del DXF').isHidden());
  assert.equal(await modal.getByLabel('Nome', {exact: true}).inputValue(), 'Traccia');
  const importNonces = [];
  await page.route('**/import-file/', async route => {
    importNonces.push(route.request().postDataJSON().nonce);
    if (importNonces.length > 1) return route.continue();
    assert.equal((await route.fetch()).status(), 200);
    await route.abort('failed');
  });
  await modal.getByRole('button', {name: 'Importa', exact: true}).click();
  await modal.locator('.form-error').waitFor();
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/import-file/') && r.ok()),
    modal.getByRole('button', {name: 'Importa', exact: true}).click(),
  ]);
  const traces = drawings.getByRole('button', {name: 'Traccia', exact: true});
  await traces.waitFor();
  assert.equal(await traces.count(), 1);
  assert.equal(importNonces.length, 2);
  assert.equal(importNonces[0], importNonces[1]);

  async function checkFilterList(target, hasSwatch = false) {
    const row = loc(`ul[data-target="${target}"] li.map-workspace-list-row`).first();
    const checkbox = row.getByRole('checkbox');
    await checkbox.waitFor();
    const style = await row.evaluate(li => {
      const label = li.querySelector('label');
      const text = label.querySelector('.map-workspace-list-label');
      const swatch = label.querySelector('.bosco-legend-dot');
      return {
        font: getComputedStyle(text).fontSize, weight: getComputedStyle(text).fontWeight,
        padding: parseFloat(getComputedStyle(label).paddingInlineStart),
        gap: parseFloat(getComputedStyle(label).columnGap),
        swatch: swatch?.getBoundingClientRect().width ?? 0,
      };
    });
    assert.equal(style.font, '16px');
    assert.equal(style.weight, '400');
    assert.ok(style.padding >= 6 && style.gap >= 6);
    if (hasSwatch) assert.equal(style.swatch, 12);
    await checkbox.uncheck();
    assert.ok(!await checkbox.isChecked());
  }
  await page.goto(url + `/bosco?c=${regionId}&m=3`);
  await checkFilterList('pai-parcels');
  await checkFilterList('pai-species', true);
  await page.goto(url + `/bosco?c=${regionId}&m=4`);
  await checkFilterList('observation-categories');
  await page.route(url + `/api/bosco/terrain/${regionId}/hillshade.png`, route => route.fulfill({
    contentType: 'image/png',
    headers: {'X-Abies-Bounds': '[[37.99, 15.99], [38.01, 16.01]]'},
    body: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1sAAAAASUVORK5CYII=', 'base64'),
  }));
  await page.goto(url + `/bosco?c=${regionId}&q=11`);
  const characteristic = loc('#bosco-characteristic-select');
  assert.equal(await characteristic.inputValue(), '11');
  assert.equal(await characteristic.locator('optgroup:has(option[value="11"])').getAttribute('label'),
    'Dati satellitari e LiDAR');
  await page.waitForFunction(() => document.querySelector('.leaflet-bosco-hillshade-pane img')?.naturalWidth === 1);
  assert.match(await loc('[data-target="legend"]').textContent(), /INGV TINITALY 1.1/);
  await characteristic.evaluate(select => {
    select.value = '0';
    select.dispatchEvent(new Event('change'));
  });
  await loc('.leaflet-bosco-hillshade-pane img').waitFor({state: 'detached'});
  assert.deepEqual(errors, []);
}

(async () => {
  const browser = await chromium.launch({ignoreDefaultArgs: ['--hide-scrollbars']});
  try { await walk(browser); } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
