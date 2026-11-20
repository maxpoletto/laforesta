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
    [section.tagName, section.querySelector('label, summary, h2').textContent.trim()])),
    [['SECTION', 'Compresa'], ['SECTION', 'Particelle'], ['SECTION', 'Strade'], ['DETAILS', 'Disegni']]);
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
  assert.match(await eyeImage(), /visible\.webp/);
  await visibleDrawing.click();
  assert.match(await eyeImage(), /hidden\.webp/);
  assert.equal(await visibleDrawing.getAttribute('aria-pressed'), 'false');
  await sidebar.locator('[data-kind=drawing] [data-action=toggle-all]').click();
  assert.equal(await visibleDrawing.getAttribute('aria-pressed'), 'true');
  await drawings.first().locator('[data-action=select-item]').click();
  await loc('#cartography-roads-label').click();
  assert.equal(await sidebar.locator('[data-action=select-item][aria-pressed=true]').count(), 0);
  assert.ok(await action('edit').isDisabled());
  await drawings.first().locator('[data-action=select-item]').click();
  await action('edit').click();
  assert.deepEqual(await loc('#cartography-parts option').allTextContents(),
    ['1. Linea', '2. Punto', '3. Linea', '4. Linea']);
  await loc('#cartography-parts').selectOption('3');
  await action('copy').click();
  await modal.getByLabel('Nome').fill('Laterale');
  await modal.locator('button[type=submit]').click();
  await page.waitForFunction(() => document.querySelector('#cartography-drawings').children.length === 2);
  await action('remove-part').click();
  await loc('#cartography-parts').selectOption('1');
  await action('remove-part').click();
  await loc('#cartography-parts').selectOption('0');
  assert.ok(await action('promote').isDisabled());
  if (width < 800) await sidebarToggle.click();
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
  await vertices.first().press('Enter');
  assert.ok(await action('break-link').isDisabled());
  assert.ok(await action('remove-vertex').isEnabled());
  await vertices.nth(1).click();
  assert.ok(await action('remove-vertex').isEnabled());
  await action('break-link').click();
  if (width < 800) await sidebarToggle.click();
  assert.equal(await loc('#cartography-parts option').count(), 2);
  if (width < 800) await sidebarToggle.click();
  await action('undo').click();
  if (width < 800) await sidebarToggle.click();
  await page.getByRole('link', {name: 'Dati di base', exact: true}).click();
  await page.getByRole('link', {name: 'Cartografia', exact: true}).click();
  await action('save').waitFor();
  assert.equal(await loc('#cartography-parts option').count(), 1);
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
  assert.equal(await loc('#cartography-parts option').count(), 1);
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
  assert.deepEqual(await loc('#cartography-parts option').allTextContents(), ['1. Punto']);
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

  await action('parcel-properties').click();
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
  assert.deepEqual(errors, []);
}

(async () => {
  const browser = await chromium.launch({ignoreDefaultArgs: ['--hide-scrollbars']});
  try { await walk(browser); } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
