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
  page.on('pageerror', e => errors.push(e.message));
  page.setDefaultTimeout(7000);
  const loc = selector => page.locator(selector);
  const action = name => loc(`#settings-cartografia [data-action="${name}"]`);
  const editor = url + '/impostazioni?scheda=cartografia';
  await page.goto(editor);
  const modal = loc('#modal-container');
  const selected = loc('#cartography-region option:checked');
  async function saveRegion(name, description) {
    await modal.getByLabel('Nome', {exact: true}).fill(name);
    await modal.getByLabel('Descrizione').fill(description);
    await modal.getByRole('button', {name: 'Salva', exact: true}).click();
    await page.waitForFunction(label => document.querySelector('#cartography-region')?.selectedOptions[0]?.text === label, name);
  }

  assert.ok(await action('edit-region').isDisabled());
  await action('new-region').click();
  await saveRegion('Temporanea', '');
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
    await loc('#cartography-imports').click();
    await action('import').click();
  }
  await loc('#cartography-drawings option').first().waitFor({state: 'attached'});
  assert.equal(await loc('[data-ipso-pending-dot]:not([hidden])').count(), 0);
  await action('edit-region').click();
  await modal.getByLabel('Nome', {exact: true}).waitFor();
  assert.equal(await modal.getByRole('button', {name: 'Elimina', exact: true}).count(), 0);
  await modal.getByRole('button', {name: 'Annulla'}).click();
  assert.equal(await loc('#cartography-drawings option').count(), 1);
  assert.ok(await action('promote').isDisabled());
  await loc('#cartography-drawings').selectOption({index: 0});
  await action('edit').click();
  assert.deepEqual(await loc('#cartography-parts option').allTextContents(),
    ['1. Linea', '2. Punto', '3. Linea', '4. Linea']);
  await loc('#cartography-parts').selectOption('3');
  await action('copy').click();
  await modal.getByLabel('Nome').fill('Laterale');
  await modal.locator('button[type=submit]').click();
  await page.waitForFunction(() => document.querySelector('#cartography-drawings').options.length === 2);
  await action('remove-part').click();
  await loc('#cartography-parts').selectOption('1');
  await action('remove-part').click();
  await loc('#cartography-parts').selectOption('0');
  assert.ok(await action('promote').isDisabled());
  await action('join').click();
  await action('undo').click();
  assert.ok(await action('promote').isDisabled());
  await action('join').click();
  const sidebarToggle = loc('#cartography-map [title="Mostra/nascondi pannello"]');
  if (width < 800) await sidebarToggle.click();
  await loc('.cartography-vertex').nth(1).click();
  if (width < 800) await sidebarToggle.click();
  await action('split').click();
  assert.equal(await loc('#cartography-parts option').count(), 2);
  await action('undo').click();
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
  await modal.locator('input[name=name]').fill('Strada nord');
  await modal.locator('select').selectOption({label: 'Pista di prova'});
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/road/') && r.request().method() === 'POST' && r.ok()),
    modal.locator('button[type=submit]').click(),
  ]);
  assert.equal((await (await page.request.get(url + '/api/geo/roads.geojson')).json()).features.length, 1);
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
  await loc('#cartography-imports').click();
  await loc('#cartography-saved summary').click();
  await action('clear').click();
  await modal.locator('[data-action=confirm]').click();
  await page.waitForFunction(() => !document.querySelector('#cartography-drawings').options.length);
  await action('restore').click();
  await loc('#cartography-drawings option').first().waitFor({state: 'attached'});
  await action('new-region').click();
  await saveRegion('Costa alta', '');
  await action('polygon').click();
  if (width < 800) await sidebarToggle.click();
  const map = loc('#cartography-map .leaflet-container');
  for (const [x, y] of [[100, 150], [165, 150], [145, 215]]) {
    await map.click({position: {x, y}});
  }
  if (width < 800) await sidebarToggle.click();
  await action('save').click();
  await modal.getByLabel('Nome').fill('Confine');
  await modal.locator('[type=submit]').click();
  await action('promote').click();
  assert.match(await modal.locator('[name=eclass_id] option:checked').textContent(), /A —/);
  assert.equal(await modal.getByLabel('Area (ha)').inputValue(), '');
  await modal.getByLabel('Nome', {exact: true}).fill('2a');
  await modal.locator('[type=submit]').click();
  await page.waitForFunction(() => document.querySelector('#cartography-parcels')?.value === '2a');
  assert.equal(await loc('#cartography-drawings option').count(), 0);
  assert.deepEqual(errors, []);
}

(async () => {
  const browser = await chromium.launch();
  try { await walk(browser); } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
