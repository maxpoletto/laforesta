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
  await action('new-region').click();
  const modal = loc('#modal-container');
  await modal.locator('input[name=name]').fill('Costa');
  await modal.getByRole('button', { name: 'Pubblica compresa' }).click();
  await page.waitForSelector('#cartography-region option:checked[value]:not([value=""])', { state: 'attached' });

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
  await loc('#trace-toggle').click();
  await loc('#point-add').click();
  await loc('#point-note').fill('Cancello');
  await loc('#point-save').click();
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
  assert.ok(await action('road').isDisabled());
  await loc('#cartography-drawings').selectOption({ index: 1 });
  assert.ok(await action('road').isDisabled());
  await loc('#cartography-drawings').selectOption({ index: 0 });
  await action('edit').click();
  const sidebarToggle = loc('#cartography-map [title="Mostra/nascondi pannello"]');
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
  await action('save').click();
  await action('road').click();
  await modal.locator('input[name=name]').fill('Strada nord');
  await modal.locator('select').selectOption({label: 'Pista di prova'});
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/road/') && r.request().method() === 'POST' && r.ok()),
    modal.locator('button[type=submit]').click(),
  ]);
  assert.equal((await (await page.request.get(url + '/api/geo/roads.geojson')).json()).features.length, 0);
  await action('publish').click();
  await Promise.all([
    page.waitForResponse(r => r.url().endsWith('/publish/') && r.request().method() === 'POST' && r.ok()),
    modal.locator('[data-action=confirm]').click(),
  ]);
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
  assert.deepEqual(errors, []);
}

(async () => {
  const browser = await chromium.launch();
  try { await walk(browser); } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
