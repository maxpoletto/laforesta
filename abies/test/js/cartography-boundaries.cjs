// Run via test_cartography_browser.py against the published parcel fixtures.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.ABIES_PLAYWRIGHT_MODULE);

async function run(browser) {
  const url = process.env.TEST_URL;
  const width = Number(process.env.TEST_WIDTH);
  const context = await browser.newContext({viewport: {width, height: 850}});
  await context.addCookies([{name: 'sessionid', value: process.env.TEST_SESSION, url}]);
  await context.route(/^https?:\/\//, route => new URL(route.request().url()).origin === url
    ? route.continue() : route.abort());
  const page = await context.newPage();
  page.setDefaultTimeout(7000);
  const errors = [];
  page.on('pageerror', error => {
    errors.push(error.message);
    console.error(error.stack);
  });
  page.on('response', response => {
    const request = new URL(response.url());
    if (request.origin === url && response.status() >= 400 && request.pathname.startsWith('/api/')) {
      console.error(response.status(), request.pathname);
    }
  });
  const action = name => page.locator(`#settings-cartografia [data-action="${name}"]`);
  const modal = page.locator('#modal-container');
  const follow = action('follow-boundary');
  const shared = page.locator('.cartography-shared-vertex');
  const target = name => page.getByRole('button', {name: `Vertice sul confine di ${name}`, exact: true});
  const sidebar = page.locator('#cartography-map [title="Mostra/nascondi pannello"]');
  const toggleMobilePanel = async () => { if (width < 800) await sidebar.click(); };
  await page.goto(url + `/impostazioni?scheda=cartografia&c=${process.env.TEST_REGION}`);
  await action('new-drawing').click();
  await toggleMobilePanel();
  await action('polygon').click();
  assert.ok(await follow.isDisabled());
  await page.getByRole('button', {name: /Vertice sul confine di (?=.*Ovest)(?=.*Sud)/}).first().click();
  assert.equal(await shared.count(), 1);
  assert.ok(await follow.isEnabled());

  async function followWest() {
    await follow.click();
    const end = target('Ovest').nth(1);
    await end.hover();
    await page.locator('.cartography-trace-preview').waitFor();
    await end.click();
    assert.ok(await follow.isDisabled());
    assert.equal(await shared.count(), 3);
  }
  await followWest();
  await action('undo').click();
  assert.equal(await shared.count(), 1);
  assert.ok(await follow.isEnabled());
  await followWest();
  await target('Esterno').nth(3).click();
  await target('Esterno').first().click();
  await page.getByRole('button', {name: 'Chiudi il poligono'}).click();
  await modal.getByLabel('Nome', {exact: true}).fill('Tracciato');
  await modal.getByRole('button', {name: 'Salva', exact: true}).click();

  await toggleMobilePanel();
  await action('promote').click();
  await modal.getByLabel('Nome', {exact: true}).fill('Est');
  await modal.getByRole('button', {name: 'Salva', exact: true}).click();
  await page.locator('#cartography-parcels').getByRole('button', {name: 'Est', exact: true}).waitFor();
  await action('boundaries').click();
  await toggleMobilePanel();
  assert.ok(await follow.isDisabled());
  const vertices = page.locator('.cartography-vertex');
  await vertices.nth(4).click();
  const destination = target('Sud').last();
  const beforeSnap = await destination.boundingBox();
  await destination.click();
  const snapped = await vertices.nth(4).boundingBox();
  assert.ok(Math.abs(snapped.x + snapped.width / 2 - beforeSnap.x - beforeSnap.width / 2) <= 1);
  assert.ok(Math.abs(snapped.y + snapped.height / 2 - beforeSnap.y - beforeSnap.height / 2) <= 1);
  assert.ok(await follow.isEnabled());
  await action('undo').click();
  const junction = vertices.first();
  assert.match(await junction.getAttribute('class'), /cartography-shared-vertex/);
  await junction.click();
  assert.ok(await follow.isEnabled());
  await follow.click();
  await target('Sud').nth(1).click();
  assert.ok(await follow.isDisabled());
  assert.equal(await vertices.count(), 7);
  await action('save').click();
  await action('save').waitFor({state: 'hidden'});

  await toggleMobilePanel();
  await action('boundaries').click();
  await toggleMobilePanel();
  const {x, y, width: size, height} = await junction.boundingBox();
  const cx = x + size / 2, cy = y + height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 18, cy + 2, {steps: 5});
  await page.mouse.up();
  await action('save').click();
  await action('save').waitFor({state: 'hidden'});
  assert.deepEqual(errors, []);
}

(async () => {
  const browser = await chromium.launch();
  try { await run(browser); } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
