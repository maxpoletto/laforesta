// Adversarial tests for locale-aware Ipso display formatting.

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, 'format.js'), 'utf8');

function loadFormat(lang) {
  const context = {
    document: { documentElement: { lang } },
    module: { exports: {} },
  };
  vm.createContext(context);
  vm.runInContext(source, context, { filename: 'format.js' });
  return context.module.exports.IpsoFormat;
}

const it = loadFormat('it');
const en = loadFormat('en');

if (it.fmtDecimal2('18.25') !== '18,25') {
  throw new Error('Italian canonical decimal strings must use a comma at display time');
}
if (en.fmtDecimal2('18.25') !== '18.25') {
  throw new Error('English decimal display must retain a decimal point');
}
if (it.fmtInt(1234) !== '1234') {
  throw new Error('wire/display integers must not gain locale grouping separators');
}
if (it.fmtCoord('38.5') !== '38,50000') {
  throw new Error('numeric coordinate strings must be localized with fixed precision');
}
if (it.fmtDecimal2(null) !== '') {
  throw new Error('missing values must remain blank at the display boundary');
}

console.log('Ipso format tests passed.');
