// Tests for Bosco position formatting helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-position-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const P = await import(staticModule('bosco/js/bosco-position.js'));
const S = await import(staticModule('base/js/strings.js'));

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  if (actual === expected) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${expected}`);
    console.error(`       got: ${actual}`);
  }
}

console.log('bosco-position.js');

assertEqual(P.formatPosition(38.123456, 16.654321), '(38,12346, 16,65432)',
            'formatPosition: locale coordinates');
assertEqual(P.formatPosition(38.123456, 16.654321, 7),
            '(38,12346, 16,65432) ± 7 m',
            'formatPosition: appends accuracy when present');
assertEqual(P.formatPosition(NaN, 16.654321, 7), '',
            'formatPosition: blank when coordinates are incomplete');
assertEqual(P.positionLabelValue(38.1, 16.2, '')[0], S.BOSCO_POSITION,
            'positionLabelValue: localized label');

console.log(`
${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
