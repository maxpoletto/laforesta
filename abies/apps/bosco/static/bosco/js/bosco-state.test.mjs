// Tests for Bosco URL-state helpers.

import {
  clearMapView, formatCenter, mapTypeName, mapTypeToken, parseCenter,
  readBoscoParams, writeMapView,
} from './bosco-state.js';

let failed = 0;
let passed = 0;

function assertEqual(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) passed++;
  else {
    failed++;
    console.error(`FAIL ${msg}`);
    console.error(`  expected: ${e}`);
    console.error(`       got: ${a}`);
  }
}

console.log('bosco-state.js');

assertEqual(mapTypeName('o'), 'osm', 'mapTypeName: osm token');
assertEqual(mapTypeName('bad'), 'satellite', 'mapTypeName: default satellite');
assertEqual(mapTypeToken('topo'), 't', 'mapTypeToken: topo name');
assertEqual(mapTypeToken('bad'), 's', 'mapTypeToken: default satellite token');

assertEqual(parseCenter('38.1,16.2'), [38.1, 16.2], 'parseCenter: valid');
assertEqual(parseCenter('38.1'), null, 'parseCenter: missing lng');
assertEqual(parseCenter('x,16.2'), null, 'parseCenter: invalid lat');
assertEqual(formatCenter([38.1234567, 16.2]), '38.123457,16.200000',
            'formatCenter: six decimals');

let state = readBoscoParams({}, [7, 8]);
assertEqual(state.regionId, 7, 'readBoscoParams: region fallback');
assertEqual(state.mode, '1', 'readBoscoParams: default mode');
assertEqual(state.mt, 's', 'readBoscoParams: default map token');
assertEqual(state.center, null, 'readBoscoParams: no partial view');

state = readBoscoParams({ c: '8', m: '3', mt: 't', mc: '38,16', mz: '12' }, [7, 8]);
assertEqual(state.regionId, 8, 'readBoscoParams: valid region');
assertEqual(state.mode, '3', 'readBoscoParams: valid mode');
assertEqual(state.basemap, 'topo', 'readBoscoParams: basemap');
assertEqual(state.center, [38, 16], 'readBoscoParams: restored center');
assertEqual(state.zoom, 12, 'readBoscoParams: restored zoom');

state = readBoscoParams({ c: '99', m: '9', mt: 'bad', mc: '38,16' }, [7, 8]);
assertEqual(state.regionId, 7, 'readBoscoParams: stale region fallback');
assertEqual(state.mode, '1', 'readBoscoParams: invalid mode fallback');
assertEqual(state.mt, 's', 'readBoscoParams: invalid map fallback');
assertEqual(state.center, null, 'readBoscoParams: center ignored without zoom');

const params = new URLSearchParams();
writeMapView(params, [38.1, 16.2], 13);
assertEqual(params.toString(), 'mc=38.100000%2C16.200000&mz=13',
            'writeMapView: params');
clearMapView(params);
assertEqual(params.toString(), '', 'clearMapView: params removed');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
