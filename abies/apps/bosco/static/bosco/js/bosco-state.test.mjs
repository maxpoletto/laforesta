// Tests for Bosco URL-state helpers.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'abies-bosco-state-js-'));
const staticRoot = path.join(tmpRoot, 'static');
fs.mkdirSync(path.join(staticRoot, 'bosco'), { recursive: true });
fs.mkdirSync(path.join(staticRoot, 'base'), { recursive: true });
fs.cpSync(here, path.join(staticRoot, 'bosco', 'js'), { recursive: true });
fs.cpSync(path.resolve(here, '../../../../base/static/base/js'),
          path.join(staticRoot, 'base', 'js'), { recursive: true });
process.on('exit', () => fs.rmSync(tmpRoot, { recursive: true, force: true }));
const staticModule = rel => pathToFileURL(path.join(staticRoot, rel)).href;

const {
  clearDetailParams, clearMapView, formatCenter, mapTypeName, mapTypeToken,
  parseCenter, parseIdList, parseOptionalIdList, parseSectionTokens, readBoscoParams,
  writeMapView, writeOptionalIdList, writeSectionTokens,
} = await import(staticModule('bosco/js/bosco-state.js'));

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
assertEqual(state.q, '1', 'readBoscoParams: default characteristic');
assertEqual(state.evolutionMetric, '1', 'readBoscoParams: default evolution metric');
assertEqual(state.evolutionDate1, null, 'readBoscoParams: default evolution date 1');
assertEqual(state.evolutionDate2, null, 'readBoscoParams: default evolution date 2');
assertEqual(state.parcelAverage, false, 'readBoscoParams: default detailed raster mode');
assertEqual(state.useCadastralArea, false, 'readBoscoParams: default cadastral flag');
assertEqual(state.harvestPerHa, false, 'readBoscoParams: default per-ha flag');
assertEqual(state.detailMode, null, 'readBoscoParams: no detail overlay');
assertEqual(state.openSections, ['m'], 'readBoscoParams: default detail sections');
assertEqual(state.detailSpeciesIds, null, 'readBoscoParams: default detail species');
assertEqual(state.paiParcelIds, null, 'readBoscoParams: default PAI parcels');
assertEqual(state.paiSpeciesIds, null, 'readBoscoParams: default PAI species');

state = readBoscoParams({
  c: '8', m: '3', mt: 't', mc: '38,16', mz: '12', q: '5', fc: '1', fh: '1',
  v: '1', pa: '42', vo: 'dmx', ds: '3,5,3,bad', pp: '7,8', ps: '',
}, [7, 8]);
assertEqual(state.regionId, 8, 'readBoscoParams: valid region');
assertEqual(state.mode, '3', 'readBoscoParams: valid mode');
assertEqual(state.basemap, 'topo', 'readBoscoParams: basemap');
assertEqual(state.center, [38, 16], 'readBoscoParams: restored center');
assertEqual(state.zoom, 12, 'readBoscoParams: restored zoom');
assertEqual(state.q, '5', 'readBoscoParams: characteristic');
assertEqual(state.evolutionMetric, '1', 'readBoscoParams: invalid evolution metric fallback outside m2');
assertEqual(state.useCadastralArea, true, 'readBoscoParams: cadastral flag');
assertEqual(state.harvestPerHa, false, 'readBoscoParams: per-ha flag ignored outside characteristics');
assertEqual(state.detailMode, '1', 'readBoscoParams: parcel detail overlay');
assertEqual(state.parcelId, 42, 'readBoscoParams: detail parcel');
assertEqual(state.openSections, ['d', 'm'], 'readBoscoParams: detail sections');
assertEqual(state.detailSpeciesIds, [3, 5], 'readBoscoParams: detail species ids');
assertEqual(state.paiParcelIds, [7, 8], 'readBoscoParams: PAI parcel ids');
assertEqual(state.paiSpeciesIds, [], 'readBoscoParams: explicit empty PAI species');

state = readBoscoParams({ ds: '' }, [7, 8]);
assertEqual(state.detailSpeciesIds, [], 'readBoscoParams: explicit empty detail species');

state = readBoscoParams({ c: '99', m: '9', mt: 'bad', mc: '38,16', q: '99', v: '9' }, [7, 8]);
assertEqual(state.regionId, 7, 'readBoscoParams: stale region fallback');
assertEqual(state.mode, '1', 'readBoscoParams: invalid mode fallback');
assertEqual(state.mt, 's', 'readBoscoParams: invalid map fallback');
assertEqual(state.center, null, 'readBoscoParams: center ignored without zoom');
assertEqual(state.q, '1', 'readBoscoParams: invalid characteristic fallback');
assertEqual(state.detailMode, null, 'readBoscoParams: invalid detail fallback');

state = readBoscoParams({ m: '1', q: '4', fh: '1' }, [7, 8]);
assertEqual(state.harvestPerHa, true, 'readBoscoParams: per-ha allowed on historical harvest');

state = readBoscoParams({ m: '1', q: '1', fh: '1' }, [7, 8]);
assertEqual(state.harvestPerHa, false, 'readBoscoParams: per-ha ignored on non-harvest characteristic');

state = readBoscoParams({ m: '2', q: '4', fh: '1' }, [7, 8]);
assertEqual(state.harvestPerHa, false, 'readBoscoParams: per-ha ignored outside characteristics');

state = readBoscoParams({ m: '1', fa: '1' }, [7, 8]);
assertEqual(state.parcelAverage, false, 'readBoscoParams: parcel-average ignored outside evolution');

state = readBoscoParams({ c: '8', m: '2', q: '3', d1: '20240102', d2: '2024-07' }, [7, 8]);
assertEqual(state.mode, '2', 'readBoscoParams: evolution mode');
assertEqual(state.q, '3', 'readBoscoParams: evolution q');
assertEqual(state.evolutionMetric, '3', 'readBoscoParams: evolution metric');
assertEqual(state.evolutionDate1, '2024-01-02', 'readBoscoParams: compact date');
assertEqual(state.evolutionDate2, '2024-07-01', 'readBoscoParams: month date');
assertEqual(state.parcelAverage, false, 'readBoscoParams: evolution defaults to raster');

state = readBoscoParams({ m: '2', q: '3', fa: '1' }, [7, 8]);
assertEqual(state.parcelAverage, true, 'readBoscoParams: evolution parcel-average flag');

state = readBoscoParams({ m: '2', q: '8' }, [7, 8]);
assertEqual(state.q, '1', 'readBoscoParams: invalid evolution q fallback');

const params = new URLSearchParams();
writeMapView(params, [38.1, 16.2], 13);
assertEqual(params.toString(), 'mc=38.100000%2C16.200000&mz=13',
            'writeMapView: params');
clearMapView(params);
assertEqual(params.toString(), '', 'clearMapView: params removed');

assertEqual(parseSectionTokens(null), ['m'], 'parseSectionTokens: default');
assertEqual(parseSectionTokens('dpmxmd'), ['d', 'p', 'm'], 'parseSectionTokens: valid unique tokens');
assertEqual(parseIdList('2,1,2,bad,0,-1'), [2, 1], 'parseIdList: positive unique ints');
assertEqual(parseOptionalIdList(null), null, 'parseOptionalIdList: absent means all');
assertEqual(parseOptionalIdList(''), [], 'parseOptionalIdList: empty means none');
assertEqual(parseOptionalIdList('3,4'), [3, 4], 'parseOptionalIdList: ids');

const detailParams = new URLSearchParams('v=1&pa=2&vo=dm&ds=4,5');
clearDetailParams(detailParams);
assertEqual(detailParams.toString(), '', 'clearDetailParams: removes overlay params');
writeSectionTokens(detailParams, ['m']);
assertEqual(detailParams.toString(), '', 'writeSectionTokens: default omitted');
writeSectionTokens(detailParams, ['d', 'm']);
assertEqual(detailParams.toString(), 'vo=dm', 'writeSectionTokens: non-default encoded');
writeOptionalIdList(detailParams, 'pp', null, [1, 2]);
assertEqual(detailParams.toString(), 'vo=dm', 'writeOptionalIdList: absent all omitted');
writeOptionalIdList(detailParams, 'pp', [], [1, 2]);
assertEqual(detailParams.toString(), 'vo=dm&pp=', 'writeOptionalIdList: explicit none');
writeOptionalIdList(detailParams, 'pp', [2], [1, 2]);
assertEqual(detailParams.toString(), 'vo=dm&pp=2', 'writeOptionalIdList: subset');

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
