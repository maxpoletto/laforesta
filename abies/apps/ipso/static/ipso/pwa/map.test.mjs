// Adversarial tests for Ipso history-map point validation and fitting.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { recordLatLng, recordLatLngs, fitRecordPoints } = require('./map.js');

const points = recordLatLngs([
  { lat: '38.5', lon: '16.1' },
  { lat: 0, lon: 0 },
  { lat: '', lon: '' },
  { lat: '   ', lon: '16.2' },
  { lat: 91, lon: 16.2 },
  { lat: 38.5, lon: -181 },
  { lat: false, lon: 16.2 },
  { lat: null, lon: 16.2 },
]);
if (JSON.stringify(points) !== JSON.stringify([[38.5, 16.1], [0, 0]])) {
  throw new Error('map points must accept valid zero/numeric strings and reject corrupt coordinates');
}
if (recordLatLng({ lat: 38.5, lon: null }) !== null) {
  throw new Error('half-present coordinates must not be rendered at an invented location');
}

const calls = [];
const leaflet = {
  setView(point, zoom) { calls.push(['setView', point, zoom]); },
  fitBounds(bounds, options) { calls.push(['fitBounds', bounds, options]); },
};
if (!fitRecordPoints(leaflet, [[38.5, 16.1]], {})) {
  throw new Error('a single valid point must fit successfully');
}
if (JSON.stringify(calls[0]) !==
    JSON.stringify(['setView', [38.5, 16.1], 18])) {
  throw new Error('a single point must center at the history-map detail zoom');
}

const bounds = { isValid: () => true };
const leafletApi = {
  latLngBounds(value) {
    if (value !== points) throw new Error('all validated points must reach Leaflet');
    return bounds;
  },
};
if (!fitRecordPoints(leaflet, points, leafletApi)) {
  throw new Error('multiple valid points must fit successfully');
}
const fitCall = calls[1];
if (fitCall[0] !== 'fitBounds' || fitCall[1] !== bounds ||
    JSON.stringify(fitCall[2]) !==
      JSON.stringify({ padding: [24, 24], maxZoom: 18 })) {
  throw new Error('multi-point history maps must use the intended fit bounds');
}
if (fitRecordPoints(leaflet, [], leafletApi)) {
  throw new Error('an empty point set must fall back to contextual map centering');
}
if (fitRecordPoints(leaflet, points, {
  latLngBounds: () => ({ isValid: () => false }),
})) {
  throw new Error('invalid Leaflet bounds must not be reported as fitted');
}

console.log('Ipso map tests passed.');
