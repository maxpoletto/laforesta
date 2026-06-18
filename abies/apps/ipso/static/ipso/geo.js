// Bridge the shared Abies ES-module geometry helpers into Ipso's classic scripts.
'use strict';

const SHARED_GEO_GLOBALS = [
  'pointInRing',
  'pointInPolygon',
  'findContainingParcel',
  'parcelLabel',
  'metersToDegLat',
  'metersToDegLng',
  'featureBbox',
  'buildBboxIndex',
  'distanceToBoundaryMeters',
];

window.AbiesGeoReady = import('/static/base/js/geo.js').then((geo) => {
  for (const name of SHARED_GEO_GLOBALS) window[name] = geo[name];
});
