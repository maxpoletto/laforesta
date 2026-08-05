import {
  SATELLITE_MARKER_COLORS, markerFillColor, refreshSemanticMarkers,
  semanticMarkerStyle,
} from './map-palette.js';

let pass = 0;
const failures = [];
const check = (ok, msg) => { if (ok) pass++; else failures.push(msg); };

check(markerFillColor('topo', 'dark') === '#2d5d2c'
      && markerFillColor('osm', 'light') === '#8fbf8e',
      'OSM/topo use the default dark/light greens');
check(markerFillColor('satellite', 'dark') === SATELLITE_MARKER_COLORS.dark
      && markerFillColor('satellite', 'light') === SATELLITE_MARKER_COLORS.light,
      'satellite uses dark yellow and pale straw');

const topoStyle = semanticMarkerStyle('topo', 'dark', { fillColor: '#17613a' });
check(topoStyle.fillColor === '#17613a'
      && topoStyle.abiesMarkerTone === 'dark'
      && topoStyle.abiesStandardFillColor === '#17613a',
      'semantic style preserves and records an app-specific standard green');
const satelliteStyle = semanticMarkerStyle(
  'satellite', 'dark', { fillColor: '#17613a' },
);
check(satelliteStyle.fillColor === SATELLITE_MARKER_COLORS.dark,
      'semantic style substitutes the satellite color');

function vector(style) {
  return {
    options: { ...style },
    setStyle(update) { Object.assign(this.options, update); },
  };
}
const dark = vector(satelliteStyle);
const light = vector(semanticMarkerStyle(
  'satellite', 'light', { fillColor: '#9abc9a' },
));
const categorical = vector({ fillColor: '#cc33aa' });
const nested = {
  eachLayer(callback) {
    [dark, { eachLayer: cb => [light, categorical].forEach(cb) }]
      .forEach(callback);
  },
};
refreshSemanticMarkers(nested, 'topo');
check(dark.options.fillColor === '#17613a'
      && light.options.fillColor === '#9abc9a',
      'recursive refresh restores each semantic marker\'s exact standard color');
check(categorical.options.fillColor === '#cc33aa',
      'recursive refresh leaves categorical markers unchanged');

console.log('map-palette.js');
if (failures.length) {
  for (const failure of failures) console.error(`FAIL ${failure}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
