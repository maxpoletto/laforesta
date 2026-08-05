let pass = 0;
const failures = [];
const check = (ok, msg) => { if (ok) pass++; else failures.push(msg); };

function classList() {
  const values = new Set();
  return {
    toggle(name, enabled) {
      const add = enabled === undefined ? !values.has(name) : enabled;
      if (add) values.add(name); else values.delete(name);
      return add;
    },
    contains: name => values.has(name),
  };
}

function element(className = '') {
  const properties = {};
  return {
    className,
    classList: classList(),
    style: {
      properties,
      setProperty(name, value) { properties[name] = value; },
    },
    appendChild(child) { (this.children ||= []).push(child); },
  };
}

let leaflet = null;
global.document = {
  documentElement: { lang: 'it' },
  getElementById: () => null,
};
global.L = {
  map() {
    const handlers = {};
    leaflet = {
      container: element(),
      on(name, callback) { (handlers[name] ||= []).push(callback); return this; },
      fire(name, detail) { for (const callback of handlers[name] || []) callback(detail); },
      getContainer() { return this.container; },
      removeLayer(layer) { this.removedLayer = layer; },
      addControl(control) { control.onAdd?.(this); },
    };
    return leaflet;
  },
  tileLayer(url) { return { url, addTo(map) { this.map = map; return this; } }; },
  control: { zoom: () => ({ addTo() {} }) },
  Control: {
    extend(proto) {
      return class {
        constructor() {
          Object.assign(this, proto);
          this.initialize?.();
        }
      };
    },
  },
  DomUtil: {
    create(tag, className, parent) {
      const child = element(className);
      parent?.appendChild(child);
      return child;
    },
  },
  DomEvent: {
    disableClickPropagation() {}, on() {}, stop() {},
  },
};

const { default: MapCommon } = await import('./map-common.js');
const wrapper = MapCommon.create('map', { basemap: 'satellite' });
check(leaflet.container.classList.contains('mc-basemap-satellite'),
      'initial satellite basemap marks the map container');
check(leaflet.container.style.properties['--mc-semantic-marker-dark'] === '#d6a800'
      && leaflet.container.style.properties['--mc-semantic-marker-light'] === '#f0dda0',
      'initial basemap exposes both semantic colors to HTML markers');

const styles = [];
const userChanges = [];
leaflet.on('basemapstylechange', event => styles.push(event.name));
leaflet.on('basemapchange', event => userChanges.push(event.name));
wrapper.syncBasemap('topo');
check(wrapper.getBasemap() === 'topo'
      && !leaflet.container.classList.contains('mc-basemap-satellite')
      && leaflet.container.style.properties['--mc-semantic-marker-dark'] === '#2d5d2c',
      'synchronized topo change updates the basemap and container class');
check(JSON.stringify(styles) === JSON.stringify(['topo']) && userChanges.length === 0,
      'synchronized change fires style event without the user-sync event');

wrapper.setBasemap('satellite');
wrapper.setBasemap('satellite');
check(leaflet.container.classList.contains('mc-basemap-satellite')
      && JSON.stringify(styles) === JSON.stringify(['topo', 'satellite']),
      'direct change fires one style event and same-basemap calls do no work');

console.log('map-common.js');
if (failures.length) {
  for (const failure of failures) console.error(`FAIL ${failure}`);
  console.log(`\n${pass} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`\n${pass} passed, 0 failed`);
