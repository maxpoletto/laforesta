/**
 * Opt-in Leaflet map tools, layered on the MapCommon shim.  Each `attach*`
 * function adds its control + handlers to a Leaflet map and returns a teardown
 * handle `{ destroy() }`.  A tool that grabs map clicks also exposes
 * `capturesClicks()` (the measure tool does), so a host that listens for clicks
 * can stand down while the tool is active.  The tools know nothing about
 * ParcelMap and take a raw Leaflet map.
 *
 * All user-facing labels come from strings.js; distances are formatted through
 * format.js (locale-aware), never `.toFixed()`.  Leaflet is the runtime global
 * `L`.
 */
import * as S from './strings.js';
import { fmtDecimal1, fmtDecimal2, fmtInt } from './format.js';

const $ = id => document.getElementById(id);

const KM_THRESHOLD_M = 1000;          // switch from metres to kilometres here
const SIDEBAR_RESIZE_MS = 300;        // wait out the sidebar CSS transition

const MEASURE_ICON = '📏';
const LOCATION_ICON = '📍';
const SIDEBAR_ICON = '☰';

const MEASURE_POINT_STYLE = {
  radius: 4, fillColor: '#ff0000', color: '#fff', weight: 2, opacity: 1,
  fillOpacity: 0.8,
};
const MEASURE_LINE_STYLE = { color: '#ff0000', weight: 2, opacity: 0.7 };

const LOCATION_COLOR = '#4CAF50';
const LOCATION_MARKER_STYLE = {
  radius: 8, fillColor: LOCATION_COLOR, color: '#fff', weight: 2,
  opacity: 1, fillOpacity: 1,
};
// Accuracy circle; `radius` (in metres) is filled in per fix.
const LOCATION_CIRCLE_STYLE = {
  color: LOCATION_COLOR, fillColor: LOCATION_COLOR, fillOpacity: 0.15, weight: 2,
};

/** Human-readable distance, locale-aware ("12,3 m" / "1,23 km" in Italian). */
export function formatDistance(meters) {
  return meters >= KM_THRESHOLD_M
    ? S.MAP_DISTANCE_KM.replace('{d}', fmtDecimal2(meters / 1000))
    : S.MAP_DISTANCE_M.replace('{d}', fmtDecimal1(meters));
}

/**
 * A single-icon button in its own leaflet-bar control.  `onClick` receives the
 * button element so stateful tools can toggle its active class without a
 * (page-unique) id lookup — which matters when two maps share a page.
 */
function iconButtonControl(icon, title, onClick) {
  return L.Control.extend({
    options: { position: 'topleft' },
    onAdd() {
      const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control');
      // A custom Leaflet control must stop its own pointer events from reaching
      // the map; otherwise a tap on the button gets entangled with the map's
      // click/drag handling and the button doesn't toggle until the next map
      // interaction.  (The basemap switcher control does the same.)
      L.DomEvent.disableClickPropagation(container);
      const btn = L.DomUtil.create('a', 'mc-control-button', container);
      btn.href = '#';
      btn.title = title;
      btn.textContent = icon;   // plain-text glyph, no markup
      L.DomEvent.on(container, 'click', (e) => {
        L.DomEvent.stopPropagation(e);
        L.DomEvent.preventDefault(e);
        onClick(btn);
      });
      return container;
    },
  });
}

/** Distance-measuring tool: tap points, see the running total. */
export function attachMeasure(leaflet) {
  let active = false;
  let points = [];
  let layer = null;

  function clear() {
    points = [];
    if (layer) { leaflet.removeLayer(layer); layer = null; }
  }

  function toggle(btn) {
    active = !active;
    btn.classList.toggle('mc-active', active);
    leaflet.getContainer().style.cursor = active ? 'crosshair' : '';
    if (!active) clear();
  }

  function addPoint(latlng) {
    if (!active) return;
    points.push(latlng);
    if (!layer) layer = L.layerGroup().addTo(leaflet);
    L.circleMarker(latlng, MEASURE_POINT_STYLE).addTo(layer);
    if (points.length > 1) {
      L.polyline(points, MEASURE_LINE_STYLE).addTo(layer);
      let total = 0;
      for (let i = 1; i < points.length; i++) {
        total += leaflet.distance(points[i - 1], points[i]);
      }
      L.marker(latlng, {
        icon: L.divIcon({
          className: 'mc-measure-label',
          html: `<div class="mc-measure-label-content">${formatDistance(total)}</div>`,
          iconSize: null,
        }),
      }).addTo(layer);
    }
  }

  const control = new (iconButtonControl(MEASURE_ICON, S.MAP_MEASURE_TITLE, toggle))();
  leaflet.addControl(control);
  const onMapClick = (e) => addPoint(e.latlng);
  leaflet.on('click', onMapClick);

  return {
    capturesClicks: () => active,
    destroy() {
      leaflet.off('click', onMapClick);
      leaflet.removeControl(control);
      clear();
    },
  };
}

/** Geolocation tool: a pulsing accuracy circle + marker, live-tracked. */
export function attachLocation(leaflet) {
  let active = false;
  let button = null;
  let marker = null;
  let circle = null;

  function clearGraphics() {
    if (marker) { leaflet.removeLayer(marker); marker = null; }
    if (circle) { leaflet.removeLayer(circle); circle = null; }
  }

  function stop() {
    active = false;
    button?.classList.remove('mc-active');
    leaflet.stopLocate();
    clearGraphics();
  }

  function toggle(btn) {
    button = btn;
    active = !active;
    if (active) {
      btn.classList.add('mc-active');
      leaflet.locate({ watch: true, enableHighAccuracy: true });
    } else {
      stop();
    }
  }

  function onFound(e) {
    clearGraphics();
    circle = L.circle(e.latlng, { ...LOCATION_CIRCLE_STYLE, radius: e.accuracy })
      .addTo(leaflet);
    marker = L.circleMarker(e.latlng, LOCATION_MARKER_STYLE).addTo(leaflet);
    marker.bindTooltip(
      `<b>${S.MAP_LOCATION_CURRENT}</b><br>${
        S.MAP_LOCATION_ACCURACY.replace('{m}', fmtInt(Math.round(e.accuracy)))}`,
      { permanent: false, direction: 'top' },
    );
  }

  function onError(e) {
    alert(S.MAP_LOCATION_ERROR.replace('{msg}', e.message));
    stop();
  }

  const control = new (iconButtonControl(LOCATION_ICON, S.MAP_LOCATION_TITLE, toggle))();
  leaflet.addControl(control);
  leaflet.on('locationfound', onFound);
  leaflet.on('locationerror', onError);

  return {
    destroy() {
      leaflet.off('locationfound', onFound);
      leaflet.off('locationerror', onError);
      leaflet.removeControl(control);
      stop();
    },
  };
}

/**
 * Sidebar show/hide toggle (sidebar pages only, e.g. Bosco).  Toggles the
 * `hidden` class on the sidebar + map elements and re-measures the map once the
 * CSS transition settles.  Takes the element ids explicitly so it never reaches
 * for a page-global `#map`.
 */
export function attachSidebarToggle(leaflet, { sidebarId = 'sidebar', mapId = 'map' } = {}) {
  function toggle() {
    const sidebar = $(sidebarId);
    const mapContainer = $(mapId);
    if (!sidebar || !mapContainer) return;
    const isHidden = sidebar.classList.toggle('hidden');
    mapContainer.classList.toggle('sidebar-hidden', isHidden);
    setTimeout(() => leaflet.invalidateSize({ pan: false }), SIDEBAR_RESIZE_MS);
  }

  const control = new (iconButtonControl(SIDEBAR_ICON, S.MAP_SIDEBAR_TITLE, toggle))();
  leaflet.addControl(control);

  return {
    destroy() { leaflet.removeControl(control); },
  };
}
