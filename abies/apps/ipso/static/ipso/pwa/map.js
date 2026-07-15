// ipso orientation map - shared Abies map wrapper plus Ipso layers.
//
// Basemap creation and the chooser control come from base/js/map-common.js.
// This file only owns Ipso-specific parcel, GPS, and local-record overlays.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof S === 'undefined') {
  Object.assign(globalThis, require('./strings.js'));
}

const PAI_PANE = 'ipsoPaiPane';
const PAI_PANE_Z_INDEX = 625;

function createOrientationMap(opts) {
  const elementId = opts.elementId;
  const formatFeatureLabel = opts.formatFeatureLabel;
  const featureName = opts.featureName;
  const getActiveName = opts.getActiveName;
  const getManualName = opts.getManualName;
  const formatRecordLabel = opts.formatRecordLabel;
  const formatPaiLabel = opts.formatPaiLabel;
  const formatSampleAreaLabel = opts.formatSampleAreaLabel;
  const sampleAreaDefaultRadius = opts.sampleAreaDefaultRadius;
  const paiControlTitle = opts.paiControlTitle || S.MODE_PAI;
  const onFeatureClick = opts.onFeatureClick;

  let initPromise = null;
  let wrapper = null;
  let leaflet = null;
  let parcelsLayer = null;
  let recordsLayer = null;
  let sampleAreasLayer = null;
  let paiLayer = null;
  let positionLayer = null;
  let paiControlEl = null;
  let paiEnabled = false;
  let paiVisible = true;
  let currentFix = null;
  let currentHeading = null;

  async function ensure() {
    if (leaflet) return;
    if (!initPromise) initPromise = init();
    await initPromise;
  }

  async function init() {
    if (typeof L === 'undefined') throw new Error(S.MAP_ERROR_LEAFLET_MISSING);
    try {
      const mod = await import('/static/base/js/map-common.js');
      const MapCommon = mod.default || mod.MapCommon || mod;
      wrapper = MapCommon.create(elementId, {
        basemap: readBasemap(),
        leafletOptions: { preferCanvas: true, zoomControl: false },
      });
      leaflet = wrapper.getLeafletMap();
      leaflet.setView([38.6, 16.3], 10);
      leaflet.on('basemapchange', (e) => writeBasemap(e.name));
      L.control.scale({ metric: true, imperial: false }).addTo(leaflet);
      setupPaiPane();
      parcelsLayer = L.geoJSON(null, {
        style: featureStyle,
        onEachFeature: bindFeature,
      }).addTo(leaflet);
      recordsLayer = L.layerGroup().addTo(leaflet);
      sampleAreasLayer = L.layerGroup().addTo(leaflet);
      paiLayer = L.layerGroup().addTo(leaflet);
      positionLayer = L.layerGroup().addTo(leaflet);
      setupPaiControl();
      updatePaiLayerVisibility();
    } catch (e) {
      initPromise = null;
      throw e;
    }
  }

  function readBasemap() {
    try { return localStorage.getItem(IPSO_BASEMAP_STORAGE_KEY) || 'satellite'; }
    catch (_) { return 'satellite'; }
  }

  function writeBasemap(name) {
    try { localStorage.setItem(IPSO_BASEMAP_STORAGE_KEY, name); } catch (_) {}
  }

  function ready() { return !!leaflet; }

  function setupPaiPane() {
    const pane = leaflet.getPane(PAI_PANE) || leaflet.createPane(PAI_PANE);
    pane.style.zIndex = String(PAI_PANE_Z_INDEX);
  }

  function renderParcels(features) {
    if (!parcelsLayer) return;
    parcelsLayer.clearLayers();
    if (!features || !features.length) return;
    parcelsLayer.addData({ type: 'FeatureCollection', features });
  }

  function featureStyle(feature) {
    const name = featureName(feature);
    const activeName = getActiveName();
    const manualName = getManualName();
    const active = name && name === activeName;
    const manual = name && manualName && name === manualName;
    return {
      color: active ? '#1f5b1a' : manual ? '#c52727' : '#2e8b27',
      weight: active || manual ? 3 : 1,
      dashArray: manual && !active ? '5 4' : null,
      fillColor: active ? '#d6e8d4' : manual ? '#f2d6d6' : '#ffffff',
      fillOpacity: active || manual ? 0.45 : 0.18,
    };
  }

  function bindFeature(feature, layer) {
    const label = formatFeatureLabel(feature);
    if (label) layer.bindTooltip(label, { sticky: true });
    layer.on('click', () => {
      if (label && layer.openTooltip) layer.openTooltip();
      if (label && onFeatureClick) onFeatureClick(label, feature);
    });
  }

  function renderRecords(records) {
    if (!recordsLayer) return;
    recordsLayer.clearLayers();
    if (!records || !records.length) return;
    for (const rec of records) {
      const lat = Number(rec && rec.lat);
      const lon = Number(rec && rec.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      const marker = L.circleMarker([lat, lon], {
        radius: 5,
        color: '#1f5b1a',
        weight: 2,
        fillColor: '#d6a02a',
        fillOpacity: 0.9,
      }).addTo(recordsLayer);
      const label = formatRecordLabel ? formatRecordLabel(rec) : '';
      if (label) marker.bindTooltip(label, { sticky: true });
    }
  }

  function renderSampleAreas(areas, enabled) {
    if (!sampleAreasLayer) return;
    sampleAreasLayer.clearLayers();
    if (!enabled || !areas || !areas.length) return;
    for (const area of areas) addSampleArea(area);
  }

  function addSampleArea(area) {
    const lat = Number(area && area.lat);
    const lon = Number(area && area.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const radius = Number.isFinite(area.r_m) ? area.r_m : sampleAreaDefaultRadius;
    const circle = L.circle([lat, lon], {
      radius,
      color: '#215f9a',
      weight: 2,
      fillColor: '#d7e8f7',
      fillOpacity: 0.28,
    }).addTo(sampleAreasLayer);
    const marker = L.circleMarker([lat, lon], {
      radius: 4,
      color: '#215f9a',
      weight: 2,
      fillColor: '#ffffff',
      fillOpacity: 1,
    }).addTo(sampleAreasLayer);
    const label = formatSampleAreaLabel ? formatSampleAreaLabel(area) : '';
    if (label) {
      circle.bindTooltip(label, { sticky: true });
      marker.bindTooltip(label, { sticky: true });
    }
  }

  function renderPai(records, enabled, speciesColors) {
    paiEnabled = !!enabled;
    if (!paiLayer) return;
    paiLayer.clearLayers();
    if (paiEnabled && records && records.length) {
      for (const rec of records) addPaiMarker(rec, speciesColors);
    }
    updatePaiLayerVisibility();
  }

  function addPaiMarker(rec, speciesColors) {
    const lat = Number(rec && rec.lat);
    const lon = Number(rec && rec.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const marker = L.circleMarker([lat, lon], {
      ...IpsoPalette.paiMarkerStyle(rec && rec[FIELD_SPECIES_ID], speciesColors),
      pane: PAI_PANE,
    }).addTo(paiLayer);
    const label = formatPaiLabel ? formatPaiLabel(rec) : '';
    if (label) marker.bindTooltip(label, { sticky: true });
  }

  function setupPaiControl() {
    const Control = L.Control.extend({
      options: { position: 'topleft' },
      onAdd: function() {
        const c = L.DomUtil.create('div', 'leaflet-bar ipso-pai-control');
        const btn = L.DomUtil.create('button', 'ipso-pai-toggle', c);
        btn.type = 'button';
        btn.title = paiControlTitle;
        btn.setAttribute('aria-label', paiControlTitle);
        btn.innerHTML = '&#128065;';
        L.DomEvent.disableClickPropagation(c);
        L.DomEvent.on(btn, 'click', (e) => {
          L.DomEvent.stop(e);
          paiVisible = !paiVisible;
          updatePaiLayerVisibility();
        });
        paiControlEl = c;
        return c;
      },
    });
    leaflet.addControl(new Control());
  }

  function updatePaiLayerVisibility() {
    if (!leaflet || !paiLayer) return;
    const showControl = !!paiEnabled;
    if (paiControlEl) paiControlEl.style.display = showControl ? '' : 'none';
    if (showControl && paiVisible) {
      if (!leaflet.hasLayer(paiLayer)) paiLayer.addTo(leaflet);
    } else if (leaflet.hasLayer(paiLayer)) {
      leaflet.removeLayer(paiLayer);
    }
    if (paiControlEl) {
      paiControlEl.classList.toggle('off', !paiVisible || !showControl);
    }
  }

  function updatePosition(fix) {
    currentFix = fix || null;
    renderPosition();
  }

  function updateHeading(heading) {
    currentHeading = Number.isFinite(heading) ? heading : null;
    renderPosition();
  }

  function renderPosition() {
    if (!positionLayer) return;
    positionLayer.clearLayers();
    if (!currentFix) return;
    const latlng = [currentFix.lat, currentFix.lon];
    const radius = Math.max(3, Math.round(currentFix.acc || 0));
    L.circle(latlng, {
      radius,
      color: '#1f5b1a',
      weight: 1,
      fillColor: '#2e8b27',
      fillOpacity: 0.12,
    }).addTo(positionLayer);
    L.circleMarker(latlng, {
      radius: 7,
      color: '#ffffff',
      weight: 2,
      fillColor: '#1f5b1a',
      fillOpacity: 1,
    }).addTo(positionLayer);
    const heading = Number.isFinite(currentHeading)
      ? currentHeading
      : currentFix && Number.isFinite(currentFix.heading)
        ? currentFix.heading : null;
    if (Number.isFinite(heading)) addHeadingMarker(latlng, heading);
  }

  function addHeadingMarker(latlng, heading) {
    const marker = L.marker(latlng, {
      interactive: false,
      icon: L.divIcon({
        className: 'ipso-position-heading-icon',
        html: '<span></span>',
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      }),
      zIndexOffset: 1000,
    }).addTo(positionLayer);
    const el = marker.getElement && marker.getElement();
    if (el) el.style.setProperty('--ipso-heading', heading + 'deg');
  }

  function center(context) {
    if (!leaflet) return false;
    const fix = context && context.fix;
    if (fix) {
      leaflet.setView([fix.lat, fix.lon], Math.max(leaflet.getZoom(), 17));
      return true;
    }
    const sampleArea = context && context.sampleArea;
    if (sampleArea && Number.isFinite(sampleArea.lat) && Number.isFinite(sampleArea.lon)) {
      leaflet.setView([sampleArea.lat, sampleArea.lon], Math.max(leaflet.getZoom(), 17));
      return true;
    }
    const committedFeature = context && context.committedFeature;
    if (committedFeature) return fitFeature(committedFeature);
    if (parcelsLayer && parcelsLayer.getLayers().length > 0) {
      const bounds = parcelsLayer.getBounds();
      if (bounds.isValid()) {
        leaflet.fitBounds(bounds, { padding: [20, 20], maxZoom: 16 });
        return true;
      }
    }
    return false;
  }

  function fitFeature(feature) {
    if (!leaflet || !feature) return false;
    const bounds = L.geoJSON(feature).getBounds();
    if (!bounds.isValid()) return false;
    leaflet.fitBounds(bounds, { padding: [20, 20], maxZoom: 18 });
    return true;
  }

  function invalidate() {
    if (leaflet) leaflet.invalidateSize({ pan: false });
  }

  return {
    ensure, ready, renderParcels, renderRecords, renderSampleAreas, renderPai,
    updatePosition, updateHeading, center, invalidate,
  };
}

if (typeof module !== 'undefined') module.exports = { createOrientationMap };
