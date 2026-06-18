// ipso orientation map - shared Abies map wrapper plus Ipso layers.
//
// Basemap creation and the chooser control come from base/js/map-common.js.
// This file only owns Ipso-specific parcel, GPS, and local-record overlays.
'use strict';

function createOrientationMap(opts) {
  const elementId = opts.elementId;
  const formatFeatureLabel = opts.formatFeatureLabel;
  const featureName = opts.featureName;
  const getActiveName = opts.getActiveName;
  const getManualName = opts.getManualName;
  const formatRecordLabel = opts.formatRecordLabel;
  const formatPaiLabel = opts.formatPaiLabel;
  const paiControlTitle = opts.paiControlTitle || 'PAI';
  const onFeatureClick = opts.onFeatureClick;

  let initPromise = null;
  let wrapper = null;
  let leaflet = null;
  let parcelsLayer = null;
  let recordsLayer = null;
  let paiLayer = null;
  let positionLayer = null;
  let paiControlEl = null;
  let paiEnabled = false;
  let paiVisible = true;

  async function ensure() {
    if (leaflet) return;
    if (!initPromise) initPromise = init();
    await initPromise;
  }

  async function init() {
    if (typeof L === 'undefined') throw new Error('Leaflet not loaded');
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
      parcelsLayer = L.geoJSON(null, {
        style: featureStyle,
        onEachFeature: bindFeature,
      }).addTo(leaflet);
      recordsLayer = L.layerGroup().addTo(leaflet);
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
    try { return localStorage.getItem('ipso.basemap') || 'satellite'; }
    catch (_) { return 'satellite'; }
  }

  function writeBasemap(name) {
    try { localStorage.setItem('ipso.basemap', name); } catch (_) {}
  }

  function ready() { return !!leaflet; }

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

  function renderPai(records, enabled) {
    paiEnabled = !!enabled;
    if (!paiLayer) return;
    paiLayer.clearLayers();
    if (paiEnabled && records && records.length) {
      for (const rec of records) addPaiMarker(rec);
    }
    updatePaiLayerVisibility();
  }

  function addPaiMarker(rec) {
    const lat = Number(rec && rec.lat);
    const lon = Number(rec && rec.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const marker = L.circleMarker([lat, lon], {
      radius: 6,
      color: '#4d2f8f',
      weight: 2,
      fillColor: '#ffffff',
      fillOpacity: 0.95,
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
    if (!positionLayer) return;
    positionLayer.clearLayers();
    if (!fix) return;
    const latlng = [fix.lat, fix.lon];
    const radius = Math.max(3, Math.round(fix.acc || 0));
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
  }

  function center(context) {
    if (!leaflet) return false;
    const fix = context && context.fix;
    if (fix) {
      leaflet.setView([fix.lat, fix.lon], Math.max(leaflet.getZoom(), 17));
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
    ensure, ready, renderParcels, renderRecords, renderPai, updatePosition,
    center, invalidate,
  };
}

if (typeof module !== 'undefined') module.exports = { createOrientationMap };
