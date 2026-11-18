import { FIT_PADDING, PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { parcelIdentity } from '../../base/js/geo.js';
import { fmtDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

const drawingStyle = { color: '#ad4c00', weight: 4 };
const roadStyle = { color: '#244c86', weight: 4 };
const selectedStyle = { color: '#db9500', weight: 6 };

function fixLabel(fix) {
  const label = document.createElement('span');
  label.textContent = [S.MAP_LOCATION_ACCURACY_LABEL(fmtDecimal(fix[0], 1)), fix[1]].filter(Boolean).join(' · ');
  return label;
}

export class CartographyMap {
  constructor(container, parcels, regionName, onSelect, basemap) {
    const features = parcels.features.filter(f => parcelIdentity(f).regionName === regionName);
    this.parcels = new ParcelMap({
      container, geojson: {type: 'FeatureCollection', features}, basemap,
      onMapClick: (_, feature) => onSelect(feature ? 'parcel' : null, feature ? parcelIdentity(feature).parcelName : null),
      className: 'map-workspace-canvas',
      tools: {measure: true, location: true, sidebar: {sidebarId: 'cartography-sidebar', mapId: 'cartography-map'}},
    });
    this.parcelLayers = new Map();
    this.parcels.parcelLayer.eachLayer(layer => {
      const {parcelName} = parcelIdentity(layer.feature);
      if (!this.parcelLayers.has(parcelName)) this.parcelLayers.set(parcelName, []);
      this.parcelLayers.get(parcelName).push(layer);
    });
    this.map = this.parcels.leaflet;
    if (!features.length) this.map.fitWorld();
    this.onSelect = onSelect;
    this.objects = L.featureGroup().addTo(this.map);
    this.handles = L.layerGroup().addTo(this.map);
    this.layers = new Map();
  }

  render(state) {
    this.handles.clearLayers();
    this.objects.clearLayers();
    this.layers.clear();
    for (const [kind, rows] of [['road', state.roads], ['drawing', state.drawings]]) {
      for (const row of rows) {
        const style = kind === 'road' ? roadStyle : drawingStyle;
        const layer = L.geoJSON(row.geometry, {
          style, bubblingMouseEvents: false,
          pointToLayer: (_, ll) => L.circleMarker(ll, {...style, radius: 5, bubblingMouseEvents: false}),
        }).getLayers()[0];
        layer.addTo(this.objects);
        if (row.geometry.type === 'Point') layer.bindTooltip(fixLabel(row.fixes[0]));
        layer.on('click', () => this.onSelect(kind, row.id));
        this.layers.set(kind + row.id, layer);
      }
    }
  }

  bounds(layers) {
    const bounds = L.latLngBounds([]);
    for (const layer of layers) bounds.extend(layer.getBounds ? layer.getBounds() : layer.getLatLng());
    return bounds;
  }

  frame(kind, id) {
    const layers = kind === 'parcel' ? this.parcelLayers.get(id) : [this.layers.get(kind + id)];
    this.fit(this.bounds(layers));
  }

  fitSource(drawings, sourceId, firstImport = false) {
    const layers = drawings.filter(row => row.source_id === sourceId).map(row => this.layers.get('drawing' + row.id));
    this.fit(this.bounds(layers), firstImport);
  }

  fit(bounds, force = false) {
    if (bounds.isValid() && (force || !this.map.getBounds().contains(bounds))) {
      this.map.fitBounds(bounds, {padding: FIT_PADDING, maxZoom: 19});
    }
  }

  fitRegion() {
    const bounds = this.objects.getBounds().extend(this.parcels.parcelLayer.getBounds());
    if (bounds.isValid()) this.map.fitBounds(bounds, {padding: FIT_PADDING, maxZoom: 19});
  }

  select(kind, row, edit = false) {
    this.stopEdit();
    this.handles.clearLayers();
    for (const [key, layer] of this.layers) {
      layer.setStyle(key === kind + row?.id ? selectedStyle : key.startsWith('road') ? roadStyle : drawingStyle);
    }
    for (const [name, pieces] of this.parcelLayers) {
      for (const piece of pieces) piece.setStyle(kind === 'parcel' && name === row?.id ? selectedStyle : PARCEL_STYLE);
    }
    if (!row || kind === 'parcel') return;
    const layer = this.layers.get(kind + row.id);
    layer.bringToFront();
    if (kind !== 'drawing') return;
    const line = row.geometry.type === 'LineString';
    const coordinates = line ? row.geometry.coordinates : [row.geometry.coordinates];
    this.positions = coordinates.map(([lon, lat]) => L.latLng(lat, lon));
    this.refreshHandles = () => {
      this.handles.clearLayers();
      const visible = this.map.getBounds().pad(0.1);
      this.positions.forEach((ll, i) => {
        if (!visible.contains(ll)) return;
        const marker = edit ? L.marker(ll, {
          draggable: true, icon: L.divIcon({className: 'cartography-vertex', iconSize: [15, 15]}),
        }) : L.circleMarker(ll, {radius: 3, color: drawingStyle.color, bubblingMouseEvents: false});
        marker.addTo(this.handles).bindTooltip(fixLabel(row.fixes[i]));
        marker.on('click', () => marker.openTooltip());
        if (edit) marker.on('drag', () => {
          this.positions[i] = marker.getLatLng();
          if (line) layer.setLatLngs(this.positions);
          else layer.setLatLng(this.positions[i]);
        });
      });
    };
    this.refreshHandles();
    this.map.on('moveend', this.refreshHandles);
  }

  geometry(type) {
    const coords = this.positions.map(ll => [ll.lng, ll.lat]);
    return {type, coordinates: type === 'Point' ? coords[0] : coords};
  }

  stopEdit() {
    if (this.refreshHandles) this.map.off('moveend', this.refreshHandles);
    this.refreshHandles = null;
  }

  destroy() {
    this.stopEdit();
    this.parcels.destroy();
  }
}
