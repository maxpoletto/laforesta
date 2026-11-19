import { FIT_PADDING, PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { parcelIdentity } from '../../base/js/geo.js';
import { fmtDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

import { handPart, rings } from './drawing-parts.js';

const drawingStyle = { color: '#ad4c00', weight: 4 };
const roadStyle = { color: '#244c86', weight: 4 };
const selectedStyle = { color: '#db9500', weight: 6 };
const SNAP_RADIUS_PX = 10;

function fixLabel(fix) {
  const label = document.createElement('span');
  label.textContent = [fix[0] == null ? '' : S.MAP_LOCATION_ACCURACY_LABEL(fmtDecimal(fix[0], 1)), fix[1]].filter(Boolean).join(' · ');
  return label;
}

export class CartographyMap {
  constructor(container, parcels, regionName, onSelect, basemap, onPlace, tools, initialView) {
    const features = parcels.features.filter(f => parcelIdentity(f).regionName === regionName);
    this.parcels = new ParcelMap({
      container, geojson: {type: 'FeatureCollection', features}, basemap, initialView,
      onMapClick: (ll, feature) => {
        if (!onPlace(ll)) onSelect(feature ? 'parcel' : null, feature && parcelIdentity(feature).parcelName);
      },
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
    if (!features.length && !initialView) this.map.fitWorld();
    this.onSelect = onSelect;
    this.objects = L.featureGroup().addTo(this.map);
    this.handles = L.layerGroup().addTo(this.map);
    this.layers = new Map();
    this.preview = L.featureGroup().addTo(this.map);
    this.hiddenLayers = new Set();
    this.editedParcels = [];
    this.toolbar = L.control({position: 'topleft'});
    this.toolbar.onAdd = () => {
      L.DomEvent.disableClickPropagation(tools);
      tools.hidden = false;
      return tools;
    };
    this.toolbar.onRemove = () => {
      tools.hidden = true;
      container.append(tools);
    };
    this.toolbar.addTo(this.map);
  }

  isVisible(kind, id) {
    return !this.hiddenLayers.has(kind + id);
  }

  setVisible(kind, id, visible) {
    const key = kind + id;
    if (visible) this.hiddenLayers.delete(key);
    else this.hiddenLayers.add(key);
    const group = kind === 'parcel' ? this.parcels.parcelLayer : this.objects;
    const layers = kind === 'parcel' ? this.parcelLayers.get(id) || [] : [this.layers.get(key)];
    for (const layer of layers) {
      if (visible) group.addLayer(layer);
      else group.removeLayer(layer);
    }
  }

  render(state) {
    this.stopEdit();
    this.handles.clearLayers();
    this.preview.clearLayers();
    this.objects.clearLayers();
    this.layers.clear();
    for (const [kind, rows] of [['road', state.roads], ['drawing', state.drawings]]) {
      for (const row of rows) {
        const group = L.featureGroup();
        if (this.isVisible(kind, row.id)) group.addTo(this.objects);
        const parts = kind === 'drawing' ? row.parts : [row.geometry];
        for (const part of parts) {
          const layer = this.partLayer(part, kind === 'road' ? roadStyle : drawingStyle).addTo(group);
          if (part.type === 'Point' && part.fixes?.[0]) layer.bindTooltip(fixLabel(part.fixes[0]));
          layer.on('click', () => this.onSelect(kind, row.id));
        }
        this.layers.set(kind + row.id, group);
      }
    }
  }

  partLayer(part, style) {
    return L.geoJSON(part, {
      style: {...style, bubblingMouseEvents: false},
      pointToLayer: (_, ll) => L.circleMarker(ll, {...style, radius: 5, bubblingMouseEvents: false}),
    }).getLayers()[0];
  }

  bounds(layers) {
    const bounds = L.latLngBounds([]);
    for (const layer of layers) bounds.extend(layer.getBounds ? layer.getBounds() : layer.getLatLng());
    return bounds;
  }

  frame(kind, id) {
    const layers = kind === 'parcel' ? this.parcelLayers.get(id) || [] : [this.layers.get(kind + id)];
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

  select(kind, row) {
    this.showEditedParcels([]);
    this.stopEdit();
    this.handles.clearLayers();
    this.preview.clearLayers();
    for (const [key, layer] of this.layers) {
      if (!this.hiddenLayers.has(key)) this.objects.addLayer(layer);
      layer.setStyle(key === kind + row?.id ? selectedStyle : key.startsWith('road') ? roadStyle : drawingStyle);
    }
    for (const [name, pieces] of this.parcelLayers) {
      for (const piece of pieces) piece.setStyle(kind === 'parcel' && name === row?.id ? selectedStyle : PARCEL_STYLE);
    }
    if (!row || kind === 'parcel' || !this.isVisible(kind, row.id)) return;
    this.layers.get(kind + row.id).bringToFront();
    if (kind === 'drawing') this.watchHandles(row.parts);
  }

  nearestParcelVertex(name, point) {
    const location = this.map.latLngToContainerPoint([point[1], point[0]]);
    const nearby = this.map.getBounds().pad(0.1);
    let distance = SNAP_RADIUS_PX;
    let nearest = point;
    for (const [other, layers] of this.parcelLayers) {
      if (other === name || !this.isVisible('parcel', other)) continue;
      for (const layer of layers) {
        if (!layer.getBounds().intersects(nearby)) continue;
        const geometry = layer.feature.geometry;
        const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
        for (const coordinates of polygons) {
          for (const ring of coordinates) {
            for (const [lon, lat] of ring) {
              const vertex = L.latLng(lat, lon);
              if (!nearby.contains(vertex)) continue;
              const pixels = this.map.latLngToContainerPoint(vertex).distanceTo(location);
              if (pixels < distance) {
                nearest = [lon, lat];
                distance = pixels;
              }
            }
          }
        }
      }
    }
    return nearest;
  }

  showEditedParcels(names) {
    const layer = this.parcels.parcelLayer;
    for (const name of this.editedParcels) {
      if (this.isVisible('parcel', name)) {
        for (const piece of this.parcelLayers.get(name) || []) layer.addLayer(piece);
      }
    }
    for (const name of names) {
      for (const piece of this.parcelLayers.get(name) || []) layer.removeLayer(piece);
    }
    this.editedParcels = names;
  }

  edit(draft, callbacks) {
    this.stopEdit();
    this.handles.clearLayers();
    this.preview.clearLayers();
    if (draft.kind === 'parcel') {
      this.showEditedParcels([draft.id, ...draft.neighbors]);
      for (const feature of draft.features) {
        if (!draft.neighbors.includes(feature.name)) continue;
        for (const part of feature.parts) {
          this.partLayer(part, {...selectedStyle, weight: 3, dashArray: '6 4'}).addTo(this.preview);
        }
      }
    } else if (draft.id) {
      this.objects.removeLayer(this.layers.get((draft.kind || 'drawing') + draft.id));
    }

    draft.parts?.forEach((part, i) => {
      const layer = this.partLayer(part, i === draft.active ? selectedStyle : drawingStyle).addTo(this.preview);
      if (i === draft.active) this.activeLayer = layer;
      layer.on('click', e => draft.tool ? callbacks.place(e.latlng) : callbacks.selectPart(i));
    });
    if (draft.tool) {
      const part = handPart(draft.tool, draft.vertices);
      if (part) this.partLayer(part, selectedStyle).addTo(this.preview).on('click', e => callbacks.place(e.latlng));
    } else if (draft.parts[draft.active]) {
      this.watchHandles([draft.parts[draft.active]], callbacks);
    }
  }

  watchHandles(parts, callbacks) {
    this.refreshHandles = () => {
      this.handles.clearLayers();
      const visible = this.map.getBounds().pad(0.1);
      for (const part of parts) {
        let offset = 0;
        rings(part).forEach((ring, r) => {
          ring.forEach(([lon, lat], i) => {
            const ll = L.latLng(lat, lon);
            if (callbacks && i > 0) {
              const mid = L.latLng((lat + ring[i - 1][1]) / 2, (lon + ring[i - 1][0]) / 2);
              if (visible.contains(mid)) {
                const icon = L.divIcon({className: 'cartography-insert-vertex', iconSize: [10, 10]});
                L.marker(mid, {icon, title: S.CARTOGRAPHY_INSERT_VERTEX, bubblingMouseEvents: false})
                  .addTo(this.handles).on('click', () => callbacks.insert(r, i, [mid.lng, mid.lat]));
              }
            }
            if (!visible.contains(ll) || (part.type === 'Polygon' && i === ring.length - 1)) return;
            const fix = part.fixes?.[offset + i];
            if (!callbacks && !fix) return;
            const marker = callbacks ? L.marker(ll, {
              draggable: true, icon: L.divIcon({className: 'cartography-vertex', iconSize: [15, 15]}),
            }) : L.circleMarker(ll, {radius: 4, color: drawingStyle.color, bubblingMouseEvents: false});
            marker.addTo(this.handles);
            if (fix) marker.bindTooltip(fixLabel(fix)).on('click', () => marker.openTooltip());
            if (callbacks) {
              const selectVertex = () => {
                this.handles.eachLayer(handle => handle.getElement()?.classList.remove('cartography-active-vertex'));
                marker.getElement()?.classList.add('cartography-active-vertex');
                callbacks.vertex = i;
                callbacks.selectVertex(i);
              };
              if (part.type === 'LineString') {
                if (callbacks.vertex === i) marker.getElement()?.classList.add('cartography-active-vertex');
                marker.on('click', selectVertex);
              }
              marker.on('drag', () => {
                const layer = this.activeLayer;
                const end = marker.getLatLng();
                if (part.type === 'Point') layer.setLatLng(end);
                else {
                  const positions = layer.getLatLngs();
                  const ring = part.type === 'Polygon' ? positions[r] : positions;
                  ring[i] = end;
                  layer.setLatLngs(positions);
                }
              });
              marker.on('dragend', () => {
                const end = marker.getLatLng();
                callbacks.move(r, i, [end.lng, end.lat]);
              });
            }
          });
          offset += ring.length;
        });
      }
    };
    this.refreshHandles();
    this.map.on('moveend', this.refreshHandles);
  }

  stopEdit() {
    if (this.refreshHandles) this.map.off('moveend', this.refreshHandles);
    this.refreshHandles = null;
  }

  destroy() {
    this.stopEdit();
    this.toolbar.remove();
    this.parcels.destroy();
  }
}
