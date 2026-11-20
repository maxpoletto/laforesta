import { FIT_PADDING, PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { parcelIdentity } from '../../base/js/geo.js';
import { fmtDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

import { handPart, isEndpoint, rings } from './drawing-parts.js';

const drawingStyle = { color: '#ad4c00', weight: 4 };
const roadStyle = { color: '#244c86', weight: 4 };
const selectedStyle = { color: '#db9500', weight: 6 };

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
    if (kind === 'drawing') this.showVertexHandles(row.parts);
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
      layer.on('click', e => {
        if (draft.tool) callbacks.place(e.latlng);
        else if (!draft.join) callbacks.selectPart(i);
      });
    });
    if (draft.tool) {
      this.drawSketch(draft, callbacks);
    } else if (draft.join) {
      this.watchHandles(() => this.drawEndpoints(draft, callbacks.endpoint));
    } else if (draft.parts[draft.active]) {
      this.showVertexHandles([draft.parts[draft.active]], callbacks);
    }
  }

  drawSketch(draft, callbacks) {
    const {tool, vertices} = draft;
    const path = handPart('LineString', vertices);
    if (!path) return;
    this.partLayer(path, selectedStyle).addTo(this.preview).on('click', e => callbacks.place(e.latlng));

    if (tool === 'Polygon') {
      const [lon, lat] = vertices[0];
      const canClose = vertices.length >= 3;
      const title = canClose ? S.CARTOGRAPHY_CLOSE_SKETCH : S.CARTOGRAPHY_FIRST_VERTEX;
      const button = document.createElement('button');
      button.type = 'button';
      button.title = title;
      button.setAttribute('aria-label', title);
      if (!canClose) {
        button.setAttribute('aria-disabled', 'true');
        button.tabIndex = -1;
      }
      button.addEventListener('click', event => {
        event.stopPropagation();
        if (canClose) callbacks.finish();
      });
      const icon = L.divIcon({
        className: canClose ? 'cartography-sketch-start can-close' : 'cartography-sketch-start',
        html: button,
        iconSize: [16, 16],
      });
      L.marker([lat, lon], {icon, keyboard: false, bubblingMouseEvents: false}).addTo(this.handles);
    }
  }

  drawEndpoints(draft, select) {
    const visible = this.map.getBounds().pad(0.1);
    draft.parts.forEach((part, i) => {
      if (part.type !== 'LineString') return;
      for (const index of [0, part.coordinates.length - 1]) {
        const [lon, lat] = part.coordinates[index];
        if (!visible.contains([lat, lon])) continue;
        const title = S.CARTOGRAPHY_JOIN_ENDPOINT(i, index === 0);
        const selected = draft.join.first?.part === i && draft.join.first.index === index;
        const icon = L.divIcon({
          className: selected ? 'cartography-endpoint cartography-active-vertex' : 'cartography-endpoint',
          iconSize: [19, 19],
        });
        // Split lines can have coincident endpoints. Put the selected one
        // behind its neighbors so they can be chosen by pointer as well.
        const marker = L.marker([lat, lon], {icon, title, zIndexOffset: selected ? -1000 : 0})
          .addTo(this.handles);
        const handle = marker.getElement();
        handle.dataset.part = i;
        handle.dataset.index = index;
        this.selectableHandle(marker, title, selected, () => select(i, index));
      }
    });
  }

  selectableHandle(marker, label, selected, select) {
    const handle = marker.getElement();
    handle.setAttribute('aria-label', label);
    handle.setAttribute('aria-pressed', String(selected));
    marker.on('click', select);
    handle.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      event.stopPropagation();
      select();
    });
  }

  watchHandles(render) {
    this.refreshHandles = () => {
      this.handles.clearLayers();
      render();
    };
    this.refreshHandles();
    this.map.on('moveend', this.refreshHandles);
  }

  showVertexHandles(parts, callbacks) {
    this.watchHandles(() => {
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
              if (part.type !== 'Point' && !isEndpoint(part, i)) {
                const handle = marker.getElement();
                const selected = callbacks.vertex?.ring === r && callbacks.vertex?.index === i;
                handle.classList.toggle('cartography-active-vertex', selected);
                const selectVertex = () => {
                  this.handles.eachLayer(layer => {
                    const element = layer.getElement();
                    if (!element?.hasAttribute('aria-pressed')) return;
                    const active = element === handle;
                    element.classList.toggle('cartography-active-vertex', active);
                    element.setAttribute('aria-pressed', String(active));
                  });
                  callbacks.vertex = {ring: r, index: i};
                  callbacks.selectVertex(r, i);
                };
                this.selectableHandle(marker, S.CARTOGRAPHY_SELECT_VERTEX, selected, selectVertex);
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
    });
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
