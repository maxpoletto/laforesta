import { FIT_PADDING, PARCEL_STYLE, ParcelMap } from '../../base/js/parcel-map.js';
import { geometryPolygons, parcelIdentity } from '../../base/js/geo.js';
import { fmtDecimal } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

import { handPart, isEndpoint, rings } from './drawing-parts.js';
import { boundaryTargetMode, referenceVertices, vertexKey } from './boundary-targets.js';

const drawingStyle = { color: '#ad4c00', weight: 4 };
const roadStyle = { color: '#244c86', weight: 4 };
const selectedStyle = { color: '#db9500', weight: 6 };
const SNAP_RADIUS = 12;
const MAX_REFERENCE_MARKERS = 250;

function fixLabel(fix) {
  const label = document.createElement('span');
  label.textContent = [fix[0] == null ? '' : S.MAP_LOCATION_ACCURACY_LABEL(fmtDecimal(fix[0], 1)), fix[1]].filter(Boolean).join(' · ');
  return label;
}

export class CartographyMap {
  constructor(container, parcels, regionName, onSelect, basemap, onPlace, controls, initialView) {
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
    // Editable parts have independent pointer targets above the base canvas.
    this.editRenderer = L.svg();
    this.hiddenLayers = new Set();
    this.editedParcels = [];
    this.toolbar = L.control({position: 'topleft'});
    this.toolbar.onAdd = () => {
      L.DomEvent.disableClickPropagation(controls);
      return controls;
    };
    this.toolbar.onRemove = () => {
      controls.querySelector('[data-role=map-tools]').hidden = true;
      container.append(controls);
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

  frame(kind, id, force = false) {
    const layers = kind === 'parcel' ? this.parcelLayers.get(id) || [] : [this.layers.get(kind + id)];
    this.fit(this.bounds(layers), force);
  }

  fitSource(drawings, sourceId, firstImport = false) {
    this.fitDrawings(drawings.filter(row => row.source_id === sourceId).map(row => row.id), firstImport);
  }

  fitDrawings(ids, firstImport = false) {
    const layers = ids.map(id => this.layers.get('drawing' + id)).filter(Boolean);
    this.fit(this.bounds(layers), firstImport);
  }

  fit(bounds, force = false) {
    if (bounds.isValid() && (force || !this.map.getBounds().contains(bounds))) {
      this.map.fitBounds(bounds, {padding: FIT_PADDING, maxZoom: 19});
    }
  }

  fitRegion() {
    const bounds = this.objects.getBounds().extend(this.parcels.parcelLayer.getBounds())
      .extend(this.preview.getBounds());
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
      const style = i === draft.active ? selectedStyle : drawingStyle;
      const layer = this.partLayer(part, {
        ...style, renderer: this.editRenderer, className: 'cartography-draft-part',
      }).addTo(this.preview);
      if (i === draft.active) this.activeLayer = layer;
      layer.on('click', e => {
        if (draft.tool) callbacks.place(e.latlng);
        else if (!draft.join && !draft.following) callbacks.selectPart(i);
      });
    });
    if (draft.tool) {
      this.drawSketch(draft, callbacks);
    } else if (draft.join) {
      this.watchHandles(() => this.drawEndpoints(draft, callbacks.endpoint));
    } else if (draft.parts[draft.active]) {
      this.showVertexHandles([draft.parts[draft.active]], callbacks, draft);
    }
  }

  drawSketch(draft, callbacks) {
    const {tool, vertices} = draft;
    if (draft.completedSketch) {
      this.partLayer(draft.completedSketch, selectedStyle).addTo(this.preview);
      return;
    }
    const path = handPart('LineString', vertices);
    if (path) this.partLayer(path, selectedStyle).addTo(this.preview).on('click', e => callbacks.place(e.latlng));
    if (tool === 'Polygon') {
      this.watchHandles(() => {
        this.drawBoundaryTargets(draft, callbacks);
        if (path) this.drawSketchStart(draft, callbacks.finish);
      });
    }
  }

  drawSketchStart(draft, finish) {
    const {vertices} = draft;
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
      if (canClose) finish();
    });
    const shared = this.referenceTargets.has(vertexKey(vertices[0])) ? ' cartography-shared-vertex' : '';
    const icon = L.divIcon({
      className: `cartography-sketch-start${canClose ? ' can-close' : ''}${shared}`,
      html: button,
      iconSize: [16, 16],
    });
    L.marker([lat, lon], {icon, keyboard: false, zIndexOffset: 1000, bubblingMouseEvents: false}).addTo(this.handles);
  }

  boundaryReferences(draft, visible, anchor) {
    const working = new Map();
    for (const feature of draft.features ?? []) {
      if (!working.has(feature.name)) working.set(feature.name, []);
      working.get(feature.name).push(...feature.parts);
    }
    const references = [];
    for (const [name, layers] of this.parcelLayers) {
      if ((draft.kind === 'parcel' && name === draft.id) || !this.isVisible('parcel', name)) continue;
      const parts = working.get(name) ?? layers.filter(layer => {
        const bounds = layer.getBounds();
        return bounds.intersects(visible) || (anchor && bounds.contains(anchor));
      }).map(layer => layer.feature.geometry);
      for (const part of parts) {
        for (const polygon of geometryPolygons(part)) {
          for (const ring of polygon) references.push({name, ring});
        }
      }
    }
    return references;
  }

  drawBoundaryTargets(draft, callbacks) {
    const bounds = this.map.getBounds();
    const anchor = callbacks.anchor();
    const anchorKey = anchor && vertexKey(anchor);
    const references = this.boundaryReferences(draft, bounds, anchor && [anchor[1], anchor[0]]);
    const west = bounds.getWest(), east = bounds.getEast(), south = bounds.getSouth(), north = bounds.getNorth();
    this.referenceTargets = referenceVertices(references, (lon, lat) =>
      (west <= lon && lon <= east && south <= lat && lat <= north) || (anchor && lon === anchor[0] && lat === anchor[1]));
    this.snapHandles = [];
    const from = this.referenceTargets.get(anchorKey);
    const occupied = new Set((draft.vertices ?? rings(draft.parts[draft.active]).flat()).map(vertexKey));
    let targets = [...this.referenceTargets].filter(([key, target]) => {
      if (!draft.following) return !occupied.has(key);
      return key !== anchorKey && from?.positions.some(start => target.positions.some(end => start.ring === end.ring));
    });
    if (targets.length > MAX_REFERENCE_MARKERS) {
      const center = this.map.getSize().divideBy(2);
      targets = targets.map(entry => {
        const [lon, lat] = entry[1].point;
        return {entry, distance: center.distanceTo(this.map.latLngToContainerPoint([lat, lon]))};
      }).sort((a, b) => a.distance - b.distance).slice(0, MAX_REFERENCE_MARKERS).map(({entry}) => entry);
    }
    for (const [, target] of targets) {
      const [lon, lat] = target.point;
      const names = [...new Set(target.positions.map(position => position.name))];
      const label = S.CARTOGRAPHY_BOUNDARY_TARGET(names.join(', '));
      const icon = L.divIcon({className: 'cartography-boundary-target', iconSize: [24, 24]});
      const marker = L.marker([lat, lon], {
        icon, title: label, zIndexOffset: draft.following ? 2000 : 0, bubblingMouseEvents: false,
      }).addTo(this.handles);
      let path;
      const getPath = () => draft.following ? (path ??= callbacks.trace(from, target)) : null;
      this.selectableHandle(marker, label, null, () => callbacks.snap(target.point, getPath()));
      if (draft.following) {
        marker.on('mouseover', () => this.showTrace(getPath()));
        marker.on('mouseout', () => this.showTrace(null));
        marker.getElement().addEventListener('focus', () => this.showTrace(getPath()));
        marker.getElement().addEventListener('blur', () => this.showTrace(null));
      }
      this.snapHandles.push({marker, point: target.point});
    }
    for (const [i, point] of (draft.vertices ?? []).entries()) {
      if (i === 0 || !this.referenceTargets.has(vertexKey(point))) continue;
      const available = !draft.following && !draft.traceReady && i === draft.vertices.length - 1;
      const icon = L.divIcon({className: 'cartography-shared-vertex cartography-sketch-vertex', iconSize: [15, 15]});
      const title = available ? S.CARTOGRAPHY_FOLLOW_FROM : '';
      const marker = L.marker([point[1], point[0]], {
        icon, title, interactive: available, keyboard: available, bubblingMouseEvents: false,
      }).addTo(this.handles);
      if (available) this.selectableHandle(marker, title, null, callbacks.rearm);
    }
  }

  showTrace(path) {
    if (this.tracePreview) this.preview.removeLayer(this.tracePreview);
    this.tracePreview = path ? L.polyline(path.map(([lon, lat]) => [lat, lon]), {
      ...selectedStyle, className: 'cartography-trace-preview', dashArray: '6 4', interactive: false,
      renderer: this.editRenderer,
    }).addTo(this.preview) : null;
  }

  snapNear(ll) {
    const position = this.map.latLngToLayerPoint(ll);
    let nearest = null, distance = SNAP_RADIUS;
    for (const handle of this.snapHandles) {
      const gap = position.distanceTo(this.map.latLngToLayerPoint(handle.marker.getLatLng()));
      if (gap > distance) continue;
      nearest = handle;
      distance = gap;
    }
    this.snapHover?.marker.getElement()?.classList.remove('cartography-target-hover');
    this.snapHover = nearest;
    nearest?.marker.getElement()?.classList.add('cartography-target-hover');
    return nearest?.point;
  }

  drawEndpoints(draft, select, visible = this.map.getBounds().pad(0.1)) {
    const joining = !!draft.join;
    const fixedParts = joining ? [] : draft.parts.filter((part, i) => i === draft.active || part.type === 'Point');
    const occupied = new Set(fixedParts.flatMap(part => rings(part).flat().map(vertexKey)));
    draft.parts.forEach((part, i) => {
      if (!isEndpoint(part, 0)) return;
      if (!joining && (i === draft.active || part.type !== 'LineString')) return;
      const [coordinates] = rings(part);
      for (const index of new Set([0, coordinates.length - 1])) {
        const [lon, lat] = coordinates[index];
        if (occupied.has(vertexKey(coordinates[index])) || !visible.contains([lat, lon])) continue;
        const title = !joining ? S.CARTOGRAPHY_SELECT_LINE(i)
          : part.type === 'Point' ? S.CARTOGRAPHY_JOIN_POINT(i) : S.CARTOGRAPHY_JOIN_ENDPOINT(i, index === 0);
        const selected = draft.join?.first?.part === i && draft.join.first.index === index;
        const className = !joining ? 'cartography-inactive-endpoint'
          : selected ? 'cartography-endpoint cartography-active-vertex' : 'cartography-endpoint';
        const icon = L.divIcon({
          className,
          iconSize: [19, 19],
        });
        // Split lines can have coincident endpoints. Put the selected one
        // behind its neighbors so they can be chosen by pointer as well.
        const marker = L.marker([lat, lon], {icon, title, zIndexOffset: !joining || selected ? -1000 : 0})
          .addTo(this.handles);
        const handle = marker.getElement();
        handle.dataset.part = i;
        handle.dataset.index = index;
        this.selectableHandle(marker, title, joining ? selected : null, () => select(i, index));
      }
    });
  }

  selectableHandle(marker, label, selected, select) {
    const handle = marker.getElement();
    handle.setAttribute('aria-label', label);
    if (selected !== null) handle.setAttribute('aria-pressed', String(selected));
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
      this.showTrace(null);
      this.handles.clearLayers();
      render();
    };
    this.refreshHandles();
    this.map.on('moveend', this.refreshHandles);
  }

  showVertexHandles(parts, callbacks, draft) {
    const referenceMode = boundaryTargetMode(draft);
    this.watchHandles(() => {
      const visible = this.map.getBounds().pad(0.1);
      if (referenceMode) this.drawBoundaryTargets(draft, callbacks);
      if (draft && !draft.kind && draft.parts.length > 1) this.drawEndpoints(draft, callbacks.selectPart, visible);
      for (const part of parts) {
        let offset = 0;
        rings(part).forEach((ring, r) => {
          ring.forEach(([lon, lat], i) => {
            const ll = L.latLng(lat, lon);
            if (callbacks && !draft?.following && i > 0) {
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
            const shared = referenceMode && this.referenceTargets.has(vertexKey([lon, lat]));
            const className = shared ? 'cartography-vertex cartography-shared-vertex' : 'cartography-vertex';
            const marker = callbacks ? L.marker(ll, {
              draggable: !draft?.following, zIndexOffset: referenceMode ? 1000 : 0,
              icon: L.divIcon({className, iconSize: [15, 15]}),
            }) : L.circleMarker(ll, {radius: 4, color: drawingStyle.color, bubblingMouseEvents: false});
            marker.addTo(this.handles);
            if (fix) marker.bindTooltip(fixLabel(fix)).on('click', () => marker.openTooltip());
            if (callbacks) {
              const handle = marker.getElement();
              const selected = callbacks.vertex?.ring === r && callbacks.vertex?.index === i;
              handle.classList.toggle('cartography-active-vertex', selected);
              const selectVertex = () => {
                if (draft?.following) return;
                this.handles.eachLayer(layer => {
                  const element = layer.getElement();
                  if (!element?.hasAttribute('aria-pressed')) return;
                  const active = element === handle;
                  element.classList.toggle('cartography-active-vertex', active);
                  element.setAttribute('aria-pressed', String(active));
                });
                callbacks.vertex = {ring: r, index: i};
                callbacks.selectVertex(r, i, shared);
              };
              this.selectableHandle(marker, S.CARTOGRAPHY_SELECT_VERTEX, selected, selectVertex);
              marker.on('drag', () => {
                const layer = this.activeLayer;
                const ll = marker.getLatLng();
                const snapped = referenceMode && this.snapNear(ll);
                const end = snapped ? L.latLng(snapped[1], snapped[0]) : ll;
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
                const snapped = referenceMode && this.snapNear(end);
                callbacks.move(r, i, snapped || [end.lng, end.lat], !!snapped);
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
    this.showTrace(null);
    this.snapHover = null;
    this.snapHandles = [];
  }

  destroy() {
    this.stopEdit();
    this.toolbar.remove();
    this.parcels.destroy();
  }
}
