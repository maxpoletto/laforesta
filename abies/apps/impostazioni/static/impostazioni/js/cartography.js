import { fetchJSON, postJSON } from '../../base/js/api.js';
import { renderModalForm, showFormError } from '../../base/js/forms.js';
import { wireParcelMetadataGovernance } from '../../bosco/js/parcel-metadata.js';
import * as cache from '../../base/js/cache.js';
import { sortFeaturesByArea } from '../../base/js/geo.js';
import * as modals from '../../base/js/modals.js';
import { navigateWithParams } from '../../base/js/page-sync.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { showConfirmModal } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { mapTypeName, mapTypeToken } from '../../base/js/map-common.js';
import { showRegionForm } from './region-form.js';
import { CartographyMap } from './cartography-map.js';
import { editBoundary } from './parcel-boundaries.js';
import { updateInboxDot } from '../../ipso/js/inbox-status.js';

import { closeLine, handPart, insertVertex, joinParts, moveVertex, openPolygon, splitLine } from './drawing-parts.js';

// Unsaved edits belong to this browser tab and compresa.
const drafts = new Map();
const API = '/api/impostazioni/cartography/';
const tab = 'cartografia';
const selectionKey = 'abies.cartography.region';
const versions = rows => rows.map(row => [row.id, row.version]).sort((a, b) => a[0] - b[0]);
const sourceLabel = row => [row.name, row.date, row.operator].filter(Boolean).join(' · ');

function fill(select, rows, label = row => row.name) {
  select.replaceChildren(...rows.map(row => new Option(label(row), String(row.id))));
}

export class CartographyEditor {
  constructor(panel) {
    this.panel = panel;
    this.life = new AbortController();
    this.generation = 0;
    this.region = panel.querySelector('#cartography-region');
    this.lists = {
      drawing: panel.querySelector('#cartography-drawings'),
      parcel: panel.querySelector('#cartography-parcels'),
      road: panel.querySelector('#cartography-roads'),
    };
    this.mapTools = panel.querySelector('[data-role=map-tools]');
    this.parts = panel.querySelector('#cartography-parts');
    this.status = panel.querySelector('[role=status]');
    this.snap = panel.querySelector('[data-role=parcel-snap] input');
    this.buttons = Object.fromEntries([...panel.querySelectorAll('[data-action]')].map(b => [b.dataset.action, b]));
    const listen = (element, event, fn) => element.addEventListener(event, fn, {signal: this.life.signal});
    listen(this.snap, 'change', () => { this.draft.snap = this.snap.checked; });
    listen(this.parts, 'change', () => this.selectPart(Number(this.parts.value)));
    listen(this.region, 'change', () => this.navigate(this.region.value));
    listen(panel, 'click', e => {
      const button = e.target.closest('button[data-action]');
      if (button && !button.disabled && !this.busy) this.act(button);
    });
    this.ready = this.loadCatalog();
  }

  async loadCatalog(geo = cache.load('terreni')) {
    const [response, parcels] = await Promise.all([fetchJSON(API), geo]);
    if (this.life.signal.aborted) return;
    this.catalog = response.data;
    this.geo = sortFeaturesByArea(parcels);
    const lang = document.documentElement.lang;
    this.catalog.regions.sort((a, b) => a.name.localeCompare(b.name, lang));
    fill(this.region, this.catalog.regions);
  }

  navigate(id, mt = this.mt) {
    const params = new URLSearchParams({scheda: tab});
    if (id) params.set('c', id);
    if (mt) params.set('mt', mt);
    navigateWithParams('/impostazioni', params);
  }

  async open(params) {
    if (this.map && this.regionId === Number(params.c)) {
      this.mt = params.mt;
      this.map.parcels.wrapper.syncBasemap(mapTypeName(this.mt));
      return;
    }
    const generation = ++this.generation;
    try {
      await this.ready;
      if (!this.current(generation)) return;
      const regions = this.catalog.regions;
      const lastRegion = Number(sessionStorage.getItem(selectionKey));
      const region = regions.find(r => r.id === Number(params.c))
        || regions.find(r => r.id === lastRegion) || regions[0];
      this.mt = params.mt;
      if (!region) {
        this.busy = false;
        sessionStorage.removeItem(selectionKey);
        this.updateButtons();
        if (params.c) this.navigate(null);
        return;
      }
      sessionStorage.setItem(selectionKey, region.id);
      if (region.id === this.regionId && this.map) return;
      this.regionId = region.id;
      this.region.value = String(region.id);
      this.showRegionMap(region);
      this.state = null;
      this.draft = null;
      this.busy = false;
      this.status.textContent = '';
      this.updateButtons();
      await this.reload(generation);
      if (this.current(generation)) {
        this.map.fitRegion();
        if (params.fonte && !this.draft) {
          const drawing = this.state.drawings.find(d => d.source_id === Number(params.fonte));
          if (drawing) this.select('drawing', drawing.id, true);
        }
      }
    } catch (error) {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    }
  }

  showRegionMap(region, preserveView = false) {
    const initialView = preserveView && this.map ? {
      center: this.map.map.getCenter(), zoom: this.map.map.getZoom(),
    } : undefined;
    this.map?.destroy();
    this.map = new CartographyMap(this.panel.querySelector('#cartography-map'), this.geo, region.name,
      (kind, id) => this.select(kind, id), mapTypeName(this.mt), ll => this.place(ll), this.mapTools, initialView);
    this.map.map.on('basemapchange', e => this.navigate(region.id, mapTypeToken(e.name)));
  }

  renderList(kind, rows) {
    const list = this.lists[kind];
    list.replaceChildren(...rows.map(row => {
      const item = cloneTemplate('tmpl-cartography-row').firstElementChild;
      item.dataset.kind = kind;
      item.dataset.id = row.id;
      const select = item.querySelector('[data-action=select-item]');
      select.title = row.name;
      select.querySelector('.map-workspace-list-label').textContent = row.name;
      const toggle = item.querySelector('[data-action=toggle-item]');
      toggle.title = S.CARTOGRAPHY_VISIBILITY(row.name);
      toggle.setAttribute('aria-label', toggle.title);
      const remove = item.querySelector('[data-action=remove-drawing]');
      if (kind === 'drawing') {
        remove.title = S.CARTOGRAPHY_REMOVE_DRAWING(row.name);
        remove.setAttribute('aria-label', remove.title);
      } else {
        remove.remove();
      }
      return item;
    }));
    list.hidden = !rows.length;
    const header = this.panel.querySelector(`.cartography-list-header[data-kind=${kind}]`);
    for (const button of header.querySelectorAll('button')) button.hidden = !rows.length;
    this.panel.querySelector(`[data-empty=${kind}]`).hidden = !!rows.length;
  }

  updateLists(locked) {
    for (const [kind, list] of Object.entries(this.lists)) {
      let anyVisible = false;
      for (const item of list.children) {
        const id = item.dataset.id;
        const visible = this.map.isVisible(kind, id);
        anyVisible ||= visible;
        for (const button of item.querySelectorAll('button')) button.disabled = locked;
        item.querySelector('[data-action=toggle-item]').setAttribute('aria-pressed', visible);
        item.querySelector('[data-action=select-item]').setAttribute('aria-pressed',
          this.kind === kind && String(this.selected?.id) === id);
      }
      const toggle = this.panel.querySelector(`.cartography-list-header[data-kind=${kind}] [data-action=toggle-all]`);
      toggle.disabled = locked || !list.children.length;
      toggle.setAttribute('aria-pressed', anyVisible);
    }
  }

  current(generation) {
    return !this.life.signal.aborted && this.generation === generation;
  }

  async reload(generation, selection) {
    const {data} = await fetchJSON(API + this.regionId + '/');
    if (!this.current(generation)) return;
    this.state = data;
    const byName = (a, b) => a.name.localeCompare(b.name, document.documentElement.lang, {numeric: true});
    const names = new Set([...this.map.parcelLayers.keys(), ...data.parcels.map(row => row.name)]);
    this.renderList('parcel', [...names].map(name => ({id: name, name})).sort(byName));
    this.renderList('drawing', data.drawings.sort(byName));
    this.renderList('road', data.roads.sort(byName));
    this.map.render(data);
    const draft = drafts.get(this.regionId);
    this.draft = null;
    if (draft) {
      const kind = draft.kind || 'drawing';
      this.select(kind, draft.id);
      if (this.selected || (kind === 'drawing' && draft.tool)) {
        this.startEdit(draft);
        return;
      }
    }
    drafts.delete(this.regionId);
    this.select(selection?.kind, selection?.id);
  }

  select(kind, id, frame = false) {
    if (!this.state || this.draft) return;
    this.kind = kind;
    const rows = kind === 'drawing' ? this.state.drawings : this.state.roads;
    const record = this.state.parcels.find(parcel => parcel.name === id);
    this.selected = kind === 'parcel' && (record || this.map.parcelLayers.has(id))
      ? {id, parcelId: record?.id} : rows.find(row => row.id === id);
    this.map.select(kind, this.selected);
    if (frame && this.selected) this.map.frame(kind, id);
    this.updateButtons();
  }

  updateButtons() {
    const b = this.buttons, draft = this.draft;
    const locked = !!(this.busy || draft);
    const drawing = this.state && this.kind === 'drawing' && this.selected;
    const parcel = this.state && this.kind === 'parcel' && this.selected?.parcelId;
    const road = this.state && this.kind === 'road' && this.selected;
    const drawingDraft = draft && !draft.kind;
    const parts = draft?.parts ?? drawing?.parts;
    const part = drawingDraft && !draft.tool && parts[draft.active];
    b.edit.disabled = locked || !drawing;
    b['parcel-properties'].disabled = b.boundaries.disabled = locked || !parcel;
    b['road-properties'].disabled = b.points.disabled = locked || !road;
    const drawingTools = !draft || (draft.kind === 'parcel' && !draft.tool);
    b.line.hidden = !!draft;
    b.line.disabled = !!this.busy || !this.state;
    b.polygon.disabled = !!this.busy || !this.state;
    this.mapTools.querySelector('[data-role=draw-tools]').hidden = !drawingTools;
    const promotableLine = parts?.length === 1 && parts[0].type === 'LineString' && this.catalog?.road_types.length;
    const promotablePolygon = parts?.length && parts.every(piece => piece.type === 'Polygon');
    b.promote.disabled = !!this.busy || !!draft?.kind || !(promotableLine || promotablePolygon);
    this.panel.querySelector('[data-role=edit-actions]').hidden = !part;
    this.panel.querySelector('[data-role=parts]').hidden = !part;
    for (const kind of ['parcel', 'road']) {
      const editing = draft?.kind === kind && !draft.tool;
      this.panel.querySelector(`[data-role=${kind}-edit-actions]`).hidden = !editing;
      b[`save-${kind}`].disabled = !!this.busy || !draft?.dirty;
      b[`cancel-${kind}`].disabled = !!this.busy;
    }
    this.snap.parentElement.hidden = draft?.kind !== 'parcel';
    this.snap.checked = !!draft?.snap;
    this.snap.disabled = !!this.busy;
    this.mapTools.querySelector('[data-role=sketch-tools]').hidden = !draft?.tool;
    b.undo.hidden = !draft;
    b.undo.disabled = !!this.busy || !(draft?.tool ? draft.vertices.length : draft?.previous);
    b.finish.disabled = !!this.busy || !draft?.tool || draft.vertices.length < (draft.tool === 'Polygon' ? 3 : 2);
    b.save.disabled = b.cancel.disabled = b.copy.disabled = b['cancel-sketch'].disabled = !!this.busy;
    b['remove-part'].disabled = !!this.busy || !part || parts.length < 2;
    b.join.disabled = !!this.busy || part?.type !== 'LineString' || parts[draft.active + 1]?.type !== 'LineString';
    b.split.disabled = !!this.busy || part?.type !== 'LineString'
      || !(draft.vertex > 0 && draft.vertex < part.coordinates.length - 1);
    b['open-polygon'].disabled = !!this.busy || part?.type !== 'Polygon';
    b['close-line'].disabled = !!this.busy || part?.type !== 'LineString' || part.coordinates.length < 3;
    b.clear.disabled = locked || !this.state?.drawings.length;
    b['open-import'].disabled = locked || !this.state;
    b['new-region'].disabled = locked;
    b['edit-region'].disabled = locked || !this.state;
    this.region.disabled = locked;
    this.updateLists(locked);
  }

  startEdit(draft) {
    this.draft = draft;
    drafts.set(this.regionId, draft);
    this.status.textContent = draft.tool ? S.CARTOGRAPHY_PLACE : S.CARTOGRAPHY_DRAG;
    this.refreshEdit();
  }

  selectPart(index) {
    this.draft.active = index;
    this.draft.vertex = null;
    this.refreshEdit();
  }

  changeParts(change) {
    const d = this.draft;
    const previous = {parts: d.parts, active: d.active, dirty: d.dirty};
    if (d.kind === 'parcel') Object.assign(previous, {
      features: d.features, edits: d.edits, added: d.added, neighbors: d.neighbors,
    });
    d.previous = structuredClone(previous);
    change(d);
    d.dirty = true;
    d.vertex = null;
    this.refreshEdit();
  }

  place(ll) {
    if (!this.draft?.tool || this.busy) return false;
    const point = [ll.wrap().lng, ll.lat];
    this.draft.vertices.push(this.draft.snap ? this.map.nearestParcelVertex(this.draft.id, point) : point);
    this.refreshEdit();
    return true;
  }

  refreshEdit() {
    const d = this.draft;
    if (!d.tool && !d.kind) {
      this.parts.replaceChildren(...d.parts.map((p, i) => new Option(S.CARTOGRAPHY_PART(i + 1, p.type), i)));
      this.parts.value = String(d.active);
    }
    this.map.edit(d, {
      vertex: d.vertex,
      selectVertex: i => { d.vertex = i; this.updateButtons(); },
      selectPart: i => this.selectPart(i), place: ll => this.place(ll),
      move: (r, i, pt) => this.editVertex('move', r, i, pt),
      insert: (r, i, pt) => this.editVertex('insert', r, i, pt),
    });
    this.updateButtons();
  }

  editVertex(operation, ring, index, point) {
    if (this.busy) return;
    this.changeParts(draft => {
      if (draft.kind === 'parcel') {
        const destination = operation === 'move' && draft.snap
          ? this.map.nearestParcelVertex(draft.id, point) : point;
        editBoundary(draft, operation, ring, index, destination);
      } else {
        const edit = operation === 'move' ? moveVertex : insertVertex;
        edit(draft.parts[draft.active], ring, index, point);
      }
    });
  }

  async loadParcel(row, view) {
    const generation = this.generation;
    this.busy = true;
    this.updateButtons();
    try {
      const {data} = await fetchJSON(`${API}${this.regionId}/parcels/${row.parcelId}/${view}/`);
      if (this.current(generation)) return data;
    } catch {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  async editBoundaries(row) {
    const data = await this.loadParcel(row, 'boundaries');
    if (!data) return;
    const {features, review} = data;
    const parts = features.filter(feature => feature.parcel_id === row.parcelId).flatMap(feature => feature.parts);
    this.startEdit({kind: 'parcel', id: row.id, parcelId: row.parcelId, features, review,
      parts, active: 0, edits: [], added: [], neighbors: []});
  }

  wireProperties(form, kind, id, version) {
    const name = form.elements.name;
    const oldName = name.value;
    const rename = form.elements.confirm_rename;
    const remove = form.querySelector('[data-action=delete]');
    const submit = form.querySelector('[type=submit]');
    form.querySelector('[data-action=cancel]').onclick = modals.dismiss;
    if (rename) {
      name.oninput = () => {
        const changed = name.value.trim() !== oldName;
        rename.closest('.form-row').hidden = !changed;
        rename.required = changed;
        rename.checked = false;
      };
    }
    if (remove) {
      remove.hidden = false;
      remove.onclick = () => showConfirmModal(S.DELETE_REFERENCE_CONFIRM(oldName),
        () => this.write(`remove-${kind}`, {id, version}));
    }
    form.onsubmit = async event => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const values = Object.fromEntries(new FormData(form));
      if (rename) values.confirm_rename = rename.checked;
      submit.disabled = true;
      try { await this.write(`${kind}-properties`, {...values, id, version}, form); }
      finally { submit.disabled = false; }
    };
    name.focus();
  }

  async showParcelProperties(row) {
    const data = await this.loadParcel(row, 'properties');
    if (!data) return;
    const form = renderModalForm(data.html);
    wireParcelMetadataGovernance(form);
    this.wireProperties(form, 'parcel', row.parcelId, Number(form.elements.version.value));
  }

  showRoadProperties(row) {
    const form = cloneTemplate('tmpl-cartography-road').querySelector('form');
    form.querySelector('h2').textContent = S.CARTOGRAPHY_ROAD_PROPERTIES;
    form.querySelector('[type=submit]').textContent = S.SAVE;
    const types = [...this.catalog.road_types];
    if (!types.some(type => type.id === row.road_type_id)) {
      types.push({id: row.road_type_id, name: row.road_type__name});
    }
    fill(form.elements.road_type_id, types);
    form.elements.road_type_id.value = row.road_type_id;
    form.elements.name.value = row.name;
    modals.show(form);
    this.wireProperties(form, 'road', row.id, row.version);
  }

  form(template, submit) {
    const form = cloneTemplate(template).querySelector('form');
    form.querySelector('[data-action=delete]')?.remove();
    const select = form.querySelector('select');
    if (select) fill(select, this.catalog.road_types);
    form.querySelector('[data-action=cancel]').onclick = modals.dismiss;
    form.onsubmit = event => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const body = Object.fromEntries(new FormData(form));
      modals.dismiss();
      submit(body);
    };
    modals.show(form);
    form.querySelector('input').focus();
  }

  async promoteParcel(drawing) {
    const generation = this.generation;
    this.busy = true;
    this.updateButtons();
    try {
      const {data} = await fetchJSON(API + this.regionId + '/parcel-form/');
      if (!this.current(generation)) return;
      const form = renderModalForm(data.html);
      wireParcelMetadataGovernance(form);
      const select = form.elements.parcel_id;
      const fields = form.querySelector('fieldset');
      if (select) select.onchange = () => { fields.hidden = fields.disabled = !!select.value; };
      form.querySelector('[data-action=cancel]').onclick = modals.dismiss;
      form.onsubmit = event => {
        event.preventDefault();
        const body = Object.fromEntries(new FormData(form));
        modals.dismiss();
        this.write('parcel', {...body, id: drawing.id, version: drawing.version, parts: drawing.parts,
          parcel_version: select?.selectedOptions[0]?.dataset.version});
      };
      (select || form.elements.name).focus();
    } catch {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  showImport() {
    const generation = this.generation;
    const modal = cloneTemplate('tmpl-cartography-import').firstElementChild;
    const uploads = modal.querySelector('#cartography-uploads');
    const sources = modal.querySelector('#cartography-sources');
    fill(uploads, this.state.uploads, sourceLabel);
    fill(sources, this.state.sources, sourceLabel);
    for (const button of modal.querySelectorAll('[data-action]')) {
      const {action} = button.dataset;
      const select = action === 'import' ? uploads : sources;
      button.disabled = action !== 'cancel' && !select.options.length;
      button.onclick = () => {
        const id = Number(select.value);
        modals.dismiss();
        if (!this.current(generation) || action === 'cancel') return;
        if (action === 'forget') {
          const source = this.state.sources.find(row => row.id === id);
          showConfirmModal(S.CARTOGRAPHY_FORGET(source.name), () => {
            if (this.current(generation)) return this.write(action, {source_id: id});
          });
        } else {
          this.write(action, action === 'import' ? {upload_id: id} : {source_id: id});
        }
      };
    }
    modals.show(modal);
    (uploads.options.length ? uploads : sources.options.length ? sources
      : modal.querySelector('[data-action=cancel]')).focus();
  }

  async act(button) {
    const {action} = button.dataset;
    const target = button.closest('[data-kind]')?.dataset;
    const row = this.selected;
    switch (action) {
      case 'select-item':
        this.select(target.kind, target.kind === 'parcel' ? target.id : Number(target.id), true);
        break;
      case 'toggle-item':
      case 'toggle-all': {
        const {kind, id} = target;
        const show = button.getAttribute('aria-pressed') === 'false';
        const items = id == null ? [...this.lists[kind].children].map(item => item.dataset.id) : [id];
        for (const item of items) this.map.setVisible(kind, item, show);
        this.map.select(this.kind, this.selected);
        this.updateButtons();
        break;
      }
      case 'new-region':
      case 'edit-region': {
        const generation = this.generation;
        this.busy = true;
        this.updateButtons();
        try {
          await showRegionForm(API, action === 'new-region' ? null : this.regionId,
            () => this.current(generation), id => this.regionChanged(id));
        } finally {
          if (this.current(generation)) { this.busy = false; this.updateButtons(); }
        }
        break;
      }
      case 'parcel-properties':
        await this.showParcelProperties(row);
        break;
      case 'road-properties':
        this.showRoadProperties(row);
        break;
      case 'boundaries':
        await this.editBoundaries(row);
        break;
      case 'points':
        this.startEdit({kind: 'road', id: row.id, version: row.version,
          parts: [structuredClone(row.geometry)], active: 0});
        break;
      case 'save-road':
        this.write('road-points', {id: row.id, version: this.draft.version, parts: this.draft.parts});
        break;
      case 'save-parcel': {
        const {parcelId, review, edits, added} = this.draft;
        this.write('parcel-boundaries', {id: parcelId, review, edits, added});
        break;
      }
      case 'open-import':
        this.showImport();
        break;
      case 'line':
      case 'polygon':
        if (this.draft?.kind === 'parcel') {
          Object.assign(this.draft, {tool: 'Polygon', vertices: []});
          this.startEdit(this.draft);
          break;
        }
        this.select(null);
        this.startEdit({tool: action === 'line' ? 'LineString' : 'Polygon', vertices: []});
        break;
      case 'edit':
        this.startEdit({id: row.id, version: row.version, parts: structuredClone(row.parts), active: 0});
        break;
      case 'join':
        this.changeParts(d => joinParts(d.parts, d.active));
        break;
      case 'split':
        this.changeParts(d => d.parts.splice(d.active, 1, ...splitLine(d.parts[d.active], d.vertex)));
        break;
      case 'open-polygon':
        this.changeParts(d => d.parts.splice(d.active, 1, ...openPolygon(d.parts[d.active])));
        break;
      case 'close-line':
        this.changeParts(d => { d.parts[d.active] = closeLine(d.parts[d.active]); });
        break;
      case 'remove-part':
        this.changeParts(d => {
          d.parts.splice(d.active, 1);
          d.active = Math.min(d.active, d.parts.length - 1);
        });
        break;
      case 'undo': {
        const d = this.draft;
        if (d.tool) d.vertices.pop();
        else { Object.assign(d, d.previous); d.previous = null; }
        this.refreshEdit();
        break;
      }
      case 'cancel':
      case 'cancel-road':
      case 'cancel-parcel':
      case 'cancel-sketch': {
        const d = this.draft;
        const kind = d.kind || 'drawing';
        const message = action === 'cancel-parcel' ? S.CARTOGRAPHY_CANCEL_BOUNDARIES :
          action === 'cancel-road' ? S.CARTOGRAPHY_CANCEL_POINTS : S.CARTOGRAPHY_CANCEL;
        showConfirmModal(message, async () => {
          if (action === 'cancel-sketch' && kind === 'parcel') {
            delete d.tool;
            delete d.vertices;
            this.startEdit(d);
            return;
          }
          drafts.delete(this.regionId);
          this.draft = null;
          this.map.render(this.state);
          this.select(kind, row?.id);
          this.status.textContent = '';
          if (d.refreshOnCancel) {
            try { await this.reload(this.generation, {kind, id: row?.id}); }
            catch { modals.showError(S.ERROR_NETWORK); }
          }
        }, {confirmLabel: S.CARTOGRAPHY_DISCARD});
        break;
      }
      case 'copy': {
        const d = this.draft;
        this.form('tmpl-cartography-drawing', body => this.write('copy', {
          ...body, id: d.id, version: d.version, parts: [d.parts[d.active]],
        }));
        break;
      }
      case 'save':
      case 'finish': {
        const d = this.draft;
        if (d.tool && d.kind === 'parcel') {
          const part = handPart(d.tool, d.vertices);
          delete d.tool;
          delete d.vertices;
          this.status.textContent = S.CARTOGRAPHY_DRAG;
          this.changeParts(draft => {
            draft.added.push(part);
            draft.parts.push(part);
            draft.active = draft.parts.length - 1;
          });
        } else if (d.tool) {
          this.form('tmpl-cartography-drawing', body =>
            this.write('create', {...body, parts: [handPart(d.tool, d.vertices)]}));
        } else {
          this.write('adjust', {id: d.id, version: d.version, parts: d.parts});
        }
        break;
      }
      case 'promote': {
        const drawing = this.draft || row;
        if (drawing.parts[0].type === 'Polygon') {
          await this.promoteParcel(drawing);
          break;
        }
        this.form('tmpl-cartography-road', body => this.write('road', {
          ...body, id: drawing.id, version: drawing.version, parts: drawing.parts,
        }));
        break;
      }
      case 'remove-drawing': {
        const drawing = this.state.drawings.find(item => item.id === Number(target.id));
        showConfirmModal(S.DELETE_REFERENCE_CONFIRM(drawing.name),
          () => this.write(action, {id: drawing.id, version: drawing.version}));
        break;
      }
      case 'clear': {
        const snapshot = versions(this.state.drawings);
        const name = this.region.selectedOptions[0].textContent;
        showConfirmModal(S.CARTOGRAPHY_CLEAR(name, snapshot.length), () => this.write(action, {versions: snapshot}));
        break;
      }

    }
  }

  async regionChanged(id) {
    const generation = ++this.generation;
    this.busy = true;
    this.updateButtons();
    try {
      await this.loadCatalog(cache.get('terreni') || cache.load('terreni'));
      if (!this.current(generation)) return;
      this.map?.destroy();
      this.map = this.state = this.selected = this.regionId = null;
      for (const kind of Object.keys(this.lists)) this.renderList(kind, []);
      this.status.textContent = '';
      this.navigate(id || this.catalog.regions[0]?.id);
    } catch {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  async write(action, body, form = null) {
    if (this.busy || this.life.signal.aborted) return;
    const generation = this.generation;
    const regionId = this.regionId, submitted = this.draft;
    this.busy = true;
    this.updateButtons();
    try {
      const path = this.regionId + '/' + action;
      const {data, status} = await postJSON(API + path + '/', {...body, nonce: crypto.randomUUID()});
      if (status >= 400) {
        if (this.current(generation)) {
          if (form?.isConnected) {
            showFormError(form, data.message || S.ERROR_GENERIC);
          } else {
            if (this.draft) this.draft.refreshOnCancel = true;
            else await this.reload(generation);
            modals.showError(data.message || S.ERROR_GENERIC);
          }
        }
        return;
      }
      cache.applyResponseChanges(data);
      if (action === 'import') updateInboxDot(data);
      if (['adjust', 'road', 'create', 'parcel', 'road-points', 'parcel-boundaries'].includes(action) && drafts.get(regionId) === submitted) {
        drafts.delete(regionId);
      }
      if (!this.current(generation)) return;
      if (form?.isConnected) modals.dismiss();
      let selection = data.drawing_id && {kind: 'drawing', id: data.drawing_id};
      if (action === 'adjust') selection = {kind: 'drawing', id: body.id};
      if (data.road_id) selection = {kind: 'road', id: data.road_id};
      if (data.parcel_id || data.parcel_map_changed) {
        const geo = await cache.load('terreni');
        if (!this.current(generation)) return;
        this.geo = sortFeaturesByArea(geo);
        this.showRegionMap(this.catalog.regions.find(region => region.id === this.regionId), true);
        if (data.parcel_id) selection = {kind: 'parcel', id: data.parcel_name};
      }
      const firstImport = !this.state.drawings.length && !this.state.roads.length
        && !this.map.parcelLayers.size;
      await this.reload(generation, selection);
      if (!this.current(generation)) return;
      if (!this.draft) this.status.textContent = data.added == null ? '' : S.CARTOGRAPHY_IMPORT_COUNT(data.added);
      if (data.parcel_id) this.map.frame('parcel', data.parcel_name);
      if (data.added) this.map.fitSource(this.state.drawings, data.source_id || body.source_id, firstImport);
    } catch (error) {
      if (this.current(generation)) {
        if (form?.isConnected) showFormError(form, S.ERROR_NETWORK);
        else modals.showError(S.ERROR_NETWORK);
      }
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  destroy() {
    this.life.abort();
    this.map?.destroy();
  }
}
