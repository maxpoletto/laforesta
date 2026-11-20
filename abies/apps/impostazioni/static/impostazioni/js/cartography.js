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
import { wireDrawingImportTabs } from './cartography-import.js';
import { addSketchPoint, cancelSketch, changeGeometry, undoEdit } from './draft-edits.js';
import { boundaryVertexRemoval, detachBoundary, editBoundary, removeBoundaryVertex } from './parcel-boundaries.js';
import { updateInboxDot } from '../../ipso/js/inbox-status.js';

import {
  breakPart, canBreakPart, canJoinEndpoints, canRemoveVertex, handPart, insertVertex,
  joinEndpoints, moveVertex, removeVertex,
} from './drawing-parts.js';

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

function removeActivePart(draft) {
  draft.parts.splice(draft.active, 1);
  draft.active = Math.min(draft.active, draft.parts.length - 1);
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
    this.buttons = Object.fromEntries([...panel.querySelectorAll('[data-action]')].map(b => [b.dataset.action, b]));
    const listen = (element, event, fn) => element.addEventListener(event, fn, {signal: this.life.signal});
    listen(this.parts, 'change', () => this.selectPart(Number(this.parts.value)));
    listen(this.region, 'change', () => this.navigate(this.region.value));
    listen(panel.querySelector('#cartography-sidebar'), 'click', e => {
      if (!this.busy && !e.target.closest('button, label, select, input, .map-workspace-list')) this.select(null);
    });
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
      const remove = item.querySelector('[data-action=remove-item]');
      remove.title = row.deleteBlock || S.CARTOGRAPHY_REMOVE_ITEM(row.name);
      remove.setAttribute('aria-label', remove.title);
      remove.toggleAttribute('data-unavailable', !!row.deleteBlock);
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
        for (const button of item.querySelectorAll('button')) {
          button.disabled = locked || button.hasAttribute('data-unavailable');
        }
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
    const parcels = new Map(data.parcels.map(row => [row.name, row]));
    const names = new Set([...this.map.parcelLayers.keys(), ...parcels.keys()]);
    this.renderList('parcel', [...names].map(name => ({
      id: name, name, deleteBlock: !parcels.has(name) ? S.CARTOGRAPHY_PARCEL_UNLINKED
        : parcels.get(name).in_use ? S.CARTOGRAPHY_PARCEL_IN_USE : null,
    })).sort(byName));
    this.renderList('drawing', data.drawings.sort(byName));
    this.renderList('road', data.roads.sort(byName));
    this.map.render(data);
    const draft = drafts.get(this.regionId);
    this.draft = null;
    if (draft) {
      const kind = draft.kind || 'drawing';
      const selection = draft.returnSelection ?? {kind, id: draft.id};
      this.select(selection.kind, selection.id);
      if (draft.returnSelection || this.selected) {
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
    const drawingTools = draft && !draft.tool && draft.kind !== 'road';
    this.mapTools.hidden = !draft;
    b.line.hidden = !drawingDraft;
    b.line.disabled = !!this.busy || !this.state;
    b.polygon.disabled = !!this.busy || !this.state;
    this.mapTools.querySelector('[data-role=draw-tools]').hidden = !drawingTools;
    const promotableLine = parts?.length === 1 && parts[0].type === 'LineString' && this.catalog?.road_types.length;
    const promotablePolygon = parts?.length && parts.every(piece => piece.type === 'Polygon');
    b.promote.disabled = !!this.busy || !!draft?.kind || !(promotableLine || promotablePolygon);
    this.panel.querySelector('[data-role=parts]').hidden = !part;
    const incompleteRoad = draft?.kind === 'road' && parts.length
      && (parts.length !== 1 || parts[0].type !== 'LineString');
    b.save.hidden = !draft || !!draft.tool || !!draft.returnSelection;
    b.save.disabled = !!this.busy || !!incompleteRoad || (!!draft?.kind && !draft.dirty);
    b.save.title = incompleteRoad ? S.CARTOGRAPHY_ROAD_NEEDS_LINE : S.SAVE;
    b.cancel.hidden = !draft;
    b.undo.hidden = !draft;
    b.undo.disabled = !!this.busy || !draft?.history?.length;
    const activePart = draft?.parts?.[draft.active];
    const selectedVertex = !draft?.join && draft?.vertex;
    const canRemove = selectedVertex && (draft.kind === 'parcel' && activePart?.type === 'Polygon'
      ? !!boundaryVertexRemoval(draft, selectedVertex.ring, selectedVertex.index)
      : canRemoveVertex(activePart, selectedVertex.ring, selectedVertex.index));
    const remove = b['remove-vertex'];
    const editingVertices = draft && !draft.tool && !draft.returnSelection;
    remove.hidden = !editingVertices;
    remove.disabled = !!this.busy || !canRemove;
    remove.title = selectedVertex && !canRemove ? S.CARTOGRAPHY_VERTEX_REQUIRED : S.CARTOGRAPHY_REMOVE_VERTEX;
    b['break-link'].hidden = b['join-link'].hidden = remove.hidden;
    b['break-link'].disabled = !!this.busy || !selectedVertex
      || !canBreakPart(activePart, selectedVertex.ring, selectedVertex.index);
    b['join-link'].disabled = !!this.busy || !editingVertices
      || !canJoinEndpoints(draft.parts, {close: draft.kind !== 'road'});
    b['join-link'].setAttribute('aria-pressed', String(!!draft?.join));
    b.finish.hidden = draft?.tool !== 'LineString';
    b.finish.disabled = !!this.busy || !draft?.tool || draft.vertices.length < 2;
    b.cancel.disabled = b.copy.disabled = !!this.busy;
    b['remove-part'].disabled = !!this.busy || !part || parts.length < 2;
    b.clear.disabled = locked || !this.state?.drawings.length;
    b['new-drawing'].disabled = b['open-import'].disabled = locked || !this.state;
    b['new-region'].disabled = locked;
    b['edit-region'].disabled = locked || !this.state;
    this.region.disabled = locked;
    this.updateLists(locked);
  }

  startEdit(draft) {
    this.draft = draft;
    drafts.set(this.regionId, draft);
    this.refreshEdit();
  }

  selectPart(index) {
    this.draft.active = index;
    this.draft.vertex = null;
    this.draft.join = null;
    this.refreshEdit();
  }

  changeParts(change) {
    const d = this.draft;
    if (!changeGeometry(d, change)) return;
    d.vertex = null;
    d.join = null;
    this.refreshEdit();
  }

  place(ll) {
    if (!this.draft?.tool || this.busy) return false;
    if (addSketchPoint(this.draft, [ll.wrap().lng, ll.lat])) this.refreshEdit();
    return true;
  }

  finishSketch() {
    const d = this.draft;
    if (this.busy || !d?.tool || d.vertices.length < (d.tool === 'Polygon' ? 3 : 2)) return;
    const part = handPart(d.tool, d.vertices);
    if (d.returnSelection) {
      d.completedSketch = part;
      this.refreshEdit();
      this.form('tmpl-cartography-drawing', body => this.write('create', {...body, parts: [part]}));
      modals.onDismiss(() => {
        delete d.completedSketch;
        if (this.draft === d && !this.life.signal.aborted) this.refreshEdit();
      });
    } else {
      this.changeParts(draft => {
        delete draft.tool;
        delete draft.vertices;
        if (draft.kind === 'parcel') draft.added.push(part);
        draft.parts.push(part);
        draft.active = draft.parts.length - 1;
      });
    }
  }

  refreshEdit() {
    const d = this.draft;
    this.status.textContent = d.join ? (d.join.first ? S.CARTOGRAPHY_JOIN_SECOND : S.CARTOGRAPHY_JOIN_FIRST)
      : d.tool === 'Polygon' ? S.CARTOGRAPHY_PLACE_POLYGON
      : d.tool ? S.CARTOGRAPHY_PLACE
      : d.returnSelection ? S.CARTOGRAPHY_CHOOSE_TOOL
      : d.parts.length ? S.CARTOGRAPHY_DRAG : S.CARTOGRAPHY_EMPTY_EDIT;
    if (!d.tool && !d.kind) {
      this.parts.replaceChildren(...d.parts.map((p, i) => new Option(S.CARTOGRAPHY_PART(i + 1, p.type), i)));
      this.parts.value = String(d.active);
    }
    this.map.edit(d, {
      vertex: d.vertex,
      selectVertex: (ring, index) => { d.vertex = {ring, index}; this.updateButtons(); },
      selectPart: i => this.selectPart(i), place: ll => this.place(ll),
      finish: () => this.finishSketch(),
      endpoint: (part, index) => this.selectEndpoint(part, index),
      move: (r, i, pt) => this.editVertex('move', r, i, pt),
      insert: (r, i, pt) => this.editVertex('insert', r, i, pt),
    });
    this.updateButtons();
  }

  editVertex(operation, ring, index, point) {
    if (this.busy) return;
    this.changeParts(draft => {
      if (draft.kind === 'parcel' && draft.parts[draft.active].type === 'Polygon') {
        editBoundary(draft, operation, ring, index, point);
      } else {
        const edit = operation === 'move' ? moveVertex : insertVertex;
        edit(draft.parts[draft.active], ring, index, point);
      }
    });
  }

  removeSelectedVertex() {
    const d = this.draft;
    const {ring, index} = d.vertex;
    const part = d.parts[d.active];
    if (d.kind === 'parcel' && part.type === 'Polygon') {
      const removal = boundaryVertexRemoval(d, ring, index);
      if (removal) this.changeParts(draft => removeBoundaryVertex(draft, ring, index, removal));
    } else if (canRemoveVertex(part, ring, index)) {
      this.changeParts(draft => {
        if (removeVertex(part, ring, index) === null) removeActivePart(draft);
      });
    }
  }

  saveFeature(kind, action) {
    const {id, version, parts} = this.draft;
    if (parts.length) {
      this.write(action, {id, version, parts});
    } else {
      showConfirmModal(S.DELETE_REFERENCE_CONFIRM(this.selected.name),
        () => this.write(`remove-${kind}`, {id, version}));
    }
  }

  breakSelectedPart() {
    const d = this.draft;
    const {ring, index} = d.vertex;
    const part = d.parts[d.active], pieces = breakPart(part, ring, index);
    if (!pieces) return;
    this.changeParts(draft => {
      if (draft.kind === 'parcel' && part.type === 'Polygon') detachBoundary(draft, part);
      draft.parts.splice(draft.active, 1, ...pieces);
      draft.active += ring;
    });
  }

  selectEndpoint(part, index) {
    const d = this.draft;
    if (this.busy || !d?.join) return;
    const second = {part, index};
    const first = d.join.first;
    if (!first) {
      d.join.first = second;
      this.refreshEdit();
      return;
    }
    if (first.part === part && first.index === index) {
      d.join.first = null;
      this.refreshEdit();
      return;
    }
    this.changeParts(draft => {
      const joined = joinEndpoints(draft.parts, first, second, {close: draft.kind !== 'road'});
      if (joined === null) {
        this.status.textContent = draft.kind === 'road' ? S.CARTOGRAPHY_JOIN_ROAD : S.CARTOGRAPHY_JOIN_NEEDS_VERTICES;
        return false;
      }
      draft.active = joined;
      const result = draft.parts[joined];
      if (draft.kind === 'parcel' && result.type === 'Polygon') draft.added.push(result);
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
    const regionId = this.regionId;
    wireDrawingImportTabs(modal, {
      current: () => modal.isConnected && this.current(generation),
      save: (action, body, form) => this.write(action, body, form),
      loadSurveys: async () => (await fetchJSON(`${API}${regionId}/surveys/`)).data.surveys,
    });
    modal.querySelector('[data-action=cancel]').onclick = modals.dismiss;
    for (const button of modal.querySelectorAll('.modal-tab-body[data-path=ipso] [data-action]')) {
      const {action} = button.dataset;
      const select = action === 'import' ? uploads : sources;
      button.disabled = !select.options.length;
      button.onclick = () => {
        const id = Number(select.value);
        modals.dismiss();
        if (!this.current(generation)) return;
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
      case 'open-import':
        this.showImport();
        break;
      case 'new-drawing':
        this.startEdit({parts: [], active: 0, returnSelection: {kind: this.kind, id: row?.id}});
        break;
      case 'line':
      case 'polygon':
        Object.assign(this.draft, {
          tool: action === 'line' ? 'LineString' : 'Polygon', vertices: [], vertex: null, join: null,
        });
        this.refreshEdit();
        break;
      case 'edit':
        this.startEdit({id: row.id, version: row.version, parts: structuredClone(row.parts), active: 0});
        break;
      case 'break-link':
        this.breakSelectedPart();
        break;
      case 'join-link':
        this.draft.join = this.draft.join ? null : {first: null};
        this.draft.vertex = null;
        this.refreshEdit();
        break;
      case 'remove-part':
        this.changeParts(removeActivePart);
        break;
      case 'remove-vertex':
        this.removeSelectedVertex();
        break;
      case 'undo': {
        const d = this.draft;
        if (!undoEdit(d)) break;
        d.vertex = d.join = null;
        this.refreshEdit();
        break;
      }
      case 'cancel': {
        const d = this.draft;
        const selection = d.returnSelection ?? {kind: d.kind || 'drawing', id: d.id};
        const message = d.kind === 'parcel' ? S.CARTOGRAPHY_CANCEL_BOUNDARIES :
          d.kind === 'road' ? S.CARTOGRAPHY_CANCEL_POINTS : S.CARTOGRAPHY_CANCEL;
        showConfirmModal(message, async () => {
          if (d.tool && !d.returnSelection) {
            cancelSketch(d);
            this.refreshEdit();
            return;
          }
          drafts.delete(this.regionId);
          this.draft = null;
          this.map.render(this.state);
          this.select(selection.kind, selection.id);
          this.status.textContent = '';
          if (d.refreshOnCancel) {
            try { await this.reload(this.generation, selection); }
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
      case 'save': {
        const {kind, parcelId, review, edits, added, parts} = this.draft;
        if (kind !== 'parcel') {
          this.saveFeature(kind || 'drawing', kind === 'road' ? 'road-points' : 'adjust');
          break;
        }
        const save = () => this.write('parcel-boundaries', {id: parcelId, review, edits, added});
        if (parts.some(part => part.type !== 'Polygon')) {
          showConfirmModal(S.CARTOGRAPHY_SAVE_OPEN_BOUNDARIES, save, {confirmLabel: S.SAVE, intent: 'confirm'});
        } else {
          save();
        }
        break;
      }
      case 'finish':
        this.finishSketch();
        break;
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
      case 'remove-item': {
        const {kind, id} = target;
        const item = kind === 'parcel' ? this.state.parcels.find(parcel => parcel.name === id)
          : this.state[kind === 'road' ? 'roads' : 'drawings'].find(entry => entry.id === Number(id));
        showConfirmModal(S.DELETE_REFERENCE_CONFIRM(item.name),
          () => this.write(`remove-${kind}`, {id: item.id, version: item.version}));
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
      if (['adjust', 'road', 'create', 'parcel', 'road-points', 'parcel-boundaries',
        'remove-drawing', 'remove-road'].includes(action) && drafts.get(regionId) === submitted) {
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
      if (action === 'import-file' || action === 'import-survey') {
        this.map.frame('drawing', data.drawing_id, firstImport);
      }
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
