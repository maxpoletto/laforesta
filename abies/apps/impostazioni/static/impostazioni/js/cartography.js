import { fetchJSON, postJSON } from '../../base/js/api.js';
import { renderModalForm } from '../../base/js/forms.js';
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
import { updateInboxDot } from '../../ipso/js/inbox-status.js';

import { closeLine, handPart, insertVertex, joinParts, moveVertex } from './drawing-parts.js';

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
    this.drawings = panel.querySelector('#cartography-drawings');
    this.roads = panel.querySelector('#cartography-roads');
    this.parcels = panel.querySelector('#cartography-parcels');
    this.uploads = panel.querySelector('#cartography-uploads');
    this.sources = panel.querySelector('#cartography-sources');
    this.parts = panel.querySelector('#cartography-parts');
    this.status = panel.querySelector('[role=status]');
    this.buttons = Object.fromEntries([...panel.querySelectorAll('[data-action]')].map(b => [b.dataset.action, b]));
    const listen = (element, event, fn) => element.addEventListener(event, fn, {signal: this.life.signal});
    listen(this.parts, 'change', () => this.selectPart(Number(this.parts.value)));
    listen(this.region, 'change', () => this.navigate(this.region.value));
    listen(this.drawings, 'change', () => this.select('drawing', Number(this.drawings.value), true));
    listen(this.parcels, 'change', () => this.select('parcel', this.parcels.value, true));
    listen(this.roads, 'change', () => this.select('road', Number(this.roads.value), true));
    listen(panel, 'click', e => {
      const action = e.target.closest('button')?.dataset.action;
      if (action && !this.busy) this.act(action);
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

  showRegionMap(region) {
    this.map?.destroy();
    this.map = new CartographyMap(this.panel.querySelector('#cartography-map'), this.geo, region.name,
      (kind, id) => this.select(kind, id), mapTypeName(this.mt), ll => this.place(ll));
    const names = [...this.map.parcelLayers.keys()].sort((a, b) =>
      a.localeCompare(b, document.documentElement.lang, {numeric: true}));
    this.parcels.replaceChildren(...names.map(name => new Option(name, name)));
    this.map.map.on('basemapchange', e => this.navigate(region.id, mapTypeToken(e.name)));
  }

  current(generation) {
    return !this.life.signal.aborted && this.generation === generation;
  }

  async reload(generation, selection) {
    const {data} = await fetchJSON(API + this.regionId + '/');
    if (!this.current(generation)) return;
    this.state = data;
    fill(this.uploads, data.uploads, sourceLabel);
    fill(this.sources, data.sources, sourceLabel);
    const byName = (a, b) => a.name.localeCompare(b.name, document.documentElement.lang, {numeric: true});
    fill(this.drawings, data.drawings.sort(byName));
    fill(this.roads, data.roads.sort(byName));
    this.map.render(data);
    const draft = drafts.get(this.regionId);
    this.draft = null;
    if (draft && (draft.tool || data.drawings.some(d => d.id === draft.id))) {
      this.select('drawing', draft.id);
      this.startEdit(draft);
    } else {
      drafts.delete(this.regionId);
      this.select(selection?.kind, selection?.id);
    }
  }

  select(kind, id, frame = false) {
    if (!this.state || this.draft) return;
    this.kind = kind;
    const rows = kind === 'drawing' ? this.state.drawings : this.state.roads;
    this.selected = kind === 'parcel' ? {id} : rows.find(row => row.id === id);
    this.parcels.value = kind === 'parcel' ? id : '';
    this.drawings.value = kind === 'drawing' ? String(id) : '';
    this.roads.value = kind === 'road' ? String(id) : '';
    this.map.select(kind, this.selected);
    if (frame && this.selected) this.map.frame(kind, id);
    this.updateButtons();
  }

  updateButtons() {
    const b = this.buttons, draft = this.draft;
    const locked = !!(this.busy || draft);
    const drawing = this.kind === 'drawing' && this.selected;
    const parts = draft?.parts ?? drawing?.parts;
    const part = draft?.parts?.[draft.active];
    b.edit.disabled = locked || !drawing;
    b.line.disabled = b.polygon.disabled = locked || !this.state;
    const road = parts?.length === 1 && parts[0].type === 'LineString' && this.catalog?.road_types.length;
    const parcel = parts?.length && parts.every(part => part.type === 'Polygon');
    b.promote.disabled = !!this.busy || !(road || parcel);
    this.panel.querySelector('[data-role=edit-actions]').hidden = !draft;
    this.panel.querySelector('[data-role=parts]').hidden = !part;
    b.save.disabled = !!this.busy || !!(draft?.tool && draft.vertices.length < (draft.tool === 'Polygon' ? 3 : 2));
    b.undo.disabled = !!this.busy || !(draft?.tool ? draft.vertices.length : draft?.previous);
    b.cancel.disabled = b.copy.disabled = !!this.busy;
    b['remove-part'].disabled = !!this.busy || !part || parts.length < 2;
    b.join.disabled = !!this.busy || part?.type !== 'LineString' || parts[draft.active + 1]?.type !== 'LineString';
    b['close-line'].disabled = !!this.busy || part?.type !== 'LineString' || part.coordinates.length < 3;
    b['remove-drawing'].disabled = locked || !drawing;
    b.clear.disabled = locked || !this.state?.drawings.length;
    b.import.disabled = locked || !this.state?.uploads.length;
    b.restore.disabled = b.forget.disabled = locked || !this.state?.sources.length;
    b['new-region'].disabled = locked;
    b['edit-region'].disabled = locked || !this.state;
    this.region.disabled = this.parcels.disabled = this.drawings.disabled = this.roads.disabled = locked;
  }

  startEdit(draft) {
    this.draft = draft;
    drafts.set(this.regionId, draft);
    this.status.textContent = draft.tool ? S.CARTOGRAPHY_PLACE : S.CARTOGRAPHY_DRAG;
    this.refreshEdit();
  }

  selectPart(index) {
    this.draft.active = index;
    this.refreshEdit();
  }

  changeParts(change) {
    const d = this.draft;
    d.previous = {parts: structuredClone(d.parts), active: d.active};
    change(d);
    this.refreshEdit();
  }

  place(ll) {
    if (!this.draft?.tool || this.busy) return false;
    this.draft.vertices.push([ll.wrap().lng, ll.lat]);
    this.refreshEdit();
    return true;
  }

  refreshEdit() {
    const d = this.draft;
    if (!d.tool) {
      this.parts.replaceChildren(...d.parts.map((p, i) => new Option(S.CARTOGRAPHY_PART(i + 1, p.type), i)));
      this.parts.value = String(d.active);
    }
    this.map.edit(d, {
      selectPart: i => this.selectPart(i), place: ll => this.place(ll),
      move: (r, i, pt) => this.changeParts(d => moveVertex(d.parts[d.active], r, i, pt)),
      insert: (r, i, pt) => this.changeParts(d => insertVertex(d.parts[d.active], r, i, pt)),
    });
    this.updateButtons();
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

  async act(action) {
    const row = this.selected;
    switch (action) {
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
      case 'import':
        this.write(action, {upload_id: Number(this.uploads.value)});
        break;
      case 'restore':
        this.write(action, {source_id: Number(this.sources.value)});
        break;
      case 'forget': {
        const id = Number(this.sources.value);
        const source = this.state.sources.find(s => s.id === id);
        showConfirmModal(S.CARTOGRAPHY_FORGET(source.name), () => this.write(action, {source_id: id}));
        break;
      }
      case 'line':
      case 'polygon':
        this.select(null);
        this.startEdit({tool: action === 'line' ? 'LineString' : 'Polygon', vertices: []});
        break;
      case 'edit':
        this.startEdit({id: row.id, version: row.version, parts: structuredClone(row.parts), active: 0});
        break;
      case 'join':
        this.changeParts(d => joinParts(d.parts, d.active));
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
        showConfirmModal(S.CARTOGRAPHY_CANCEL, () => {
          drafts.delete(this.regionId);
          this.draft = null;
          this.map.render(this.state);
          this.select(row ? 'drawing' : null, row?.id);
          this.status.textContent = '';
        }, {confirmLabel: S.CARTOGRAPHY_DISCARD});
        break;
      case 'copy': {
        const d = this.draft;
        this.form('tmpl-cartography-drawing', body => this.write('copy', {
          ...body, id: d.id, version: d.version, parts: [d.parts[d.active]],
        }));
        break;
      }
      case 'save': {
        const d = this.draft;
        if (d.tool) this.form('tmpl-cartography-drawing', body =>
          this.write('create', {...body, parts: [handPart(d.tool, d.vertices)]}));
        else this.write('adjust', {id: d.id, version: d.version, parts: d.parts});
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
      case 'remove-drawing':
        showConfirmModal(S.DELETE_REFERENCE_CONFIRM(row.name), () => this.write(action, {id: row.id, version: row.version}));
        break;
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
      for (const select of [this.drawings, this.parcels, this.roads, this.uploads, this.sources]) {
        select.replaceChildren();
      }
      this.status.textContent = '';
      this.navigate(id || this.catalog.regions[0]?.id);
    } catch {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  async write(action, body) {
    if (this.busy || this.life.signal.aborted) return;
    const generation = this.generation;
    this.busy = true;
    this.updateButtons();
    try {
      const path = this.regionId + '/' + action;
      const {data, status} = await postJSON(API + path + '/', {...body, nonce: crypto.randomUUID()});
      if (status >= 400) {
        if (this.current(generation)) {
          if (!this.draft) await this.reload(generation);
          modals.showError(data.message || S.ERROR_GENERIC);
        }
        return;
      }
      cache.applyResponseChanges(data);
      if (action === 'import') updateInboxDot(data);
      if (!this.current(generation)) return;
      if (['adjust', 'road', 'create', 'parcel'].includes(action)) drafts.delete(this.regionId);
      let selection = data.drawing_id && {kind: 'drawing', id: data.drawing_id};
      if (action === 'adjust') selection = {kind: 'drawing', id: body.id};
      if (action === 'road') selection = {kind: 'road', id: data.road_id};
      if (data.parcel_id) {
        const geo = await cache.load('terreni');
        if (!this.current(generation)) return;
        this.geo = sortFeaturesByArea(geo);
        this.showRegionMap(this.catalog.regions.find(region => region.id === this.regionId));
        selection = {kind: 'parcel', id: data.parcel_name};
      }
      const firstImport = !this.state.drawings.length && !this.state.roads.length
        && !this.map.parcels.parcelLayer.getBounds().isValid();
      await this.reload(generation, selection);
      if (!this.current(generation)) return;
      if (!this.draft) this.status.textContent = data.added == null ? '' : S.CARTOGRAPHY_IMPORT_COUNT(data.added);
      if (data.parcel_id) this.map.frame('parcel', data.parcel_name);
      if (data.added) this.map.fitSource(this.state.drawings, data.source_id || body.source_id, firstImport);
    } catch (error) {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    } finally {
      if (this.current(generation)) { this.busy = false; this.updateButtons(); }
    }
  }

  destroy() {
    this.life.abort();
    this.map?.destroy();
  }
}
