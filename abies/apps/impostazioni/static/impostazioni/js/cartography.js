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
    this.buttons = Object.fromEntries([...panel.querySelectorAll('[data-action]')].map(b => [b.dataset.action, b]));
    const listen = (element, event, fn) => element.addEventListener(event, fn, {signal: this.life.signal});
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

  showRegionMap(region) {
    this.map?.destroy();
    this.map = new CartographyMap(this.panel.querySelector('#cartography-map'), this.geo, region.name,
      (kind, id) => this.select(kind, id), mapTypeName(this.mt), ll => this.place(ll), this.mapTools);
    const names = [...this.map.parcelLayers.keys()].sort((a, b) =>
      a.localeCompare(b, document.documentElement.lang, {numeric: true}));
    this.renderList('parcel', names.map(name => ({id: name, name})));
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
    this.renderList('drawing', data.drawings.sort(byName));
    this.renderList('road', data.roads.sort(byName));
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
    this.map.select(kind, this.selected);
    if (frame && this.selected) this.map.frame(kind, id);
    this.updateButtons();
  }

  updateButtons() {
    const b = this.buttons, draft = this.draft;
    const locked = !!(this.busy || draft);
    const drawing = this.state && this.kind === 'drawing' && this.selected;
    const parts = draft?.parts ?? drawing?.parts;
    const part = draft?.parts?.[draft.active];
    b.edit.disabled = locked || !drawing;
    b.line.disabled = b.polygon.disabled = locked || !this.state;
    const road = parts?.length === 1 && parts[0].type === 'LineString' && this.catalog?.road_types.length;
    const parcel = parts?.length && parts.every(part => part.type === 'Polygon');
    b.promote.disabled = !!this.busy || !(road || parcel);
    this.panel.querySelector('[data-role=edit-actions]').hidden = !part;
    this.panel.querySelector('[data-role=parts]').hidden = !part;
    this.mapTools.querySelector('[data-role=draw-tools]').hidden = !!draft;
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
    d.previous = {parts: structuredClone(d.parts), active: d.active};
    change(d);
    d.vertex = null;
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
      vertex: d.vertex,
      selectVertex: i => { d.vertex = i; this.updateButtons(); },
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
      case 'open-import':
        this.showImport();
        break;
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
      case 'cancel-sketch':
        showConfirmModal(S.CARTOGRAPHY_CANCEL, async () => {
          const refresh = this.draft.refreshOnCancel;
          drafts.delete(this.regionId);
          this.draft = null;
          this.map.render(this.state);
          this.select(row ? 'drawing' : null, row?.id);
          this.status.textContent = '';
          if (refresh) {
            try { await this.reload(this.generation, {kind: 'drawing', id: row?.id}); }
            catch { modals.showError(S.ERROR_NETWORK); }
          }
        }, {confirmLabel: S.CARTOGRAPHY_DISCARD});
        break;
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

  async write(action, body) {
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
          if (this.draft) this.draft.refreshOnCancel = true;
          else await this.reload(generation);
          modals.showError(data.message || S.ERROR_GENERIC);
        }
        return;
      }
      cache.applyResponseChanges(data);
      if (action === 'import') updateInboxDot(data);
      if (['adjust', 'road', 'create', 'parcel'].includes(action) && drafts.get(regionId) === submitted) {
        drafts.delete(regionId);
      }
      if (!this.current(generation)) return;
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
        && !this.map.parcelLayers.size;
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
