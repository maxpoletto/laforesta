import { fetchJSON, postJSON } from '../../base/js/api.js';
import * as cache from '../../base/js/cache.js';
import { sortFeaturesByArea } from '../../base/js/geo.js';
import * as modals from '../../base/js/modals.js';
import { navigateWithParams } from '../../base/js/page-sync.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { showConfirmModal } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { mapTypeName, mapTypeToken } from '../../base/js/map-common.js';
import { CartographyMap } from './cartography-map.js';

const API = '/api/impostazioni/cartography/';
const tab = 'cartografia';
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
    this.status = panel.querySelector('[role=status]');
    this.buttons = Object.fromEntries([...panel.querySelectorAll('[data-action]')].map(b => [b.dataset.action, b]));
    const listen = (element, event, fn) => element.addEventListener(event, fn, {signal: this.life.signal});
    listen(this.region, 'change', () => this.navigate(this.region.value));
    listen(this.drawings, 'change', () => this.select('drawing', Number(this.drawings.value), true));
    listen(this.parcels, 'change', () => this.select('parcel', this.parcels.value, true));
    listen(this.roads, 'change', () => this.select('road', Number(this.roads.value), true));
    listen(panel, 'click', e => {
      const action = e.target.closest('button')?.dataset.action;
      if (action && !this.busy) this.act(action);
    });
    this.ready = Promise.all([fetchJSON(API), cache.load('terreni')]).then(([response, geo]) => {
      this.catalog = response.data;
      this.geo = sortFeaturesByArea(geo);
      fill(this.region, this.catalog.regions);
    });
  }

  navigate(id, mt = this.mt) {
    const params = new URLSearchParams({scheda: tab, c: id});
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
      const region = this.catalog.regions.find(r => r.id === Number(params.c)) || this.catalog.regions[0];
      this.mt = params.mt;
      if (!region) { this.updateButtons(); return; }
      if (region.id === this.regionId && this.map) return;
      this.regionId = region.id;
      this.region.value = String(region.id);
      this.map?.destroy();
      this.map = new CartographyMap(this.panel.querySelector('#cartography-map'), this.geo, region.name,
        (kind, id) => this.select(kind, id), mapTypeName(params.mt));
      this.parcels.replaceChildren(...[...this.map.parcelLayers.keys()].map(name => new Option(name, name)));
      this.map.map.on('basemapchange', e => this.navigate(region.id, mapTypeToken(e.name)));
      this.state = null;
      this.editing = false;
      this.busy = false;
      this.status.textContent = '';
      this.updateButtons();
      await this.reload(generation);
      if (this.current(generation)) {
        this.map.fitRegion();
        if (params.fonte) this.map.fitSource(this.state.drawings, Number(params.fonte), true);
      }
    } catch (error) {
      if (this.current(generation)) modals.showError(S.ERROR_NETWORK);
    }
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
    fill(this.drawings, data.drawings, row => S.CARTOGRAPHY_DRAWING(row.name, row.sequence,
      row.geometry.type === 'Point' ? S.IPSO_POINT_LABEL : S.IPSO_TRACE_LABEL));
    fill(this.roads, data.roads, row => row.published ? row.name : S.CARTOGRAPHY_DRAFT(row.name));
    this.map.render(data);
    this.editing = false;
    this.select(selection?.kind, selection?.id);
  }

  select(kind, id, frame = false) {
    if (!this.state || this.editing) return;
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
    const b = this.buttons;
    const locked = this.busy || this.editing;
    const drawing = this.kind === 'drawing' && this.selected;
    b.edit.disabled = locked || !drawing;
    b.edit.hidden = !!this.editing;
    b.save.hidden = b.cancel.hidden = !this.editing;
    b.save.disabled = b.cancel.disabled = !!this.busy;
    b.road.disabled = locked || drawing?.geometry?.type !== 'LineString' || !this.catalog?.road_types.length;
    b['remove-drawing'].disabled = locked || !drawing;
    b['discard-road'].disabled = locked || this.kind !== 'road' || !this.selected || this.selected.published;
    b.clear.disabled = locked || !this.state?.drawings.length;
    b.publish.disabled = locked || !this.state?.roads.some(r => !r.published);
    b.import.disabled = locked || !this.state?.uploads.length;
    b.restore.disabled = b.forget.disabled = locked || !this.state?.sources.length;
    b['new-region'].disabled = !!locked;
    this.region.disabled = this.parcels.disabled = this.drawings.disabled = this.roads.disabled = !!locked;
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

  act(action) {
    const row = this.selected;
    switch (action) {
      case 'new-region':
        this.form('tmpl-cartography-region', body => this.write('regions', body));
        break;
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
      case 'edit':
        this.editing = true;
        this.map.select('drawing', row, true);
        this.status.textContent = S.CARTOGRAPHY_DRAG;
        this.updateButtons();
        break;
      case 'cancel':
        this.editing = false;
        this.map.render(this.state);
        this.select('drawing', row.id);
        this.status.textContent = '';
        break;
      case 'save':
        this.write('adjust', {id: row.id, version: row.version, geometry: this.map.geometry(row.geometry.type)});
        break;
      case 'road':
        this.form('tmpl-cartography-road', body => this.write(action, {...body, id: row.id, version: row.version}));
        break;
      case 'remove-drawing':
      case 'discard-road': {
        const label = action === 'remove-drawing' ? this.drawings.selectedOptions[0].textContent : row.name;
        showConfirmModal(S.DELETE_REFERENCE_CONFIRM(label), () => this.write(action, {id: row.id, version: row.version}));
        break;
      }
      case 'clear': {
        const snapshot = versions(this.state.drawings);
        const name = this.region.selectedOptions[0].textContent;
        showConfirmModal(S.CARTOGRAPHY_CLEAR(name, snapshot.length), () => this.write(action, {versions: snapshot}));
        break;
      }
      case 'publish': {
        const drafts = this.state.roads.filter(r => !r.published);
        const snapshot = versions(drafts);
        showConfirmModal(S.CARTOGRAPHY_PUBLISH(drafts.map(r => r.name)), () => this.write(action, {versions: snapshot}),
          {confirmLabel: S.CARTOGRAPHY_PUBLISH_LABEL, intent: 'confirm'});
        break;
      }
    }
  }

  async write(action, body) {
    if (this.busy || this.life.signal.aborted) return;
    const generation = this.generation;
    this.busy = true;
    this.updateButtons();
    try {
      const path = action === 'regions' ? 'regions' : this.regionId + '/' + action;
      const {data, status} = await postJSON(API + path + '/', {...body, nonce: crypto.randomUUID()});
      if (status >= 400) {
        if (this.current(generation)) {
          if (action !== 'regions') await this.reload(generation);
          modals.showError(data.message || S.ERROR_GENERIC);
        }
        return;
      }
      cache.applyResponseChanges(data);
      if (!this.current(generation)) return;
      if (action === 'regions') {
        this.catalog.regions.push({id: data.region_id, name: body.name.trim()});
        fill(this.region, this.catalog.regions);
        this.navigate(data.region_id);
        return;
      }
      let selection;
      if (action === 'adjust') selection = {kind: 'drawing', id: body.id};
      if (action === 'road') selection = {kind: 'road', id: data.road_id};
      const firstImport = !this.state.drawings.length && !this.state.roads.length
        && !this.map.parcels.parcelLayer.getBounds().isValid();
      await this.reload(generation, selection);
      if (!this.current(generation)) return;
      this.status.textContent = data.added == null ? '' : S.CARTOGRAPHY_IMPORT_COUNT(data.added);
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
