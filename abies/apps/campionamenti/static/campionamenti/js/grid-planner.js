/**
 * "Genera automaticamente" grid planner — Campionamenti modal.
 *
 * Logic vendored from `bosco/pac/app.js`, generalized to:
 *   - support multiple comprese in one run (spec §1, lines 21–22);
 *   - take a radius (matching schema r_m) instead of pac's diameter;
 *   - persist the result via /api/campionamenti/grid/save-auto/
 *     instead of exporting a CSV.
 */

import MapCommon from '../../base/js/map-common.js';
import {
  featureArea, planGridForTarget,
} from '../../base/js/geo.js';
import { fetchJSON, postJSON } from '../../base/js/api.js';
import { showError } from '../../base/js/modals.js';
import * as S from '../../base/js/strings.js';

const TERRENI_URL = '/api/geo/terreni.geojson';
const SAVE_URL = '/api/campionamenti/grid/save-auto/';

const POINT_STYLE = {
  radius: 5, fillColor: '#ff4444', color: '#000', weight: 1, opacity: 1,
  fillOpacity: 0.8,
};
const PARCEL_STYLE = {
  color: '#3388ff', weight: 1.5, opacity: 0.8, fillOpacity: 0.08,
};

export class GridPlanner {
  /**
   * @param {object} opts
   * @param {HTMLElement} opts.host — element to render the planner UI into.
   * @param {function(number): void} opts.onCreated — called after a
   *   successful save with the new SampleGrid.id.
   */
  constructor(opts) {
    this.host = opts.host;
    this.onCreated = opts.onCreated;
    this.featuresByCompresa = {};   // compresa → [GeoJSON features]
    this.points = [];                // {lat, lng, compresa, particella}
    this.leaflet = null;
    this.parcelLayer = null;
    this.pointLayer = null;
    this.statusEl = null;
    this.statsEl = null;
    this.submitBtn = null;
    this._built = false;
  }

  async init() {
    if (this._built) return;
    this._built = true;
    this._buildUI();
    try {
      const { data } = await fetchJSON(TERRENI_URL);
      this._loadFeatures(data);
    } catch {
      showError(S.ERROR_NETWORK);
    }
  }

  _buildUI() {
    const h = this.host;
    h.replaceChildren();
    h.classList.add('grid-planner');

    // Grid name + description.
    h.appendChild(this._labelInput('Nome', 'input', {
      id: 'grid-auto-name', required: true, maxlength: 100,
    }));
    h.appendChild(this._labelInput('Descrizione (opzionale)', 'textarea', {
      id: 'grid-auto-description', rows: 2,
    }));

    // Compresa multi-select.
    const compresaGroup = document.createElement('div');
    compresaGroup.className = 'form-group';
    const compresaLabel = document.createElement('label');
    compresaLabel.textContent = 'Comprese';
    compresaGroup.appendChild(compresaLabel);
    this.compreseListEl = document.createElement('div');
    this.compreseListEl.className = 'grid-planner-compresa-list';
    this.compreseListEl.textContent = S.LOADING;
    compresaGroup.appendChild(this.compreseListEl);
    h.appendChild(compresaGroup);

    // Parameters: raggio (m) + copertura %.
    const params = document.createElement('div');
    params.className = 'form-row grid-planner-params';
    params.appendChild(this._labelInput('Raggio (m)', 'input', {
      id: 'grid-auto-radius', type: 'number', min: 1, step: 1, value: 12,
    }));
    params.appendChild(this._labelInput('Copertura (%)', 'input', {
      id: 'grid-auto-coverage', type: 'number',
      min: 0.01, max: 100, step: 0.1, value: 1,
    }));
    h.appendChild(params);

    // Pianifica button.
    const planBtn = document.createElement('button');
    planBtn.type = 'button';
    planBtn.className = 'btn btn-secondary';
    planBtn.textContent = 'Pianifica';
    planBtn.addEventListener('click', () => this._plan());
    h.appendChild(planBtn);

    // Stats / status panel.
    this.statusEl = document.createElement('div');
    this.statusEl.className = 'grid-planner-status';
    h.appendChild(this.statusEl);

    this.statsEl = document.createElement('div');
    this.statsEl.className = 'grid-planner-stats';
    h.appendChild(this.statsEl);

    // Leaflet map.
    const mapEl = document.createElement('div');
    mapEl.id = 'grid-planner-map';
    mapEl.className = 'grid-planner-map';
    h.appendChild(mapEl);
    this.wrapper = MapCommon.create(mapEl, { basemap: 'satellite' });
    this.leaflet = this.wrapper.getLeafletMap();
    this.pointLayer = L.layerGroup().addTo(this.leaflet);

    // Crea griglia submit.
    this.submitBtn = document.createElement('button');
    this.submitBtn.type = 'button';
    this.submitBtn.className = 'btn btn-primary';
    this.submitBtn.textContent = 'Crea griglia';
    this.submitBtn.disabled = true;
    this.submitBtn.addEventListener('click', () => this._save());
    h.appendChild(this.submitBtn);
  }

  _labelInput(labelText, tag, attrs) {
    const group = document.createElement('div');
    group.className = 'form-group';
    const label = document.createElement('label');
    label.textContent = labelText;
    label.htmlFor = attrs.id;
    group.appendChild(label);
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'value') el.value = v;
      else el.setAttribute(k, String(v));
    }
    group.appendChild(el);
    return group;
  }

  _loadFeatures(geojson) {
    const polys = (geojson.features || []).filter(
      f => f.properties && f.properties.type === 'polygon'
        && f.geometry && f.geometry.type === 'Polygon',
    );
    this.featuresByCompresa = {};
    for (const f of polys) {
      const name = f.properties.layer;
      if (!name) continue;
      if (!this.featuresByCompresa[name]) this.featuresByCompresa[name] = [];
      this.featuresByCompresa[name].push(f);
    }
    // Render the compresa checklist.
    this.compreseListEl.replaceChildren();
    const names = Object.keys(this.featuresByCompresa).sort();
    for (const name of names) {
      const lab = document.createElement('label');
      lab.className = 'grid-planner-compresa-item';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = name;
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(
        ` ${name} (${this.featuresByCompresa[name].length} particelle)`,
      ));
      this.compreseListEl.appendChild(lab);
    }
    // Draw parcel polygons (all comprese) for visual context.
    this.parcelLayer = L.geoJSON(
      { type: 'FeatureCollection', features: polys },
      { style: PARCEL_STYLE },
    ).addTo(this.leaflet);
    if (this.parcelLayer.getBounds().isValid()) {
      this.leaflet.fitBounds(this.parcelLayer.getBounds());
    }
  }

  _selectedComprese() {
    return [...this.compreseListEl.querySelectorAll('input:checked')]
      .map(cb => cb.value);
  }

  _selectedFeatures() {
    return this._selectedComprese().flatMap(
      name => this.featuresByCompresa[name] || [],
    );
  }

  _plan() {
    const features = this._selectedFeatures();
    if (!features.length) {
      this._setStatus('Seleziona almeno una compresa.');
      return;
    }
    const radius = parseFloat(this.host.querySelector('#grid-auto-radius').value);
    const pct = parseFloat(this.host.querySelector('#grid-auto-coverage').value);
    if (!(radius > 0)) { this._setStatus('Raggio deve essere > 0.'); return; }
    if (!(pct > 0 && pct <= 100)) {
      this._setStatus('Copertura deve essere tra 0 e 100%.'); return;
    }
    const totalAreaM2 = features.reduce((s, f) => s + featureArea(f), 0);
    const perPointAreaM2 = Math.PI * radius * radius;
    const targetN = Math.round((totalAreaM2 * pct / 100) / perPointAreaM2);
    if (targetN < 1) { this._setStatus('Parametri danno 0 punti.'); return; }

    this.points = planGridForTarget(features, targetN);
    this._renderPoints();
    this._renderStats(totalAreaM2, perPointAreaM2, targetN);
    this.submitBtn.disabled = this.points.length === 0;
    this._setStatus(
      `Pianificazione completata: ${this.points.length} punti.`,
    );
  }

  _renderPoints() {
    this.pointLayer.clearLayers();
    this.points.forEach((pt, i) => {
      const m = L.circleMarker([pt.lat, pt.lng], POINT_STYLE);
      m.bindTooltip(
        `adc ${i + 1} · ${pt.compresa} ${pt.particella}`,
      );
      m.addTo(this.pointLayer);
    });
  }

  _renderStats(totalAreaM2, perPointAreaM2, targetN) {
    this.statsEl.replaceChildren();
    const lines = [
      `Punti: ${this.points.length} (obiettivo: ${targetN})`,
      `Superficie totale: ${(totalAreaM2 / 10000).toFixed(2)} ha`,
      `Area singola adc: ${perPointAreaM2.toFixed(1)} m²`,
    ];
    for (const t of lines) {
      const div = document.createElement('div');
      div.textContent = t;
      this.statsEl.appendChild(div);
    }
  }

  _setStatus(msg) {
    if (this.statusEl) this.statusEl.textContent = msg;
  }

  async _save() {
    const name = this.host.querySelector('#grid-auto-name').value.trim();
    const description = this.host.querySelector('#grid-auto-description').value.trim();
    const radius = parseFloat(this.host.querySelector('#grid-auto-radius').value);
    if (!name) { this._setStatus('Nome richiesto.'); return; }
    if (!this.points.length) {
      this._setStatus('Esegui prima "Pianifica".'); return;
    }
    this.submitBtn.disabled = true;
    this._setStatus('Salvataggio in corso...');
    try {
      const { data, status } = await postJSON(SAVE_URL, {
        name, description, r_m: Math.round(radius),
        points: this.points,
      });
      if (status !== 200) {
        this.submitBtn.disabled = false;
        this._setStatus(data?.message || S.ERROR_GENERIC);
        return;
      }
      this.onCreated?.(data.row_id);
    } catch (err) {
      this.submitBtn.disabled = false;
      this._setStatus(err?.message || S.ERROR_GENERIC);
    }
  }

  destroy() {
    if (this.leaflet) {
      this.leaflet.remove();
      this.leaflet = null;
    }
    this.host?.replaceChildren();
    this._built = false;
  }
}
