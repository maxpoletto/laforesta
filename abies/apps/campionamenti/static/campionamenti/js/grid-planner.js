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
import { parseDecimal } from '../../base/js/format.js';
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
   * @param {function(): void} [opts.onCancel] — called when the user clicks
   *   the planner's [Annulla] button.  The planner builds that button lazily
   *   (on the auto-path switch), after the modal-level wireCancelButtons()
   *   has run, so it wires the button itself rather than relying on it.
   * @param {string} [opts.basemap] — MapCommon basemap key for the modal
   *   map's initial layer.  Defaults to 'satellite' to preserve previous
   *   behaviour when the caller doesn't pass one.
   */
  constructor(opts) {
    this.host = opts.host;
    this.onCreated = opts.onCreated;
    this.onCancel = opts.onCancel;
    this.basemap = opts.basemap || 'satellite';
    this.featuresByCompresa = {};   // compresa → [GeoJSON features]
    this.points = [];                // {lat, lon, compresa, particella}
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

    // Grid name + description.  `type: 'text'` is mandatory: the
    // common.css selectors are attribute-explicit (`input[type="text"]`,
    // `input[type="number"]`, …) and don't match a bare `<input>`, so
    // without it Nome falls back to user-agent default styling and
    // visibly mismatches Raggio / Copertura below.
    h.appendChild(this._labelInput(S.LABEL_NAME, 'input', {
      id: 'grid-auto-name', type: 'text', required: true, maxlength: 100,
    }));
    h.appendChild(this._labelInput(S.LABEL_DESCRIPTION_OPTIONAL, 'textarea', {
      id: 'grid-auto-description', rows: 2,
    }));

    // Compresa multi-select.
    const compresaGroup = document.createElement('div');
    compresaGroup.className = 'form-group';
    const compresaLabel = document.createElement('label');
    compresaLabel.textContent = S.LABEL_REGIONS;
    compresaGroup.appendChild(compresaLabel);
    this.compreseListEl = document.createElement('div');
    this.compreseListEl.className = 'grid-planner-compresa-list';
    this.compreseListEl.textContent = S.LOADING;
    compresaGroup.appendChild(this.compreseListEl);
    h.appendChild(compresaGroup);

    // Parameters: raggio (m) + copertura %.
    const params = document.createElement('div');
    params.className = 'form-row grid-planner-params';
    params.appendChild(this._labelInput(S.LABEL_RADIUS_M, 'input', {
      id: 'grid-auto-radius', type: 'number', min: 1, step: 1, value: 12,
    }));
    params.appendChild(this._labelInput(S.LABEL_COVERAGE_PCT, 'input', {
      id: 'grid-auto-coverage', type: 'number',
      min: 0.01, max: 100, step: 0.1, value: 1,
    }));
    h.appendChild(params);

    const planBtn = document.createElement('button');
    planBtn.type = 'button';
    planBtn.className = 'btn';
    planBtn.textContent = S.ACTION_PLAN;
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
    this.wrapper = MapCommon.create(mapEl, { basemap: this.basemap });
    this.leaflet = this.wrapper.getLeafletMap();
    this.pointLayer = L.layerGroup().addTo(this.leaflet);

    // Bottom button row: [Annulla] [Crea].  Both are wired here: the planner
    // is built lazily on the auto-path switch, after the modal-level
    // wireCancelButtons() has already run, so the cancel button calls back
    // through onCancel (mirroring onCreated for [Crea]).
    const actions = document.createElement('div');
    actions.className = 'form-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn';
    cancelBtn.textContent = S.CANCEL;
    cancelBtn.addEventListener('click', () => this.onCancel?.());
    actions.appendChild(cancelBtn);

    this.submitBtn = document.createElement('button');
    this.submitBtn.type = 'button';
    this.submitBtn.className = 'btn btn-save';
    this.submitBtn.textContent = S.ACTION_CREATE;
    this.submitBtn.disabled = true;
    this.submitBtn.addEventListener('click', () => this._save());
    actions.appendChild(this.submitBtn);

    h.appendChild(actions);
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
      this._setStatus(S.ERR_SELECT_REGION);
      return;
    }
    const radius = parseInt(this.host.querySelector('#grid-auto-radius').value, 10);
    const pct = parseDecimal(this.host.querySelector('#grid-auto-coverage').value);
    if (!(radius > 0)) { this._setStatus(S.ERR_RADIUS_POSITIVE); return; }
    if (!(pct > 0 && pct <= 100)) {
      this._setStatus(S.ERR_COVERAGE_RANGE); return;
    }
    const totalAreaM2 = features.reduce((s, f) => s + featureArea(f), 0);
    const perPointAreaM2 = Math.PI * radius * radius;
    const targetN = Math.round((totalAreaM2 * pct / 100) / perPointAreaM2);
    if (targetN < 1) { this._setStatus(S.ERR_PARAMS_ZERO_POINTS); return; }

    // geo.js emits Leaflet-convention {lat, lng, …}; the rest of this grid
    // pipeline (render, _save payload, server, DB) speaks `lon`.  Normalize at
    // this single boundary so the points carry `lon` everywhere downstream.
    this.points = planGridForTarget(features, targetN).map(
      ({ lat, lng, compresa, particella }) =>
        ({ lat, lon: lng, compresa, particella }),
    );
    this._renderPoints();
    this._renderStats(totalAreaM2, perPointAreaM2, targetN);
    this.submitBtn.disabled = this.points.length === 0;
    this._setStatus(
      S.STATUS_PLAN_COMPLETE.replace('{n}', this.points.length),
    );
  }

  _renderPoints() {
    this.pointLayer.clearLayers();
    // Number each point per compresa, restarting at 1, to mirror
    // grid_save_auto_view's server-side numbering — otherwise the preview
    // adc numbers wouldn't match the saved grid.
    const countByCompresa = {};
    this.points.forEach((pt) => {
      const n = (countByCompresa[pt.compresa] || 0) + 1;
      countByCompresa[pt.compresa] = n;
      const m = L.circleMarker([pt.lat, pt.lon], POINT_STYLE);
      m.bindTooltip(
        S.TOOLTIP_ADC
          .replace('{n}', n)
          .replace('{compresa}', pt.compresa)
          .replace('{particella}', pt.particella),
      );
      m.addTo(this.pointLayer);
    });
  }

  _renderStats(totalAreaM2, perPointAreaM2, targetN) {
    this.statsEl.replaceChildren();
    const lines = [
      S.STATS_POINTS
        .replace('{n}', this.points.length)
        .replace('{target}', targetN),
      S.STATS_TOTAL_AREA_HA.replace('{ha}', (totalAreaM2 / 10000).toFixed(2)),
      S.STATS_AREA_PER_POINT_M2.replace('{area}', perPointAreaM2.toFixed(1)),
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
    const radius = parseInt(this.host.querySelector('#grid-auto-radius').value, 10);
    if (!name) { this._setStatus(S.ERR_GRID_NAME_REQUIRED); return; }
    if (!this.points.length) {
      this._setStatus(S.ERR_PLAN_FIRST); return;
    }
    this.submitBtn.disabled = true;
    this._setStatus(S.STATUS_SAVING);
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
      this.onCreated?.(data.row_id, data);
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
