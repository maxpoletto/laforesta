/**
 * Shared tree-detail visualization: point map plus dendrometric summary.
 *
 * Pages retain ownership of their TableWrapper and pass its filtered rows to
 * setRows(). The map can use a separate host or be disabled for an area-map
 * survey. This component owns presentation and delegates row actions
 * and export policy to callbacks supplied by the page.
 */

import { cloneTemplate } from './templates.js';
import {
  aggregateTreeDendrometry, clearDendrometrySummaryInfo,
  renderDendrometryBarCharts, renderDendrometryLegend,
  renderDendrometrySummaryInfo,
} from './dendrometry.js';
import { FIELD_DIAMETER_CLASS_MODE } from './constants.js';
import { compareParcelNames } from './natural-sort.js';
import { TreePointsMap, treePointsFromDigest } from './tree-points-map.js';
import * as S from './strings.js';

export class TreeDetail {
  constructor({
    container,
    mapContainer = container,
    mapEnabled = true,
    digest,
    geojson = null,
    basemap = undefined,
    speciesNames = [],
    pointColumnNames = undefined,
    onTreeClick = null,
    onExport = null,
    parcelSummary = false,
    emptyMessage = S.TREE_DENDROMETRY_EMPTY,
  }) {
    this.container = container;
    this.mapContainer = mapContainer;
    this.mapEnabled = mapEnabled;
    this.columns = digest?.columns || [];
    this.diameterClassMode = digest?.[FIELD_DIAMETER_CLASS_MODE];
    this.geojson = geojson;
    this.basemap = basemap;
    this.speciesNames = speciesNames;
    this.pointColumnNames = pointColumnNames;
    this.onTreeClick = onTreeClick;
    this.onExport = onExport;
    this.emptyMessage = emptyMessage;
    this.parcelSummary = parcelSummary;
    this.rows = [];
    this.map = null;
    this.mapHost = null;
    this.charts = {};
    this.parcelColumn = this.columns.indexOf(S.COL_PARCEL);
    this.selectedParcel = '';

    if (mapEnabled && digest?.rows?.length && geojson && typeof L !== 'undefined') {
      this._mountMap();
    }
    const fragment = cloneTemplate('tmpl-tree-dendrometry-summary');
    this.root = fragment.querySelector('.tree-dendrometry-summary');
    this.exportButton = fragment.querySelector('[data-action="export-dendrometry"]');
    this.exportButton?.addEventListener('click', () => this._export());
    if (parcelSummary) {
      this.root.querySelector('[data-target="dendrometry-parcel-select"]')
        ?.addEventListener('change', event => {
          this.selectedParcel = event.target.value;
          this._renderSummary();
        });
    }
    container.appendChild(fragment);
    this.setData(digest);
  }

  /** Update source membership/metadata separately from the table's filtered rows. */
  setData(digest, { rows = digest?.rows || [], geojson = this.geojson } = {}) {
    this.columns = digest?.columns || [];
    this.diameterClassMode = digest?.[FIELD_DIAMETER_CLASS_MODE];
    this.parcelColumn = this.columns.indexOf(S.COL_PARCEL);
    this.geojson = geojson;
    if (this.parcelSummary) this._syncParcelSelector(digest?.rows || []);
    if (!digest?.rows?.length) this._destroyMap();
    else if (this.mapEnabled && !this.map && geojson && typeof L !== 'undefined') this._mountMap();
    this.setRows(rows);
  }

  setRows(rows) {
    this.rows = [...(rows || [])];
    this.map?.setTrees(treePointsFromDigest(
      this.rows, this.columns, this.pointColumnNames,
    ));
    this._renderSummary();
  }

  /** Reflow a map that was initially mounted inside a hidden section. */
  showMap() {
    if (!this.map) return;
    this.map.invalidateSize();
    this.map.fitParcels();
  }

  /** Mirror a basemap change made on another map on the same page. */
  syncBasemap(name) {
    this.basemap = name;
    this.map?.wrapper?.syncBasemap(name);
  }

  _destroyMap() {
    this.map?.destroy();
    this.map = null;
    this.mapHost?.remove();
    this.mapHost = null;
  }

  destroy() {
    this._destroyMap();
    this._destroyCharts();
    this.root?.remove();
    this.root = null;
  }

  _mountMap() {
    this.mapHost = document.createElement('div');
    this.mapHost.className = 'tree-detail-map-host';
    if (this.root && this.mapContainer === this.container) this.root.before(this.mapHost);
    else this.mapContainer.appendChild(this.mapHost);
    this.map = new TreePointsMap({
      container: this.mapHost,
      className: 'tree-detail-map',
      geojson: this.geojson,
      onTreeClick: this.onTreeClick,
      basemap: this.basemap,
    });
  }

  _syncParcelSelector(rows) {
    const parcels = this.parcelColumn < 0 ? [] : [...new Set(rows
      .map(row => row[this.parcelColumn])
      .filter(name => name != null && name !== '')
      .map(String))].sort(compareParcelNames);
    const control = this.root?.querySelector('[data-target="dendrometry-parcels"]');
    const select = control?.querySelector('select');
    if (!select) return;
    // Keep the localized "all parcels" option supplied by the template.
    select.replaceChildren(select.children[0]);
    for (const parcel of parcels) {
      const option = document.createElement('option');
      option.value = parcel;
      option.textContent = parcel;
      select.appendChild(option);
    }
    if (parcels.length < 2 || !parcels.includes(this.selectedParcel)) this.selectedParcel = '';
    select.value = this.selectedParcel;
    control.hidden = parcels.length < 2;
  }

  _renderSummary() {
    if (!this.root) return;
    const status = this.root.querySelector('[data-target="dendrometry-status"]');
    const chartGrid = this.root.querySelector('[data-target="dendrometry-chart-grid"]');
    const legendRow = this.root.querySelector('[data-target="dendrometry-species-row"]');
    const legend = this.root.querySelector('[data-target="dendrometry-species"]');
    const sourceRows = this.selectedParcel
      ? this.rows.filter(row => String(row[this.parcelColumn]) === this.selectedParcel)
      : this.rows;
    const rows = aggregateTreeDendrometry(sourceRows, this.columns, {
      allSpeciesNames: this.speciesNames,
      diameterClassMode: this.diameterClassMode,
    });

    if (this.exportButton) this.exportButton.disabled = !this.rows.length;
    if (!rows.length) {
      this._destroyCharts();
      if (status) {
        status.textContent = this.emptyMessage;
        status.hidden = false;
      }
      if (chartGrid) chartGrid.hidden = true;
      if (legendRow) legendRow.hidden = true;
      legend?.replaceChildren();
      clearDendrometrySummaryInfo(this._infoHosts());
      return;
    }

    if (status) {
      status.textContent = '';
      status.hidden = true;
    }
    if (chartGrid) chartGrid.hidden = false;
    if (legendRow) legendRow.hidden = false;
    renderDendrometryLegend(legend, rows);
    this.charts = renderDendrometryBarCharts({
      rows,
      canvases: {
        treeCount: this.root.querySelector('[data-target="dendrometry-tree-count-chart"]'),
        volume: this.root.querySelector('[data-target="dendrometry-volume-chart"]'),
        basalArea: this.root.querySelector('[data-target="dendrometry-basal-area-chart"]'),
      },
      yTitles: {
        treeCount: S.BOSCO_TREE_COUNT,
        volume: S.COL_VOLUME_M3,
        basalArea: S.COL_BASAL_AREA_M2,
      },
      existing: this.charts,
    });
    renderDendrometrySummaryInfo(this._infoHosts(), rows);
  }

  _infoHosts() {
    return {
      treeCount: this.root?.querySelector('[data-target="dendrometry-tree-count-info"]'),
      volume: this.root?.querySelector('[data-target="dendrometry-volume-info"]'),
      basalArea: this.root?.querySelector('[data-target="dendrometry-basal-area-info"]'),
    };
  }

  _destroyCharts() {
    for (const chart of Object.values(this.charts)) chart?.destroy?.();
    this.charts = {};
  }

  async _export() {
    if (!this.onExport || !this.rows.length || !this.exportButton) return;
    this.exportButton.disabled = true;
    try {
      await this.onExport([...this.rows], this.exportButton);
    } finally {
      if (this.exportButton) this.exportButton.disabled = !this.rows.length;
    }
  }
}
