/**
 * Shared tree-detail visualisation: point map plus dendrometric summary.
 *
 * Pages retain ownership of their TableWrapper and pass its filtered rows to
 * setRows(). This component owns only presentation and delegates row actions
 * and export policy to callbacks supplied by the page.
 */

import { cloneTemplate } from './templates.js';
import {
  aggregateTreeDendrometry, clearDendrometrySummaryInfo,
  renderDendrometryBarCharts, renderDendrometryLegend,
  renderDendrometrySummaryInfo,
} from './dendrometry.js';
import { TreePointsMap, treePointsFromDigest } from './tree-points-map.js';
import * as S from './strings.js';

export class TreeDetail {
  constructor({
    container,
    digest,
    geojson = null,
    basemap = undefined,
    speciesNames = [],
    pointColumnNames = undefined,
    onTreeClick = null,
    onExport = null,
    emptyMessage = S.TREE_DENDROMETRY_EMPTY,
  }) {
    this.container = container;
    this.columns = digest?.columns || [];
    this.geojson = geojson;
    this.basemap = basemap;
    this.speciesNames = speciesNames;
    this.pointColumnNames = pointColumnNames;
    this.onTreeClick = onTreeClick;
    this.onExport = onExport;
    this.emptyMessage = emptyMessage;
    this.rows = [];
    this.map = null;
    this.mapHost = null;
    this.charts = {};

    if (digest?.rows?.length && geojson && typeof L !== 'undefined') {
      this._mountMap();
    }
    const fragment = cloneTemplate('tmpl-tree-dendrometry-summary');
    this.root = fragment.querySelector('.tree-dendrometry-summary');
    this.exportButton = fragment.querySelector('[data-action="export-dendrometry"]');
    this.exportButton?.addEventListener('click', () => this._export());
    container.appendChild(fragment);
    this.setRows(digest?.rows || []);
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
    this.map?.wrapper?.syncBasemap(name);
  }

  destroy() {
    this.map?.destroy();
    this.map = null;
    this.mapHost?.remove();
    this.mapHost = null;
    this._destroyCharts();
    this.root?.remove();
    this.root = null;
  }

  _mountMap() {
    this.mapHost = document.createElement('div');
    this.mapHost.className = 'tree-detail-map-host';
    this.container.appendChild(this.mapHost);
    this.map = new TreePointsMap({
      container: this.mapHost,
      className: 'tree-detail-map',
      geojson: this.geojson,
      onTreeClick: this.onTreeClick,
      basemap: this.basemap,
    });
  }

  _renderSummary() {
    if (!this.root) return;
    const status = this.root.querySelector('[data-target="dendrometry-status"]');
    const chartGrid = this.root.querySelector('[data-target="dendrometry-chart-grid"]');
    const legendRow = this.root.querySelector('[data-target="dendrometry-species-row"]');
    const legend = this.root.querySelector('[data-target="dendrometry-species"]');
    const rows = aggregateTreeDendrometry(this.rows, this.columns, {
      allSpeciesNames: this.speciesNames,
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
