import { ROW_ID } from '../../base/js/constants.js';
import { fmtArea, fmtDecimal2, fmtInt, fmtVolume } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';
import { TableWrapper } from '../../base/js/table.js';
import { regionalProductionYears, regionalYield } from './bosco-production.js';

const COLUMNS = [
  ROW_ID, S.BOSCO_YIELD_HARVEST_YEARS, S.COL_PARCEL, S.COL_GOVERNANCE, S.COL_VOLUME_M3,
  S.BOSCO_YIELD_AREA, S.BOSCO_YIELD_AREA_SOURCE,
];
const COLUMN_DEFS = {
  [S.BOSCO_YIELD_HARVEST_YEARS]: {
    type: 'natural', width: '140px', className: 'numeric', cellClassName: 'numeric',
  },
  [S.COL_PARCEL]: { type: 'natural', searchMatch: 'exact', width: '100px' },
  [S.COL_GOVERNANCE]: { width: '95px' },
  [S.COL_VOLUME_M3]: { type: 'number', formatter: fmtDecimal2, width: '120px' },
  [S.BOSCO_YIELD_AREA]: {
    type: 'number', formatter: value => fmtDecimal2(value) || '-', width: '155px',
  },
  [S.BOSCO_YIELD_AREA_SOURCE]: { width: '170px' },
};

export function wireProductionReport(root, onFilterChange) {
  if (!root) return null;
  const governance = root.querySelector('[data-role="production-governance"]');
  const firstYear = root.querySelector('[data-role="production-year-from"]');
  const lastYear = root.querySelector('[data-role="production-year-to"]');
  const summary = root.querySelector('[data-target="production-yield"]');
  const context = root.querySelector('[data-target="production-yield-context"]');
  const missing = root.querySelector('[data-target="production-yield-missing"]');
  const breakdown = root.querySelector('[data-target="production-breakdown"]');
  const tableHost = root.querySelector('[data-target="production-breakdown-table"]');
  let years = [];
  let detailRows = [];
  let table = null;

  for (const control of [governance, firstYear, lastYear]) {
    control.addEventListener('change', () => {
      if (Number(firstYear.value) > Number(lastYear.value)) {
        if (control === firstYear) lastYear.value = firstYear.value;
        else firstYear.value = lastYear.value;
      }
      const from = Number(firstYear.value);
      const to = Number(lastYear.value);
      onFilterChange({
        governance: governance.value,
        from: from === years[0] ? null : from,
        to: to === years.at(-1) ? null : to,
      });
    });
  }
  breakdown.addEventListener('toggle', renderBreakdown);

  return {
    hide() {
      root.hidden = true;
      breakdown.open = false;
      table?.destroy();
      table = null;
    },
    render(data, filters, error = false) {
      root.hidden = false;
      years = data ? regionalProductionYears(data.rows) : [];
      const from = years.includes(filters.from) ? filters.from : years[0] ?? null;
      const to = Math.max(from, years.includes(filters.to) ? filters.to : years.at(-1) ?? from);
      const selection = { governance: filters.governance, from, to: from == null ? null : to };
      governance.value = selection.governance;
      setYearOptions(firstYear, years, selection.from);
      setYearOptions(lastYear, years, selection.to);
      governance.disabled = !years.length;

      const report = regionalYield(data?.rows || [], selection);
      summary.textContent = error ? S.BOSCO_HISTORICAL_PRODUCTION_UNAVAILABLE
        : !data ? S.LOADING
          : !report.rows.length ? S.BOSCO_YIELD_NO_HARVESTS
            : !report.areaHa ? S.BOSCO_YIELD_NO_AREA
              : S.BOSCO_YIELD_MEAN(fmtDecimal2(report.volumeM3 / report.areaHa));
      context.textContent = report.areaHa ? S.BOSCO_YIELD_CONTEXT(
        fmtVolume(report.volumeM3), fmtArea(report.areaHa),
        fmtInt(report.parcelCount), fmtInt(report.yearCount),
      ) : '';
      missing.hidden = !report.unknownAreaCount;
      missing.textContent = missing.hidden ? '' : S.BOSCO_YIELD_MISSING_AREAS(fmtInt(report.unknownAreaCount));

      detailRows = report.rows.map(row => [
        `${row.parcelId}:${row.year}`,
        row.year === row.lastYear ? String(row.year) : `${row.year}–${row.lastYear}`,
        row.parcel, row.coppice ? S.TYPE_COPPICE : S.TYPE_HIGHFOREST, row.volumeM3, row.areaHa,
        row.areaSource === 'intervention' ? S.BOSCO_YIELD_INTERVENTION
          : row.areaSource === 'parcel' ? S.BOSCO_YIELD_CADASTRAL : S.BOSCO_YIELD_AREA_UNKNOWN,
      ]);
      breakdown.hidden = !detailRows.length;
      renderBreakdown();
      return selection;
    },
  };

  function renderBreakdown() {
    if (!breakdown.open) return;
    const digest = { columns: COLUMNS, rows: detailRows };
    if (table) table.setData(digest);
    else if (detailRows.length) table = new TableWrapper({
      container: tableHost, digest, columnDefs: COLUMN_DEFS, rowsPerPage: 15,
      toolbar: { search: false }, labels: S.TABLE_LABELS, csvFormat: S.TABLE_CSV_FORMAT,
      csvFilename: S.CSV_BOSCO_YIELD,
    });
  }
}

function setYearOptions(select, years, selected) {
  select.replaceChildren(...years.map(year => {
    const option = document.createElement('option');
    option.value = String(year);
    option.textContent = String(year);
    return option;
  }));
  select.disabled = !years.length;
  select.value = selected == null ? '' : String(selected);
}
