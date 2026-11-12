/** Metric-specific print content; layout and preview/download remain shared. */
import { FIELD_SELECTED_PLAN } from '../../base/js/constants.js';
import * as S from '../../base/js/strings.js';
import { Q_AGE, Q_TYPE, Q_CUTTING_CALENDAR, usesPlanSource } from './bosco-metrics.js';
import { formatMetricValue, metricLabel, metricValue } from './bosco-characteristics.js';
import { calendarPlanSubtitle, calendarPrintItems } from './bosco-calendar-print.js';
import { parcelPrintItems } from './bosco-parcel-print.js';

const PRINT_FIELDS = new Map([
  [Q_AGE, S.BOSCO_PRINT_AGE_FILE], [Q_TYPE, S.BOSCO_PRINT_TYPE_FILE],
]);

export function printAvailability(metricId, { hasRegion = false, data } = {}) {
  const visible = metricId === Q_CUTTING_CALENDAR || PRINT_FIELDS.has(metricId);
  const reason = !hasRegion ? S.BOSCO_NO_REGION
    : usesPlanSource(metricId) && !data?.[FIELD_SELECTED_PLAN] ? S.BOSCO_PRINT_SELECT_PLAN : '';
  return { visible, disabled: !visible || Boolean(reason), reason };
}

export function buildMetricPrint({ geojson, region, entries, metricId, data,
  planEntries = entries, context = {} }) {
  if (metricId === Q_CUTTING_CALENDAR) {
    return {
      items: calendarPrintItems(geojson, region.name, entries, data),
      title: S.BOSCO_CALENDAR_MAP_TITLE(region.name),
      subtitle: calendarPlanSubtitle(planEntries, data),
      legend: S.BOSCO_CALENDAR_MAP_LEGEND,
      filename: S.BOSCO_CALENDAR_PDF_FILENAME(region.name, data[FIELD_SELECTED_PLAN]),
      metricLabel: S.BOSCO_METRIC_CUTTING_CALENDAR,
    };
  }
  const stem = PRINT_FIELDS.get(metricId);
  if (!stem) throw new Error(S.BOSCO_PRINT_UNSUPPORTED);
  const label = metricLabel(metricId);
  return {
    items: parcelPrintItems(geojson, region.name, entries, entry => {
      const value = metricValue(entry, metricId, context);
      const missing = value == null || value === '' || (typeof value === 'number' && !Number.isFinite(value));
      return [missing ? '-' : formatMetricValue(metricId, value, context)];
    }),
    title: S.BOSCO_PRINT_METRIC_TITLE(label, region.name), subtitle: '',
    legend: S.BOSCO_PRINT_METRIC_LEGEND,
    filename: S.BOSCO_PRINT_METRIC_FILENAME(stem, region.name), metricLabel: label,
  };
}
