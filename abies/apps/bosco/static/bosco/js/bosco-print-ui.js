import { FIELD_SELECTED_PLAN } from '../../base/js/constants.js';
import { localISODate } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { show, dismiss } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { calendarPrintItems } from './bosco-calendar-print.js';
import {
  buildPrintMapLayout, buildPrintMapPreview, downloadPrintMap,
} from './characteristic-print.js';

export function showCalendarPrint({ geojson, region, entries, data, date = new Date() }) {
  const plan = data?.[FIELD_SELECTED_PLAN];
  const items = calendarPrintItems(geojson, region.name, entries, data);
  const layout = buildPrintMapLayout({
    items, title: S.BOSCO_CALENDAR_MAP_TITLE(region.name),
    subtitle: S.BOSCO_CALENDAR_MAP_PLAN(plan),
    footer: S.BOSCO_CALENDAR_MAP_FOOTER(localISODate(date)),
  });
  // The preview and download retain this snapshot even if the source is refreshed.
  const filename = S.BOSCO_CALENDAR_PDF_FILENAME(region.name, plan);
  const fragment = cloneTemplate('tmpl-bosco-calendar-print');
  fragment.querySelector('[data-target="print-preview"]').appendChild(buildPrintMapPreview(layout));
  const error = fragment.querySelector('[data-target="print-error"]');
  fragment.querySelector('[data-action="download-map"]').addEventListener('click', () => {
    try {
      downloadPrintMap(layout, filename);
      error.textContent = '';
    } catch {
      error.textContent = S.BOSCO_PRINT_DOWNLOAD_ERROR;
    }
  });
  wireCancelButtons(fragment, dismiss);
  show(fragment);
}
