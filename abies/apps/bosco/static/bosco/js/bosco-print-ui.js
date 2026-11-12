import { localISODate } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { show, dismiss } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { buildMetricPrint } from './bosco-metric-print.js';
import {
  buildPrintMapLayout, buildPrintMapPreview, downloadPrintMap,
} from './characteristic-print.js';

export function showMetricPrint({ date = new Date(), ...options }) {
  const content = buildMetricPrint(options);
  const layout = buildPrintMapLayout({
    ...content, footer: S.BOSCO_PRINT_MAP_FOOTER(localISODate(date)),
  });
  // The preview and download retain this snapshot even if the source is refreshed.
  const filename = content.filename;
  const fragment = cloneTemplate('tmpl-bosco-map-print');
  fragment.querySelector('[data-field="print-title"]').textContent = S.BOSCO_PRINT_DIALOG_TITLE(content.metricLabel);
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
