import { fmtDecimal1, localISODate } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { show, dismiss } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { buildMetricPrint } from './bosco-metric-print.js';
import {
  buildPrintMapLayout, buildPrintMapPreview, downloadPrintMap,
} from './characteristic-print.js';

export function showMetricPrint({ date = new Date(), ...options }) {
  // Scale edits, preview and download all retain the same source snapshot.
  const content = structuredClone(buildMetricPrint(options));
  content.footer = S.BOSCO_PRINT_MAP_FOOTER(localISODate(date));
  let layout = null;
  const filename = content.filename;
  const fragment = cloneTemplate('tmpl-bosco-map-print');
  fragment.querySelector('[data-field="print-title"]').textContent = S.BOSCO_PRINT_DIALOG_TITLE(content.metricLabel);
  const preview = fragment.querySelector('[data-target="print-preview"]');
  const scale = fragment.querySelector('[data-role="print-scale"]');
  const dimensions = fragment.querySelector('[data-field="print-dimensions"]');
  const paper = fragment.querySelector('[data-field="print-paper"]');
  const error = fragment.querySelector('[data-target="print-error"]');
  const download = fragment.querySelector('[data-action="download-map"]');
  const update = () => {
    layout = null;
    preview.replaceChildren();
    dimensions.textContent = '';
    paper.textContent = '';
    error.textContent = '';
    download.disabled = true;
    try {
      const next = buildPrintMapLayout({ ...content, scaleDenominator: Number(scale.value) });
      preview.appendChild(buildPrintMapPreview(next));
      dimensions.textContent = S.BOSCO_PRINT_DIMENSIONS(fmtDecimal1(next.mapWidthCm), fmtDecimal1(next.mapHeightCm));
      paper.textContent = S.BOSCO_PRINT_PAPER(next.paper, fmtDecimal1(next.width / 72 * 2.54),
        fmtDecimal1(next.height / 72 * 2.54), next.landscape);
      layout = next;
      scale.setAttribute('aria-invalid', 'false');
      download.disabled = false;
      download.title = '';
    } catch (failure) {
      error.textContent = failure.message || S.BOSCO_PRINT_DOWNLOAD_ERROR;
      scale.setAttribute('aria-invalid', 'true');
      download.title = error.textContent;
    }
  };
  scale.addEventListener('input', update);
  download.addEventListener('click', () => {
    if (!layout) return;
    try {
      downloadPrintMap(layout, filename);
      error.textContent = '';
    } catch {
      error.textContent = S.BOSCO_PRINT_DOWNLOAD_ERROR;
    }
  });
  wireCancelButtons(fragment, dismiss);
  update();
  show(fragment);
  scale.focus();
}
