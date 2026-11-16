import { fmtDecimal1, localISODate } from '../../base/js/format.js';
import { cloneTemplate } from '../../base/js/templates.js';
import { show, dismiss, onDismiss } from '../../base/js/modals.js';
import { wireCancelButtons } from '../../base/js/ui-widgets.js';
import * as S from '../../base/js/strings.js';
import { buildMetricPrint } from './bosco-metric-print.js';
import { CONTEXT_LAYERS, contextCredits, loadPrintContext } from './bosco-print-context.js';
import {
  buildPrintMapLayout, buildPrintMapPreview, downloadPrintMap,
} from './characteristic-print.js';

export function showMetricPrint({ date = new Date(), loadContext = loadPrintContext, ...options }) {
  // Scale edits, preview and download all retain the same source snapshot.
  const content = structuredClone(buildMetricPrint(options));
  content.footer = S.BOSCO_PRINT_MAP_FOOTER(localISODate(date));
  let layout = null;
  let context = null, loading = true, closed = false;
  const filename = content.filename;
  const fragment = cloneTemplate('tmpl-bosco-map-print');
  fragment.querySelector('[data-field="print-title"]').textContent = S.BOSCO_PRINT_DIALOG_TITLE(content.metricLabel);
  const preview = fragment.querySelector('[data-target="print-preview"]');
  const scale = fragment.querySelector('[data-role="print-scale"]');
  const dimensions = fragment.querySelector('[data-field="print-dimensions"]');
  const paper = fragment.querySelector('[data-field="print-paper"]');
  const error = fragment.querySelector('[data-target="print-error"]');
  const download = fragment.querySelector('[data-action="download-map"]');
  const contextStatus = fragment.querySelector('[data-field="print-context-status"]');
  const contourNote = fragment.querySelector('[data-field="print-contour-note"]');
  const layerControls = CONTEXT_LAYERS.map(layer => fragment.querySelector(`[data-layer="${layer}"]`));
  const update = () => {
    layout = null;
    preview.replaceChildren();
    dimensions.textContent = '';
    paper.textContent = '';
    contourNote.textContent = '';
    error.textContent = '';
    download.disabled = true;
    try {
      const selected = layerControls.filter(control => control.checked && !control.disabled).map(control => control.dataset.layer);
      const features = context?.features?.filter(feature => selected.includes(feature.properties.layer)) || [];
      const next = buildPrintMapLayout({ ...content, scaleDenominator: Number(scale.value), contextFeatures: features,
        credits: selected.length ? contextCredits(context, selected) : [] });
      preview.appendChild(buildPrintMapPreview(next));
      dimensions.textContent = S.BOSCO_PRINT_DIMENSIONS(fmtDecimal1(next.mapWidthCm), fmtDecimal1(next.mapHeightCm));
      paper.textContent = S.BOSCO_PRINT_PAPER(next.paper, fmtDecimal1(next.width / 72 * 2.54),
        fmtDecimal1(next.height / 72 * 2.54), next.landscape);
      layout = next;
      scale.setAttribute('aria-invalid', 'false');
      download.disabled = loading;
      download.title = loading ? S.BOSCO_PRINT_CONTEXT_LOADING : '';
      contourNote.textContent = selected.includes('contours')
        ? [S.BOSCO_PRINT_CONTEXT_CONTOURS(context.build.contour_interval_m),
          context.build.dem_source === 'copernicus-glo-30' && S.BOSCO_PRINT_DSM_NOTE].filter(Boolean).join(' ') : '';
    } catch (failure) {
      error.textContent = failure.message || S.BOSCO_PRINT_DOWNLOAD_ERROR;
      scale.setAttribute('aria-invalid', 'true');
      download.title = error.textContent;
    }
  };
  scale.addEventListener('input', update);
  layerControls.forEach(control => control.addEventListener('change', update));
  download.addEventListener('click', () => {
    if (!layout || loading) return;
    try {
      downloadPrintMap(layout, filename);
      error.textContent = '';
    } catch {
      error.textContent = S.BOSCO_PRINT_DOWNLOAD_ERROR;
    }
  });
  wireCancelButtons(fragment, dismiss);
  contextStatus.textContent = S.BOSCO_PRINT_CONTEXT_LOADING;
  update();
  show(fragment);
  scale.focus();
  onDismiss(() => { closed = true; });
  const complete = data => {
    if (closed) return;
    loading = false;
    if (data?.available && data.schema_version === 1 && Array.isArray(data.features)) {
      context = structuredClone(data);
      contextStatus.textContent = '';
      layerControls.forEach(control => {
        control.disabled = !context.features.some(feature => feature.properties.layer === control.dataset.layer);
        control.checked = !control.disabled;
      });
    } else contextStatus.textContent = data?.stale ? S.BOSCO_PRINT_CONTEXT_STALE : S.BOSCO_PRINT_CONTEXT_MISSING;
    update();
  };
  Promise.resolve().then(() => loadContext(options.region.id)).then(complete).catch(() => {
    if (closed) return;
    complete(null);
    contextStatus.textContent = S.BOSCO_PRINT_CONTEXT_ERROR;
  });
}
