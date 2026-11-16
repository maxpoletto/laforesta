/** Local vector context: loading, cartographic styles and shared SVG/PDF paths. */
import * as cache from '../../base/js/cache.js';
import { PDFDocument } from '../../base/js/pdf.js';
import { fmtInt } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

const measure = new PDFDocument();
export const CONTEXT_LAYERS = ['roads', 'water', 'contours'];
const COLORS = { road: [.45, .45, .45], water: [.2, .46, .66], contour: [.62, .47, .32] };

export async function loadPrintContext(regionId) {
  if (!regionId) return { available: false };
  const key = `map_context_${regionId}`;
  cache.register(key, `/api/bosco/map-context/${regionId}/`);
  return cache.load(key);
}

export function contextCredits(data, selected) {
  const credits = [];
  if (selected.includes('roads') || selected.includes('water')) credits.push(S.BOSCO_PRINT_OSM_CREDIT);
  if (selected.includes('contours')) {
    if (data.build.dem_source === 'copernicus-glo-30') {
      credits.push(S.BOSCO_PRINT_DSM_DESCRIPTION, S.BOSCO_PRINT_DSM_CREDIT);
    } else credits.push(S.BOSCO_PRINT_ELEVATION_CREDIT);
  }
  return credits;
}

export function contextStyle(properties, polygon) {
  if (properties.layer === 'water') return { stroke: COLORS.water,
    fill: polygon ? [.89, .95, .98] : null, width: .55, dash: [] };
  if (properties.layer === 'contours') return { stroke: COLORS.contour,
    width: properties.elevation % 100 === 0 ? .65 : .25, dash: [] };
  const minor = ['track', 'path', 'footway', 'bridleway', 'steps'].includes(properties.kind);
  return { stroke: COLORS.road, width: minor ? .5 : .9, dash: minor ? [3, 2] : [] };
}

export function layoutPrintContext(features, project, mapBox, parcelLabels) {
  const shapes = [], labels = [];
  for (const feature of features) {
    const { type, coordinates } = feature.geometry;
    const polygon = type === 'Polygon' || type === 'MultiPolygon';
    const groups = type === 'Polygon' ? [coordinates]
      : type === 'MultiPolygon' ? coordinates
      : type === 'LineString' ? [[coordinates]]
      : type === 'MultiLineString' ? coordinates.map(line => [line]) : [];
    for (const group of groups) {
      const rings = group.map(ring => ring.map(project));
      const style = contextStyle(feature.properties, polygon);
      shapes.push({ rings, close: polygon, ...style });
      if (feature.properties.layer !== 'contours' || feature.properties.elevation % 100) continue;
      const line = rings[0];
      const length = line.slice(1).reduce((sum, point, i) => sum + Math.hypot(point[0] - line[i][0], point[1] - line[i][1]), 0);
      const text = S.BOSCO_PRINT_ELEVATION(fmtInt(feature.properties.elevation));
      const width = measure.textWidth(text, { size: 8 }) + 4, height = 12;
      if (length < width * 3) continue;
      for (const fraction of [.5, .25, .75]) {
        const anchor = line[Math.floor((line.length - 1) * fraction)];
        const candidate = { text, width, height, x: anchor[0] - width / 2, y: anchor[1] - height / 2 };
        if (candidate.x < mapBox.x || candidate.y < mapBox.y
            || candidate.x + width > mapBox.x + mapBox.width || candidate.y + height > mapBox.y + mapBox.height) continue;
        if ([...parcelLabels, ...labels].some(other => candidate.x < other.x + other.width + 4
          && other.x < candidate.x + width + 4 && candidate.y < other.y + other.height + 4
          && other.y < candidate.y + height + 4)) continue;
        labels.push(candidate);
        break;
      }
    }
  }
  return { shapes, labels, box: mapBox };
}

export function drawContextPDF(doc, context) {
  if (!context) return;
  const { box } = context;
  doc.clipRect(box.x, box.y, box.width, box.height);
  for (const shape of context.shapes) doc.shape(shape.rings, shape);
  for (const label of context.labels) {
    doc.whiteRect(label.x, label.y, label.width, label.height);
    doc.text(label.x + 2, label.y + 9, label.text, { size: 8 });
  }
  doc.endClip();
}

export function appendContextSVG(svg, context, element) {
  if (!context) return;
  const { box } = context;
  const clip = element('clipPath', { id: 'bosco-print-context-clip', clipPathUnits: 'userSpaceOnUse' });
  clip.appendChild(element('rect', { x: box.x, y: box.y, width: box.width, height: box.height }));
  const definitions = element('defs');
  definitions.appendChild(clip);
  svg.appendChild(definitions);
  const group = element('g', { 'clip-path': 'url(#bosco-print-context-clip)' });
  const color = rgb => `rgb(${rgb.map(c => Math.round(c * 255)).join(',')})`;
  for (const shape of context.shapes) {
    group.appendChild(element('path', {
      d: shape.rings.map(ring => ring.map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ')
        + (shape.close ? ' Z' : '')).join(' '),
      fill: shape.fill ? color(shape.fill) : 'none', 'fill-rule': 'evenodd',
      stroke: color(shape.stroke), 'stroke-width': shape.width,
      'stroke-dasharray': shape.dash?.join(' ') || 'none',
    }));
  }
  for (const label of context.labels) {
    group.appendChild(element('rect', { x: label.x, y: label.y, width: label.width, height: label.height, fill: 'white' }));
    group.appendChild(element('text', { x: label.x + 2, y: label.y + 9,
      'font-family': 'Helvetica, Arial, sans-serif', 'font-size': 8,
      textLength: measure.textWidth(label.text, { size: 8 }), lengthAdjust: 'spacingAndGlyphs' }, label.text));
  }
  svg.appendChild(group);
}
