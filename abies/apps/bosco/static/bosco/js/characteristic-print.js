/** Fixed-scale vector-map layout shared by the SVG preview and PDF export. */
import { PDFDocument, buildPDF } from '../../base/js/pdf.js';
import { localProjection, pointInPolygon } from '../../base/js/geo.js';
import { fmtDecimal1, fmtInt } from '../../base/js/format.js';
import * as S from '../../base/js/strings.js';

const MARGIN = 24;
const TOP = 70;
const BOTTOM = 32;
const CALLOUT_WIDTH = 82;
const GAP = 4;
const POINTS_PER_METER = 72 / .0254;
const PAPERS = [
  ['A4', 210, 297], ['A3', 297, 420], ['A2', 420, 594],
  ['A1', 594, 841], ['A0', 841, 1189],
].map(([name, width, height]) => ({ name,
  width: Math.round(width * 72 / 25.4 * 100) / 100,
  height: Math.round(height * 72 / 25.4 * 100) / 100,
}));
const measure = new PDFDocument();

function polygons(geometry) {
  if (geometry?.type === 'Polygon') return [geometry.coordinates];
  if (geometry?.type === 'MultiPolygon') return geometry.coordinates;
  return [];
}

function mapGeometry(geometry, transform) {
  return { type: geometry.type, coordinates: geometry.type === 'Polygon'
    ? geometry.coordinates.map(ring => ring.map(transform))
    : geometry.coordinates.map(rings => rings.map(ring => ring.map(transform))) };
}

function bounds(points) {
  return points.reduce((box, [x, y]) => ({
    minX: Math.min(box.minX, x), maxX: Math.max(box.maxX, x),
    minY: Math.min(box.minY, y), maxY: Math.max(box.maxY, y),
  }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
}

function coordinates(geometry) {
  return polygons(geometry).flat(2);
}

function overlaps(a, b, gap = GAP) {
  return a.x < b.x + b.width + gap && b.x < a.x + a.width + gap
    && a.y < b.y + b.height + gap && b.y < a.y + a.height + gap;
}

function segmentDistanceSquared(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const t = dx || dy ? Math.max(0, Math.min(1,
    ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy))) : 0;
  return (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
}

function clearance(point, geometry) {
  if (!pointInPolygon(...point, geometry)) return -1;
  let distance = Infinity;
  for (const rings of polygons(geometry)) {
    for (const ring of rings) {
      for (let i = 1; i < ring.length; i++) {
        distance = Math.min(distance, segmentDistanceSquared(point, ring[i - 1], ring[i]));
      }
    }
  }
  return Math.sqrt(distance);
}

/** Interior candidates from every component, with seeds for narrow polygons. */
function interiorCandidates(geometry) {
  const candidates = [];
  for (const rings of polygons(geometry)) {
    const box = bounds(rings[0]);
    const dx = (box.maxX - box.minX) / 16, dy = (box.maxY - box.minY) / 16;
    for (let x = 0; x < 16; x++) {
      for (let y = 0; y < 16; y++) {
        const point = [box.minX + (x + .5) * dx, box.minY + (y + .5) * dy];
        const distance = clearance(point, geometry);
        if (distance > 0) candidates.push({ point, distance });
      }
    }
    // A long, narrow diagonal may fall between the grid points. Polygon-edge
    // midpoints nudged inward provide seeds without assuming a valid centroid.
    for (let i = 1; i < rings[0].length; i++) {
      const a = rings[0][i - 1], b = rings[0][i];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (!length) continue;
      for (const sign of [-1, 1]) {
        const epsilon = Math.min(dx, dy) / 8;
        const point = [(a[0] + b[0]) / 2 + sign * (b[1] - a[1]) / length * epsilon,
          (a[1] + b[1]) / 2 - sign * (b[0] - a[0]) / length * epsilon];
        const distance = clearance(point, geometry);
        if (distance > 0) candidates.push({ point, distance });
      }
    }
  }
  candidates.sort((a, b) => b.distance - a.distance);
  if (!candidates.length) throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
  return candidates;
}

function rectangleInside(box, geometry) {
  const corners = [[box.x, box.y], [box.x + box.width, box.y],
    [box.x + box.width, box.y + box.height], [box.x, box.y + box.height]];
  if (!corners.every(point => pointInPolygon(...point, geometry))) return false;
  // Corners alone miss holes and concave boundaries crossing the rectangle.
  for (const rings of polygons(geometry)) {
    for (const ring of rings) {
      for (let i = 1; i < ring.length; i++) {
        const a = ring[i - 1], b = ring[i];
        let lo = 0, hi = 1;
        for (const [start, delta, min, max] of [
          [a[0], b[0] - a[0], box.x, box.x + box.width],
          [a[1], b[1] - a[1], box.y, box.y + box.height],
        ]) {
          if (delta === 0) { if (start < min || start > max) hi = -1; }
          else {
            const t1 = (min - start) / delta, t2 = (max - start) / delta;
            lo = Math.max(lo, Math.min(t1, t2));
            hi = Math.min(hi, Math.max(t1, t2));
          }
        }
        if (hi >= lo) return false;
      }
    }
  }
  return true;
}

function wrap(value, width, opts) {
  const lines = [];
  let line = '';
  for (const word of String(value).split(/\s+/)) {
    const combined = line ? `${line} ${word}` : word;
    if (line && measure.textWidth(combined, opts) > width) {
      lines.push(line); line = word;
    } else line = combined;
    if (measure.textWidth(line, opts) > width) throw new Error(S.BOSCO_PRINT_LABEL_ERROR);
  }
  if (line) lines.push(line);
  return lines;
}

function labelFor(item, maxWidth = CALLOUT_WIDTH) {
  const lines = [
    ...wrap(item.name, maxWidth - 8, { size: 9, bold: true })
      .map(text => ({ text, size: 9, bold: true })),
    ...item.lines.flatMap(value => wrap(value, maxWidth - 8, { size: 8 }))
      .map(text => ({ text, size: 8, bold: false })),
  ];
  return { lines, width: Math.max(...lines.map(line => measure.textWidth(line.text, line))) + 6,
    height: lines.length * 10 + 4 };
}

function layoutFor(map, paper, landscape, header, gutters) {
  const doc = new PDFDocument({ landscape, size: paper });
  const { width, height } = doc;
  const headerLines = [];
  let headerY = 30;
  for (const [value, size, bold, spacing] of [
    [header.title, 14, true, 18], [header.subtitle, 10, false, 13],
    [header.legend, 8, false, 11],
    [S.BOSCO_PRINT_SCALE(fmtInt(map.scaleDenominator)), 8, false, 11],
  ]) {
    for (const text of wrap(value, width - 2 * MARGIN, { size, bold })) {
      headerLines.push({ text, size, bold, y: headerY });
      headerY += spacing;
    }
  }
  const top = Math.max(TOP, headerY + 8);
  const mapBox = { x: MARGIN + gutters[0], y: top,
    width: width - 2 * MARGIN - gutters[0] - gutters[1], height: height - top - BOTTOM };
  if (map.width > mapBox.width || map.height > mapBox.height) return null;
  const offsetX = mapBox.x + (mapBox.width - map.width) / 2;
  const offsetY = mapBox.y + (mapBox.height - map.height) / 2;
  const translate = ([x, y]) => [x + offsetX, y + offsetY];
  const parcels = map.parcels.map(item => ({ ...item,
    geometry: mapGeometry(item.geometry, translate) }));
  const translatedLabel = label => ({ ...label, x: label.x + offsetX, y: label.y + offsetY,
    anchor: translate(label.anchor) });
  const labels = map.labels.map(translatedLabel);
  const columns = [[], []];
  const used = [0, 0];
  for (const label of map.outside.map(translatedLabel).sort((a, b) => b.height - a.height)) {
    const fits = side => gutters[side] && used[side] + label.height <= mapBox.height;
    let side = label.anchor[0] < offsetX + map.width / 2 ? 0 : 1;
    if (!fits(side)) side = 1 - side;
    if (!fits(side)) throw new Error(S.BOSCO_PRINT_LABEL_ERROR);
    columns[side].push(label);
    used[side] += label.height + GAP;
  }
  columns.forEach((column, side) => {
    column.sort((a, b) => a.anchor[1] - b.anchor[1]);
    let bottom = top;
    for (const label of column) {
      label.x = side ? width - MARGIN - label.width : MARGIN;
      label.y = Math.max(bottom, label.anchor[1] - label.height / 2);
      bottom = label.y + label.height + GAP;
    }
    bottom = height - BOTTOM;
    for (const label of [...column].reverse()) {
      label.y = Math.min(label.y, bottom - label.height);
      bottom = label.y - GAP;
      const edge = side ? label.x : label.x + label.width;
      label.leader = [label.anchor, [edge + (side ? -4 : 4), label.y + label.height / 2],
        [edge, label.y + label.height / 2]];
      labels.push(label);
    }
  });
  return { width, height, landscape, paper: paper.name, scale: map.scale,
    scaleDenominator: map.scaleDenominator,
    mapWidthCm: map.width / POINTS_PER_METER * 100,
    mapHeightCm: map.height / POINTS_PER_METER * 100,
    gutterWidth: gutters[0] + gutters[1],
    fit: Math.min(mapBox.width / map.width, mapBox.height / map.height),
    parcels, labels, headerLines, ...header };
}

function interiorLabels(parcels, width) {
  const pending = parcels.map(item => ({ item, candidates: interiorCandidates(item.geometry) }));
  // Give narrow parcels first choice of the limited interior space.
  pending.sort((a, b) => a.candidates[0].distance - b.candidates[0].distance);
  const labels = [], outside = [];
  for (const label of pending) {
    let placed;
    // At larger map scales a full value line can fit inside a parcel. Try it
    // before wrapping to the narrower shape used by margin callouts.
    const widest = Math.max(CALLOUT_WIDTH, width);
    for (const maxWidth of new Set([widest, Math.min(widest, 2 * CALLOUT_WIDTH), CALLOUT_WIDTH])) {
      const shape = labelFor(label.item, maxWidth);
      for (const { point: anchor } of label.candidates) {
        const rect = { ...shape, name: label.item.name, anchor,
          x: anchor[0] - shape.width / 2, y: anchor[1] - shape.height / 2 };
        if (!labels.some(other => overlaps(rect, other)) && rectangleInside(rect, label.item.geometry)) {
          placed = rect;
          break;
        }
      }
      if (placed) break;
    }
    if (placed) labels.push(placed);
    else {
      const shape = labelFor(label.item), anchor = label.candidates[0].point;
      outside.push({ ...shape, name: label.item.name, anchor,
        x: anchor[0] - shape.width / 2, y: anchor[1] - shape.height / 2 });
    }
  }
  return { labels, outside };
}

export function buildPrintMapLayout({ items, scaleDenominator = 10000,
  title = '', subtitle = '', legend = '', footer = '' }) {
  if (!Number.isSafeInteger(scaleDenominator) || scaleDenominator <= 0) {
    throw new Error(S.BOSCO_PRINT_SCALE_ERROR);
  }
  if (!items.length || items.some(item => !polygons(item.geometry).length
      || polygons(item.geometry).some(rings => !rings.length || rings.some(ring => ring.length < 4))
      || coordinates(item.geometry).some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y)
        || Math.abs(x) > 180 || Math.abs(y) >= 85))) {
    throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
  }
  const geographic = bounds(items.flatMap(item => coordinates(item.geometry)));
  const plane = localProjection((geographic.minY + geographic.maxY) / 2,
    (geographic.minX + geographic.maxX) / 2);
  const projected = items.map(item => ({ ...item,
    geometry: mapGeometry(item.geometry, ([lon, lat]) => {
      const { x, y } = plane.project(lat, lon);
      return [x, -y];
    }),
  }));
  const box = bounds(projected.flatMap(item => coordinates(item.geometry)));
  const scale = POINTS_PER_METER / scaleDenominator;
  const width = (box.maxX - box.minX) * scale, height = (box.maxY - box.minY) * scale;
  if (!(width > 0 && height > 0)) throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
  const sizeError = new Error(S.BOSCO_PRINT_SIZE_ERROR(
    fmtDecimal1(width / POINTS_PER_METER * 100), fmtDecimal1(height / POINTS_PER_METER * 100)));
  const a0 = PAPERS.at(-1);
  if (Math.min(width, height) > a0.width - 2 * MARGIN
      || Math.max(width, height) > a0.height - 2 * MARGIN) throw sizeError;
  const parcels = projected.map(item => ({ ...item,
    geometry: mapGeometry(item.geometry, ([x, y]) => [(x - box.minX) * scale, (y - box.minY) * scale]),
  }));
  const map = { parcels, width, height, scale, scaleDenominator, ...interiorLabels(parcels, width) };
  const gutter = CALLOUT_WIDTH + GAP;
  const gutterOptions = map.outside.length ? [[gutter, 0], [0, gutter], [gutter, gutter]] : [[0, 0]];
  let lastError = sizeError;
  for (const paper of PAPERS) {
    const layouts = [];
    for (const landscape of [false, true]) {
      for (const gutters of gutterOptions) {
        try {
          const layout = layoutFor(map, paper, landscape, { title, subtitle, legend, footer }, gutters);
          if (layout) layouts.push(layout);
        } catch (error) { lastError = error; }
      }
    }
    if (layouts.length) return layouts.sort((a, b) => a.gutterWidth - b.gutterWidth || b.fit - a.fit)[0];
  }
  throw lastError;
}

function drawPDF(layout) {
  const doc = new PDFDocument({ size: layout });
  for (const line of layout.headerLines) doc.text(MARGIN, line.y, line.text, line);
  doc.text(MARGIN, doc.height - 15, layout.footer, { size: 8 });
  for (const parcel of layout.parcels) {
    for (const rings of polygons(parcel.geometry)) {
      for (const ring of rings) doc.path(ring, { close: true });
    }
  }
  for (const label of layout.labels) {
    if (!label.leader) continue;
    doc.path(label.leader, { width: .4 });
    doc.path([[label.anchor[0] - .7, label.anchor[1]],
      [label.anchor[0] + .7, label.anchor[1]]], { width: 1.4 });
  }
  for (const label of layout.labels) {
    doc.whiteRect(label.x, label.y, label.width, label.height);
    label.lines.forEach((line, i) => {
      doc.text(label.x + (label.width - doc.textWidth(line.text, line)) / 2,
        label.y + 10 + i * 10, line.text, line);
    });
  }
  return doc;
}

export function printMapPDF(layout) {
  const doc = drawPDF(layout);
  return buildPDF(doc.width, doc.height, doc.pages);
}

export function downloadPrintMap(layout, filename) {
  drawPDF(layout).save(filename);
}

/** DOM creation keeps parcel/plan names as text, never interprets them as markup. */
export function buildPrintMapPreview(layout) {
  const element = (name, attrs = {}, text = null) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== null) el.textContent = text;
    return el;
  };
  const svg = element('svg', { viewBox: `0 0 ${layout.width} ${layout.height}`,
    role: 'img', 'aria-label': layout.title, class: 'bosco-print-preview' });
  svg.appendChild(element('rect', { width: layout.width, height: layout.height, fill: 'white' }));
  const text = (x, y, value, size, bold) => svg.appendChild(element('text', {
    x, y, 'font-family': 'Helvetica, Arial, sans-serif', 'font-size': size,
    'font-weight': bold ? 'bold' : 'normal',
    textLength: measure.textWidth(value, { size, bold }), lengthAdjust: 'spacingAndGlyphs',
  }, value));
  for (const line of layout.headerLines) text(MARGIN, line.y, line.text, line.size, line.bold);
  text(MARGIN, layout.height - 15, layout.footer, 8, false);
  const path = (points, close, width) => svg.appendChild(element('path', {
    d: points.map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ') + (close ? ' Z' : ''),
    fill: 'none', stroke: 'black', 'stroke-width': width,
  }));
  for (const parcel of layout.parcels) {
    for (const rings of polygons(parcel.geometry)) for (const ring of rings) path(ring, true, .6);
  }
  for (const label of layout.labels) {
    if (!label.leader) continue;
    path(label.leader, false, .4);
    svg.appendChild(element('rect', { x: label.anchor[0] - .7, y: label.anchor[1] - .7,
      width: 1.4, height: 1.4, fill: 'black' }));
  }
  for (const label of layout.labels) {
    svg.appendChild(element('rect', { x: label.x, y: label.y, width: label.width,
      height: label.height, fill: 'white' }));
    label.lines.forEach((line, i) => text(
      label.x + (label.width - measure.textWidth(line.text, line)) / 2,
      label.y + 10 + i * 10, line.text, line.size, line.bold,
    ));
  }
  return svg;
}
