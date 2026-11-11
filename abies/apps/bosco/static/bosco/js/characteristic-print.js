/** A4 vector-map layout shared by the SVG preview and PDF export. */
import { PDFDocument, buildPDF } from '../../base/js/pdf.js';
import { pointInPolygon } from '../../base/js/geo.js';
import * as S from '../../base/js/strings.js';

const MARGIN = 24;
const TOP = 70;
const BOTTOM = 32;
const CALLOUT_WIDTH = 82;
const GAP = 4;
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

function labelFor(item) {
  const lines = [
    ...wrap(item.name, CALLOUT_WIDTH - 8, { size: 9, bold: true })
      .map(text => ({ text, size: 9, bold: true })),
    ...item.lines.flatMap(value => wrap(value, CALLOUT_WIDTH - 8, { size: 8 }))
      .map(text => ({ text, size: 8, bold: false })),
  ];
  return { lines, width: Math.max(...lines.map(line => measure.textWidth(line.text, line))) + 6,
    height: lines.length * 10 + 4 };
}

function layoutFor(items, landscape, header) {
  const doc = new PDFDocument({ landscape });
  const { width, height } = doc;
  const headerLines = [];
  let headerY = 30;
  for (const [value, size, bold, spacing] of [
    [header.title, 14, true, 18], [header.subtitle, 10, false, 13],
  ]) {
    for (const text of wrap(value, width - 2 * MARGIN, { size, bold })) {
      headerLines.push({ text, size, bold, y: headerY });
      headerY += spacing;
    }
  }
  const top = Math.max(TOP, headerY + 8);
  const mapBox = { x: MARGIN + CALLOUT_WIDTH + GAP, y: top,
    width: width - 2 * (MARGIN + CALLOUT_WIDTH + GAP), height: height - top - BOTTOM };
  if (mapBox.height < 100) throw new Error(S.BOSCO_PRINT_LABEL_ERROR);
  const projected = items.map(item => ({ ...item,
    geometry: mapGeometry(item.geometry, ([lon, lat]) => [lon * Math.PI / 180,
      -Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360))]),
  }));
  const box = bounds(projected.flatMap(item => coordinates(item.geometry)));
  const scale = Math.min(mapBox.width / (box.maxX - box.minX),
    mapBox.height / (box.maxY - box.minY));
  if (!Number.isFinite(scale)) throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
  const transform = ([x, y]) => [
    mapBox.x + mapBox.width / 2 + (x - (box.minX + box.maxX) / 2) * scale,
    mapBox.y + mapBox.height / 2 + (y - (box.minY + box.maxY) / 2) * scale,
  ];
  const parcels = projected.map(item => ({ ...item, geometry: mapGeometry(item.geometry, transform) }));
  const pending = parcels.map(item => ({ ...labelFor(item), name: item.name,
    geometry: item.geometry, candidates: interiorCandidates(item.geometry) }));
  // Give narrow parcels first choice of the limited interior space.
  pending.sort((a, b) => a.candidates[0].distance - b.candidates[0].distance);
  const labels = [], outside = [];
  for (const label of pending) {
    const candidate = label.candidates.slice(0, 80).find(({ point }) => {
      const rect = { ...label, x: point[0] - label.width / 2, y: point[1] - label.height / 2 };
      return !labels.some(other => overlaps(rect, other)) && rectangleInside(rect, label.geometry);
    });
    const anchor = (candidate || label.candidates[0]).point;
    const result = { name: label.name, lines: label.lines, width: label.width,
      height: label.height, anchor, x: anchor[0] - label.width / 2, y: anchor[1] - label.height / 2 };
    if (candidate) labels.push(result);
    else outside.push(result);
  }
  const columns = [[], []];
  const used = [0, 0];
  for (const label of outside.sort((a, b) => b.height - a.height)) {
    let side = label.anchor[0] < width / 2 ? 0 : 1;
    if (used[side] + label.height + GAP > mapBox.height) side = 1 - side;
    if (used[side] + label.height + GAP > mapBox.height) throw new Error(S.BOSCO_PRINT_LABEL_ERROR);
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
  return { width, height, landscape, scale, parcels, labels, headerLines, ...header };
}

export function buildPrintMapLayout({ items, title = '', subtitle = '', footer = '' }) {
  if (!items.length || items.some(item => !polygons(item.geometry).length
      || coordinates(item.geometry).some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y)
        || Math.abs(x) > 180 || Math.abs(y) >= 85))) {
    throw new Error(S.BOSCO_PRINT_GEOMETRY_ERROR);
  }
  const layouts = [];
  let lastError;
  for (const landscape of [false, true]) {
    try { layouts.push(layoutFor(items, landscape, { title, subtitle, footer })); }
    catch (error) { lastError = error; }
  }
  if (!layouts.length) throw lastError;
  return layouts.sort((a, b) => b.scale - a.scale)[0];
}

function drawPDF(layout) {
  const doc = new PDFDocument({ landscape: layout.landscape });
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
