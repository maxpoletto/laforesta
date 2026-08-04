import * as S from '../../base/js/strings.js';
import {
  FIELD_CATEGORIES, FIELD_CONTENT_TYPE, FIELD_DATE, FIELD_LAT, FIELD_LON,
  FIELD_NAME, FIELD_PHOTOS, FIELD_TEXT, FIELD_URL,
} from '../../base/js/constants.js';
import { fmtCoord } from '../../base/js/format.js';
import { PDFDocument } from '../../base/js/pdf.js';

const PAGE_MARGIN = 40;
const CONTENT_WIDTH = 515;
const TITLE_Y = 46;
const BODY_START_Y = 92;
const LINE_HEIGHT = 15;
const PHOTO_GAP = 14;
const PHOTO_WIDTH = (CONTENT_WIDTH - PHOTO_GAP) / 2;
const PHOTO_MAX_HEIGHT = 170;
const MAP_WIDTH = CONTENT_WIDTH;
const MAP_HEIGHT = 205;
const CANVAS_PHOTO_MAX_PX = 1400;
const MAP_CANVAS_WIDTH = 1200;
const MAP_CANVAS_HEIGHT = 480;
const OSM_TILE_SIZE = 256;
const OSM_OVERVIEW_ZOOM = 15;
const IMAGE_JPEG_QUALITY = 0.84;
const PDF_TITLE_SIZE = 18;
const PDF_BODY_SIZE = 10;
const PDF_META_SIZE = 11;

export async function generateObservationPDF(observation, { regionName = '', parcelsGeo = null } = {}) {
  const categories = observationCategories(observation);
  const category = primaryCategory(categories);
  const doc = new PDFDocument();
  doc.text(PAGE_MARGIN, TITLE_Y, S.BOSCO_OBSERVATION_REPORT_TITLE(category), {
    size: PDF_TITLE_SIZE, bold: true,
  });

  let y = BODY_START_Y;
  doc.text(PAGE_MARGIN, y, `${S.COL_DATE}:`, { size: PDF_META_SIZE, bold: true });
  doc.text(PAGE_MARGIN + 52, y, observation[FIELD_DATE] || '', { size: PDF_META_SIZE });
  doc.text(PAGE_MARGIN + 230, y, `${S.COL_REGION}:`, { size: PDF_META_SIZE, bold: true });
  doc.text(PAGE_MARGIN + 305, y, regionName || '', { size: PDF_META_SIZE });
  y += 20;
  doc.text(PAGE_MARGIN, y, `${S.BOSCO_POSITION}:`, { size: PDF_META_SIZE, bold: true });
  doc.text(PAGE_MARGIN + 72, y, observationPositionText(observation), { size: PDF_META_SIZE });
  y += 34;

  for (const line of wrapText(doc, observation[FIELD_TEXT] || '', CONTENT_WIDTH, {
    size: PDF_BODY_SIZE,
  })) {
    y = ensureSpace(doc, y, LINE_HEIGHT);
    doc.text(PAGE_MARGIN, y, line, { size: PDF_BODY_SIZE });
    y += LINE_HEIGHT;
  }
  y += 16;

  const photos = await observationPDFPhotos(observation);
  for (let i = 0; i < photos.length; i += 2) {
    const row = photos.slice(i, i + 2).map(photo => fittedSize(photo, PHOTO_WIDTH, PHOTO_MAX_HEIGHT));
    const rowHeight = Math.max(...row.map(item => item.height));
    y = ensureSpace(doc, y, rowHeight + 18);
    row.forEach((item, col) => {
      const x = PAGE_MARGIN + col * (PHOTO_WIDTH + PHOTO_GAP);
      const image = doc.addJPEGImage(item.photo);
      doc.image(x, y, item.width, item.height, image);
    });
    y += rowHeight + 18;
  }

  y = ensureSpace(doc, y, MAP_HEIGHT + 12);
  const mapImage = await observationOverviewMap(observation, parcelsGeo);
  if (mapImage) {
    const image = doc.addJPEGImage(mapImage);
    doc.image(PAGE_MARGIN, y, MAP_WIDTH, MAP_HEIGHT, image);
  }

  doc.save(observationPDFFilename(observation, { regionName, category }));
}

export function observationPDFFilename(observation, { regionName = '', category = '' } = {}) {
  const date = String(observation?.[FIELD_DATE] || '').replace(/-/g, '')
    || slugFilenamePart(S.BOSCO_NO_DATE);
  const region = slugFilenamePart(regionName || S.COL_REGION);
  const categoryPart = slugFilenamePart(category || primaryCategory(observationCategories(observation)));
  return `osservazione_${date}_${region}_${categoryPart}.pdf`;
}

export function observationCategories(observation) {
  return Array.isArray(observation?.[FIELD_CATEGORIES])
    ? observation[FIELD_CATEGORIES].map(row => row?.[FIELD_NAME]).filter(Boolean)
    : [];
}

export function primaryCategory(categories) {
  return categories && categories.length ? categories[0] : S.BOSCO_OBSERVATION_CATEGORY;
}

export function observationPositionText(observation) {
  const lat = observation?.[FIELD_LAT];
  const lon = observation?.[FIELD_LON];
  return `(${fmtCoord(lat)}, ${fmtCoord(lon)})`;
}

export function wrapText(doc, text, maxWidth, opts = {}) {
  const lines = [];
  const paragraphs = String(text || '').split(/\r?\n/);
  for (const paragraph of paragraphs) {
    const words = paragraph.trim().split(/\s+/).filter(Boolean);
    if (!words.length) {
      lines.push('');
      continue;
    }
    let line = '';
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (doc.textWidth(candidate, opts) <= maxWidth || !line) {
        line = candidate;
      } else {
        lines.push(line);
        line = word;
      }
    }
    if (line) lines.push(line);
  }
  return lines;
}

export function slugFilenamePart(value) {
  const out = String(value || '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace(/[^a-z0-9_-]/g, '');
  return out || 'osservazione';
}

function ensureSpace(doc, y, needed) {
  if (y + needed <= doc.height - PAGE_MARGIN) return y;
  doc.addPage();
  return PAGE_MARGIN;
}

async function observationPDFPhotos(observation) {
  const photos = Array.isArray(observation?.[FIELD_PHOTOS]) ? observation[FIELD_PHOTOS] : [];
  const out = [];
  for (const photo of photos) {
    if (!String(photo?.[FIELD_CONTENT_TYPE] || '').startsWith('image/')) continue;
    try {
      out.push(await imageURLToJPEG(photo[FIELD_URL], CANVAS_PHOTO_MAX_PX));
    } catch {
      // Report generation must not fail because one optional photo cannot load.
    }
  }
  return out;
}

async function imageURLToJPEG(url, maxDimension) {
  const response = await fetch(url, { credentials: 'same-origin' });
  if (!response.ok) throw new Error('image_fetch_failed');
  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = await loadImage(objectUrl);
    const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth, image.naturalHeight));
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));
    return imageToJPEG(image, width, height);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function imageToJPEG(image, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(image, 0, 0, width, height);
  return canvasToJPEG(canvas);
}

async function observationOverviewMap(observation, parcelsGeo) {
  const lat = Number(observation?.[FIELD_LAT]);
  const lon = Number(observation?.[FIELD_LON]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  const canvas = document.createElement('canvas');
  canvas.width = MAP_CANVAS_WIDTH;
  canvas.height = MAP_CANVAS_HEIGHT;
  const ctx = canvas.getContext('2d');
  try {
    await drawOSMTiles(ctx, lat, lon, canvas.width, canvas.height, OSM_OVERVIEW_ZOOM);
    drawObservationMarker(ctx, canvas.width / 2, canvas.height / 2);
    return canvasToJPEG(canvas);
  } catch {
    drawFallbackMap(ctx, lat, lon, canvas.width, canvas.height, parcelsGeo);
    drawObservationMarker(ctx, canvas.width / 2, canvas.height / 2);
    return canvasToJPEG(canvas);
  }
}

async function drawOSMTiles(ctx, lat, lon, width, height, zoom) {
  const center = latLonToPixel(lat, lon, zoom);
  const left = center.x - width / 2;
  const top = center.y - height / 2;
  const minX = Math.floor(left / OSM_TILE_SIZE);
  const maxX = Math.floor((left + width) / OSM_TILE_SIZE);
  const minY = Math.floor(top / OSM_TILE_SIZE);
  const maxY = Math.floor((top + height) / OSM_TILE_SIZE);
  const maxTile = 2 ** zoom;
  const tasks = [];
  for (let x = minX; x <= maxX; x += 1) {
    for (let y = minY; y <= maxY; y += 1) {
      if (y < 0 || y >= maxTile) continue;
      const wrappedX = ((x % maxTile) + maxTile) % maxTile;
      tasks.push(loadTile(wrappedX, y, zoom).then(image => ({ image, x, y })));
    }
  }
  const tiles = await Promise.all(tasks);
  for (const tile of tiles) {
    ctx.drawImage(
      tile.image,
      Math.round(tile.x * OSM_TILE_SIZE - left),
      Math.round(tile.y * OSM_TILE_SIZE - top),
    );
  }
}

function drawFallbackMap(ctx, lat, lon, width, height, parcelsGeo) {
  ctx.fillStyle = '#eef2ec';
  ctx.fillRect(0, 0, width, height);
  const center = latLonToPixel(lat, lon, OSM_OVERVIEW_ZOOM);
  const left = center.x - width / 2;
  const top = center.y - height / 2;
  ctx.strokeStyle = '#6a8065';
  ctx.lineWidth = 2;
  for (const feature of parcelsGeo?.features || []) drawFeature(ctx, feature, left, top);
}

function drawFeature(ctx, feature, left, top) {
  const geom = feature?.geometry;
  if (!geom) return;
  const polygons = geom.type === 'Polygon'
    ? [geom.coordinates]
    : geom.type === 'MultiPolygon' ? geom.coordinates : [];
  for (const polygon of polygons) {
    for (const ring of polygon) {
      let started = false;
      ctx.beginPath();
      for (const coord of ring) {
        const point = latLonToPixel(coord[1], coord[0], OSM_OVERVIEW_ZOOM);
        const x = point.x - left;
        const y = point.y - top;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
  }
}

function drawObservationMarker(ctx, x, y) {
  ctx.fillStyle = '#17613a';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(x, y, 15, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function loadTile(x, y, z) {
  return loadImage(`https://a.tile.openstreetmap.org/${z}/${x}/${y}.png`, 'anonymous');
}

function loadImage(url, crossOrigin = '') {
  return new Promise((resolve, reject) => {
    const image = new Image();
    if (crossOrigin) image.crossOrigin = crossOrigin;
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('image_load_failed'));
    image.src = url;
  });
}

function canvasToJPEG(canvas) {
  const dataUrl = canvas.toDataURL('image/jpeg', IMAGE_JPEG_QUALITY);
  return {
    dataBase64: dataUrl.split(',')[1] || '',
    width: canvas.width,
    height: canvas.height,
  };
}

function fittedSize(image, maxWidth, maxHeight) {
  const scale = Math.min(maxWidth / image.width, maxHeight / image.height);
  return {
    photo: image,
    width: image.width * scale,
    height: image.height * scale,
  };
}

function latLonToPixel(lat, lon, zoom) {
  const sinLat = Math.sin(lat * Math.PI / 180);
  const scale = OSM_TILE_SIZE * 2 ** zoom;
  return {
    x: (lon + 180) / 360 * scale,
    y: (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale,
  };
}
