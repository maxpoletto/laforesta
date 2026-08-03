// Observation photo preparation for Ipso.
//
// Extracts the GPS metadata we need before optional browser-side image
// downscaling strips EXIF from the uploaded JPEG.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof FIELD_LAT === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

const PHOTO_MAX_DIMENSION_PX = 2000;
const PHOTO_JPEG_QUALITY = 0.82;
const PHOTO_OUTPUT_TYPE = 'image/jpeg';
const PHOTO_POSITION_PROMPT_DISTANCE_M = 100;

async function prepareObservationPhoto(file, opts = {}) {
  const metadata = await extractPhotoMetadataFromFile(file);
  const conversion = await compressImageFile(file, opts).catch((err) => (
    imageConversionResult(
      file, null, null, null, null, PHOTO_CONVERSION_FAILED, errorReason(err)
    )
  ));
  const out = validBlob(conversion.blob) ? conversion.blob : file;
  return {
    blob: out,
    metadata,
    contentType: out.type || file.type || '',
    sizeBytes: Number.isInteger(out.size) ? out.size : 0,
    originalFilename: file.name || '',
    [FIELD_WIDTH_PX]: conversion[FIELD_WIDTH_PX] || null,
    [FIELD_HEIGHT_PX]: conversion[FIELD_HEIGHT_PX] || null,
    [FIELD_ORIGINAL_SIZE_BYTES]: Number.isInteger(file && file.size)
      ? file.size : 0,
    [FIELD_ORIGINAL_WIDTH_PX]: conversion[FIELD_ORIGINAL_WIDTH_PX] || null,
    [FIELD_ORIGINAL_HEIGHT_PX]: conversion[FIELD_ORIGINAL_HEIGHT_PX] || null,
    [FIELD_CONVERSION_STATUS]: conversion[FIELD_CONVERSION_STATUS] || '',
    [FIELD_CONVERSION_REASON]: conversion[FIELD_CONVERSION_REASON] || '',
  };
}

async function extractPhotoMetadataFromFile(file) {
  if (!file || typeof file.arrayBuffer !== 'function') return {};
  try {
    return extractPhotoMetadata(await file.arrayBuffer());
  } catch (_) {
    return {};
  }
}

function extractPhotoMetadata(buffer) {
  const view = buffer instanceof DataView ? buffer : new DataView(buffer);
  if (view.byteLength < 4 || view.getUint16(0, false) !== 0xffd8) return {};
  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    if (view.getUint8(offset) !== 0xff) return {};
    const marker = view.getUint8(offset + 1);
    if (marker === 0xda || marker === 0xd9) return {};
    const length = view.getUint16(offset + 2, false);
    if (length < 2) return {};
    const payloadStart = offset + 4;
    const payloadEnd = offset + 2 + length;
    if (payloadEnd > view.byteLength) return {};
    if (marker === 0xe1 && hasExifHeader(view, payloadStart)) {
      return parseExifTiff(view, payloadStart + 6, payloadEnd);
    }
    offset = payloadEnd;
  }
  return {};
}

function hasExifHeader(view, offset) {
  return offset + 6 <= view.byteLength &&
    view.getUint8(offset) === 0x45 &&
    view.getUint8(offset + 1) === 0x78 &&
    view.getUint8(offset + 2) === 0x69 &&
    view.getUint8(offset + 3) === 0x66 &&
    view.getUint8(offset + 4) === 0x00 &&
    view.getUint8(offset + 5) === 0x00;
}

function parseExifTiff(view, tiffStart, tiffEnd) {
  if (tiffStart + 8 > tiffEnd) return {};
  const byteOrder = view.getUint16(tiffStart, false);
  const little = byteOrder === 0x4949;
  if (!little && byteOrder !== 0x4d4d) return {};
  if (view.getUint16(tiffStart + 2, little) !== 42) return {};
  const ifd0Offset = view.getUint32(tiffStart + 4, little);
  const ifd0 = readIfd(view, tiffStart, tiffEnd, ifd0Offset, little);
  const gpsOffset = ifd0.get(0x8825)?.uint32;
  const metadata = {};
  const orientation = ifd0.get(0x0112)?.uint16;
  if (Number.isInteger(orientation)) metadata.orientation = orientation;
  if (Number.isInteger(gpsOffset) && gpsOffset > 0) {
    Object.assign(metadata, gpsMetadata(view, tiffStart, tiffEnd, gpsOffset, little));
  }
  return metadata;
}

function gpsMetadata(view, tiffStart, tiffEnd, gpsOffset, little) {
  const gps = readIfd(view, tiffStart, tiffEnd, gpsOffset, little);
  const lat = gpsCoordinate(gps.get(0x0001)?.ascii, gps.get(0x0002)?.rationals);
  const lon = gpsCoordinate(gps.get(0x0003)?.ascii, gps.get(0x0004)?.rationals);
  const out = {};
  if (Number.isFinite(lat) && Number.isFinite(lon)) {
    out[FIELD_LAT] = lat;
    out[FIELD_LON] = lon;
  }
  const takenAt = gpsTimestamp(gps.get(0x001d)?.ascii, gps.get(0x0007)?.rationals);
  if (takenAt) out[FIELD_TAKEN_AT] = takenAt;
  return out;
}

function readIfd(view, tiffStart, tiffEnd, ifdOffset, little) {
  const entries = new Map();
  const start = tiffStart + ifdOffset;
  if (!Number.isInteger(ifdOffset) || start + 2 > tiffEnd) return entries;
  const count = view.getUint16(start, little);
  for (let i = 0; i < count; i += 1) {
    const entry = start + 2 + i * 12;
    if (entry + 12 > tiffEnd) break;
    const tag = view.getUint16(entry, little);
    const type = view.getUint16(entry + 2, little);
    const valueCount = view.getUint32(entry + 4, little);
    const valueOffset = valueDataOffset(
      view, tiffStart, tiffEnd, entry, type, valueCount, little
    );
    if (valueOffset == null) continue;
    entries.set(tag, readExifValue(
      view, tiffEnd, valueOffset, type, valueCount, little
    ));
  }
  return entries;
}

function valueDataOffset(view, tiffStart, tiffEnd, entry, type, count, little) {
  const bytes = typeSize(type) * count;
  if (!Number.isFinite(bytes) || bytes < 0) return null;
  if (bytes <= 4) return entry + 8;
  const offset = tiffStart + view.getUint32(entry + 8, little);
  return offset >= tiffStart && offset + bytes <= tiffEnd ? offset : null;
}

function readExifValue(view, tiffEnd, offset, type, count, little) {
  if (type === 2) return { ascii: readAscii(view, offset, count, tiffEnd) };
  if (type === 3 && count >= 1 && offset + 2 <= tiffEnd) {
    return { uint16: view.getUint16(offset, little) };
  }
  if (type === 4 && count >= 1 && offset + 4 <= tiffEnd) {
    return { uint32: view.getUint32(offset, little) };
  }
  if (type === 5 && offset + count * 8 <= tiffEnd) {
    return { rationals: readRationals(view, offset, count, little) };
  }
  return {};
}

function typeSize(type) {
  switch (type) {
    case 1:
    case 2:
    case 7:
      return 1;
    case 3:
      return 2;
    case 4:
    case 9:
      return 4;
    case 5:
    case 10:
      return 8;
    default:
      return NaN;
  }
}

function readAscii(view, offset, count, limit) {
  const end = Math.min(offset + count, limit);
  let out = '';
  for (let i = offset; i < end; i += 1) {
    const c = view.getUint8(i);
    if (c === 0) break;
    out += String.fromCharCode(c);
  }
  return out.trim();
}

function readRationals(view, offset, count, little) {
  const values = [];
  for (let i = 0; i < count; i += 1) {
    const pos = offset + i * 8;
    const numerator = view.getUint32(pos, little);
    const denominator = view.getUint32(pos + 4, little);
    values.push(denominator ? numerator / denominator : NaN);
  }
  return values;
}

function gpsCoordinate(ref, rationals) {
  if (!Array.isArray(rationals) || rationals.length < 3) return null;
  const value = rationals[0] + rationals[1] / 60 + rationals[2] / 3600;
  if (!Number.isFinite(value)) return null;
  const negative = String(ref || '').toUpperCase() === 'S' ||
    String(ref || '').toUpperCase() === 'W';
  return roundCoord(negative ? -value : value);
}

function roundCoord(value) {
  return Math.round(value * 1e7) / 1e7;
}

function gpsTimestamp(dateStamp, time) {
  if (!dateStamp || !Array.isArray(time) || time.length < 3) return '';
  const match = String(dateStamp).match(/^(\d{4}):(\d{2}):(\d{2})$/);
  if (!match) return '';
  const h = Math.floor(time[0]);
  const m = Math.floor(time[1]);
  const s = Math.floor(time[2]);
  if (![h, m, s].every(Number.isFinite)) return '';
  if (h < 0 || h > 23 || m < 0 || m > 59 || s < 0 || s > 60) return '';
  return `${match[1]}-${match[2]}-${match[3]}T${pad2(h)}:${pad2(m)}:${pad2(s)}Z`;
}

function pad2(value) { return String(value).padStart(2, '0'); }

async function compressImageFile(file, opts = {}) {
  if (!isLikelyImageFile(file)) {
    return unavailableConversion(file, 'unsupported_type');
  }
  if (typeof document === 'undefined') {
    return unavailableConversion(file, 'canvas_unavailable');
  }
  const decoded = await decodeImage(file);
  try {
    const sourceWidth = decoded.width || 0;
    const sourceHeight = decoded.height || 0;
    if (!sourceWidth || !sourceHeight) {
      return failedConversion(file, 'source_dimensions_missing');
    }
    const maxDimension = positiveNumber(opts.maxDimension, PHOTO_MAX_DIMENSION_PX);
    const scale = Math.min(
      1, maxDimension / Math.max(sourceWidth, sourceHeight)
    );
    if (!Number.isFinite(scale) || scale <= 0) {
      return failedConversion(file, 'invalid_scale', sourceWidth, sourceHeight);
    }
    const targetWidth = Math.max(1, Math.round(sourceWidth * scale));
    const targetHeight = Math.max(1, Math.round(sourceHeight * scale));
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return unavailableConversion(
        file, 'canvas_context_unavailable', sourceWidth, sourceHeight
      );
    }
    if ('fillStyle' in ctx && typeof ctx.fillRect === 'function') {
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, targetWidth, targetHeight);
    }
    ctx.drawImage(decoded.image, 0, 0, targetWidth, targetHeight);
    const quality = jpegQuality(opts.quality);
    const blob = await canvasToBlob(canvas, PHOTO_OUTPUT_TYPE, quality);
    if (!validBlob(blob)) {
      return failedConversion(file, 'canvas_empty_blob', sourceWidth, sourceHeight);
    }

    // If dimensions were reduced, keep the converted image even if the encoded
    // byte size is unexpectedly larger: the max-dimension rule is authoritative.
    const out = scale < 1 || blob.size < file.size ? blob : file;
    const converted = out !== file;
    const width = converted ? targetWidth : sourceWidth;
    const height = converted ? targetHeight : sourceHeight;
    return imageConversionResult(
      out, sourceWidth, sourceHeight, width, height,
      converted ? PHOTO_CONVERSION_CONVERTED : PHOTO_CONVERSION_ORIGINAL,
      converted ? '' : 'not_smaller'
    );
  } finally {
    if (decoded && typeof decoded.close === 'function') decoded.close();
  }
}

function unavailableConversion(
    file, reason, originalWidth = null, originalHeight = null) {
  return imageConversionResult(
    file, originalWidth, originalHeight, originalWidth, originalHeight,
    PHOTO_CONVERSION_UNAVAILABLE, reason
  );
}

function failedConversion(file, reason, originalWidth = null, originalHeight = null) {
  return imageConversionResult(
    file, originalWidth, originalHeight, originalWidth, originalHeight,
    PHOTO_CONVERSION_FAILED, reason
  );
}

function imageConversionResult(
    blob, originalWidth, originalHeight, width, height, status = '', reason = '') {
  return {
    blob,
    [FIELD_WIDTH_PX]: ipsoPositiveInt(width == null ? originalWidth : width),
    [FIELD_HEIGHT_PX]: ipsoPositiveInt(height == null ? originalHeight : height),
    [FIELD_ORIGINAL_WIDTH_PX]: ipsoPositiveInt(originalWidth),
    [FIELD_ORIGINAL_HEIGHT_PX]: ipsoPositiveInt(originalHeight),
    [FIELD_CONVERSION_STATUS]: status,
    [FIELD_CONVERSION_REASON]: reason,
  };
}

function isLikelyImageFile(file) {
  if (!file) return false;
  if (String(file.type || '').startsWith('image/')) return true;
  return /\.(avif|heic|heif|jpe?g|png|webp)$/i.test(String(file.name || ''));
}

function positiveNumber(value, fallback) {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}


function jpegQuality(value) {
  if (!Number.isFinite(value)) return PHOTO_JPEG_QUALITY;
  return Math.min(1, Math.max(0.1, value));
}

function validBlob(blob) {
  return !!blob && Number.isInteger(blob.size) && blob.size > 0;
}

async function decodeImage(file) {
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
      return {
        image: bitmap, width: bitmap.width, height: bitmap.height,
        close: () => {
          if (typeof bitmap.close === 'function') bitmap.close();
        },
      };
    } catch (_) {
      // Fall through to the HTMLImageElement decoder.
    }
  }
  return loadHtmlImage(file);
}

function loadHtmlImage(file) {
  return new Promise((resolve, reject) => {
    if (typeof URL === 'undefined') {
      reject(new Error('url_api_unavailable'));
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({
        image: img,
        width: img.naturalWidth || img.width || 0,
        height: img.naturalHeight || img.height || 0,
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('image_load_failed'));
    };
    img.src = url;
  });
}

function errorReason(err) {
  return err && err.message ? String(err.message) : 'unknown';
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve) => {
    if (!canvas || typeof canvas.toBlob !== 'function') {
      resolve(null);
      return;
    }
    canvas.toBlob(resolve, type, quality);
  });
}

function hasPhotoPosition(photo) {
  return Number.isFinite(photo && photo[FIELD_LAT]) &&
    Number.isFinite(photo && photo[FIELD_LON]);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${Math.round(bytes)} B`;
}

const IpsoPhotos = {
  PHOTO_MAX_DIMENSION_PX,
  PHOTO_JPEG_QUALITY,
  PHOTO_POSITION_PROMPT_DISTANCE_M,
  prepareObservationPhoto,
  extractPhotoMetadata,
  hasPhotoPosition,
  formatBytes,
};

if (typeof module !== 'undefined') module.exports = IpsoPhotos;
