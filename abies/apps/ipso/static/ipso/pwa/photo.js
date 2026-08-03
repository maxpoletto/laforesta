// Observation photo preparation for Ipso.
//
// Photos are downscaled/recompressed before upload. Photo EXIF GPS is
// intentionally ignored: observation location comes only from the device GPS
// fix captured by Ipso while the observation is being recorded.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof FIELD_WIDTH_PX === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

const PHOTO_MAX_DIMENSION_PX = 2000;
const PHOTO_JPEG_QUALITY = 0.82;
const PHOTO_OUTPUT_TYPE = 'image/jpeg';

async function prepareObservationPhoto(file, opts = {}) {
  const conversion = await compressImageFile(file, opts).catch((err) => (
    imageConversionResult(
      file, null, null, null, null, PHOTO_CONVERSION_FAILED, errorReason(err)
    )
  ));
  const out = validBlob(conversion.blob) ? conversion.blob : file;
  return {
    blob: out,
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

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${Math.round(bytes)} B`;
}

const IpsoPhotos = {
  PHOTO_MAX_DIMENSION_PX,
  PHOTO_JPEG_QUALITY,
  prepareObservationPhoto,
  formatBytes,
};

if (typeof module !== 'undefined') module.exports = IpsoPhotos;
