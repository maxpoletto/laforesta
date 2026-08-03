import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const photos = require('./photo.js');
const {
  FIELD_CONVERSION_REASON, FIELD_CONVERSION_STATUS, FIELD_HEIGHT_PX,
  FIELD_ORIGINAL_HEIGHT_PX, FIELD_ORIGINAL_SIZE_BYTES,
  FIELD_ORIGINAL_WIDTH_PX, FIELD_WIDTH_PX,
  PHOTO_CONVERSION_CONVERTED, PHOTO_CONVERSION_FAILED,
  PHOTO_CONVERSION_ORIGINAL, PHOTO_CONVERSION_UNAVAILABLE,
} = require('./constants.js');

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

function close(actual, expected, msg) {
  if (Math.abs(actual - expected) < 0.000001) pass += 1;
  else failures.push(`${msg} (got ${actual}, expected ${expected})`);
}

check(
  photos.formatBytes(30 * 1024 * 1024) === '30.0 MB',
  'formatBytes formats MiB',
);


class MockFile {
  constructor({ size, name = 'photo.jpg', type = 'image/jpeg' }) {
    this.size = size;
    this.name = name;
    this.type = type;
  }
  async arrayBuffer() { return new Uint8Array([1, 2, 3]).buffer; }
}

class MockBlob {
  constructor(size, type) {
    this.size = size;
    this.type = type;
  }
}

async function withImageConversionEnv({ width, height, blobSize, bitmap }, fn) {
  const priorDocument = globalThis.document;
  const priorUrl = globalThis.URL;
  const priorImage = globalThis.Image;
  const priorCreateImageBitmap = globalThis.createImageBitmap;
  const calls = [];
  globalThis.document = {
    createElement(tag) {
      if (tag !== 'canvas') throw new Error(`unexpected element ${tag}`);
      const canvas = {
        width: 0,
        height: 0,
        getContext(type) {
          if (type !== '2d') return null;
          return {
            fillStyle: '',
            fillRect() {},
            drawImage() {
              calls.push({
                kind: 'drawImage', width: canvas.width, height: canvas.height,
              });
            },
          };
        },
        toBlob(resolve, type, quality) {
          calls.push({
            kind: 'toBlob', width: canvas.width, height: canvas.height,
            type, quality,
          });
          resolve(new MockBlob(blobSize, type));
        },
      };
      return canvas;
    },
  };
  globalThis.URL = {
    createObjectURL() { return 'blob:test'; },
    revokeObjectURL() {},
  };
  if (bitmap) {
    globalThis.createImageBitmap = async () => {
      calls.push({ kind: 'createImageBitmap' });
      if (bitmap === 'fail') throw new Error('bitmap decode failed');
      return { width, height, close() { calls.push({ kind: 'closeBitmap' }); } };
    };
  } else {
    delete globalThis.createImageBitmap;
  }
  globalThis.Image = class {
    set src(_value) {
      this.naturalWidth = width;
      this.naturalHeight = height;
      queueMicrotask(() => this.onload());
    }
  };
  try {
    await fn(calls);
  } finally {
    if (priorDocument === undefined) delete globalThis.document;
    else globalThis.document = priorDocument;
    if (priorUrl === undefined) delete globalThis.URL;
    else globalThis.URL = priorUrl;
    if (priorImage === undefined) delete globalThis.Image;
    else globalThis.Image = priorImage;
    if (priorCreateImageBitmap === undefined) delete globalThis.createImageBitmap;
    else globalThis.createImageBitmap = priorCreateImageBitmap;
  }
}

await withImageConversionEnv(
  { width: 4000, height: 2000, blobSize: 500000 },
  async (calls) => {
    const file = new MockFile({ size: 5000000, name: 'large.jpg' });
    const prepared = await photos.prepareObservationPhoto(file);
    const encoded = calls.find(call => call.kind === 'toBlob');
    check(
      prepared.blob !== file,
      'prepareObservationPhoto stores converted large photo',
    );
    check(
      prepared.sizeBytes === 500000,
      'prepareObservationPhoto reports converted size',
    );
    check(
      prepared[FIELD_ORIGINAL_SIZE_BYTES] === 5000000,
      'prepareObservationPhoto reports original size',
    );
    check(
      prepared[FIELD_ORIGINAL_WIDTH_PX] === 4000 &&
        prepared[FIELD_ORIGINAL_HEIGHT_PX] === 2000,
      'prepareObservationPhoto reports original dimensions',
    );
    check(
      prepared[FIELD_WIDTH_PX] === 2000 && prepared[FIELD_HEIGHT_PX] === 1000,
      'prepareObservationPhoto reports converted dimensions',
    );
    check(
      prepared[FIELD_CONVERSION_STATUS] === PHOTO_CONVERSION_CONVERTED,
      'prepareObservationPhoto reports converted status for resized photos',
    );
    check(
      encoded.width === 2000 && encoded.height === 1000,
      'prepareObservationPhoto caps large photo dimensions at 2000 px',
    );
    close(
      encoded.quality, photos.PHOTO_JPEG_QUALITY,
      'prepareObservationPhoto encodes large photo with configured JPEG quality',
    );
  },
);

await withImageConversionEnv(
  { width: 1000, height: 500, blobSize: 600000 },
  async (calls) => {
    const file = new MockFile({ size: 1000000, name: 'small.jpg' });
    const prepared = await photos.prepareObservationPhoto(file);
    const encoded = calls.find(call => call.kind === 'toBlob');
    check(
      prepared.blob !== file,
      'prepareObservationPhoto re-encodes small photo when smaller',
    );
    check(
      encoded.width === 1000 && encoded.height === 500,
      'prepareObservationPhoto keeps small photo dimensions while re-encoding',
    );
    close(
      encoded.quality, photos.PHOTO_JPEG_QUALITY,
      'prepareObservationPhoto applies JPEG quality to small photos',
    );
  },
);

await withImageConversionEnv(
  { width: 1000, height: 500, blobSize: 1200000 },
  async () => {
    const file = new MockFile({ size: 1000000, name: 'small.jpg' });
    const prepared = await photos.prepareObservationPhoto(file);
    check(
      prepared.blob === file,
      'prepareObservationPhoto keeps original small photo if re-encode is larger',
    );
    check(
      prepared[FIELD_CONVERSION_STATUS] === PHOTO_CONVERSION_ORIGINAL,
      'prepareObservationPhoto reports original status when re-encode is larger',
    );
  },
);

await withImageConversionEnv(
  { width: 4000, height: 2000, blobSize: 500000, bitmap: true },
  async (calls) => {
    const file = new MockFile({ size: 5000000, name: 'large.jpg' });
    await photos.prepareObservationPhoto(file);
    check(
      calls.some(call => call.kind === 'createImageBitmap'),
      'prepareObservationPhoto prefers createImageBitmap when available',
    );
    check(
      calls.some(call => call.kind === 'closeBitmap'),
      'prepareObservationPhoto closes decoded ImageBitmap after drawing',
    );
  },
);
const textFile = new MockFile({
  size: 1000, name: 'note.txt', type: 'text/plain',
});
let preparedUnsupported = await photos.prepareObservationPhoto(textFile);
check(
  preparedUnsupported.blob === textFile,
  'prepareObservationPhoto keeps unsupported file types unchanged',
);
check(
  preparedUnsupported[FIELD_CONVERSION_STATUS] === PHOTO_CONVERSION_UNAVAILABLE &&
    preparedUnsupported[FIELD_CONVERSION_REASON] === 'unsupported_type',
  'prepareObservationPhoto reports unsupported file type conversion status',
);

await withImageConversionEnv(
  { width: 1000, height: 500, blobSize: 0 },
  async () => {
    const file = new MockFile({ size: 1000000, name: 'empty-blob.jpg' });
    const prepared = await photos.prepareObservationPhoto(file);
    check(
      prepared.blob === file,
      'prepareObservationPhoto keeps original when canvas returns empty blob',
    );
    check(
      prepared[FIELD_CONVERSION_STATUS] === PHOTO_CONVERSION_FAILED &&
        prepared[FIELD_CONVERSION_REASON] === 'canvas_empty_blob',
      'prepareObservationPhoto reports empty canvas blob failure',
    );
    check(
      prepared[FIELD_WIDTH_PX] === 1000 && prepared[FIELD_HEIGHT_PX] === 500,
      'prepareObservationPhoto reports original dimensions on conversion failure',
    );
  },
);

await withImageConversionEnv(
  { width: 1000, height: 500, blobSize: 600000, bitmap: 'fail' },
  async (calls) => {
    const file = new MockFile({ size: 1000000, name: 'bitmap-fallback.jpg' });
    await photos.prepareObservationPhoto(file);
    check(
      calls.some(call => call.kind === 'createImageBitmap'),
      'prepareObservationPhoto attempts createImageBitmap before fallback',
    );
    check(
      calls.some(call => call.kind === 'toBlob'),
      'prepareObservationPhoto falls back to HTMLImageElement after bitmap failure',
    );
  },
);

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`photo.js\n\n${pass} passed, 0 failed`);
