/**
 * HTTP helpers: conditional GET and POST with CSRF.
 */

function csrfToken() {
  return document.body.dataset.csrf;
}

/**
 * Conditional GET.  Returns {data, status} where data is parsed JSON
 * (or null on 304) and status is the HTTP status code.
 *
 * @param {string} url
 * @param {string|null} lastModified  — value for If-Modified-Since header
 * @returns {Promise<{data: any, status: number, lastModified: string|null}>}
 */
export async function fetchJSON(url, lastModified = null) {
  const headers = { 'Accept': 'application/json' };
  if (lastModified) {
    headers['If-Modified-Since'] = lastModified;
  }
  const resp = await fetch(url, { headers });
  if (resp.status === 304) {
    return { data: null, status: 304, lastModified };
  }
  if (!resp.ok) {
    throw new Error(`GET ${url}: ${resp.status}`);
  }
  const data = await resp.json();
  return {
    data,
    status: resp.status,
    lastModified: resp.headers.get('Last-Modified'),
  };
}

/**
 * POST JSON payload.  Includes CSRF token.
 *
 * @param {string} url
 * @param {object} body
 * @returns {Promise<{data: any, status: number}>}
 */
export async function postJSON(url, body) {
  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': csrfToken(),
    },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  return { data, status: resp.status };
}

/**
 * Read a browser File/Blob as base64 so upload writes can stay on the
 * same JSON+nonce protocol as every other mutating endpoint.
 *
 * @param {Blob} file
 * @returns {Promise<string>}
 */
export async function fileToBase64(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

