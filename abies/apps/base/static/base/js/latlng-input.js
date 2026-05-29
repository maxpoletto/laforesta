/**
 * Shared lat/lng entry helper.
 *
 * Used by Campionamenti (manual tree entry, new sample-area form),
 * and (planned) Bosco PAI add + Piano di taglio mark-tree add — see
 * `docs/page-campionamenti.md` §"Lat/lng entry component".
 *
 * Server-rendered form supplies two number inputs.  The client calls
 * `mountUseLocationButton(latEl, lngEl)` to append a "Usa posizione
 * attuale" button to the same row.  The button is only inserted when
 * `navigator.geolocation` is available.
 */

import * as S from './strings.js';

/**
 * Append a geolocation-button to the row that contains `latEl` and
 * `lngEl`.  The button is hidden when geolocation isn't supported.
 *
 * @param {HTMLInputElement} latEl
 * @param {HTMLInputElement} lngEl
 * @param {object} [opts]
 * @param {HTMLElement} [opts.appendTo] — where to append the button.
 *   Defaults to lngEl.parentElement (the lng form-group).
 */
export function mountUseLocationButton(latEl, lngEl, opts = {}) {
  if (!latEl || !lngEl) return;
  if (!('geolocation' in navigator)) return;

  const host = opts.appendTo || lngEl.parentElement;
  if (!host) return;

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'btn latlng-use-location';
  btn.textContent = S.USE_CURRENT_LOCATION;
  btn.addEventListener('click', () => {
    btn.disabled = true;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        latEl.value = pos.coords.latitude.toFixed(5);
        lngEl.value = pos.coords.longitude.toFixed(5);
        latEl.dispatchEvent(new Event('change'));
        lngEl.dispatchEvent(new Event('change'));
        btn.disabled = false;
      },
      () => {
        btn.disabled = false;
        // Permission denied or position unavailable — silent failure,
        // user can still type in the inputs.
      },
      { enableHighAccuracy: true, timeout: 8000 },
    );
  });
  host.appendChild(btn);
}
