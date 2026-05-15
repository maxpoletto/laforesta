/**
 * Bosco — placeholder page (Stage 2).
 *
 * Displays a link to the existing Boscoscopio app until the full
 * forest visualization is ported into Abies.
 */

import * as S from '../../base/js/strings.js';

const BOSCOSCOPIO_URL = 'https://laforesta.it/bosco/b/';

export function mount() {
  const el = document.getElementById('content');
  el.replaceChildren();

  const p = document.createElement('p');
  p.textContent = S.BOSCO_PLACEHOLDER_MESSAGE;

  const link = document.createElement('a');
  link.href = BOSCOSCOPIO_URL;
  link.target = '_blank';
  link.rel = 'noopener';
  link.textContent = S.BOSCO_OPEN_BOSCOSCOPIO;

  const p2 = document.createElement('p');
  p2.appendChild(link);

  el.appendChild(p);
  el.appendChild(p2);
}

export function unmount() {
  document.getElementById('content').replaceChildren();
}

export function onQueryChange() {}
