import { PENDING_COUNT } from '../../base/js/constants.js';

export function updateInboxDot(data) {
  for (const dot of document.querySelectorAll('[data-ipso-pending-dot]')) {
    dot.hidden = !data?.[PENDING_COUNT];
  }
}
