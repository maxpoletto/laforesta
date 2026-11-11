import { fetchJSON, postJSON } from '../../base/js/api.js';
import { applyResponseChanges } from '../../base/js/cache.js';
import {
  FIELD_ACTIVE_ID, FIELD_HARVEST_PLAN_ID, FIELD_ID, FIELD_NAME,
  FIELD_NONCE, FIELD_PLANS, FIELD_YEAR_START, FIELD_YEAR_END,
} from '../../base/js/constants.js';
import * as S from '../../base/js/strings.js';

const SOURCE_URL = '/api/bosco/plan-source/';

/** A single saved source shared by both planned-harvest map views. */
export function wirePlanSource(host, { canEdit, onRefresh }) {
  const select = host.querySelector('select');
  const message = host.querySelector('[data-role="plan-source-message"]');
  let activeId = null;
  let loaded = false;
  let pending = null;
  let busy = false;
  let disposed = false;

  function updateDisabled() {
    select.disabled = !canEdit || busy || !loaded || !select.options.length;
  }

  function load(force = false) {
    if (disposed || (loaded && !force)) return Promise.resolve();
    if (pending) return pending;
    select.disabled = true;
    pending = (async () => {
      try {
        const { data } = await fetchJSON(`${SOURCE_URL}data/`);
        if (disposed) return;
        activeId = data[FIELD_ACTIVE_ID];
        select.replaceChildren();
        for (const plan of data[FIELD_PLANS]) {
          const option = document.createElement('option');
          option.value = String(plan[FIELD_ID]);
          option.textContent = `${plan[FIELD_NAME]} (${plan[FIELD_YEAR_START]}–${plan[FIELD_YEAR_END]})`;
          select.appendChild(option);
        }
        select.value = String(activeId ?? '');
        loaded = true;
        message.textContent = select.options.length ? '' : S.BOSCO_NO_HARVEST_PLANS;
      } catch {
        if (!disposed) message.textContent = S.BOSCO_PLAN_SOURCE_ERROR;
      } finally {
        pending = null;
        if (!disposed) updateDisabled();
      }
    })();
    return pending;
  }

  async function change() {
    if (!canEdit || busy || disposed) return;
    const planId = Number(select.value);
    if (!Number.isInteger(planId) || planId <= 0 || planId === activeId) return;
    busy = true;
    updateDisabled();
    message.textContent = S.LOADING;
    try {
      const { data, status } = await postJSON(`${SOURCE_URL}save/`, {
        [FIELD_HARVEST_PLAN_ID]: planId, [FIELD_NONCE]: crypto.randomUUID(),
      });
      if (status >= 400) throw new Error('Plan source save failed');
      activeId = planId;
      applyResponseChanges(data);
      await onRefresh();
      if (!disposed) await load(true);
    } catch {
      if (!disposed) {
        select.value = String(activeId ?? '');
        message.textContent = S.BOSCO_PLAN_SOURCE_ERROR;
      }
    } finally {
      busy = false;
      if (!disposed) updateDisabled();
    }
  }

  select.addEventListener('change', change);
  updateDisabled();
  return {
    load,
    destroy() { disposed = true; select.removeEventListener('change', change); },
  };
}
