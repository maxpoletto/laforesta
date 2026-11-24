import { fetchJSON } from './api.js';
import { VERSION } from './constants.js';
import { show } from './modals.js';
import { cloneTemplate } from './templates.js';

const CHECK_INTERVAL_MS = 5 * 60 * 1000;

export function startVersionCheck() {
  const { appVersion, appVersionUrl } = document.body.dataset;
  if (!appVersion) return;
  const overlay = document.getElementById('modal-container');
  setInterval(async () => {
    let version;
    try {
      const { data } = await fetchJSON(appVersionUrl);
      version = data?.[VERSION];
    } catch {
      return;
    }
    if (!version || version === appVersion || overlay.classList.contains('open')) return;

    const prompt = cloneTemplate('tmpl-app-update');
    prompt.querySelector('[data-action="reload"]').addEventListener('click', () => window.location.reload());
    show(prompt);
  }, CHECK_INTERVAL_MS);
}
