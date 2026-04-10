/**
 * Abies — application entry point.
 *
 * Imports shared infrastructure and all domain page modules,
 * registers routes, and boots the client-side router.
 */

import * as router from './router.js';
import * as cache from './cache.js';
import * as prelievi from '../../prelievi/js/prelievi.js';
import * as controllo from '../../controllo/js/controllo.js';

// Domain page modules — all loaded eagerly at boot.
// Each exports { mount(params), unmount(), onQueryChange(params) }.
// Placeholders for domains not yet implemented.

function placeholder(name) {
  const el = document.getElementById('content');
  return {
    mount() { el.textContent = name; },
    unmount() { el.replaceChildren(); },
    onQueryChange() {},
  };
}

// Register routes.
router.addRoute('bosco', placeholder('Bosco'));
router.addRoute('prelievi', prelievi);
router.addRoute('controllo', controllo);
router.addRoute('impostazioni', placeholder('Impostazioni'));

// Boot.
router.init();
cache.startRefresh();
