/**
 * Abies — application entry point.
 *
 * Imports shared infrastructure and all domain page modules,
 * registers routes, and boots the client-side router.
 */

import * as router from './router.js';
import * as cache from './cache.js';
import * as bosco from '../../bosco/js/bosco.js';
import * as prelievi from '../../prelievi/js/prelievi.js';
import * as controllo from '../../controllo/js/controllo.js';
import * as impostazioni from '../../impostazioni/js/impostazioni.js';

// Domain page modules — all loaded eagerly at boot.
// Each exports { mount(params), unmount(), onQueryChange(params) }.

// Register routes.
router.addRoute('bosco', bosco);
router.addRoute('prelievi', prelievi);
router.addRoute('controllo', controllo);
router.addRoute('impostazioni', impostazioni);

// Boot.
router.init();
cache.startRefresh();
