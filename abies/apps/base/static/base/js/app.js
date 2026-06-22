/**
 * Abies — application entry point.
 *
 * Imports shared infrastructure and all domain page modules,
 * registers routes, and boots the client-side router.
 */

import * as router from './router.js';
import * as cache from './cache.js';
import * as bosco from '../../bosco/js/bosco.js';
import * as squadre from '../../squadre/js/squadre.js';
import * as prelievi from '../../prelievi/js/prelievi.js';
import * as campionamenti from '../../campionamenti/js/campionamenti.js';
import * as pianoDiTaglio from '../../piano_di_taglio/js/piano-di-taglio.js';
import * as controllo from '../../controllo/js/controllo.js';
import * as impostazioni from '../../impostazioni/js/impostazioni.js';
import * as importazione from '../../ipso/js/importazione.js';

// Domain page modules — all loaded eagerly at boot.
// Each exports { mount(params), unmount(), onQueryChange(params) }.

// Register routes.
router.addRoute('bosco', bosco);
router.addRoute('squadre', squadre);
router.addRoute('prelievi', prelievi);
router.addRoute('campionamenti', campionamenti);
router.addRoute('piano-di-taglio', pianoDiTaglio);
router.addRoute('controllo', controllo);
router.addRoute('importazione', importazione);
router.addRoute('impostazioni', impostazioni);

// Boot.
router.init();
cache.startRefresh();
