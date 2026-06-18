// ipso mode registry.
//
// Only martellate is implemented today. Keeping mode identity in one place
// makes the shared shell explicit before Campionamenti and PAI are added.
'use strict';

const IPSO_MODE_MARTELLATE = 'martellate';
const IPSO_MODE_SAMPLES = 'samples';
const IPSO_MODE_PAI = 'pai';

const IpsoModes = (function() {
  const defs = {
    [IPSO_MODE_MARTELLATE]: {
      id: IPSO_MODE_MARTELLATE,
      labelKey: 'MODE_MARTELLATE',
      preTitleKey: 'PRE_NEW_SESSION',
      enabled: true,
    },
    [IPSO_MODE_SAMPLES]: {
      id: IPSO_MODE_SAMPLES,
      labelKey: 'MODE_SAMPLES',
      preTitleKey: 'MODE_SAMPLES',
      enabled: false,
    },
    [IPSO_MODE_PAI]: {
      id: IPSO_MODE_PAI,
      labelKey: 'MODE_PAI',
      preTitleKey: 'MODE_PAI',
      enabled: false,
    },
  };

  function get(id) {
    return defs[id] || defs[IPSO_MODE_MARTELLATE];
  }

  function defaultMode() {
    return defs[IPSO_MODE_MARTELLATE];
  }

  function all() {
    return [defs[IPSO_MODE_MARTELLATE], defs[IPSO_MODE_SAMPLES], defs[IPSO_MODE_PAI]];
  }

  return {
    MARTELLATE: IPSO_MODE_MARTELLATE,
    SAMPLES: IPSO_MODE_SAMPLES,
    PAI: IPSO_MODE_PAI,
    get, defaultMode, all,
  };
})();

if (typeof module !== 'undefined') {
  module.exports = {
    IPSO_MODE_MARTELLATE, IPSO_MODE_SAMPLES, IPSO_MODE_PAI, IpsoModes,
  };
}
