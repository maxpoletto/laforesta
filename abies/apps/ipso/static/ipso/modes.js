// ipso mode registry.
//
// Only martellate is implemented today. Keeping mode identity in one place
// makes the shared shell explicit before Campionamenti and PAI are added.
'use strict';

const IPSO_MODE_MARTELLATE = 'martellate';

const IpsoModes = (function() {
  const defs = {
    [IPSO_MODE_MARTELLATE]: {
      id: IPSO_MODE_MARTELLATE,
      preTitleKey: 'PRE_NEW_SESSION',
    },
  };

  function get(id) {
    return defs[id] || defs[IPSO_MODE_MARTELLATE];
  }

  function defaultMode() {
    return defs[IPSO_MODE_MARTELLATE];
  }

  return { MARTELLATE: IPSO_MODE_MARTELLATE, get, defaultMode };
})();

if (typeof module !== 'undefined') {
  module.exports = { IPSO_MODE_MARTELLATE, IpsoModes };
}
