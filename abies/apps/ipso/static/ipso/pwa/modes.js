// ipso mode registry.
//
// Mode behavior lives here so the shell can share session, GPS, map, local
// storage, and upload flow while individual modes keep their recording rules.
'use strict';

const IPSO_MODE_MARTELLATE = 'martellate';
const IPSO_MODE_SAMPLES = 'samples';
const IPSO_MODE_PAI = 'pai';
const IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX = 'sampling_survey:';

const IpsoModes = (function() {
  const defs = {
    [IPSO_MODE_MARTELLATE]: {
      id: IPSO_MODE_MARTELLATE,
      labelKey: 'MODE_MARTELLATE',
      preTitleKey: 'PRE_NEW_SESSION',
      buttonId: 'btn-mode-martellate',
      autoHeight: true,
      blankSmallNumber: true,
      dRequired: true,
      hRequired: true,
      persistNumber: true,
      enabled: true,
    },
    [IPSO_MODE_SAMPLES]: {
      id: IPSO_MODE_SAMPLES,
      labelKey: 'MODE_SAMPLES',
      preTitleKey: 'PRE_NEW_SAMPLES',
      buttonId: 'btn-mode-samples',
      autoHeight: false,
      blankSmallNumber: false,
      dRequired: true,
      hRequired: true,
      numberRequired: true,
      sampleAreaRequired: true,
      firstNumber: 1,
      persistNumber: false,
      enabled: true,
    },
    [IPSO_MODE_PAI]: {
      id: IPSO_MODE_PAI,
      labelKey: 'MODE_PAI',
      preTitleKey: 'PRE_NEW_PAI',
      buttonId: 'btn-mode-pai',
      autoHeight: false,
      blankSmallNumber: false,
      dRequired: true,
      hRequired: true,
      persistNumber: false,
      enabled: true,
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
    IPSO_MODE_MARTELLATE, IPSO_MODE_SAMPLES, IPSO_MODE_PAI,
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX, IpsoModes,
  };
}
