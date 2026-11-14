// ipso mode registry.
//
// Mode behavior lives here so the shell can share session, GPS, map, local
// storage, and upload flow while individual modes keep their recording rules.
'use strict';

if (typeof module !== 'undefined' && typeof require !== 'undefined' &&
    typeof UPLOAD_SCHEMA_VERSION === 'undefined') {
  Object.assign(globalThis, require('./constants.js'));
}

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
      parcelRequired: true,
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
    [IPSO_MODE_FREE_SURVEY]: {
      id: IPSO_MODE_FREE_SURVEY,
      labelKey: 'MODE_FREE_SURVEYS',
      preTitleKey: 'PRE_NEW_FREE_SURVEY',
      buttonId: 'btn-mode-free-survey',
      autoHeight: true,
      blankSmallNumber: false,
      dRequired: true,
      hRequired: true,
      numberRequired: false,
      parcelRequired: true,
      persistNumber: false,
      freeSurvey: true,
      enabled: true,
    },
    [IPSO_MODE_OBSERVATIONS]: {
      id: IPSO_MODE_OBSERVATIONS,
      labelKey: 'MODE_OBSERVATIONS',
      preTitleKey: 'PRE_NEW_OBSERVATION',
      buttonId: 'btn-mode-observations',
      autoHeight: false,
      dRequired: false,
      hRequired: false,
      numberRequired: false,
      parcelRequired: false,
      persistNumber: false,
      observations: true,
      enabled: true,
    },
    [IPSO_MODE_MAP]: {
      id: IPSO_MODE_MAP,
      labelKey: 'MODE_MAP',
      buttonId: 'btn-mode-map',
      mapOnly: true,
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
    return [
      defs[IPSO_MODE_SAMPLES],
      defs[IPSO_MODE_FREE_SURVEY],
      defs[IPSO_MODE_MARTELLATE],
      defs[IPSO_MODE_OBSERVATIONS],
      defs[IPSO_MODE_MAP],
    ];
  }

  return {
    MARTELLATE: IPSO_MODE_MARTELLATE,
    SAMPLES: IPSO_MODE_SAMPLES,
    FREE_SURVEY: IPSO_MODE_FREE_SURVEY,
    OBSERVATIONS: IPSO_MODE_OBSERVATIONS,
    MAP: IPSO_MODE_MAP,
    get, defaultMode, all,
  };
})();

if (typeof module !== 'undefined') {
  module.exports = {
    IPSO_MODE_MARTELLATE, IPSO_MODE_SAMPLES, IPSO_MODE_FREE_SURVEY,
    IPSO_MODE_OBSERVATIONS, IPSO_MODE_MAP,
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX,
    IpsoModes,
  };
}
