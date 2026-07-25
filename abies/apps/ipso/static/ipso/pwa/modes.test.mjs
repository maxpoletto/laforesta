import { createRequire } from "module";

const require = createRequire(import.meta.url);
const {
  IPSO_MODE_MARTELLATE,
  IPSO_MODE_SAMPLES,
  IPSO_MODE_FREE_SURVEY,
  IPSO_MODE_OBSERVATIONS,
  IPSO_MODE_MAP,
  IpsoModes,
} = require("./modes.js");

let pass = 0;
const failures = [];

function check(ok, msg) {
  if (ok) pass += 1;
  else failures.push(msg);
}

function eq(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  check(a === e, msg + ": expected " + e + ", got " + a);
}

const allModes = IpsoModes.all();

eq(allModes.map((mode) => mode.id), [
  IPSO_MODE_SAMPLES,
  IPSO_MODE_FREE_SURVEY,
  IPSO_MODE_MARTELLATE,
  IPSO_MODE_OBSERVATIONS,
  IPSO_MODE_MAP,
], "landing mode order keeps predefined surveys first");

const freeSurvey = allModes.find((mode) =>
  mode.id === IPSO_MODE_FREE_SURVEY
);
check(Boolean(freeSurvey), "free-survey mode is present");
check(freeSurvey.enabled === true, "free-survey mode is enabled");
check(freeSurvey.labelKey === "MODE_FREE_SURVEYS",
      "free-survey mode uses a localized label key");
check(freeSurvey.autoHeight === true,
      "free-survey mode can derive unmeasured heights");
check(IpsoModes.get(IPSO_MODE_FREE_SURVEY).localOnly !== true,
      "free-survey mode is uploadable");


const observations = allModes.find((mode) =>
  mode.id === IPSO_MODE_OBSERVATIONS
);
check(Boolean(observations), "observations mode is present");
check(observations.enabled === true, "observations mode is enabled");
check(observations.labelKey === "MODE_OBSERVATIONS",
      "observations mode uses a localized label key");
check(observations.observations === true,
      "observations mode is marked as observation-shaped");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(pass + " mode-registry tests passed");
