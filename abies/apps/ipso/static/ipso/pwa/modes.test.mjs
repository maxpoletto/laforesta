import { createRequire } from "module";

const require = createRequire(import.meta.url);
const {
  IPSO_MODE_MARTELLATE,
  IPSO_MODE_SAMPLES,
  IPSO_MODE_FREE_SURVEY_PLACEHOLDER,
  IPSO_MODE_PAI,
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
  IPSO_MODE_FREE_SURVEY_PLACEHOLDER,
  IPSO_MODE_MARTELLATE,
  IPSO_MODE_PAI,
  IPSO_MODE_MAP,
], "landing mode order keeps predefined surveys first");

const freeSurvey = allModes.find((mode) =>
  mode.id === IPSO_MODE_FREE_SURVEY_PLACEHOLDER
);
check(Boolean(freeSurvey), "free-survey placeholder is present");
check(freeSurvey.enabled === false, "free-survey placeholder is disabled");
check(freeSurvey.labelKey === "MODE_FREE_SURVEYS",
      "free-survey placeholder uses a localized label key");
check(IpsoModes.get(IPSO_MODE_FREE_SURVEY_PLACEHOLDER).id === IPSO_MODE_MARTELLATE,
      "free-survey placeholder is not an upload/session mode");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(pass + " mode-registry tests passed");
