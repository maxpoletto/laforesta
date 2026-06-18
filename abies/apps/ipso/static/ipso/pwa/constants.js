// Ipso PWA wire identifiers.
//
// Keep these aligned with config/constants.py and base/js/constants.js. The
// offline PWA uses classic scripts, so it cannot synchronously import the
// shared ES module constants at parse time.
'use strict';

const UPLOAD_SCHEMA_VERSION = 1;
const DEFAULT_SAMPLE_RADIUS_M = 12;

const IPSO_MODE_MARTELLATE = 'martellate';
const IPSO_MODE_SAMPLES = 'samples';
const IPSO_MODE_PAI = 'pai';
const IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX = 'sampling_survey:';

const IPSO_REF_SAMPLING = 'sampling';
const IPSO_REF_SURVEYS = 'surveys';
const IPSO_REF_SAMPLE_AREAS = 'sample_areas';
const IPSO_REF_PAI = 'pai';
const IPSO_REF_PRESERVED_TREES = 'preserved_trees';
const IPSO_REF_SPECIES = 'species';
const IPSO_REF_PARCELS = 'parcels';
const IPSO_REF_HYPSOMETRY = 'ipsometrica';

const FIELD_SESSION_ID = 'session_id';
const FIELD_MODE = 'mode';
const FIELD_SCHEMA_VERSION = 'schema_version';
const FIELD_REFERENCE_VERSION = 'reference_version';
const FIELD_WORK_PACKAGE_ID = 'work_package_id';
const FIELD_OPERATOR = 'operator';
const FIELD_CREATED_AT = 'created_at';
const FIELD_COMPLETED_AT = 'completed_at';
const FIELD_DAMAGED = 'damaged';
const FIELD_REGION_ID = 'region_id';
const FIELD_CLIENT_RECORD_ID = 'client_record_id';
const FIELD_DATE = 'date';
const FIELD_PARCEL_ID = 'parcel_id';
const FIELD_SPECIES_ID = 'species_id';
const FIELD_NUMBER = 'number';
const FIELD_D_CM = 'd_cm';
const FIELD_H_M = 'h_m';
const FIELD_H_MEASURED = 'h_measured';
const FIELD_HYPSO_PARAM_SET_ID = 'hypso_param_set_id';
const FIELD_LAT = 'lat';
const FIELD_LON = 'lon';
const FIELD_ACC_M = 'acc_m';
const FIELD_SAMPLE_AREA_ID = 'sample_area_id';
const FIELD_SAMPLE_GRID_ID = 'sample_grid_id';
const FIELD_SURVEY_ID = 'survey_id';
const FIELD_R_M = 'r_m';
const FIELD_COPPICE = 'coppice';
const FIELD_SHOOT = 'shoot';
const FIELD_STANDARD = 'standard';
const FIELD_L10_MM = 'l10_mm';
const FIELD_PRESSLER_COEFF = 'pressler_coeff';
const FIELD_PRESERVED = 'preserved';
const FIELD_ESTIMATED_BIRTH_YEAR = 'estimated_birth_year';
const FIELD_NOTE = 'note';
const FIELD_CSV_TEXT = 'csv_text';

const SESSION = 'session';
const RECORDS = 'records';
const IPSO_BOOTSTRAP_BEARER_TOKEN = 'bearer_token';

const PRESSLER_DEFAULT = '2.00';

if (typeof module !== 'undefined') {
  module.exports = {
    UPLOAD_SCHEMA_VERSION,
    DEFAULT_SAMPLE_RADIUS_M,
    IPSO_MODE_MARTELLATE,
    IPSO_MODE_SAMPLES,
    IPSO_MODE_PAI,
    IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX,
    IPSO_REF_SAMPLING,
    IPSO_REF_SURVEYS,
    IPSO_REF_SAMPLE_AREAS,
    IPSO_REF_PAI,
    IPSO_REF_PRESERVED_TREES,
    IPSO_REF_SPECIES,
    IPSO_REF_PARCELS,
    IPSO_REF_HYPSOMETRY,
    FIELD_SESSION_ID,
    FIELD_MODE,
    FIELD_SCHEMA_VERSION,
    FIELD_REFERENCE_VERSION,
    FIELD_WORK_PACKAGE_ID,
    FIELD_OPERATOR,
    FIELD_CREATED_AT,
    FIELD_COMPLETED_AT,
    FIELD_DAMAGED,
    FIELD_REGION_ID,
    FIELD_CLIENT_RECORD_ID,
    FIELD_DATE,
    FIELD_PARCEL_ID,
    FIELD_SPECIES_ID,
    FIELD_NUMBER,
    FIELD_D_CM,
    FIELD_H_M,
    FIELD_H_MEASURED,
    FIELD_HYPSO_PARAM_SET_ID,
    FIELD_LAT,
    FIELD_LON,
    FIELD_ACC_M,
    FIELD_SAMPLE_AREA_ID,
    FIELD_SAMPLE_GRID_ID,
    FIELD_SURVEY_ID,
    FIELD_R_M,
    FIELD_COPPICE,
    FIELD_SHOOT,
    FIELD_STANDARD,
    FIELD_L10_MM,
    FIELD_PRESSLER_COEFF,
    FIELD_PRESERVED,
    FIELD_ESTIMATED_BIRTH_YEAR,
    FIELD_NOTE,
    FIELD_CSV_TEXT,
    SESSION,
    RECORDS,
    IPSO_BOOTSTRAP_BEARER_TOKEN,
    PRESSLER_DEFAULT,
  };
  Object.assign(globalThis, module.exports);
}
