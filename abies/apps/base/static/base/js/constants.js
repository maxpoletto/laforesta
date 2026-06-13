/**
 * Internal symbolic identifiers — wire format, NOT user-facing.
 *
 * Defines JSON field names, API envelope keys, status enum values, and
 * other identifiers that travel only between machine boundaries (HTTP
 * request/response bodies, digests, form `name=` attributes).  These
 * are locale-independent contracts; the client and server must agree
 * on every value exactly.
 *
 * For translatable user-facing text (labels, error messages, button
 * copy, column headers), see `strings.js` / `strings_it.js`.
 */

// API envelope keys.
export const ROW_ID  = 'row_id';
export const VERSION = 'version';

export const RECORD   = 'record';
export const DATA_ID  = 'data_id';
export const PATCHES  = 'patches';
export const DELETES  = 'deletes';
export const COLUMNS  = 'columns';
export const ROWS     = 'rows';

// Locale-independent digest column identifiers for internal fields.
export const COL_REGION_ID  = 'Region id';
export const COL_PARCEL_ID  = 'Parcel id';
export const COL_SURVEY_ID  = 'Survey id';
export const COL_SPECIES_ID = 'Species id';
export const COL_TREE_ID    = 'Tree id';
export const COL_COPPICE    = 'Coppice';

export const STATUS   = 'status';
export const MESSAGE  = 'message';
export const HTML     = 'html';


// JSON API `status` field values.
export const STATUS_CONFLICT         = 'conflict';
export const STATUS_VALIDATION_ERROR = 'validation_error';
export const STATUS_RATE_LIMITED     = 'rate_limited';
export const STATUS_NOT_FOUND        = 'not_found';

// Role identifiers — mirror apps/base/models.py Role.TextChoices values.
export const ROLE_ADMIN  = 'admin';
export const ROLE_WRITER = 'writer';
export const ROLE_READER = 'reader';

// Login-method identifiers — mirror apps/base/models.py LoginMethod values.
export const LOGIN_METHOD_PASSWORD = 'password';
export const LOGIN_METHOD_OAUTH    = 'oauth';

// Digest filesystem identifiers (the digest file is `<name>.json.gz`).
export const DIGEST_PARCELS = 'parcels';
export const DIGEST_FUTURE_PRODUCTION = 'future_production';
export const DIGEST_PARCEL_DENDROMETRY = 'parcel_dendrometry';
export const DIGEST_PARCEL_DENDROMETRY_POINTS = 'parcel_dendrometry_points';
export const DIGEST_PRESERVED_TREES = 'preserved_trees';

// Hypsometric-parameter set source — mirror apps/base/models.py HypsoParamSource.
export const HYPSO_SOURCE_COMPUTED = 'computed';
export const HYPSO_SOURCE_IMPORTED = 'imported';

// Regression family — mirror apps/base/models.py HYPSO_FUNC_LN.
export const HYPSO_FUNC_LN = 'ln';

// Form / JSON-body field names.  Match HTML form `name=` attributes and
// JSON request-body keys; client and server must agree on every name.
export const FIELD_NAME                  = 'name';
export const FIELD_DATE                  = 'date';
export const FIELD_DESCRIPTION           = 'description';
export const FIELD_NUMBER                = 'number';
export const FIELD_NOTE                  = 'note';
export const FIELD_NOTE_ID               = 'note_id';
export const FIELD_NOTES                 = 'notes';
export const FIELD_EXTRA_NOTE            = 'extra_note';
export const FIELD_ACTIVE                = 'active';
export const FIELD_IS_ACTIVE             = 'is_active';
export const FIELD_LAT                   = 'lat';
export const FIELD_LON                   = 'lon';
export const FIELD_ALTITUDE              = 'altitude';
export const FIELD_ALTITUDE_M            = 'altitude_m';
export const FIELD_R_M                   = 'r_m';
export const FIELD_AREA                  = 'area';
export const FIELD_PARCEL                = 'parcel';
export const FIELD_PARCEL_ID             = 'parcel_id';
export const FIELD_COMPRESA              = 'compresa';
export const FIELD_PARTICELLA            = 'particella';
export const FIELD_SPECIES               = 'species';
export const FIELD_SPECIES_ID            = 'species_id';
export const FIELD_CREW_ID               = 'crew_id';
export const FIELD_PRODUCT_ID            = 'product_id';
export const FIELD_SAMPLE_AREA_ID        = 'sample_area_id';
export const FIELD_SAMPLE_GRID_ID        = 'sample_grid_id';
export const FIELD_SURVEY_ID             = 'survey_id';
export const FIELD_TREE_PICK             = 'tree_pick';
export const FIELD_TREE_PICK_EXISTING_ID = 'tree_pick_existing_id';
export const FIELD_D_CM                  = 'd_cm';
export const FIELD_H_M                   = 'h_m';
export const FIELD_L10_MM                = 'l10_mm';
export const FIELD_QUINTALS              = 'quintals';
export const FIELD_HOURS                 = 'hours';
export const FIELD_LICENSE_PLATE         = 'license_plate';
export const FIELD_SLIP_COUNT            = 'slip_count';
export const FIELD_MONTH                 = 'month';
export const FIELD_VOLUME_M3             = 'volume_m3';
export const FIELD_MASS_Q                = 'mass_q';
export const FIELD_PRESERVED             = 'preserved';
export const FIELD_COPPICE               = 'coppice';
export const FIELD_HIGHFOREST               = 'fustaia';
export const FIELD_SHOOT                 = 'shoot';
export const FIELD_SHOOTS                = 'shoots';
export const FIELD_NEXT_SHOOT            = 'next_shoot';
export const FIELD_STANDARD              = 'standard';
export const FIELD_MANUFACTURER          = 'manufacturer';
export const FIELD_MODEL                 = 'model';
export const FIELD_YEAR                  = 'year';
export const FIELD_COMMON_NAME           = 'common_name';
export const FIELD_LATIN_NAME            = 'latin_name';
export const FIELD_DENSITY               = 'density';
export const FIELD_SORT_ORDER            = 'sort_order';
export const FIELD_ROLE                  = 'role';
export const FIELD_EMAIL                 = 'email';
export const FIELD_USERNAME              = 'username';
export const FIELD_FIRST_NAME            = 'first_name';
export const FIELD_LAST_NAME             = 'last_name';
export const FIELD_LOGIN_METHOD          = 'login_method';
export const FIELD_PASSWORD              = 'password';
export const FIELD_NONCE                 = 'nonce';
export const FIELD_POINTS                = 'points';
export const FIELD_NEXT                  = 'next';
export const FIELD_DEFAULT_DATE          = 'default_date';
export const FIELD_FILE                  = 'file';
export const FIELD_HIGHFOREST_FILE          = 'fustaia_file';
export const FIELD_COPPICE_FILE            = 'ceduo_file';
export const FIELD_PATH                  = 'path';
export const FIELD_FIRST_DATE            = 'first_date';
export const FIELD_LAST_DATE             = 'last_date';
export const FIELD_ERRORS                = 'errors';
export const FIELD_RECORD1               = 'record1';
export const FIELD_RECORD2               = 'record2';
export const FIELD_HARVEST_PLAN_ID       = 'harvest_plan_id';
export const FIELD_HARVEST_PLAN_ITEM_ID  = 'harvest_plan_item_id';
export const FIELD_YEAR_START            = 'year_start';
export const FIELD_YEAR_END              = 'year_end';
export const FIELD_OPEN                  = 'open';
export const FIELD_PASSWORD1             = 'password1';
export const FIELD_PASSWORD2             = 'password2';
export const FIELD_MIN_N                 = 'min_n';
export const FIELD_SURVEY_IDS            = 'survey_ids';
export const FIELD_SOURCE                = 'source';
export const FIELD_SURVEYS               = 'surveys';
export const FIELD_CREATED_AT            = 'created_at';
export const FIELD_USE_FOR_HEIGHT_PLOTS  = 'use_for_height_plots';

// Domain defaults — locale-independent values the client and server must agree
// on exactly (mirror of constants.py).
export const DEFAULT_RADIUS_M = 12;  // sample-area radius (m) when none is supplied
