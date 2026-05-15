/**
 * App-wide string constants.
 *
 * Locale-specific strings live in `strings_<lang>.js` and are re-exported
 * below — change the import line to switch language.  Locale-INDEPENDENT
 * identifiers (wire markers, internal column names, etc.) are defined
 * directly in this file so they aren't duplicated across language files.
 */

export * from './strings_it.js';

// Internal identifiers — wire format, never displayed.
// Used both as digest-column keys and as JSON API field names.
export const ROW_ID  = 'row_id';
export const VERSION = 'version';

// JSON API field names shared between Python views and JS clients.
export const RECORD   = 'record';
export const RECORDS  = 'records';
export const DATA_ID  = 'data_id';
export const COLUMNS  = 'columns';
export const ROWS     = 'rows';
export const STATUS   = 'status';
export const MESSAGE  = 'message';
export const HTML     = 'html';

// Side-effect record keys returned alongside the primary RECORD so the
// client can update related digests.  See CLAUDE.md §"Optimistic table
// updates".
export const SAMPLE_RECORD   = 'sample_record';
export const SURVEY_RECORD   = 'survey_record';
export const SURVEY_RECORDS  = 'survey_records';
export const GRID_RECORD     = 'grid_record';
export const AREA_RECORDS    = 'area_records';

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
export const FIELD_VOLUME_M3             = 'volume_m3';
export const FIELD_MASS_Q                = 'mass_q';
export const FIELD_PRESERVED             = 'preserved';
export const FIELD_COPPICE               = 'coppice';
export const FIELD_FUSTAIA               = 'fustaia';
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
export const FIELD_PATH                  = 'path';
export const FIELD_FIRST_DATE            = 'first_date';
export const FIELD_LAST_DATE             = 'last_date';
export const FIELD_ERRORS                = 'errors';
export const FIELD_RECORD1               = 'record1';
export const FIELD_RECORD2               = 'record2';
