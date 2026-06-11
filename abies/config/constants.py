"""Internal symbolic identifiers — wire format, NOT user-facing.

Defines JSON field names, API envelope keys, status enum values, and
other identifiers that travel only between machine boundaries (HTTP
request/response bodies, digests, form `name=` attributes).  These are
locale-independent contracts; the client and server must agree on every
value exactly.

For translatable user-facing text (labels, error messages, button copy,
column headers), see `config/strings.py` / `config/strings_it.py`.
"""

# API envelope keys.
ROW_ID  = 'row_id'
VERSION = 'version'

RECORD   = 'record'
DATA_ID  = 'data_id'
PATCHES  = 'patches'
DELETES  = 'deletes'
COLUMNS  = 'columns'
ROWS     = 'rows'

# Locale-independent digest column identifiers for internal fields.
COL_REGION_ID  = 'Region id'
COL_PARCEL_ID  = 'Parcel id'
COL_SURVEY_ID  = 'Survey id'
COL_SPECIES_ID = 'Species id'
COL_TREE_ID    = 'Tree id'
COL_COPPICE    = 'Coppice'

STATUS   = 'status'
MESSAGE  = 'message'
HTML     = 'html'

# Additional response keys for read-only metadata endpoints.
TRANSITION_RECORDS   = 'transition_records'

# JSON API `status` field values.
STATUS_CONFLICT         = 'conflict'
STATUS_VALIDATION_ERROR = 'validation_error'
STATUS_RATE_LIMITED     = 'rate_limited'
STATUS_NOT_FOUND        = 'not_found'

# Form / JSON-body field names.  Match HTML form `name=` attributes and
# JSON request-body keys; client and server must agree on every name.
# Lowercase to match the wire format.
FIELD_NAME                  = 'name'
FIELD_DATE                  = 'date'
FIELD_DESCRIPTION           = 'description'
FIELD_NUMBER                = 'number'
FIELD_NOTE                  = 'note'
FIELD_NOTE_ID               = 'note_id'
FIELD_NOTES                 = 'notes'
FIELD_EXTRA_NOTE            = 'extra_note'
FIELD_ACTIVE                = 'active'
FIELD_IS_ACTIVE             = 'is_active'
FIELD_LAT                   = 'lat'
FIELD_LON                   = 'lon'
FIELD_ALTITUDE              = 'altitude'
FIELD_ALTITUDE_M            = 'altitude_m'
FIELD_R_M                   = 'r_m'
FIELD_AREA                  = 'area'
FIELD_PARCEL                = 'parcel'
FIELD_PARCEL_ID             = 'parcel_id'
FIELD_COMPRESA              = 'compresa'
FIELD_PARTICELLA            = 'particella'
FIELD_SPECIES               = 'species'
FIELD_SPECIES_ID            = 'species_id'
FIELD_CREW_ID               = 'crew_id'
FIELD_PRODUCT_ID            = 'product_id'
FIELD_SAMPLE_AREA_ID        = 'sample_area_id'
FIELD_SAMPLE_GRID_ID        = 'sample_grid_id'
FIELD_SURVEY_ID             = 'survey_id'
FIELD_TREE_PICK             = 'tree_pick'
FIELD_TREE_PICK_EXISTING_ID = 'tree_pick_existing_id'
FIELD_D_CM                  = 'd_cm'
FIELD_H_M                   = 'h_m'
FIELD_L10_MM                = 'l10_mm'
FIELD_QUINTALS              = 'quintals'
FIELD_HOURS                 = 'hours'
FIELD_LICENSE_PLATE         = 'license_plate'
FIELD_SLIP_COUNT            = 'slip_count'
FIELD_MONTH                 = 'month'
FIELD_VOLUME_M3             = 'volume_m3'
FIELD_MASS_Q                = 'mass_q'
FIELD_PRESERVED             = 'preserved'
FIELD_COPPICE               = 'coppice'
FIELD_HIGHFOREST               = 'fustaia'
FIELD_SHOOT                 = 'shoot'
FIELD_SHOOTS                = 'shoots'
FIELD_NEXT_SHOOT            = 'next_shoot'
FIELD_STANDARD              = 'standard'
FIELD_MANUFACTURER          = 'manufacturer'
FIELD_MODEL                 = 'model'
FIELD_YEAR                  = 'year'
FIELD_COMMON_NAME           = 'common_name'
FIELD_LATIN_NAME            = 'latin_name'
FIELD_DENSITY               = 'density'
FIELD_MINOR                 = 'minor'
FIELD_SORT_ORDER            = 'sort_order'
FIELD_ROLE                  = 'role'
FIELD_EMAIL                 = 'email'
FIELD_USERNAME              = 'username'
FIELD_FIRST_NAME            = 'first_name'
FIELD_LAST_NAME             = 'last_name'
FIELD_LOGIN_METHOD          = 'login_method'
FIELD_PASSWORD              = 'password'
FIELD_PASSWORD1             = 'password1'
FIELD_PASSWORD2             = 'password2'
FIELD_NONCE                 = 'nonce'
FIELD_POINTS                = 'points'
FIELD_NEXT                  = 'next'
FIELD_DEFAULT_DATE          = 'default_date'
FIELD_FILE                  = 'file'
FIELD_PATH                  = 'path'
FIELD_FIRST_DATE            = 'first_date'
FIELD_LAST_DATE             = 'last_date'
FIELD_ERRORS                = 'errors'
FIELD_RECORD1               = 'record1'
FIELD_RECORD2               = 'record2'
FIELD_HARVEST_PLAN_ID       = 'harvest_plan_id'
FIELD_HARVEST_PLAN_ITEM_ID  = 'harvest_plan_item_id'
FIELD_REGION_ID             = 'region_id'
FIELD_YEAR_START            = 'year_start'
FIELD_YEAR_END              = 'year_end'
FIELD_YEAR_PLANNED          = 'year_planned'
FIELD_VOLUME_PLANNED_M3     = 'volume_planned_m3'
FIELD_INTERVENTION_AREA_HA  = 'intervention_area_ha'
FIELD_PERIOD_Y               = 'turno_a'
FIELD_DAMAGED               = 'damaged'
FIELD_UNHEALTHY             = 'unhealthy'
FIELD_PSR                   = 'psr'
FIELD_STATE                 = 'state'
FIELD_OPEN                  = 'open'
FIELD_OPERATOR              = 'operator'
FIELD_H_MEASURED            = 'h_measured'
FIELD_ACC_M                 = 'acc_m'
FIELD_HIGHFOREST_FILE          = 'fustaia_file'
FIELD_COPPICE_FILE            = 'ceduo_file'
FIELD_MIN_N                 = 'min_n'
FIELD_SURVEY_IDS            = 'survey_ids'
FIELD_SOURCE                = 'source'
FIELD_CREATED_AT            = 'created_at'
FIELD_SURVEYS               = 'surveys'

# Digest filesystem identifiers (the digest file is `<name>.json.gz`).
DIGEST_FUTURE_PRODUCTION = 'future_production'
DIGEST_PARCEL_DENDROMETRY = 'parcel_dendrometry'
DIGEST_PARCEL_DENDROMETRY_POINTS = 'parcel_dendrometry_points'
DIGEST_PRESERVED_TREES = 'preserved_trees'
DIGEST_HYPSO_PARAMS = 'hypso_params'

BOSCO_DENDROMETRY_DIGESTS = (
    DIGEST_PARCEL_DENDROMETRY,
    DIGEST_PARCEL_DENDROMETRY_POINTS,
)
BOSCO_TREE_DIGESTS = (*BOSCO_DENDROMETRY_DIGESTS, DIGEST_PRESERVED_TREES)
BOSCO_SPECIES_DIGESTS = (*BOSCO_DENDROMETRY_DIGESTS, DIGEST_PRESERVED_TREES)

# Domain defaults — locale-independent values the client and server must agree
# on exactly (mirrored in constants.js).
DEFAULT_RADIUS_M = 12  # sample-area radius (m) when none is supplied


# Truthy tokens for both edges: form/JSON values (the HTML checkbox 'on', real
# booleans) and CSV cells (Italian sì).  The union is unambiguous — no edge
# emits a token meaningful to the other.
_TRUTHY = ('true', '1', 'yes', 'si', 'sì', 'on')

def is_truthy(value) -> bool:
    """True iff `value` is a recognised truthy token, case-insensitive and
    whitespace-trimmed: true/1/yes/si/sì/on (the bool ``True`` and int ``1``
    stringify into this set).  Safe for form/JSON values and CSV cells alike."""
    return str(value).strip().lower() in _TRUTHY
