"""Internal symbolic identifiers — wire format, NOT user-facing.

Defines JSON field names, API envelope keys, status enum values, and
other identifiers that travel only between machine boundaries (HTTP
request/response bodies, digests, form `name=` attributes).  These are
locale-independent contracts; the client and server must agree on every
value exactly.

For translatable user-facing text (labels, error messages, button copy,
column headers), see `config/strings.py` / `config/strings_it.py`.
"""

from decimal import Decimal

# API envelope keys.
ROW_ID  = 'row_id'
VERSION = 'version'

RECORD   = 'record'
DATA_ID  = 'data_id'
PATCHES  = 'patches'
DELETES  = 'deletes'
COLUMNS  = 'columns'
ROWS     = 'rows'

# Digest IDs.
DIGEST_PARCELS = 'parcels'

# Dendrometry defaults.
PRESSLER_DEFAULT = Decimal('2.00')

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

# Device / integration API envelope keys.
OK                  = 'ok'
ERROR               = 'error'
DETAIL              = 'detail'
DUPLICATE           = 'duplicate'
STORED_AS           = 'stored_as'
IMPORTED            = 'imported'
SKIPPED_DUPLICATES  = 'skipped_duplicates'
UPLOAD              = 'upload'
SESSION             = 'session'
RECORDS             = 'records'
RECORD_COUNT        = 'record_count'
FILE_ERROR          = 'file_error'
TARGETS                    = 'targets'
SUGGESTED_TARGET_ID        = 'suggested_target_id'
PENDING_COUNT              = 'pending_count'
IPSO_SECRET_HASH_PARAM = 'secret'
IPSO_BEARER_STORAGE_KEY = 'ipso.bearer_token'
IPSO_ERROR_AUTH            = 'auth'
IPSO_ERROR_INVALID_PAYLOAD = 'invalid_payload'
IPSO_ERROR_CONFLICT        = 'conflict'
IPSO_ERROR_RATE_LIMITED    = 'rate_limited'
IPSO_ERROR_TOO_LARGE       = 'too_large'

# Form / JSON-body field names.  Match HTML form `name=` attributes and
# JSON request-body keys; client and server must agree on every name.
# Lowercase to match the wire format.
FIELD_ID                    = 'id'
FIELD_NAME                  = 'name'
FIELD_DATE                  = 'date'
FIELD_DESCRIPTION           = 'description'
FIELD_NUMBER                = 'number'
FIELD_MAX_TREE_NUMBER       = 'max_tree_number'
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
FIELD_TREE_ID               = 'tree_id'
FIELD_TREE_PRESERVED_ID     = 'tree_preserved_id'
FIELD_D_CM                  = 'd_cm'
FIELD_H_M                   = 'h_m'
FIELD_L10_MM                = 'l10_mm'
FIELD_PRESSLER_COEFF         = 'pressler_coeff'
FIELD_QUINTALS              = 'quintals'
FIELD_HOURS                 = 'hours'
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
FIELD_ESTIMATED_BIRTH_YEAR  = 'estimated_birth_year'
FIELD_COMMON_NAME           = 'common_name'
FIELD_LATIN_NAME            = 'latin_name'
FIELD_DENSITY               = 'density'
FIELD_PRESSLER_DEFAULT       = 'pressler_default'
FIELD_MINOR                 = 'minor'
FIELD_SORT_ORDER            = 'sort_order'
FIELD_MIN_HARVEST_VOLUME    = 'min_harvest_volume'
FIELD_ROLE                  = 'role'
FIELD_EMAIL                 = 'email'
FIELD_USERNAME              = 'username'
FIELD_FIRST_NAME            = 'first_name'
FIELD_LAST_NAME             = 'last_name'
FIELD_LOGIN_METHOD          = 'login_method'
FIELD_PASSWORD              = 'password'
FIELD_PASSWORD1             = 'password1'
FIELD_PASSWORD2             = 'password2'
FIELD_LANDING_PAGE          = 'landing_page'
FIELD_DEFAULT_LANDING_PAGE  = 'default_landing_page'
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
FIELD_USE_FOR_HEIGHT_PLOTS  = 'use_for_height_plots'
FIELD_SESSION_ID            = 'session_id'
FIELD_MODE                  = 'mode'
FIELD_MODE_LABEL            = 'mode_label'
FIELD_SCHEMA_VERSION        = 'schema_version'
FIELD_REFERENCE_VERSION     = 'reference_version'
FIELD_REFERENCE_VERSION_LABEL = 'reference_version_label'
FIELD_WORK_PACKAGE_ID       = 'work_package_id'
FIELD_WORK_PACKAGE_LABEL    = 'work_package_label'
FIELD_COMPLETED_AT          = 'completed_at'
FIELD_CLIENT_RECORD_ID      = 'client_record_id'
FIELD_HYPSO_PARAM_SET_ID    = 'hypso_param_set_id'
FIELD_CSV_TEXT              = 'csv_text'
FIELD_CHECKSUM              = 'checksum'
FIELD_RECORD_DATE           = 'record_date'
FIELD_STATE_LABEL           = 'state_label'
FIELD_RECEIVED_AT           = 'received_at'
FIELD_IMPORTED_AT           = 'imported_at'
FIELD_TARGET_TYPE           = 'target_type'
FIELD_TARGET_ID             = 'target_id'
FIELD_TARGET_LABEL          = 'target_label'
FIELD_ERROR_SUMMARY         = 'error_summary'

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
BOSCO_SPECIES_DIGESTS = BOSCO_TREE_DIGESTS

# Domain defaults — locale-independent values the client and server must agree
# on exactly (mirrored in constants.js).
DEFAULT_RADIUS_M = 12  # sample-area radius (m) when none is supplied
M2_PER_HA = 10000  # square metres per hectare
# Quantization for tree-height measurements (centimetre precision).
TREE_H_QUANTUM = Decimal('0.01')

# Ipso integration identifiers.
DATA_ID_IPSO_UPLOADS = 'ipso_uploads'
IPSO_REFERENCE_JSON = 'reference.json'
IPSO_REFERENCE_LEGACY_CONVERTED = 'legacy-converted'
IPSO_TERRENI_GEOJSON = 'terreni.geojson'
IPSO_UPLOAD_CONFIG_JS = 'upload-config.js'
IPSO_UPLOAD_FILE_JSON = 'upload.json'
IPSO_UPLOAD_FILE_SHA256 = 'upload.sha256'
IPSO_UPLOAD_FILE_CSV = 'export.csv'
IPSO_MODE_MARTELLATE = 'martellate'
IPSO_MODE_SAMPLES = 'samples'
IPSO_MODE_PAI = 'pai'
IPSO_UPLOAD_MODES = (IPSO_MODE_MARTELLATE, IPSO_MODE_SAMPLES, IPSO_MODE_PAI)
IPSO_UPLOAD_STATE_RECEIVED = 'received'
IPSO_UPLOAD_STATE_IMPORTED = 'imported'
IPSO_UPLOAD_STATE_REJECTED = 'rejected'
IPSO_UPLOAD_STATE_CONFLICT = 'conflict'
IPSO_TARGET_HARVEST_PLAN_ITEM = 'harvest_plan_item'
IPSO_TARGET_SURVEY = 'survey'
IPSO_TARGET_PAI = 'pai'
IPSO_WORK_PACKAGE_HARVEST_PREFIX = 'harvest:'
IPSO_WORK_PACKAGE_SAMPLING_SURVEY = 'sampling_survey'
IPSO_WORK_PACKAGE_SAMPLING_SURVEY_PREFIX = 'sampling_survey:'
IPSO_REF_GENERATED_AT = 'generated_at'
IPSO_REF_SPECIES = 'species'
IPSO_REF_PARCELS = 'parcels'
IPSO_REF_HYPSOMETRY = 'ipsometrica'
IPSO_REF_SAMPLING = 'sampling'
IPSO_REF_PAI = 'pai'
IPSO_REF_WORK_PACKAGES = 'work_packages'
IPSO_REF_SURVEYS = 'surveys'
IPSO_REF_SAMPLE_AREAS = 'sample_areas'
IPSO_REF_SAMPLE_AREA_MAX_NUMBERS = 'sample_area_max_numbers'
IPSO_REF_PRESERVED_TREES = 'preserved_trees'
IPSO_REFERENCE_VERSION_KEYS = (
    IPSO_REF_SPECIES, IPSO_REF_PARCELS, IPSO_REF_HYPSOMETRY,
    IPSO_REF_SAMPLING, IPSO_REF_PAI, IPSO_REF_WORK_PACKAGES,
)


# Truthy tokens for both edges: form/JSON values (the HTML checkbox 'on', real
# booleans) and CSV cells (Italian sì).  The union is unambiguous — no edge
# emits a token meaningful to the other.
_TRUTHY = ('true', '1', 'yes', 'si', 'sì', 'on')
# Falsy tokens, mirroring _TRUTHY across both edges (English + Italian).
_FALSY = ('false', 'falso', '0', 'no', 'off')

def is_truthy(value) -> bool:
    """True iff `value` is a recognised truthy token, case-insensitive and
    whitespace-trimmed: true/1/yes/si/sì/on (the bool ``True`` and int ``1``
    stringify into this set).  Safe for form/JSON values and CSV cells alike."""
    return str(value).strip().lower() in _TRUTHY


def parse_bool(value) -> bool | None:
    """Strict boolean parse: ``True`` for a recognised truthy token, ``False``
    for a recognised falsy token, ``None`` for anything else.  Unlike
    ``is_truthy`` (which maps every non-truthy value to ``False``), this lets a
    caller flag an unrecognised cell as an error instead of silently defaulting
    it — required by strict CSV import, where a malformed boolean must not be
    read as ``False``."""
    token = str(value).strip().lower()
    if token in _TRUTHY:
        return True
    if token in _FALSY:
        return False
    return None
