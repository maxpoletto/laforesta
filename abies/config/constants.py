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
RECORDS  = 'records'
DATA_ID  = 'data_id'
COLUMNS  = 'columns'
ROWS     = 'rows'
STATUS   = 'status'
MESSAGE  = 'message'
HTML     = 'html'

# Side-effect record keys returned alongside the primary RECORD so the
# client can update related digests (e.g. a tree-save also updates the
# parent Sample's row in `samples`).  See CLAUDE.md §"Optimistic table
# updates" for the contract.
SAMPLE_RECORD   = 'sample_record'
SURVEY_RECORD   = 'survey_record'
SURVEY_RECORDS  = 'survey_records'
GRID_RECORD     = 'grid_record'
AREA_RECORDS    = 'area_records'
PLAN_RECORD          = 'plan_record'
ITEM_RECORD          = 'item_record'
ITEM_RECORDS         = 'item_records'
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
FIELD_VOLUME_M3             = 'volume_m3'
FIELD_MASS_Q                = 'mass_q'
FIELD_PRESERVED             = 'preserved'
FIELD_COPPICE               = 'coppice'
FIELD_FUSTAIA               = 'fustaia'
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
FIELD_TURNO_A               = 'turno_a'
FIELD_DAMAGED               = 'damaged'
FIELD_UNHEALTHY             = 'unhealthy'
FIELD_PSR                   = 'psr'
FIELD_STATE                 = 'state'
FIELD_OPEN                  = 'open'
FIELD_OPERATOR              = 'operator'
FIELD_H_MEASURED            = 'h_measured'
FIELD_ACC_M                 = 'acc_m'
FIELD_FUSTAIA_FILE          = 'fustaia_file'
FIELD_CEDUO_FILE            = 'ceduo_file'
FIELD_REGRESSION_FILE       = 'regression_file'


_TRUTHY = (True, 1, '1', 'true', 'on')

def is_truthy(value) -> bool:
    """Safe boolean parse for form data that may arrive as string, int, or bool."""
    return value in _TRUTHY
