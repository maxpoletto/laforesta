"""App-wide string constants.

Locale-specific strings live in `strings_<lang>.py` and are re-exported
below -- change the import line to switch language.  Locale-INDEPENDENT
identifiers (wire markers, internal column names, etc.) are defined
directly in this file so they aren't duplicated across language files.
"""

from config.strings_it import *  # noqa: F401,F403

# Internal identifiers -- wire format, never displayed.
# Used both as digest-column keys and as JSON API field names.
ROW_ID  = 'row_id'
VERSION = 'version'

# JSON API field names shared between Python views and JS clients.
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
