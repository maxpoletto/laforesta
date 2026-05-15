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

# JSON API `status` field values.
STATUS_CONFLICT         = 'conflict'
STATUS_VALIDATION_ERROR = 'validation_error'
STATUS_RATE_LIMITED     = 'rate_limited'
STATUS_NOT_FOUND        = 'not_found'
