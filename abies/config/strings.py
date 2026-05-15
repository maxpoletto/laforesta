"""App-wide string constants.

Locale-specific strings live in `strings_<lang>.py` and are re-exported
below -- change the import line to switch language.  Locale-INDEPENDENT
identifiers (wire markers, internal column names, etc.) are defined
directly in this file so they aren't duplicated across language files.
"""

from config.strings_it import *  # noqa: F401,F403

# Internal digest column identifiers -- wire format, never displayed.
COL_ROW_ID  = 'row_id'
COL_VERSION = 'version'
