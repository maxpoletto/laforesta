"""App-wide user-facing string constants.

Locale-specific text (UI labels, button copy, error messages, column
headers) lives in `strings_<lang>.py` and is re-exported below — change
the import line to switch language.

For internal symbolic identifiers (JSON field names, API envelope keys,
status enum values, etc.), see `config/constants.py`.
"""

from config.strings_it import *  # noqa: F401,F403
