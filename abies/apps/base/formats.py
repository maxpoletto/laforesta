"""Locale-aware parsing of user-entered numbers.

Decimal form inputs are rendered in the active locale (Italian "1,5") and
parsed back here before `float()`/`Decimal()` — the inverse of the
`floatformat` template rendering and the JS `Intl.NumberFormat` display.
See CLAUDE.md §"Number formatting".
"""

from django.utils.formats import sanitize_separators


def normalize_decimal(value) -> str:
    """Normalize a locale-formatted number string to canonical form
    ("1.234,5" → "1234.5" in the Italian locale), ready for `float()`/`Decimal()`.

    Apply only to form-submitted strings — never to CSV rows or JSON payloads,
    which are already canonical (in the it locale `sanitize_separators` reads
    "." as a thousands separator, so "38.6" would become "386").
    """
    return sanitize_separators(str(value).strip())
