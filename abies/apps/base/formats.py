"""Locale-aware parsing of user-entered numbers.

Decimal form inputs are rendered in the active locale (Italian "1,5") and
parsed back here before storage.  Parsing maps the active locale's decimal
separator to '.', so a literal '.' is always accepted (an Italian user may
type either "3,14" or "3.14"); in a dot-decimal locale a comma is not a
decimal separator and is rejected.  Thousands separators are out of scope.

The inverse — display — is Django's `{{ value|floatformat:N }}` (server) and
the JS `Intl.NumberFormat` formatters in `format.js` (client).  Values are
stored and transmitted canonical (dot decimal); see CLAUDE.md §"Number
formatting".
"""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from django.utils.formats import get_format

_BLANK = (None, '', 'null')

# Reject finite-but-absurd magnitudes like "1E999999": they pass `is_finite()`
# yet a later `quantize()` would try to materialize ~10**6 digits.  Real
# quantities sit well within ±10**30 (and above ~10**-30).
_MAX_ABS_EXPONENT = 30


def _locale_separator() -> str:
    return str(get_format('DECIMAL_SEPARATOR'))


def _canonical(value, separator: str) -> str | None:
    """Map `separator` to '.'; None for blanks."""
    if value in _BLANK:
        return None
    s = str(value).strip()
    if not s:
        return None
    return s.replace(separator, '.') if separator != '.' else s


def to_decimal(value, separator: str) -> Decimal | None:
    """Parse a number whose decimal mark is `separator` to a Decimal, or None.

    Returns None for blank/invalid and rejects the non-finite values Decimal
    otherwise accepts ("NaN", "Infinity", …): NaN slips past `<= 0` guards and
    a huge exponent makes a later `quantize()` ruinously expensive.

    Separator-parameterized so the locale-driven form parser (`parse_decimal`)
    and the delimiter-driven CSV reader (`apps.base.csv_io`) share one core.
    """
    s = _canonical(value, separator)
    if s is None:
        return None
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite() or abs(d.adjusted()) > _MAX_ABS_EXPONENT:
        return None
    return d


def parse_decimal(value) -> Decimal | None:
    """Parse a locale-formatted number to a Decimal, or None if blank/invalid."""
    return to_decimal(value, _locale_separator())


_COORD_QUANTUM = Decimal('0.00001')  # 5 dp ≈ 1 m; coordinates persist to float


def coord_float(value: Decimal | None) -> float | None:
    """Quantize a parsed coordinate Decimal to 5 dp and convert to `float` for
    the model's FloatField boundary (see CLAUDE.md §"Number formatting").
    `None` passes through.  Coordinates are the one quantity stored as float;
    everything else stays Decimal.
    """
    if value is None:
        return None
    return float(value.quantize(_COORD_QUANTUM, rounding=ROUND_HALF_UP))
