"""Number parsing: the shared core plus the form/JSON edge.

Number conversion is organized **by edge**, not by direction:

  * This module is the **parsing core** — `to_decimal` / `to_int`, parameterized
    by decimal separator — and the **form/JSON edge** layered on it
    (`parse_decimal` / `int_or_none` / `coord_float`, driven by the *active
    locale*).
  * `apps.base.csv_io` is the **CSV edge**: the same core driven by the
    *delimiter-detected* separator, and the only place numbers are *formatted*
    in Python (`csv_io.format_decimal`).

There is deliberately no formatter here, because the form/JSON edge never
formats in Python: display is Django's `{{ value|floatformat:N }}` (server) and
the JS `Intl.NumberFormat` formatters in `format.js` (client).  Values are
stored and transmitted canonical (dot decimal); only the edges localize.  See
CLAUDE.md §"Number formatting".

Parsing is lenient: it maps the relevant decimal separator to '.', so a literal
'.' is always accepted (an Italian user may type "3,14" or "3.14"); in a
dot-decimal locale a comma is not a separator and is rejected.  Thousands
separators are out of scope.
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


def to_int(value, separator: str) -> int | None:
    """Parse an integer whose decimal mark is `separator` to int, or None.

    Goes through `to_decimal`, so a float-formatted integer ("12" or "12,0") is
    accepted while a genuinely fractional value ("12,5") is not integral and
    yields None.  This is the core behind the CSV edge's integer parser
    (`csv_io.CsvReader.integer`); the form edge uses the stricter `int_or_none`.
    """
    d = to_decimal(value, separator)
    if d is None or d != d.to_integral_value():
        return None
    return int(d)


# --- Form/JSON edge (active locale) ----------------------------------------

def parse_decimal(value) -> Decimal | None:
    """Parse a locale-formatted number to a Decimal, or None if blank/invalid."""
    return to_decimal(value, _locale_separator())


def int_or_none(value) -> int | None:
    """Parse a form/JSON integer field to int, or None for blank/``'null'``/
    non-integer.  Integers carry no decimal separator, so this is
    locale-independent (unlike `parse_decimal`).

    Stricter than the CSV edge's `to_int`: a form's integer field should hold a
    bare integer, so a decimal literal like "12.0" is rejected here, whereas
    `to_int` tolerates the spreadsheet artifact "12,0"."""
    if value in _BLANK:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
