"""The CSV edge: locale-tolerant CSV reading and writing, shared by every
importer and exporter.

This is the **only** CSV codec.  It is the CSV-side counterpart to
`apps.base.numparse` (the parsing core + the form/JSON edge): cell parsing reuses
that core's `to_decimal` / `to_int`, driven by the delimiter-detected separator
rather than the active locale.  All CSV cell parsing goes through a `CsvReader`;
CSV export goes through `csv_buffer()` and `format_decimal` here — nothing
reimplements delimiter, decimal handling, or spreadsheet-formula hardening.

CSV is I/O, not canonical transmission (see CLAUDE.md §"Number formatting").
Input is lenient: the field delimiter is auto-detected and fixes the decimal
separator as a *pair* — ``;`` ⇒ ``,``
decimal, ``,`` ⇒ ``.`` decimal.  Numbers are parsed against the *detected*
separator, independent of the install locale, so a dot-decimal file imports on
an Italian install and a comma-decimal file imports on a US install.

``read()`` raises ``CsvError`` (its message is a user-facing ``S.ERR_CSV_*``
string) for an undecodable, empty, or column-incomplete file.  Per-cell parsing
(``decimal()`` / ``integer()``) returns ``None`` for blank/invalid so the
caller can report a per-row error.
"""

import base64
import binascii
import csv
import io
import re
import zipfile
from collections.abc import Iterable
from decimal import Decimal

from django.http import HttpResponse
from django.utils.formats import get_format

from apps.base.numparse import to_decimal, to_int
from config import strings as S
from config.constants import parse_bool

# Field delimiter ⇒ decimal separator.  ';' is the Italian /
# Excel pairing (comma decimal); ',' is the canonical / US pairing (dot decimal).
_DECIMAL_FOR_DELIMITER = {';': ',', ',': '.'}
_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r', '\n')
_NUMERIC_LITERAL_RE = re.compile(r'^[+-]?(?:\d+|\d*[.,]\d+)(?:[eE][+-]?\d+)?$')


class CsvError(Exception):
    """Malformed CSV; ``str(exc)`` is a user-facing message."""


class CsvReader:
    """Header-keyed rows plus the decimal separator implied by the delimiter."""

    def __init__(self, rows: list[dict], delimiter: str, fieldnames=None):
        self.rows = rows
        self.delimiter = delimiter
        self.fieldnames = fieldnames
        self.decimal_separator = _DECIMAL_FOR_DELIMITER[delimiter]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def decimal(self, value):
        """Cell value → Decimal (using the detected separator), or None."""
        return to_decimal(value, self.decimal_separator)

    def integer(self, value):
        """Cell value → int (via the detected separator), or None if blank,
        invalid, or not integral.  Tolerates a float-formatted integer ("12" or
        "12,0"); a fractional value ("12,5") yields None.  See `numparse.to_int`.
        """
        return to_int(value, self.decimal_separator)

    def opt_int(self, value):
        """Optional integer cell, distinguishing blank from invalid: blank →
        ``(None, True)``; a non-blank value that is not an integer →
        ``(None, False)``; otherwise ``(int, True)``.  Unlike ``integer`` (which
        returns ``None`` for both), this lets a caller accept a blank optional
        cell but flag a present-but-invalid one."""
        raw = (value or '').strip()
        if not raw:
            return None, True
        parsed = self.integer(raw)
        return parsed, parsed is not None

    def opt_decimal(self, value):
        """Optional decimal cell; blank → ``(None, True)``, present-but-invalid →
        ``(None, False)``, else ``(Decimal, True)``.  See ``opt_int``."""
        raw = (value or '').strip()
        if not raw:
            return None, True
        parsed = self.decimal(raw)
        return parsed, parsed is not None

    def opt_bool(self, value):
        """Optional boolean cell; blank → ``(None, True)``, an unrecognised
        non-blank token → ``(None, False)``, else ``(bool, True)``.  Same
        blank-vs-invalid contract as ``opt_int``; a required boolean treats a
        ``None`` value (blank) as an error, an optional one defaults it."""
        raw = (value or '').strip()
        if not raw:
            return None, True
        parsed = parse_bool(raw)
        return parsed, parsed is not None


def json_file_bytes(body: dict, field: str) -> bytes | None:
    """Decode a base64 CSV field from a JSON write payload.

    Returns ``None`` when the field is absent/blank so callers can use their
    existing "file required" validation. Raises ``CsvError`` for malformed
    base64; ``read()`` still owns UTF-8 validation after bytes are decoded.
    """
    encoded = body.get(field)
    if encoded in (None, ''):
        return None
    if not isinstance(encoded, str):
        raise CsvError(S.ERR_CSV_NOT_UTF8)
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CsvError(S.ERR_CSV_NOT_UTF8) from exc


def read(source, required_cols=None) -> CsvReader:
    """Parse `source` (a decoded str or an uploaded file) into a CsvReader.

    Raises CsvError if the file is undecodable, empty, or missing a required
    column.
    """
    text = _decode(source)
    delimiter = ';' if ';' in text.split('\n', 1)[0] else ','
    dict_reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if dict_reader.fieldnames is None:
        raise CsvError(S.ERR_CSV_EMPTY)
    try:
        if required_cols:
            missing = [c for c in required_cols if c not in dict_reader.fieldnames]
            if missing:
                raise CsvError(S.ERR_CSV_MISSING_COLS.format(', '.join(missing)))
        rows = list(dict_reader)
    except csv.Error as exc:
        raise CsvError(S.ERR_CSV_INVALID) from exc
    return CsvReader(rows, delimiter, dict_reader.fieldnames)


def _decode(source) -> str:
    if isinstance(source, str):
        return source
    raw = source.read() if hasattr(source, 'read') else source
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise CsvError(S.ERR_CSV_NOT_UTF8) from exc


def export_format() -> tuple[str, str]:
    """(field delimiter, decimal separator) for CSV *output* in the active
    locale — the inverse of `read`'s detection.  A comma
    decimal pairs with a ';' delimiter; a dot decimal with ','.
    """
    decimal_sep = str(get_format('DECIMAL_SEPARATOR'))
    delimiter = ';' if decimal_sep == ',' else ','
    return delimiter, decimal_sep


class SafeCSVWriter:
    """csv.writer wrapper that neutralizes spreadsheet formulas in text cells."""

    def __init__(self, writer):
        self._writer = writer

    def writerow(self, row):
        return self._writer.writerow([harden_formula_cell(value) for value in row])

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def harden_formula_cell(value):
    """Return a CSV-safe cell value for spreadsheet applications.

    Text beginning with formula sigils is prefixed with an apostrophe. Numeric
    literals such as ``-4`` and ``+3,14`` are left unchanged so exported numeric
    data remains round-trippable.
    """
    if value is None:
        return ''
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return value
    s = str(value)
    if s.startswith(_FORMULA_PREFIXES) and not _NUMERIC_LITERAL_RE.match(s.strip()):
        return f"'{s}"
    return s


def csv_buffer(delimiter: str):
    """Return a text buffer and hardened CSV writer using the delimiter."""
    buf = io.StringIO()
    return buf, SafeCSVWriter(csv.writer(buf, delimiter=delimiter))


def zip_response(
        files: Iterable[tuple[str, str | bytes]],
        filename: str,
) -> HttpResponse:
    """Build a no-store ZIP download response from rendered file contents."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for archive_name, content in files:
            zf.writestr(archive_name, content)

    response = HttpResponse(zip_buf.getvalue(), content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    response['Cache-Control'] = 'no-store'
    return response


def zip_csv_response(
        files: Iterable[tuple[str, str | bytes]],
        filename: str,
) -> HttpResponse:
    """Build a no-store ZIP download response from already-rendered CSV files."""
    return zip_response(files, filename)


def format_decimal(value, decimal_sep: str) -> str:
    """Render a number for a CSV cell: trailing zeros stripped, '.' replaced by
    `decimal_sep`; '' for None.  (Dates stay ISO — formatted by the caller.)
    """
    if value is None:
        return ''
    if isinstance(value, Decimal):
        s = format(value, 'f')
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        s = s or '0'
    else:
        s = str(value)
    return s.replace('.', decimal_sep) if decimal_sep != '.' and '.' in s else s
