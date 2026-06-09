"""Tests for the shared locale-tolerant CSV reader (`apps.base.csv_io`).

The delimiter is auto-detected and fixes the decimal separator as a pair:
';' ⇒ ',' decimal (Italian/Excel), ',' ⇒ '.' decimal (canonical/US).  Numbers
parse against the *detected* separator, independent of the install locale.
"""

import io
import zipfile

from decimal import Decimal

import pytest

from django.utils import translation

from apps.base import csv_io
from apps.base.csv_io import CsvError

COMMA_CSV = 'a,b\n1.5,2\n3.25,4\n'      # ',' delimiter ⇒ '.' decimal
SEMI_CSV = 'a;b\n1,5;2\n3,25;4\n'       # ';' delimiter ⇒ ',' decimal


def test_detects_comma_delimiter_dot_decimal():
    r = csv_io.read(COMMA_CSV)
    assert r.delimiter == ',' and r.decimal_separator == '.'
    assert r.rows[0]['a'] == '1.5'
    assert r.decimal(r.rows[0]['a']) == Decimal('1.5')


def test_detects_semicolon_delimiter_comma_decimal():
    r = csv_io.read(SEMI_CSV)
    assert r.delimiter == ';' and r.decimal_separator == ','
    assert r.decimal(r.rows[0]['a']) == Decimal('1.5')
    assert r.decimal(r.rows[1]['a']) == Decimal('3.25')


def test_dot_decimal_rejects_comma_value():
    # In ',' (dot-decimal) mode, a comma is not a decimal separator.
    assert csv_io.read(COMMA_CSV).decimal('3,14') is None


def test_comma_decimal_accepts_dot_too():
    # ';' mode is lenient: a dot value is unambiguous and still parses.
    assert csv_io.read(SEMI_CSV).decimal('3.14') == Decimal('3.14')


def test_thousands_grouping_rejected():
    assert csv_io.read(SEMI_CSV).decimal('1.234,5') is None


@pytest.mark.parametrize('bad', ['', '   ', None, 'abc', 'NaN', 'Infinity'])
def test_blank_and_garbage_decimal(bad):
    assert csv_io.read(COMMA_CSV).decimal(bad) is None


def test_integer_parsing():
    r = csv_io.read(SEMI_CSV)
    assert r.integer('12') == 12
    assert r.integer('12,0') == 12      # float-formatted integer accepted
    assert r.integer('12,5') is None    # genuinely fractional → not an int
    assert r.integer('') is None


def test_bom_stripped_from_bytes():
    r = csv_io.read('﻿a,b\n1,2\n'.encode('utf-8'))
    assert r.rows[0]['a'] == '1' and r.rows[0]['b'] == '2'


def test_reads_uploaded_file_object():
    upload = io.BytesIO('a,b\n9,9\n'.encode('utf-8'))
    r = csv_io.read(upload, required_cols=['a', 'b'])
    assert len(r) == 1 and r[0]['a'] == '9'


def test_missing_required_cols_raises():
    with pytest.raises(CsvError):
        csv_io.read(COMMA_CSV, required_cols=['a', 'zzz'])


def test_empty_raises():
    with pytest.raises(CsvError):
        csv_io.read('')


def test_non_utf8_raises():
    with pytest.raises(CsvError):
        csv_io.read(b'\xff\xfe bad bytes')


def test_format_decimal_strips_zeros_and_localizes():
    assert csv_io.format_decimal(Decimal('9.50'), ',') == '9,5'
    assert csv_io.format_decimal(Decimal('12'), ',') == '12'
    assert csv_io.format_decimal(Decimal('3.14'), '.') == '3.14'
    assert csv_io.format_decimal(38.5, ',') == '38,5'
    assert csv_io.format_decimal(None, ',') == ''


def test_export_format_pairs_with_locale():
    with translation.override('it'):
        assert csv_io.export_format() == (';', ',')
    with translation.override('en'):
        assert csv_io.export_format() == (',', '.')


def test_csv_buffer_uses_export_delimiter():
    buf, writer = csv_io.csv_buffer(';')
    writer.writerow(['a', 'b'])
    assert buf.getvalue() == 'a;b\r\n'


def test_zip_csv_response_builds_no_store_download():
    response = csv_io.zip_csv_response(
        [('a.csv', 'one\n'), ('b.csv', b'two\n')],
        'bundle.zip',
    )
    assert response.status_code == 200
    assert response['Content-Type'] == 'application/zip'
    assert response['Content-Disposition'] == 'attachment; filename="bundle.zip"'
    assert response['Cache-Control'] == 'no-store'

    zf = zipfile.ZipFile(io.BytesIO(response.content))
    assert set(zf.namelist()) == {'a.csv', 'b.csv'}
    assert zf.read('a.csv').decode() == 'one\n'
    assert zf.read('b.csv') == b'two\n'
