"""Tests for `apps.base.numparse` — the number-parsing core and form/JSON edge.

`parse_decimal` maps the active locale's decimal separator to '.' and then
parses; a literal '.' is always accepted (lenient), so an Italian user may type
either "3,14" or "3.14".  In a locale whose decimal separator is '.', a comma is
not a decimal and is rejected.  Thousands separators are out of scope.

`to_int` is the separator-parameterized integer core (tolerates a
float-formatted integer like "12,0"); `int_or_none` is the stricter form/JSON
integer parser (rejects "12.0").
"""

from decimal import Decimal

import pytest
from django.utils import translation

from apps.base.numparse import (
    coord_float, float_or_none, int_or_none, parse_decimal, to_int,
)


class TestParseDecimal:
    @pytest.mark.parametrize('raw, expected', [
        ('3,14', Decimal('3.14')),      # native Italian comma
        ('3.14', Decimal('3.14')),      # lenient: dot accepted too
        ('9,50', Decimal('9.50')),
        ('0,65', Decimal('0.65')),
        ('-16,98765', Decimal('-16.98765')),
        ('5', Decimal('5')),
        ('  3,14  ', Decimal('3.14')),  # surrounding whitespace
    ])
    def test_it_locale(self, raw, expected):
        with translation.override('it'):
            assert parse_decimal(raw) == expected

    @pytest.mark.parametrize('raw, expected', [
        ('3.14', Decimal('3.14')),
        ('-16.98765', Decimal('-16.98765')),
        ('5', Decimal('5')),
    ])
    def test_en_locale_accepts_dot(self, raw, expected):
        with translation.override('en'):
            assert parse_decimal(raw) == expected

    def test_en_locale_rejects_comma(self):
        """In a dot-decimal locale a comma is not a decimal separator."""
        with translation.override('en'):
            assert parse_decimal('3,14') is None

    @pytest.mark.parametrize('raw', [
        '', '   ', None, 'null', 'abc', '1.2.3',
        '1.234,5',                       # thousands grouping is out of scope
        'NaN', 'Infinity', '-Infinity', '1E999999',   # non-finite rejected
    ])
    def test_blank_or_invalid_is_none(self, raw):
        with translation.override('it'):
            assert parse_decimal(raw) is None


class TestCoordFloat:
    """coord_float quantizes a parsed Decimal to 5 dp and returns a float —
    the FloatField boundary for coordinates."""

    @pytest.mark.parametrize('value, expected', [
        (Decimal('38.123456'), 38.12346),   # rounds to 5 dp
        (Decimal('-16.98765'), -16.98765),  # already 5 dp
        (Decimal('16'), 16.0),
    ])
    def test_quantizes_to_5dp(self, value, expected):
        assert coord_float(value) == pytest.approx(expected)

    def test_none_passes_through(self):
        assert coord_float(None) is None

    def test_parse_then_coord_locale(self):
        with translation.override('it'):
            assert coord_float(parse_decimal('38,123456')) == pytest.approx(38.12346)


class TestToInt:
    """`to_int` is the separator-parameterized integer core (the CSV edge's
    `CsvReader.integer` delegates to it).  It tolerates a float-formatted
    integer ("12,0") but rejects a genuinely fractional value ("12,5")."""

    @pytest.mark.parametrize('raw, sep, expected', [
        ('12', '.', 12),
        ('12', ',', 12),
        ('12.0', '.', 12),      # float-formatted integer tolerated (dot sep)
        ('12,0', ',', 12),      # float-formatted integer tolerated (comma sep)
        ('-7', '.', -7),
        ('0', '.', 0),
        ('  12  ', '.', 12),    # surrounding whitespace
    ])
    def test_integral(self, raw, sep, expected):
        assert to_int(raw, sep) == expected

    @pytest.mark.parametrize('raw, sep', [
        ('12.5', '.'),          # fractional is not integral
        ('12,5', ','),
        ('12,0', '.'),          # comma is not a decimal under '.' sep → invalid
        ('', '.'), ('   ', '.'), (None, '.'), ('null', '.'), ('abc', '.'),
        ('NaN', '.'),
    ])
    def test_blank_fractional_or_invalid_is_none(self, raw, sep):
        assert to_int(raw, sep) is None


class TestIntOrNone:
    """`int_or_none` is the form/JSON integer parser: locale-independent and
    stricter than `to_int` — a decimal literal like "12.0" is rejected."""

    @pytest.mark.parametrize('raw, expected', [
        ('12', 12), ('-7', -7), ('0', 0), (12, 12),
    ])
    def test_parses_bare_integer(self, raw, expected):
        assert int_or_none(raw) == expected

    @pytest.mark.parametrize('raw', ['', None, 'null', 'abc', '12.0', '3,14', '12.5'])
    def test_blank_or_non_integer_is_none(self, raw):
        assert int_or_none(raw) is None

    def test_differs_from_to_int_on_float_formatted_integer(self):
        """The deliberate edge asymmetry: CSV tolerates "12.0", a form does not."""
        assert to_int('12.0', '.') == 12
        assert int_or_none('12.0') is None


def test_float_or_none_preserves_null_and_converts_decimal():
    assert float_or_none(None) is None
    assert float_or_none(Decimal('18.50')) == 18.5
