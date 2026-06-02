"""Tests for locale-aware parsing of user-entered numbers.

The parser maps the active locale's decimal separator to '.' and then parses;
a literal '.' is always accepted (lenient), so an Italian user may type either
"3,14" or "3.14".  In a locale whose decimal separator is '.', a comma is not a
decimal and is rejected.  Thousands separators are out of scope.
"""

from decimal import Decimal

import pytest
from django.utils import translation

from apps.base.formats import coord_float, parse_decimal


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
