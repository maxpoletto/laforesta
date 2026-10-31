"""Different protocol decoders feed the same sampled-measurement policy."""
from decimal import Decimal

import pytest

from apps.base import csv_io
from apps.base.numparse import int_or_none, parse_decimal
from apps.campionamenti.tree_validation import normalize_sample_tree_values
from apps.ipso.views import UploadValidationError, _bool, _int


@pytest.mark.parametrize('value', ['12', 12.0, True, False, '12.5'])
def test_native_integer_does_not_inherit_form_or_spreadsheet_coercion(value):
    with pytest.raises(UploadValidationError):
        _int({'value': value}, 'value')
    assert _int({'value': 12}, 'value') == 12


@pytest.mark.parametrize('value', [0, 1, 'true', 'false', '1'])
def test_native_boolean_requires_boolean(value):
    with pytest.raises(UploadValidationError):
        _bool({'value': value}, 'value')
    assert _bool({'value': True}, 'value') is True


def test_form_and_csv_decode_distinct_representations_to_one_domain_value():
    csv = csv_io.read('number;height\n12,0;20,125\n')
    row = csv[0]
    assert int_or_none('12.0') is None and int_or_none('12') == 12
    assert csv.integer(row['number']) == 12
    decoded = [
        (int_or_none('12'), parse_decimal('20,125')),
        (csv.integer(row['number']), csv.decimal(row['height'])),
        (_int({'number': 12}, 'number'), Decimal('20.125')),
    ]
    normalized = [normalize_sample_tree_values(number=n, d_cm=30, h_m=h) for n, h in decoded]
    assert normalized[0] == normalized[1] == normalized[2]
    assert normalized[0].h_m == Decimal('20.13')
