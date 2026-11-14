"""Field-specific diagnostics shared by every CSV import path."""
from datetime import date
from decimal import Decimal

import pytest

from apps.base import csv_io
from config import strings as S


def cells(values):
    from apps.base.csv_validation import CsvRow

    values = {**values, '__delimiter_fixture': ''}
    buffer, writer = csv_io.csv_buffer(';')
    writer.writerow(values)
    writer.writerow(values.values())
    reader = csv_io.read(buffer.getvalue())
    return CsvRow(reader, reader[0], 2)


def test_csv_row_collects_specific_errors_once_per_source_row():
    row = cells({S.CSV_COL_D_CM: '-1', S.CSV_COL_H_M: '', S.CSV_COL_DATA: 'wrong'})
    assert row.integer(S.CSV_COL_D_CM, positive=True) is None
    assert row.decimal(S.CSV_COL_H_M, positive=True) is None
    assert row.date(S.CSV_COL_DATA) is None
    errors = []
    assert not row.validate(errors)
    assert len(errors) == 1 and errors[0].startswith('Riga 2:')
    assert all(text in errors[0] for text in ('Diametro', 'intero positivo', 'Altezza', 'mancante', 'AAAA-MM-GG'))
    assert 'parsing' not in errors[0]


@pytest.mark.parametrize('raw, expected', [('', None), ('12,0', 12), ('12', 12)])
def test_csv_row_preserves_optional_integer_and_spreadsheet_input(raw, expected):
    row = cells({'number': raw})
    assert row.integer('number', required=False) == expected
    errors = []
    assert row.validate(errors) and errors == []


@pytest.mark.parametrize('raw', ['1,5', 'bad', '1e100', 'NaN'])
def test_csv_row_rejects_invalid_or_unstorable_integers(raw):
    row = cells({'number': raw})
    assert row.integer('number') is None
    errors = []
    assert not row.validate(errors)
    assert 'number' in errors[0]


def test_csv_row_validates_decimal_storage_without_changing_rounding():
    row = cells({'height': '12,345', 'too_large': '1000'})
    assert row.decimal('height', storage=('base.TreeSample', 'h_m')) == Decimal('12.345')
    assert row.decimal('too_large', storage=('base.TreeSample', 'h_m')) is None
    errors = []
    assert not row.validate(errors)
    assert 'too_large' in errors[0] and 'memorizzato' in errors[0]


def test_csv_row_distinguishes_boolean_date_and_coordinate_errors():
    row = cells({'active': 'maybe', 'date': '2026-02-30', 'lat': '91', 'lon': '16'})
    assert row.boolean('active') is None
    assert row.date('date') is None
    assert row.coordinates('lat', 'lon') == (None, 16.0)
    errors = []
    assert not row.validate(errors)
    assert all(text in errors[0] for text in ('active', 'sì/no', 'AAAA-MM-GG', 'lat', '-90', '90'))


def test_csv_row_preserves_date_fallback_and_italian_mark_dates():
    row = cells({'blank': '', 'italian': '14/11/2026'})
    assert row.date('blank', default=date(2026, 11, 14)) == date(2026, 11, 14)
    assert row.date('italian', formats=('%d/%m/%Y',)) == date(2026, 11, 14)
    assert row.errors == []


def test_csv_row_names_the_missing_coordinate_without_a_grouped_parse_error():
    row = cells({'lat': '38,5', 'lon': ''})
    assert row.coordinates('lat', 'lon') == (38.5, None)
    errors = []
    assert not row.validate(errors)
    assert 'lon' in errors[0] and 'insieme' in errors[0]


def test_csv_row_resolves_alias_labels_in_diagnostics():
    from apps.base.csv_validation import CsvRow

    reader = csv_io.read('Anno previsto;D_cm\nwrong;30\n')
    row = CsvRow(reader, {'year': 'wrong'}, 2, labels={'year': 'Anno previsto'})
    row.integer('year')
    errors = []
    assert not row.validate(errors)
    assert 'Anno previsto' in errors[0] and 'year' not in errors[0]


@pytest.mark.parametrize('field', ['h_m', 'pressler_coeff'])
def test_positive_csv_measurements_cannot_round_to_zero_in_storage(field):
    row = cells({'measurement': '0,004'})
    assert row.decimal('measurement', positive=True, storage=('base.TreeSample', field)) is None
    errors = []
    assert not row.validate(errors)
    assert 'measurement' in errors[0] and 'positivo' in errors[0]


def test_csv_errors_keep_the_rejected_cell_value_for_correction():
    row = cells({'planned volume': 'nope'})
    row.decimal('planned volume')
    errors = []
    assert not row.validate(errors)
    assert 'planned volume' in errors[0] and 'nope' in errors[0]
    assert 'numero finito' in errors[0]
