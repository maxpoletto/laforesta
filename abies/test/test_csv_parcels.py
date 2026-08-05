"""Unit tests for the parcel CSV import core (apps/base/csv_parcels)."""

from decimal import Decimal

import pytest

from apps.base import csv_io, csv_parcels
from apps.base.models import Parcel
from config import strings as S


def _reader(text):
    return csv_io.read(text)


HEADER = (f'{S.CSV_COL_REGION},{S.CSV_COL_CLASS},{S.CSV_COL_PARCEL},'
          f'{S.CSV_COL_AREA_HA},{S.CSV_COL_AVE_AGE}')


@pytest.mark.django_db
def test_validate_and_apply_happy_path(regions, eclasses):
    region = regions[0]            # Capistrano
    eclass = eclasses[0]           # A
    reader = _reader(f'{HEADER}\n{region.name},{eclass.name},7,12.5,40\n')
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert errors == []
    assert csv_parcels.apply(parsed) == 1
    p = Parcel.objects.get(name='7', region=region)
    assert p.eclass == eclass
    assert p.area_ha == Decimal('12.50')
    assert p.ave_age == 40


@pytest.mark.django_db
def test_unknown_region_flagged(regions, eclasses):
    reader = _reader(f'{HEADER}\nNowhere,A,7,12.5,40\n')
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_unknown_eclass_flagged(regions, eclasses):
    reader = _reader(f'{HEADER}\n{regions[0].name},Z,7,12.5,40\n')
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_blank_name_flagged(regions, eclasses):
    reader = _reader(f'{HEADER}\n{regions[0].name},A,,12.5,40\n')
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_bad_area_flagged(regions, eclasses):
    reader = _reader(f'{HEADER}\n{regions[0].name},A,7,abc,40\n')
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_duplicate_key_within_file_flagged(regions, eclasses):
    """Same (name, region) twice → second flagged; same name, other region OK."""
    r0, r1 = regions[0].name, regions[1].name
    reader = _reader(
        f'{HEADER}\n'
        f'{r0},A,7,12.5,40\n'
        f'{r0},A,7,9.0,30\n'    # duplicate (name, region)
        f'{r1},A,7,8.0,20\n'    # same name, different region → OK
    )
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert len(parsed) == 2
    assert len(errors) == 1


@pytest.mark.django_db
def test_optional_columns_absent_default_null(regions, eclasses):
    """ave_age absent entirely → stored NULL (nullable field)."""
    reader = _reader(
        f'{S.CSV_COL_REGION},{S.CSV_COL_CLASS},{S.CSV_COL_PARCEL},{S.CSV_COL_AREA_HA}\n'
        f'{regions[0].name},A,7,12.5\n'
    )
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert errors == []
    csv_parcels.apply(parsed)
    assert Parcel.objects.get(name='7', region=regions[0]).ave_age is None


@pytest.mark.django_db
def test_invalid_optional_numeric_flagged(regions, eclasses):
    reader = _reader(
        f'{S.CSV_COL_REGION},{S.CSV_COL_CLASS},{S.CSV_COL_PARCEL},'
        f'{S.CSV_COL_AREA_HA},{S.CSV_COL_AVE_AGE}\n'
        f'{regions[0].name},A,7,12.5,abc\n'      # ave_age non-blank, invalid
    )
    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert parsed == []
    assert len(errors) == 1


@pytest.mark.django_db
def test_no_synthetic_parcels(regions, eclasses):
    """apply creates exactly the rows given — no synthetic 'X' parcels."""
    reader = _reader(
        f'{HEADER}\n'
        f'{regions[0].name},A,7,12.5,40\n'
        f'{regions[0].name},A,8,5.0,20\n'
    )
    parsed, _ = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    csv_parcels.apply(parsed)
    assert Parcel.objects.count() == 2
    assert not Parcel.objects.filter(name='X').exists()


@pytest.mark.django_db
def test_apply_idempotent(regions, eclasses):
    reader = _reader(f'{HEADER}\n{regions[0].name},A,7,12.5,40\n')
    parsed, _ = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert csv_parcels.apply(parsed) == 1
    reader2 = _reader(f'{HEADER}\n{regions[0].name},A,7,12.5,40\n')
    parsed2, _ = csv_parcels.validate_rows(reader2, csv_parcels.db_indexes())
    assert csv_parcels.apply(parsed2) == 0    # already present
    assert Parcel.objects.filter(name='7', region=regions[0]).count() == 1


@pytest.mark.django_db
def test_apply_does_not_update_existing(regions, eclasses):
    """Create-only: a re-import with changed fields leaves the existing row
    unchanged (bootstrap loads into an empty instance)."""
    reader = _reader(f'{HEADER}\n{regions[0].name},A,7,12.5,40\n')
    parsed, _ = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    csv_parcels.apply(parsed)
    reader2 = _reader(f'{HEADER}\n{regions[0].name},A,7,99.0,99\n')  # changed
    parsed2, _ = csv_parcels.validate_rows(reader2, csv_parcels.db_indexes())
    assert csv_parcels.apply(parsed2) == 0
    p = Parcel.objects.get(name='7', region=regions[0])
    assert p.area_ha == Decimal('12.50')   # original retained, not 99.00
    assert p.ave_age == 40


@pytest.mark.django_db
def test_coppice_metadata_imported(regions, eclasses):
    reader = _reader(
        f'{HEADER},{S.CSV_COL_CUTTING_PLAN},{S.CSV_COL_HARVEST_MECHANISM},'
        f'{S.CSV_COL_INTERVAL},{S.CSV_COL_STANDARDS}\n'
        f'{regions[0].name},F,7,12.5,40,Taglio ceduo,Verricello,18,75\n'
    )

    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())
    assert errors == []
    csv_parcels.apply(parsed)
    p = Parcel.objects.get(name='7', region=regions[0])
    assert p.cutting_plan == 'Taglio ceduo'
    assert p.harvest_mechanism == 'Verricello'
    assert p.intervention_interval == 18
    assert p.standards_per_ha == 75


@pytest.mark.django_db
def test_coppice_metadata_required_for_coppice(regions, eclasses):
    reader = _reader(f'{HEADER}\n{regions[0].name},F,7,12.5,40\n')

    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())

    assert parsed == []
    assert S.CSV_COL_INTERVAL in errors[0]
    assert S.CSV_COL_STANDARDS in errors[1]


@pytest.mark.django_db
def test_coppice_metadata_rejected_for_highforest(regions, eclasses):
    reader = _reader(
        f'{HEADER},{S.CSV_COL_INTERVAL},{S.CSV_COL_STANDARDS}\n'
        f'{regions[0].name},A,7,12.5,40,18,75\n'
    )

    parsed, errors = csv_parcels.validate_rows(reader, csv_parcels.db_indexes())

    assert parsed == []
    assert S.CSV_COL_INTERVAL in errors[0]
    assert S.CSV_COL_STANDARDS in errors[1]


@pytest.mark.django_db
def test_export_import_export_import_round_trip(regions, eclasses):
    Parcel.objects.create(
        name='7', region=regions[0], eclass=eclasses[0],
        area_ha=Decimal('12.75'), ave_age=41,
        location_name='Versante; nord', altitude_min_m=410,
        altitude_max_m=780, aspect='NE', grade_pct=35,
        desc_geo='Roccia, affiorante', desc_veg='Abetina\na gruppi',
        cutting_plan='Diradamento selettivo', harvest_mechanism='Verricello',
    )
    Parcel.objects.create(
        name='C1', region=regions[1], eclass=eclasses[2],
        area_ha=Decimal('4.25'), ave_age=None,
        intervention_interval=18, standards_per_ha=75,
        cutting_plan='Taglio ceduo', harvest_mechanism='Trattore',
    )

    def parcels():
        return (Parcel.objects.select_related('region', 'eclass')
                .order_by('region__name', 'name'))

    def snapshot():
        return list(parcels().values_list(
            'region__name', 'name', 'eclass__name', 'area_ha', 'ave_age',
            'location_name', 'altitude_min_m', 'altitude_max_m', 'aspect',
            'grade_pct', 'desc_geo', 'desc_veg', 'cutting_plan',
            'harvest_mechanism', 'intervention_interval', 'standards_per_ha',
        ))

    def import_text(content):
        reader = csv_io.read(content, csv_parcels.PARCEL_CSV_REQUIRED)
        parsed, errors = csv_parcels.validate_rows(
            reader, csv_parcels.db_indexes(),
        )
        assert errors == []
        assert csv_parcels.apply(parsed) == 2

    expected = snapshot()
    first_export = csv_parcels.render_csv(parcels())
    assert csv_io.read(first_export).fieldnames == (
        csv_parcels.PARCEL_EXPORT_COLUMNS
    )

    Parcel.objects.all().delete()
    import_text(first_export)
    assert snapshot() == expected

    second_export = csv_parcels.render_csv(parcels())
    assert second_export == first_export
    Parcel.objects.all().delete()
    import_text(second_export)
    assert snapshot() == expected
