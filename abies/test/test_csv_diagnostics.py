"""Import entry points use the shared field-specific CSV diagnostics."""
import pytest
from apps.base import csv_io
from config import strings as S


@pytest.mark.django_db
@pytest.mark.parametrize('kind, field, rule', [
    ('grid', S.CSV_COL_RADIUS, 'numero intero'),
    ('preserved', 'Altezza', 'numero positivo'),
    ('harvest', S.CSV_COL_QUINTALS, 'numero positivo'),
    ('parcel', S.CSV_COL_AVE_AGE, 'numero intero'),
    ('reference', S.CSV_COL_SORT_ORDER, 'numero intero'),
    ('survey', S.CSV_COL_ACTIVE, 'sì/no'),
    ('highforest_calendar', S.CSV_COL_YEAR, 'numero intero'),
    ('coppice_calendar', S.CSV_COL_YEAR, 'numero intero'),
    ('canonical_calendar', S.CSV_COL_HARVEST_PSR, 'sì/no'),
    ('legacy_height', 'a', 'numero finito'),
])
def test_all_csv_cores_explain_the_actual_field(kind, field, rule, parcels, species):
    from apps.base import csv_containers, csv_parcels, csv_reference, hypsometry
    from apps.base.models import Crew, HarvestPlan, Product, SampleGrid
    from apps.campionamenti import csv_grid, csv_preserved
    from apps.piano_di_taglio import csv_plan
    from apps.prelievi import csv_harvests

    parcel = parcels[0]
    location = {S.CSV_COL_REGION: parcel.region.name, S.CSV_COL_PARCEL: parcel.name}
    grid = SampleGrid.objects.create(name='Diagnostic grid')
    plan = HarvestPlan.objects.create(name='Diagnostic plan', year_start=2026, year_end=2030)
    errors = []

    def reader(values):
        buffer, writer = csv_io.csv_buffer(';')
        writer.writerow(values)
        writer.writerow(values.values())
        return csv_io.read(buffer.getvalue())

    if kind == 'grid':
        source = reader({**location, S.CSV_COL_SAMPLE_AREA: '1', S.CSV_COL_LAT: '38',
                         S.CSV_COL_LON: '16', S.CSV_COL_RADIUS: 'wrong', S.CSV_COL_ALT: ''})
        columns, missing = csv_grid.resolve_columns(source.fieldnames)
        assert not missing
        _, errors = csv_grid.validate_rows(source, columns, csv_grid.db_indexes(grid))
    elif kind == 'preserved':
        source = reader({**location, S.CSV_COL_NUMBER: '1', S.CSV_COL_SPECIES: species[0].common_name,
                         S.CSV_COL_LAT: '38', S.CSV_COL_LON: '16', S.CSV_COL_D_CM: '30', S.CSV_COL_H_M: '0'})
        _, errors = csv_preserved.validate_rows(source, csv_preserved.db_indexes())
    elif kind == 'harvest':
        crew = Crew.objects.create(name='Diagnostic crew')
        product = Product.objects.create(name='Diagnostic logs')
        source = reader({**location, S.CSV_COL_DATA: '2026-11-14', S.CSV_COL_CREW: crew.name,
                         S.CSV_COL_PRODUCT: product.name, S.CSV_COL_QUINTALS: '-1',
                         f'{S.CSV_COL_SPECIES_PREFIX}{species[0].common_name}': '100'})
        columns, dynamic, missing = csv_harvests.resolve_columns(source.fieldnames)
        assert not missing
        _, errors = csv_harvests.validate_rows(source, columns, dynamic, csv_harvests.db_indexes())
    elif kind == 'parcel':
        source = reader({**location, S.CSV_COL_CLASS: parcel.eclass.name,
                         S.CSV_COL_AREA_HA: '10', S.CSV_COL_AVE_AGE: 'wrong'})
        _, errors = csv_parcels.validate_rows(source, csv_parcels.db_indexes())
    elif kind == 'reference':
        source = reader({S.CSV_COL_SPECIES: species[0].common_name, S.CSV_COL_SORT_ORDER: 'wrong'})
        columns, _ = csv_reference.resolve_columns(csv_reference.SPECIES, source.fieldnames)
        _, errors = csv_reference.validate_rows(csv_reference.SPECIES, source, columns)
    elif kind == 'survey':
        source = reader({S.CSV_COL_SURVEY: 'Diagnostic survey', S.CSV_COL_GRID: grid.name, S.CSV_COL_ACTIVE: 'maybe'})
        _, errors = csv_containers.validate_surveys(source, csv_containers.survey_db_indexes())
    elif kind in ('highforest_calendar', 'coppice_calendar'):
        source = reader({**location, S.CSV_COL_YEAR: 'wrong', S.CSV_COL_HARVEST_M3: '10',
                         S.COL_INTERVENTION_AREA_HA: '2', S.CSV_COL_PERIOD_Y: '20'})
        indexes = csv_plan.db_indexes()
        if kind == 'highforest_calendar':
            data = csv_plan.read_optional(csv_text(source), csv_plan.HIGHFOREST_REQUIRED, csv_plan.HIGHFOREST_OPTIONAL)
            csv_plan.parse_highforest_rows(data, indexes.parcels, indexes.regions, errors)
        else:
            data = csv_plan.read_optional(csv_text(source), csv_plan.COPPICE_REQUIRED, csv_plan.COPPICE_OPTIONAL)
            csv_plan.parse_coppice_rows(data, indexes.parcels, errors)
    elif kind == 'canonical_calendar':
        source = reader({**location, S.CSV_COL_PLAN: plan.name, S.CSV_COL_YEAR: '2026', S.CSV_COL_HARVEST_PSR: 'maybe'})
        _, errors = csv_plan.validate_canonical_items(source, csv_plan.db_indexes(), {plan.name: plan})
    else:
        source = reader({S.CSV_COL_REGION: parcel.region.name, S.CSV_COL_SPECIES: species[0].common_name,
                         S.CSV_COL_FUNCTION: 'ln', 'a': 'wrong', 'b': '-4'})
        _, errors = hypsometry.parse_param_csv(csv_text(source))
    assert len(errors) == 1, errors
    assert errors[0].startswith('Riga 2:'), errors
    assert field in errors[0] and rule in errors[0], errors
    assert 'parsing' not in errors[0], errors


def csv_text(reader):
    buffer, writer = csv_io.csv_buffer(reader.delimiter)
    writer.writerow(reader.fieldnames)
    writer.writerows([row[name] for name in reader.fieldnames] for row in reader)
    return buffer.getvalue()


@pytest.mark.django_db
def test_legacy_mark_staging_reports_the_same_field_rule(parcels, species, tmp_path):
    from django.core.management.base import CommandError
    from apps.ipso.management.commands.stage_marks_uploads import _payload_from_csv

    parcel = parcels[0]
    source = tmp_path / 'legacy-marks.csv'
    source.write_text('Data;Compresa;Particella;Genere;D_cm;H_m\n'
        f'2026-11-14;{parcel.region.name};{parcel.name};{species[0].common_name};30;wrong\n')
    with pytest.raises(CommandError) as problem:
        _payload_from_csv(source,
            {(parcel.region.name.lower(), parcel.name.lower()): parcel},
            {species[0].common_name.lower(): species[0]})
    message = str(problem.value)
    assert 'Riga 2:' in message and 'Altezza' in message and 'numero positivo' in message
