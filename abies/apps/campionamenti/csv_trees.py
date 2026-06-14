"""Sampled-trees CSV import core: the three-phase contract
(``validate_rows`` → ``apply``) shared by the in-app upload view and (later)
bootstrap.  ``validate_rows`` is pure (no DB writes); foreign keys resolve
against an injected ``TreeIndexes``, so the same code can back a ``--check``
dry-run.  Mirrors ``apps.campionamenti.csv_grid``.
"""

from dataclasses import dataclass
from datetime import date as date_type
from decimal import ROUND_HALF_UP

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import Sample, SampleArea, Species, Tree, TreeSample, tree_mass_q
from config import strings as S
from config.constants import (
    BOSCO_TREE_DIGESTS, FIELD_AREA, FIELD_COPPICE, FIELD_DATE, FIELD_D_CM,
    FIELD_H_M, FIELD_L10_MM, FIELD_MASS_Q, FIELD_NUMBER, FIELD_PARCEL,
    FIELD_PRESERVED, FIELD_SHOOT, FIELD_SPECIES, FIELD_STANDARD, FIELD_VOLUME_M3,
    TREE_H_QUANTUM,
)

TREE_CSV_REQUIRED = [S.CSV_COL_REGION, S.CSV_COL_PARCEL,
                     S.CSV_COL_SAMPLE_AREA, S.CSV_COL_TREE,
                     S.CSV_COL_COPPICE_SHOOT, S.CSV_COL_COPPICE_STD,
                     S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_L10_MM,
                     S.CSV_COL_SPECIES, S.CSV_COL_HIGHFOREST]
TREE_CSV_OPTIONAL = [S.CSV_COL_DATA, S.CSV_COL_PRESERVED]


@dataclass
class TreeIndexes:
    """Lookups ``validate_rows`` needs, injected so validation runs against the
    live DB (view) or a staged index (bootstrap)."""
    area_cache: dict       # (region.lower(), parcel_name, number) -> SampleArea
    species_cache: dict    # common_name.lower() -> Species
    existing_sample_by_area: dict  # sample_area_id -> Sample (already on survey)


def db_indexes(survey) -> TreeIndexes:
    """Build ``TreeIndexes`` from the live database for the given survey."""
    area_cache = {
        (sa.parcel.region.name.lower(), sa.parcel.name, sa.number): sa
        for sa in SampleArea.objects.filter(sample_grid=survey.sample_grid)
                       .select_related('parcel__region')
    }
    species_cache = {s.common_name.lower(): s for s in Species.objects.all()}
    existing_sample_by_area = {
        s.sample_area_id: s for s in Sample.objects.filter(survey=survey)
    }
    return TreeIndexes(area_cache, species_cache, existing_sample_by_area)


def validate_rows(reader, idx: TreeIndexes, *, has_date_column, default_date):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed_rows, errors)``.  Each parsed row is a dict ready for
    ``apply``.  ``errors`` are user-facing strings keyed to the 1-based data
    row number (header is row 1, first data row is row 2).
    """
    # Tabacchi inputs use English-side species names; reuse the GENERE_MAP
    # from the management command to handle minor naming drift.
    from apps.base.management.commands.import_sampled_trees import GENERE_MAP
    from apps.base.tabacchi import has_species, tabacchi_volume_m3

    csv_date_by_area = {}
    errors = []
    parsed = []
    for i, row in enumerate(reader, 2):
        compresa = row[S.CSV_COL_REGION].strip()
        particella = row[S.CSV_COL_PARCEL].strip()
        adc = row[S.CSV_COL_SAMPLE_AREA].strip()
        area = idx.area_cache.get((compresa.lower(), particella, adc))
        if area is None:
            errors.append(S.ERR_CSV_ROW_AREA.format(i, compresa, particella, adc))
            continue
        number = reader.integer(row.get(S.CSV_COL_TREE))
        d_cm = reader.integer(row.get(S.CSV_COL_D_CM))
        h_dec = reader.decimal(row.get(S.CSV_COL_H_M))
        if number is None or d_cm is None or h_dec is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_TREE}/{S.CSV_COL_D_CM}/{S.CSV_COL_H_M}'))
            continue
        shoot, shoot_ok = reader.opt_int(row.get(S.CSV_COL_COPPICE_SHOOT))
        l10, l10_ok = reader.opt_int(row.get(S.CSV_COL_L10_MM))
        standard, std_ok = reader.opt_bool(row.get(S.CSV_COL_COPPICE_STD))
        preserved, pai_ok = reader.opt_bool(row.get(S.CSV_COL_PRESERVED, ''))
        if not (shoot_ok and l10_ok and std_ok and pai_ok):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_COPPICE_SHOOT}/{S.CSV_COL_L10_MM}/'
                   f'{S.CSV_COL_COPPICE_STD}/{S.CSV_COL_PRESERVED}'))
            continue
        shoot = shoot or 0
        l10_mm = l10 or 0
        standard = bool(standard)        # required column; blank → False
        preserved = bool(preserved)      # optional; absent/blank → False
        h_m = h_dec.quantize(TREE_H_QUANTUM, rounding=ROUND_HALF_UP)
        # Fustaia is required: a blank or unrecognised value is an error.
        fustaia, fustaia_ok = reader.opt_bool(row[S.CSV_COL_HIGHFOREST])
        if not fustaia_ok or fustaia is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_HIGHFOREST))
            continue
        coppice = not fustaia

        genere = row[S.CSV_COL_SPECIES].strip()
        mapped = GENERE_MAP.get(genere, genere)
        species = idx.species_cache.get(mapped.lower())
        if species is None:
            errors.append(S.ERR_CSV_ROW_SPECIES.format(i, genere))
            continue

        # Per-row date (if column present) else default.
        if has_date_column and row.get(S.CSV_COL_DATA, '').strip():
            try:
                row_date = date_type.fromisoformat(row[S.CSV_COL_DATA].strip())
            except ValueError:
                errors.append(S.ERR_CSV_ROW_PARSE.format(
                    i, f'{S.CSV_COL_DATA}: {row[S.CSV_COL_DATA]}',
                ))
                continue
        else:
            row_date = default_date
        if row_date is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(i, S.CSV_COL_DATA))
            continue

        existing_sample = idx.existing_sample_by_area.get(area.id)
        if existing_sample and existing_sample.date != row_date:
            errors.append(S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
                i, compresa, particella, adc, existing_sample.date.isoformat(),
            ))
            continue
        previous_date = csv_date_by_area.get(area.id)
        if previous_date is not None and previous_date != row_date:
            errors.append(S.ERR_CSV_ROW_SAMPLE_DATE_CONFLICT.format(
                i, compresa, particella, adc, previous_date.isoformat(),
            ))
            continue
        csv_date_by_area.setdefault(area.id, row_date)

        if coppice or not has_species(mapped):
            volume_m3 = None
            mass_q = None
        else:
            volume_m3 = tabacchi_volume_m3(d_cm, h_m, mapped)
            mass_q = tree_mass_q(volume_m3, species.density)

        parsed.append({
            FIELD_AREA: area, FIELD_DATE: row_date, FIELD_PARCEL: area.parcel,
            FIELD_SPECIES: species, FIELD_COPPICE: coppice, FIELD_PRESERVED: preserved,
            FIELD_NUMBER: number, FIELD_SHOOT: shoot, FIELD_STANDARD: standard,
            FIELD_D_CM: d_cm, FIELD_H_M: h_m, FIELD_L10_MM: l10_mm,
            FIELD_VOLUME_M3: volume_m3, FIELD_MASS_Q: mass_q,
        })
    return parsed, errors


def apply(survey, parsed) -> dict:
    """Persist validated rows: one Sample per (survey, area), then a Tree +
    TreeSample per row.  Returns ``{'n_samples', 'n_trees'}`` for the response.
    """
    with transaction.atomic():
        # One Sample per (survey, area); all rows for that area share its date.
        sample_by_area = {}
        for r in parsed:
            area_id = r[FIELD_AREA].id
            if area_id in sample_by_area:
                continue
            sample, _ = Sample.objects.get_or_create(
                sample_area=r[FIELD_AREA], survey=survey,
                defaults={FIELD_DATE: r[FIELD_DATE]},
            )
            sample_by_area[area_id] = sample

        n_trees = 0
        for r in parsed:
            sample = sample_by_area[r[FIELD_AREA].id]
            tree = Tree.objects.create(
                species=r[FIELD_SPECIES], parcel=r[FIELD_PARCEL],
                preserved=r[FIELD_PRESERVED], coppice=r[FIELD_COPPICE],
            )
            TreeSample.objects.create(
                sample=sample, tree=tree, shoot=r[FIELD_SHOOT],
                standard=r[FIELD_STANDARD], number=r[FIELD_NUMBER],
                d_cm=r[FIELD_D_CM], h_m=r[FIELD_H_M], l10_mm=r[FIELD_L10_MM],
                volume_m3=r[FIELD_VOLUME_M3], mass_q=r[FIELD_MASS_Q],
            )
            n_trees += 1

        mark_stale(
            f'sampled_trees_{survey.id}', 'samples', 'surveys',
            *BOSCO_TREE_DIGESTS, 'audit',
        )
    return {'n_samples': len(sample_by_area), 'n_trees': n_trees}
