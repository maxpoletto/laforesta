"""Sampled-trees CSV import core: the three-phase contract
(``validate_rows`` → ``apply``) shared by the in-app upload view and (later)
bootstrap.  ``validate_rows`` is pure (no DB writes); foreign keys resolve
against an injected ``TreeIndexes``, so the same code can back a ``--check``
dry-run.  Mirrors ``apps.campionamenti.csv_grid``.
"""

from dataclasses import dataclass, field
from datetime import date as date_type
from decimal import ROUND_HALF_UP

from django.db import transaction

from apps.base.digests import mark_stale
from apps.base.models import (
    Parcel, Sample, SampleArea, Species, Tree, TreeSample, tree_mass_q,
)
from apps.base.numparse import coord_float
from apps.base.preserved_trees import latest_preserved_tree_samples
from apps.base.refdata import GENERE_MAP
from apps.campionamenti.tree_validation import normalize_sample_tree_values
from config import strings as S
from config.constants import (
    BOSCO_TREE_DIGESTS, FIELD_ACC_M, FIELD_AREA, FIELD_COPPICE, FIELD_DATE,
    FIELD_D_CM, FIELD_H_M, FIELD_H_MEASURED, FIELD_L10_MM, FIELD_LAT, FIELD_LON,
    FIELD_MASS_Q, FIELD_NOTE,
    FIELD_NUMBER, FIELD_OPERATOR, FIELD_PARCEL, FIELD_PRESERVED_NUMBER,
    FIELD_PRESERVED, FIELD_PRESSLER_COEFF, FIELD_SHOOT, FIELD_SPECIES,
    FIELD_STANDARD, FIELD_VOLUME_M3,
    TREE_H_QUANTUM,
)

TREE_CSV_REQUIRED = [S.CSV_COL_REGION, S.CSV_COL_PARCEL,
                     S.CSV_COL_SAMPLE_AREA, S.CSV_COL_TREE,
                     S.CSV_COL_COPPICE_SHOOT, S.CSV_COL_COPPICE_STD,
                     S.CSV_COL_D_CM, S.CSV_COL_H_M, S.CSV_COL_L10_MM,
                     S.CSV_COL_PRESSLER, S.CSV_COL_SPECIES,
                     S.CSV_COL_HIGHFOREST]
FREE_TREE_CSV_REQUIRED = [S.CSV_COL_REGION, S.CSV_COL_PARCEL,
                          S.CSV_COL_TREE, S.CSV_COL_COPPICE_SHOOT,
                          S.CSV_COL_COPPICE_STD, S.CSV_COL_D_CM,
                          S.CSV_COL_H_M, S.CSV_COL_L10_MM,
                          S.CSV_COL_PRESSLER, S.CSV_COL_SPECIES,
                          S.CSV_COL_HIGHFOREST]
TREE_CSV_OPTIONAL = [
    S.CSV_COL_DATA, S.CSV_COL_PRESERVED, S.CSV_COL_H_MEASURED,
    S.CSV_COL_LON, S.CSV_COL_LAT, S.CSV_COL_ACC_M,
    S.CSV_COL_OPERATOR, S.CSV_COL_NOTE,
]


@dataclass
class TreeIndexes:
    """Lookups ``validate_rows`` needs, injected so validation runs against the
    live DB (view) or a staged index (bootstrap)."""
    area_cache: dict       # (region.lower(), parcel_name, number) -> SampleArea
    species_cache: dict    # common_name.lower() -> Species
    existing_sample_by_area: dict  # sample_area_id -> Sample (already on survey)
    existing_number_shoots: set     # (sample_area_id, number, shoot)
    parcel_cache: dict = field(default_factory=dict)  # (region.lower(), parcel) -> Parcel
    preserved_tree_by_key: dict = field(default_factory=dict)  # (parcel_id, number) -> Tree
    is_unstructured: bool = False


def db_indexes(survey) -> TreeIndexes:
    """Build ``TreeIndexes`` from the live database for the given survey."""
    is_unstructured = survey.sample_grid_id is None
    area_cache = {}
    if not is_unstructured:
        area_cache = {
            (sa.parcel.region.name.lower(), sa.parcel.name, sa.number): sa
            for sa in SampleArea.objects.filter(sample_grid=survey.sample_grid)
                           .select_related('parcel__region')
        }
    parcels = Parcel.objects.select_related('region')
    parcel_cache = {(p.region.name.lower(), p.name): p for p in parcels}
    species_cache = {s.common_name.lower(): s for s in Species.objects.all()}
    preserved_tree_by_key = {
        (ts.parcel_id, ts.preserved_number): ts.tree
        for ts in latest_preserved_tree_samples().select_related('tree__species')
    }
    existing_sample_by_area = {
        s.sample_area_id: s for s in Sample.objects.filter(survey=survey)
    }
    existing_number_shoots = set(
        TreeSample.objects
        .filter(sample__survey=survey)
        .values_list('sample__sample_area_id', FIELD_NUMBER, FIELD_SHOOT)
    )
    return TreeIndexes(
        area_cache, species_cache, existing_sample_by_area, existing_number_shoots,
        parcel_cache=parcel_cache, preserved_tree_by_key=preserved_tree_by_key,
        is_unstructured=is_unstructured,
    )


def validate_rows(reader, idx: TreeIndexes, *, has_date_column, default_date):
    """Validate parsed CSV rows against ``idx``.  Pure: no DB writes.

    Returns ``(parsed_rows, errors)``.  Each parsed row is a dict ready for
    ``apply``.  ``errors`` are user-facing strings keyed to the 1-based data
    row number (header is row 1, first data row is row 2).
    """
    csv_date_by_area = {}
    seen_number_shoots = set(idx.existing_number_shoots)
    errors = []
    parsed = []
    for i, row in enumerate(reader, 2):
        compresa = row[S.CSV_COL_REGION].strip()
        particella = row[S.CSV_COL_PARCEL].strip()
        if idx.is_unstructured:
            adc = ''
            area = None
            parcel = idx.parcel_cache.get((compresa.lower(), particella))
            if parcel is None:
                errors.append(S.ERR_CSV_PARCEL_NOT_FOUND.format(
                    i, compresa, particella,
                ))
                continue
        else:
            adc = row[S.CSV_COL_SAMPLE_AREA].strip()
            area = idx.area_cache.get((compresa.lower(), particella, adc))
            if area is None:
                errors.append(S.ERR_CSV_ROW_AREA.format(i, compresa, particella, adc))
                continue
            parcel = area.parcel
        number = reader.integer(row.get(S.CSV_COL_TREE))
        d_cm = reader.integer(row.get(S.CSV_COL_D_CM))
        h_dec = reader.decimal(row.get(S.CSV_COL_H_M))
        if number is None or d_cm is None or h_dec is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_TREE}/{S.CSV_COL_D_CM}/{S.CSV_COL_H_M}'))
            continue
        if number <= 0 or d_cm <= 0 or h_dec <= 0:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_TREE}/{S.CSV_COL_D_CM}/{S.CSV_COL_H_M}'))
            continue
        shoot, shoot_ok = reader.opt_int(row.get(S.CSV_COL_COPPICE_SHOOT))
        l10, l10_ok = reader.opt_int(row.get(S.CSV_COL_L10_MM))
        pressler = reader.decimal(row.get(S.CSV_COL_PRESSLER))
        standard, std_ok = reader.opt_bool(row.get(S.CSV_COL_COPPICE_STD))
        preserved, pai_ok = reader.opt_bool(row.get(S.CSV_COL_PRESERVED, ''))
        h_measured, h_measured_ok = reader.opt_bool(row.get(S.CSV_COL_H_MEASURED, ''))
        lat, lat_ok = _optional_coord(reader, row.get(S.CSV_COL_LAT, ''))
        lon, lon_ok = _optional_coord(reader, row.get(S.CSV_COL_LON, ''))
        acc_m, acc_ok = reader.opt_int(row.get(S.CSV_COL_ACC_M, ''))
        if (
                not (shoot_ok and l10_ok and std_ok and pai_ok and h_measured_ok
                     and lat_ok and lon_ok and acc_ok)
                or pressler is None or pressler <= 0
        ):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_COPPICE_SHOOT}/{S.CSV_COL_L10_MM}/'
                   f'{S.CSV_COL_PRESSLER}/{S.CSV_COL_COPPICE_STD}/'
                   f'{S.CSV_COL_PRESERVED}/{S.CSV_COL_H_MEASURED}/'
                   f'{S.CSV_COL_LAT}/{S.CSV_COL_LON}/{S.CSV_COL_ACC_M}'))
            continue
        if (lat is None) != (lon is None):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_LAT}/{S.CSV_COL_LON}',
            ))
            continue
        if (shoot is not None and shoot < 0) or (l10 is not None and l10 < 0):
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_COPPICE_SHOOT}/{S.CSV_COL_L10_MM}'))
            continue
        values = normalize_sample_tree_values(
            number=number,
            d_cm=d_cm,
            h_m=h_dec,
            shoot=shoot or 0,
            l10_mm=l10 or 0,
            pressler_coeff=pressler,
            h_measured=h_measured,
        )
        if values is None:
            errors.append(S.ERR_CSV_ROW_PARSE.format(
                i, f'{S.CSV_COL_TREE}/{S.CSV_COL_COPPICE_SHOOT}/'
                   f'{S.CSV_COL_D_CM}/{S.CSV_COL_H_M}/'
                   f'{S.CSV_COL_L10_MM}/{S.CSV_COL_PRESSLER}',
            ))
            continue
        standard = bool(standard)        # required column; blank → False
        preserved = bool(preserved)      # optional; absent/blank → False
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
        if preserved:
            existing_tree = idx.preserved_tree_by_key.get((parcel.id, values.number))
            if existing_tree is not None and existing_tree.species_id != species.id:
                errors.append(S.ERR_CSV_ROW_PAI_SPECIES_CONFLICT.format(
                    i, compresa, particella, values.number,
                    existing_tree.species.common_name,
                ))
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

        if idx.is_unstructured:
            previous_date = csv_date_by_area.get(None)
            if previous_date is not None and previous_date != row_date:
                errors.append(S.ERR_CSV_ROW_FREE_SAMPLE_DATE_CONFLICT.format(
                    i, previous_date.isoformat(),
                ))
                continue
            csv_date_by_area.setdefault(None, row_date)
            number_shoot_key = (None, values.number, values.shoot)
        else:
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
            number_shoot_key = (area.id, values.number, values.shoot)
        if number_shoot_key in seen_number_shoots:
            errors.append(S.ERR_CSV_ROW_TREE_NUMBER_DUPLICATE.format(
                i, values.number, values.shoot,
            ))
            continue
        seen_number_shoots.add(number_shoot_key)

        parsed.append(parsed_tree_row(
            area=area, parcel=parcel, row_date=row_date, species=species,
            coppice=coppice, preserved=preserved, number=values.number,
            shoot=values.shoot,
            standard=standard, d_cm=values.d_cm, h_m=values.h_m,
            h_measured=values.h_measured, l10_mm=values.l10_mm,
            pressler_coeff=values.pressler_coeff,
            lat=lat, lon=lon, acc_m=acc_m,
            operator=(row.get(S.CSV_COL_OPERATOR) or '').strip(),
            note=(row.get(S.CSV_COL_NOTE) or '').strip(),
            volume_species_name=mapped,
        ))
    return parsed, errors


def _optional_coord(reader, value):
    raw = (value or '').strip()
    if not raw:
        return None, True
    parsed = coord_float(reader.decimal(raw))
    return parsed, parsed is not None


def tree_volume_and_mass(coppice, d_cm, h_m, species, species_name=None):
    if coppice or d_cm is None or h_m is None:
        return None, None
    from apps.base.tabacchi import has_species, tabacchi_volume_m3
    name = species_name or species.common_name
    if not has_species(name):
        return None, None
    volume_m3 = tabacchi_volume_m3(d_cm, h_m, name)
    return volume_m3, tree_mass_q(volume_m3, species.density)


def parsed_tree_row(
        *, area, row_date, species, coppice, preserved, number, shoot, standard,
        d_cm, h_m, l10_mm, pressler_coeff, parcel=None, h_measured=False,
        lat=None, lon=None,
        acc_m=None, operator='', note='',
        volume_species_name=None,
):
    h_m = h_m.quantize(TREE_H_QUANTUM, rounding=ROUND_HALF_UP)
    volume_m3, mass_q = tree_volume_and_mass(
        coppice, d_cm, h_m, species, species_name=volume_species_name,
    )
    return {
        FIELD_AREA: area,
        FIELD_DATE: row_date,
        FIELD_PARCEL: parcel or area.parcel,
        FIELD_SPECIES: species,
        FIELD_COPPICE: coppice,
        FIELD_PRESERVED: preserved,
        FIELD_PRESERVED_NUMBER: number if preserved else None,
        FIELD_NUMBER: number,
        FIELD_SHOOT: shoot,
        FIELD_STANDARD: standard,
        FIELD_D_CM: d_cm,
        FIELD_H_M: h_m,
        FIELD_H_MEASURED: bool(h_measured),
        FIELD_L10_MM: l10_mm,
        FIELD_PRESSLER_COEFF: pressler_coeff,
        FIELD_VOLUME_M3: volume_m3,
        FIELD_MASS_Q: mass_q,
        FIELD_LAT: lat,
        FIELD_LON: lon,
        FIELD_ACC_M: acc_m,
        FIELD_OPERATOR: operator,
        FIELD_NOTE: note,
    }


def apply(survey, parsed) -> dict:
    """Persist validated rows: one Sample per (survey, area), then a
    TreeSample per row. Physical Tree identity is shared by sample area and
    tree number across surveys, including coppice shoots.

    Returns ``{'n_samples', 'n_trees'}`` for the response.
    """
    if survey.sample_grid_id is None:
        return apply_unstructured(survey, parsed)

    with transaction.atomic():
        area_ids = sorted({r[FIELD_AREA].id for r in parsed})
        list(
            SampleArea.objects
            .select_for_update()
            .filter(id__in=area_ids)
            .order_by('id')
        )

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

        tree_by_identity = _tree_identity_map(sample_by_area)
        tree_by_preserved_key = _preserved_tree_identity_map(parsed, for_update=True)
        n_trees = 0
        for r in parsed:
            sample = sample_by_area[r[FIELD_AREA].id]
            identity = (r[FIELD_AREA].id, r[FIELD_NUMBER])
            preserved_key = _preserved_key(r)
            tree = tree_by_preserved_key.get(preserved_key) if preserved_key else None
            if tree is None:
                tree = tree_by_identity.get(identity)
            if tree is None:
                tree = Tree.objects.create(
                    species=r[FIELD_SPECIES], coppice=r[FIELD_COPPICE],
                )
            tree_by_identity[identity] = tree
            if preserved_key:
                tree_by_preserved_key[preserved_key] = tree
            TreeSample.objects.create(
                sample=sample, tree=tree, parcel=r[FIELD_PARCEL],
                shoot=r[FIELD_SHOOT], standard=r[FIELD_STANDARD],
                number=r[FIELD_NUMBER],
                preserved_number=r[FIELD_PRESERVED_NUMBER],
                d_cm=r[FIELD_D_CM], h_m=r[FIELD_H_M],
                h_measured=r[FIELD_H_MEASURED], l10_mm=r[FIELD_L10_MM],
                pressler_coeff=r[FIELD_PRESSLER_COEFF],
                volume_m3=r[FIELD_VOLUME_M3], mass_q=r[FIELD_MASS_Q],
                lat=r[FIELD_LAT], lon=r[FIELD_LON], acc_m=r[FIELD_ACC_M],
                operator=r[FIELD_OPERATOR], note=r[FIELD_NOTE],
            )
            n_trees += 1

        mark_stale(
            f'sampled_trees_{survey.id}', 'samples', 'surveys',
            *BOSCO_TREE_DIGESTS, 'audit',
        )
    return {'n_samples': len(sample_by_area), 'n_trees': n_trees}


def apply_unstructured(survey, parsed) -> dict:
    """Persist CSV rows into one null-area Sample for a free survey."""
    if not parsed:
        return {'n_samples': 0, 'n_trees': 0}
    with transaction.atomic():
        sample = Sample.objects.create(
            sample_area=None, survey=survey, date=parsed[0][FIELD_DATE],
        )
        tree_by_number = {}
        tree_by_preserved_key = _preserved_tree_identity_map(parsed, for_update=True)
        for r in parsed:
            preserved_key = _preserved_key(r)
            tree = tree_by_preserved_key.get(preserved_key) if preserved_key else None
            if tree is None:
                tree = tree_by_number.get(r[FIELD_NUMBER])
            if tree is None:
                tree = Tree.objects.create(
                    species=r[FIELD_SPECIES], coppice=r[FIELD_COPPICE],
                )
            tree_by_number.setdefault(r[FIELD_NUMBER], tree)
            if preserved_key:
                tree_by_preserved_key[preserved_key] = tree
            TreeSample.objects.create(
                sample=sample, tree=tree, parcel=r[FIELD_PARCEL],
                shoot=r[FIELD_SHOOT], standard=r[FIELD_STANDARD],
                number=r[FIELD_NUMBER],
                preserved_number=r[FIELD_PRESERVED_NUMBER],
                d_cm=r[FIELD_D_CM], h_m=r[FIELD_H_M],
                h_measured=r[FIELD_H_MEASURED], l10_mm=r[FIELD_L10_MM],
                pressler_coeff=r[FIELD_PRESSLER_COEFF],
                volume_m3=r[FIELD_VOLUME_M3], mass_q=r[FIELD_MASS_Q],
                lat=r[FIELD_LAT], lon=r[FIELD_LON], acc_m=r[FIELD_ACC_M],
                operator=r[FIELD_OPERATOR], note=r[FIELD_NOTE],
            )

        mark_stale(
            f'sampled_trees_{survey.id}', 'samples', 'surveys',
            *BOSCO_TREE_DIGESTS, 'audit',
        )
    return {'n_samples': 1, 'n_trees': len(parsed)}


def _tree_identity_map(sample_by_area):
    sample_ids = [sample.id for sample in sample_by_area.values()]
    area_ids = list(sample_by_area)
    tree_by_identity = {}

    target_rows = (
        TreeSample.objects
        .select_for_update()
        .filter(sample_id__in=sample_ids)
        .select_related('sample', 'tree')
        .order_by('sample__sample_area_id', 'number', '-sample__date', '-id')
    )
    for row in target_rows:
        tree_by_identity.setdefault(
            (row.sample.sample_area_id, row.number), row.tree,
        )

    historical_rows = (
        TreeSample.objects
        .select_for_update()
        .filter(sample__sample_area_id__in=area_ids)
        .exclude(sample_id__in=sample_ids)
        .select_related('sample', 'tree')
        .order_by('sample__sample_area_id', 'number', '-sample__date', '-id')
    )
    for row in historical_rows:
        tree_by_identity.setdefault(
            (row.sample.sample_area_id, row.number), row.tree,
        )

    return tree_by_identity


def _preserved_key(row):
    preserved_number = row.get(FIELD_PRESERVED_NUMBER)
    if preserved_number is None:
        return None
    return (row[FIELD_PARCEL].id, preserved_number)


def _preserved_tree_identity_map(parsed, *, for_update=False):
    keys = {_preserved_key(row) for row in parsed}
    keys.discard(None)
    if not keys:
        return {}
    parcel_ids = {parcel_id for parcel_id, _number in keys}
    numbers = {number for _parcel_id, number in keys}
    qs = latest_preserved_tree_samples(for_update=for_update).filter(
        parcel_id__in=parcel_ids, preserved_number__in=numbers,
    ).select_related('tree')
    return {
        (row.parcel_id, row.preserved_number): row.tree
        for row in qs
        if (row.parcel_id, row.preserved_number) in keys
    }
