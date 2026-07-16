"""Tests for the hypsometric log-regression fit (apps.base.hypsometry)."""

import math
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest

from apps.base import hypsometry
from apps.base.models import (
    HYPSO_FUNC_LN, HypsoParam, HypsoParamSet, HypsoParamSource, Sample,
    Survey, Tree, TreeSample,
)
from config import strings as S

CSV_HEADER = ','.join([
    S.CSV_COL_REGION.lower(), S.CSV_COL_SPECIES.lower(), S.CSV_COL_FUNCTION,
    S.CSV_COL_A, S.CSV_COL_B, S.CSV_COL_R2, S.CSV_COL_N_REGRESSION,
])

# Ground-truth coefficients for synthetic exact-fit data.
A_TRUE = 7.0306
B_TRUE = -4.2563
MIN_N = 10
# Diameters (cm) spanning a realistic range; at least MIN_N distinct points.
DIAMETERS = [8, 10, 12, 15, 18, 20, 24, 28, 32, 36, 40, 45]


def _exact_heights(diameters: Sequence[float]) -> list[float]:
    return [A_TRUE * math.log(d) + B_TRUE for d in diameters]


def test_recovers_coefficients_from_exact_data():
    result = hypsometry.fit_log(DIAMETERS, _exact_heights(DIAMETERS), MIN_N)
    assert result is not None
    a, b, r2, n = result
    assert a == pytest.approx(A_TRUE, abs=1e-6)
    assert b == pytest.approx(B_TRUE, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert n == len(DIAMETERS)


def test_returns_none_below_min_n():
    d = DIAMETERS[: MIN_N - 1]
    assert hypsometry.fit_log(d, _exact_heights(d), MIN_N) is None


def test_drops_invalid_points_before_fitting():
    # Non-positive / non-finite D or h must be ignored, leaving the valid set.
    extra_d = [0, -5, 20, 22, float("nan")]
    extra_h = [10.0, 10.0, float("nan"), -1.0, 10.0]
    d = DIAMETERS + extra_d
    h = _exact_heights(DIAMETERS) + extra_h
    result = hypsometry.fit_log(d, h, MIN_N)
    assert result is not None
    a, b, _r2, n = result
    assert n == len(DIAMETERS)
    assert a == pytest.approx(A_TRUE, abs=1e-6)
    assert b == pytest.approx(B_TRUE, abs=1e-6)


def test_returns_none_when_cleaning_drops_below_min_n():
    valid = 4
    d = DIAMETERS[:valid] + [0, -1, -2, float("nan")]
    h = _exact_heights(DIAMETERS[:valid]) + [1.0, 1.0, 1.0, 1.0]
    assert hypsometry.fit_log(d, h, MIN_N) is None


def test_returns_none_for_single_distinct_diameter():
    # All-equal D -> ln(D) constant -> degenerate fit must be rejected.
    d = [20] * MIN_N
    h = [10.0 + 0.01 * i for i in range(MIN_N)]
    assert hypsometry.fit_log(d, h, MIN_N) is None


def test_returns_none_for_empty_even_at_min_n_zero():
    # min_n <= 0 must not let an empty vector reach polyfit.
    assert hypsometry.fit_log([], [], 0) is None


def test_high_r2_on_nearly_log_data():
    # Deterministic small perturbations -> still a strong log fit.
    perturb = [0.05, -0.04, 0.03, -0.02, 0.02, -0.03,
               0.04, -0.05, 0.03, -0.02, 0.01, -0.01]
    h = [y + p for y, p in zip(_exact_heights(DIAMETERS), perturb)]
    result = hypsometry.fit_log(DIAMETERS, h, MIN_N)
    assert result is not None
    _a, _b, r2, n = result
    assert 0.95 < r2 <= 1.0
    assert n == len(DIAMETERS)


# --- compute_params (DB) ---

def test_compute_params_fits_and_excludes(hypso_samples):
    rows = hypsometry.compute_params([hypso_samples['survey'].id], min_n=5)
    # Only Abete qualifies: Castagno has 3 < 5; the coppice sample is excluded.
    assert len(rows) == 1
    row = rows[0]
    assert row.region == hypso_samples['region']
    assert row.species == hypso_samples['species']
    assert row.n == hypso_samples['n_abete']
    # h_m is stored to 2 decimals, so the recovered fit is close, not exact.
    assert row.a == pytest.approx(hypso_samples['a'], abs=0.05)
    assert row.b == pytest.approx(hypso_samples['b'], abs=0.05)


def test_compute_params_min_n_excludes_all(hypso_samples):
    assert hypsometry.compute_params([hypso_samples['survey'].id], min_n=100) == []


def test_compute_params_ignores_unstructured_samples(
        hypso_samples, parcels, species,
):
    unstructured = Survey.objects.create(name='Ad hoc tree measurements')
    sample = Sample.objects.create(
        sample_area=None, survey=unstructured, date=date(2026, 7, 16),
    )
    for number, d_cm in enumerate((30, 32, 34, 36, 38, 40), start=1):
        tree = Tree.objects.create(species=species[0], parcel=parcels[0])
        TreeSample.objects.create(
            sample=sample, tree=tree, number=number, d_cm=d_cm,
            h_m=Decimal('99.00'),
        )

    rows = hypsometry.compute_params(
        [hypso_samples['survey'].id, unstructured.id], min_n=5,
    )

    assert len(rows) == 1
    assert rows[0].n == hypso_samples['n_abete']
    assert rows[0].a == pytest.approx(hypso_samples['a'], abs=0.05)
    assert rows[0].b == pytest.approx(hypso_samples['b'], abs=0.05)


def test_compute_params_unknown_survey_is_empty(db):
    assert hypsometry.compute_params([999], min_n=1) == []


# --- parse_param_csv (DB) ---

def _csv(*data_rows: str) -> str:
    return CSV_HEADER + '\n' + '\n'.join(data_rows) + '\n'


def test_parse_param_csv_resolves_rows(db, regions, species):
    text = _csv(f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},'
                f'7.03,-4.26,0.63,48')
    rows, errors = hypsometry.parse_param_csv(text)
    assert errors == []
    assert len(rows) == 1
    assert rows[0].region == regions[0]
    assert rows[0].species == species[0]
    assert rows[0].n == 48


def test_parse_param_csv_accepts_italian_decimals_and_semicolons(db, regions, species):
    header = CSV_HEADER.replace(',', ';')
    text = (f'{header}\n'
            f'{regions[0].name};{species[0].common_name};{HYPSO_FUNC_LN};'
            f'7,03;-4,26;0,63;48\n')
    rows, errors = hypsometry.parse_param_csv(text)
    assert errors == []
    assert rows[0].a == pytest.approx(7.03)


def test_parse_param_csv_reports_unknown_region(db, species):
    text = _csv(f'Nessuna,{species[0].common_name},{HYPSO_FUNC_LN},7,-4,0.5,20')
    rows, errors = hypsometry.parse_param_csv(text)
    assert rows == []
    assert len(errors) == 1


def test_parse_param_csv_rejects_non_ln_function(db, regions, species):
    text = _csv(f'{regions[0].name},{species[0].common_name},lin,7,-4,0.5,20')
    rows, errors = hypsometry.parse_param_csv(text)
    assert rows == []
    assert len(errors) == 1


def test_parse_param_csv_rejects_out_of_range_r2(db, regions, species):
    # r2 outside [0, 1] would mislead and (large) overflow the DecimalField.
    text = _csv(f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},'
                f'7,-4,5.0,20')
    rows, errors = hypsometry.parse_param_csv(text)
    assert rows == []
    assert len(errors) == 1


def test_parse_param_csv_rejects_overflowing_coefficient(db, regions, species):
    text = _csv(f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},'
                f'1000000,-4,0.6,20')
    rows, errors = hypsometry.parse_param_csv(text)
    assert rows == []
    assert len(errors) == 1


def test_parse_param_csv_rejects_duplicate_pair(db, regions, species):
    line = (f'{regions[0].name},{species[0].common_name},{HYPSO_FUNC_LN},'
            f'7,-4,0.6,20')
    rows, errors = hypsometry.parse_param_csv(_csv(line, line))
    assert len(rows) == 1   # first row accepted
    assert len(errors) == 1  # duplicate flagged


def test_compute_params_groups_by_region(hypso_samples, regions, eclasses):
    # Abete samples in a second region must yield a second, separate row.
    from datetime import date
    from decimal import Decimal
    from apps.base.models import (
        Parcel, Sample, SampleArea, Survey, Tree, TreeSample,
    )
    grid = hypso_samples['survey'].sample_grid
    parcel2 = Parcel.objects.create(
        name='R2', region=regions[1], eclass=eclasses[0],
        area_ha=Decimal('5.0'),
    )
    area2 = SampleArea.objects.create(
        sample_grid=grid, parcel=parcel2, number='2', lat=0.0, lon=0.0,
    )
    survey2 = Survey.objects.create(name='Hypso survey 2', sample_grid=grid)
    sample2 = Sample.objects.create(
        sample_area=area2, survey=survey2, date=date(2024, 9, 16),
    )
    sp = hypso_samples['species']
    for i, d in enumerate([8, 10, 12, 15, 18, 20, 24, 28, 32, 36], start=1):
        tree = Tree.objects.create(species=sp, parcel=parcel2, coppice=False)
        TreeSample.objects.create(
            sample=sample2, tree=tree, shoot=0, standard=False,
            number=i, d_cm=d, h_m=Decimal(str(round(6.5 * math.log(d) - 3, 2))),
        )
    rows = hypsometry.compute_params(
        [hypso_samples['survey'].id, survey2.id], min_n=5,
    )
    regions_fitted = {r.region for r in rows}
    assert regions[0] in regions_fitted
    assert regions[1] in regions_fitted


# --- active-set write path (DB) ---

def _abete_rows(hypso_samples):
    return hypsometry.compute_params([hypso_samples['survey'].id], min_n=5)


def test_replace_active_set_archives_prior_and_keeps_one_active(hypso_samples):
    rows = _abete_rows(hypso_samples)
    first = hypsometry.replace_active_set(
        rows, source=HypsoParamSource.COMPUTED, min_n=5,
        survey_ids=[hypso_samples['survey'].id],
    )
    assert hypsometry.active_set() == first
    assert list(first.surveys.all()) == [hypso_samples['survey']]

    second = hypsometry.replace_active_set(
        rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
    )
    first.refresh_from_db()
    assert first.superseded_at is not None
    assert hypsometry.active_set() == second
    assert HypsoParamSet.objects.filter(superseded_at__isnull=True).count() == 1


def test_clear_active_set_archives_and_retains_params(hypso_samples):
    s = hypsometry.replace_active_set(
        _abete_rows(hypso_samples), source=HypsoParamSource.COMPUTED,
        min_n=5, survey_ids=[],
    )
    assert hypsometry.clear_active_set() is True
    assert hypsometry.active_set() is None
    s.refresh_from_db()
    assert s.superseded_at is not None
    assert HypsoParam.objects.filter(param_set=s).count() == 1
    # Clearing again when none is active is a no-op.
    assert hypsometry.clear_active_set() is False
