"""Hypsometric height-from-diameter regression.

Fits ``h = a·ln(D) + b`` over (diameter, height) samples, mirroring the
logarithmic regression in ``pdg-2026/pdg/computation.py``, and manages the
single active parameter set (archiving superseded sets).  See
``docs/hypsometry.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from config import strings as S

if TYPE_CHECKING:
    from apps.base.models import HypsoParamSet, Region, Species

_DECIMAL_PLACES = Decimal('0.0001')
# a and b persist to DecimalField(max_digits=10, decimal_places=4): |value| < 1e6.
_COEF_MAX = 10 ** 6


def fit_log(
    diameters: Sequence[float],
    heights: Sequence[float],
    min_n: int,
) -> tuple[float, float, float, int] | None:
    """Fit ``h = a·ln(D) + b`` by least squares.

    Points with non-positive or non-finite ``D`` or ``h`` are dropped first.
    Returns ``(a, b, r2, n)`` for the cleaned sample, or ``None`` if fewer
    than ``min_n`` valid points remain.
    """
    d = np.asarray(diameters, dtype=float)
    h = np.asarray(heights, dtype=float)
    keep = (d > 0) & (h > 0) & np.isfinite(d) & np.isfinite(h)
    d, h = d[keep], h[keep]

    n = len(d)
    # Need >= min_n points AND >= 2 distinct diameters: a single (or zero)
    # distinct D makes ln(D) constant, so the fit is degenerate (and an empty
    # vector would make polyfit raise).
    if n < min_n or len(np.unique(d)) < 2:
        return None

    a, b = np.polyfit(np.log(d), h, 1)

    h_pred = a * np.log(d) + b
    ss_tot = float(np.sum((h - np.mean(h)) ** 2))
    ss_res = float(np.sum((h - h_pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(a), float(b), float(r2), n


@dataclass
class ParamRow:
    """One fitted/parsed (region, species) regression, before persistence."""
    region: Region
    species: Species
    a: float
    b: float
    r2: float
    n: int


def compute_params(survey_ids: Sequence[int], min_n: int) -> list[ParamRow]:
    """Fit a regression per (region, species) over the given surveys' samples.

    Coppice samples are excluded.  A (region, species) group is fit only if it
    has at least ``min_n`` valid samples.  Sorted by region then species name.
    """
    from apps.base.models import TreeSample

    qs = (TreeSample.objects
          .filter(
              sample__survey_id__in=list(survey_ids),
              h_measured=True,
              tree__coppice=False,
          )
          .select_related('tree__parcel__region', 'tree__species'))

    groups: dict[tuple[int, int], dict] = {}
    for ts in qs:
        region = ts.tree.parcel.region
        species = ts.tree.species
        group = groups.setdefault(
            (region.id, species.id),
            {'region': region, 'species': species, 'd': [], 'h': []},
        )
        group['d'].append(ts.d_cm)
        group['h'].append(float(ts.h_m))

    out = []
    for group in groups.values():
        fit = fit_log(group['d'], group['h'], min_n)
        if fit is None:
            continue
        a, b, r2, n = fit
        out.append(ParamRow(group['region'], group['species'], a, b, r2, n))
    out.sort(key=lambda r: (r.region.name, r.species.common_name))
    return out


def parse_param_csv(source) -> tuple[list[ParamRow], list[str]]:
    """Parse an `equazioni_ipsometro.csv` (decoded text or upload) into
    ParamRows + error list.

    Reading (decode, delimiter/decimal detection) is delegated to
    `apps.base.csv_io`; headers are matched case-insensitively.  A malformed
    file, or any unresolved region/species, wrong function, or unparseable
    number, is reported as an error (the caller aborts on a non-empty list).
    """
    from apps.base import csv_io
    from apps.base.models import HYPSO_FUNC_LN, Region, Species

    region_cache = {r.name.lower(): r for r in Region.objects.all()}
    species_cache = {sp.common_name.lower(): sp for sp in Species.objects.all()}

    try:
        reader = csv_io.read(source)
    except csv_io.CsvError as exc:
        return [], [str(exc)]

    rows: list[ParamRow] = []
    errors: list[str] = []
    seen: set[tuple[int, int]] = set()
    for i, raw in enumerate(reader, 2):
        row = {(k or '').strip().lower(): v for k, v in raw.items()}
        compresa = (row.get(S.CSV_COL_REGION.lower()) or '').strip()
        genere = (row.get(S.CSV_COL_SPECIES.lower()) or '').strip()
        function = (row.get(S.CSV_COL_FUNCTION) or '').strip().lower()
        region = region_cache.get(compresa.lower())
        if region is None:
            errors.append(S.ERR_CSV_REGION_NOT_FOUND.format(i, compresa))
            continue
        species = species_cache.get(genere.lower())
        if species is None:
            errors.append(S.ERR_CSV_SPECIES_NOT_FOUND.format(i, genere))
            continue
        if function != HYPSO_FUNC_LN:
            errors.append(S.ERR_CSV_FUNCTION_INVALID.format(i, function))
            continue
        a = reader.decimal(row.get(S.CSV_COL_A))
        b = reader.decimal(row.get(S.CSV_COL_B))
        r2 = reader.decimal(row.get(S.CSV_COL_R2))
        n = reader.integer(row.get(S.CSV_COL_N_REGRESSION))
        bad = next((col for col, v in (
            (S.CSV_COL_A, a), (S.CSV_COL_B, b),
            (S.CSV_COL_R2, r2), (S.CSV_COL_N_REGRESSION, n),
        ) if v is None), None)
        if bad is not None:
            errors.append(S.ERR_CSV_VALUE_PARSE.format(i, bad, row.get(bad, '')))
            continue
        a, b, r2 = float(a), float(b), float(r2)
        range_error = _range_error(i, a, b, r2, n)
        if range_error:
            errors.append(range_error)
            continue
        key = (region.id, species.id)
        if key in seen:
            errors.append(S.ERR_CSV_DUPLICATE_PARAM.format(i, compresa, genere))
            continue
        seen.add(key)
        rows.append(ParamRow(region, species, a, b, r2, n))
    return rows, errors


def _range_error(line: int, a: float, b: float, r2: float, n: int) -> str | None:
    """Reject values that overflow storage or are out of their valid range.

    Guards the persist step: `a`/`b` must fit DecimalField(10,4); `r2` is a
    coefficient of determination in [0, 1]; `n` is a positive sample count.
    """
    checks = [
        (S.CSV_COL_A, a, abs(a) < _COEF_MAX),
        (S.CSV_COL_B, b, abs(b) < _COEF_MAX),
        (S.CSV_COL_R2, r2, 0.0 <= r2 <= 1.0),
        (S.CSV_COL_N_REGRESSION, n, n >= 1),
    ]
    for col, val, ok in checks:
        if not ok:
            return S.ERR_CSV_VALUE_RANGE.format(line, col, val)
    return None


def active_set() -> HypsoParamSet | None:
    """The currently-active parameter set, or None if none is active."""
    from apps.base.models import HypsoParamSet
    return HypsoParamSet.objects.active().first()


def _dec(x: float) -> Decimal:
    return Decimal(str(x)).quantize(_DECIMAL_PLACES)


def replace_active_set(
    rows: Sequence[ParamRow],
    *,
    source: str,
    min_n: int | None,
    survey_ids: Sequence[int],
    use_for_height_plots: bool = False,
) -> HypsoParamSet:
    """Archive the current active set (if any) and persist a new active set.

    The whole operation is one transaction, upholding the invariant that at
    most one set has ``superseded_at`` NULL.
    """
    from django.db import transaction
    from django.utils import timezone

    from apps.base.models import HypsoParam, HypsoParamSet

    with transaction.atomic():
        _archive_active_sets(HypsoParamSet, timezone.now())
        new_set = HypsoParamSet.objects.create(
            source=source, min_n=min_n,
            use_for_height_plots=use_for_height_plots,
        )
        if survey_ids:
            new_set.surveys.set(list(survey_ids))
        for r in rows:
            HypsoParam.objects.create(
                param_set=new_set, region=r.region, species=r.species,
                a=_dec(r.a), b=_dec(r.b), r2=_dec(r.r2), n=r.n,
            )
    return new_set


def clear_active_set() -> bool:
    """Archive the active set so that no parameters are active.

    Returns True if a set was archived, False if there was none.
    """
    from django.db import transaction
    from django.utils import timezone

    from apps.base.models import HypsoParamSet

    with transaction.atomic():
        archived = _archive_active_sets(HypsoParamSet, timezone.now())
    return archived > 0


def _archive_active_sets(model, superseded_at) -> int:
    archived = 0
    for param_set in model.objects.active().select_for_update():
        param_set.superseded_at = superseded_at
        param_set.version += 1
        param_set.save(update_fields=['superseded_at', 'version'])
        archived += 1
    return archived
