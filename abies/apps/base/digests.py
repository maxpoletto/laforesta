"""Central digest generation for all domains.

Each digest is a gzip-compressed JSON file with the format:

    { "columns": ["row_id", ...], "rows": [[1, ...], ...] }

Write path: after a successful save, views mark affected digests as stale.
Read path: regenerate_if_stale() checks the flag, regenerates if needed,
and returns the file path.  See CLAUDE.md "JSON digest regeneration" for the
full protocol.
"""

import gzip
import json
import os
import tempfile
from pathlib import Path

from django.conf import settings
from django.db.models import Sum
from django.http import FileResponse, HttpResponse
from django.utils.http import http_date, parse_http_date_safe

from apps.base.models import DigestStatus, render_flag_note
from config import strings as S
from config.constants import (
    FIELD_FIRST_DATE, FIELD_LAST_DATE, FIELD_NUMBER, FIELD_SAMPLE_AREA_ID,
    FIELD_SHOOT, FIELD_SORT_ORDER, FIELD_SPECIES, FIELD_SPECIES_ID,
    FIELD_SURVEY_ID, FIELD_VOLUME_M3, ROW_ID, VERSION,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _write_gzip_json(data: dict, dest: Path) -> None:
    """Atomically write *data* as gzip-compressed JSON to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix='.tmp')
    try:
        with gzip.open(os.fdopen(fd, 'wb'), 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        os.rename(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _dest(name: str) -> Path:
    return settings.DIGEST_DIR / f'{name}.json.gz'


def mark_stale(*names: str) -> None:
    """Mark one or more digests as needing regeneration."""
    for name in names:
        DigestStatus.objects.update_or_create(
            name=name, defaults={'stale': True},
        )


_DYNAMIC_PREFIX_SAMPLED_TREES = 'sampled_trees_'
_DYNAMIC_PREFIX_MARK_TREES    = 'mark_trees_'


def _resolve_generator(name: str):
    """Look up a digest generator, including dynamic per-entity names.

    Supports static names registered in `_GENERATORS` plus two
    dynamic patterns:
      - `sampled_trees_<survey_id>` (Campionamenti)
      - `mark_trees_<harvest_plan_item_id>` (Piano di taglio)
    Returns None for unknown names.
    """
    if name in _GENERATORS:
        return _GENERATORS[name]
    if name.startswith(_DYNAMIC_PREFIX_SAMPLED_TREES):
        try:
            survey_id = int(name[len(_DYNAMIC_PREFIX_SAMPLED_TREES):])
        except ValueError:
            return None
        return lambda: generate_sampled_trees_for_survey(survey_id)
    if name.startswith(_DYNAMIC_PREFIX_MARK_TREES):
        try:
            item_id = int(name[len(_DYNAMIC_PREFIX_MARK_TREES):])
        except ValueError:
            return None
        return lambda: generate_mark_trees_for_item(item_id)
    return None


def regenerate_if_stale(name: str) -> Path:
    """Return the path to *name*'s digest, regenerating first if stale."""
    dest = _dest(name)
    status, _ = DigestStatus.objects.get_or_create(name=name)
    if status.stale or not dest.exists():
        gen = _resolve_generator(name)
        if gen is None:
            raise ValueError(f'unknown digest: {name!r}')
        gen()
        # Compare-and-swap: only clear if still stale (avoids race).
        DigestStatus.objects.filter(name=name, stale=True).update(stale=False)
    return dest


def serve_digest(request, name: str):
    """Serve a named digest with conditional GET and lazy regeneration."""
    path = regenerate_if_stale(name)
    mtime = os.path.getmtime(path)
    ims = request.META.get('HTTP_IF_MODIFIED_SINCE')
    if ims:
        ims_ts = parse_http_date_safe(ims)
        if ims_ts is not None and ims_ts >= int(mtime):
            return HttpResponse(status=304)
    response = FileResponse(open(path, 'rb'), content_type='application/json')
    response['Content-Encoding'] = 'gzip'
    response['Last-Modified'] = http_date(mtime)
    return response


# ---------------------------------------------------------------------------
# Prelievi digest
# ---------------------------------------------------------------------------

def generate_prelievi() -> None:
    """De-normalized harvest table for the Prelievi sortable-table.

    Columns: row_id, version, Data, Compresa, Particella, Squadra, VDP, Tipo,
    Q.li, Volume (m³), Note, Altre note, then one quintal column per species
    (sort_order), then one quintal column per tractor (alphabetical).  Also
    carries percentage values for form pre-population (suffixed with " %").
    """
    from apps.base.models import Species, Tractor
    from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor

    # Stable sort orders for dynamic columns.
    all_species = list(Species.objects.order_by(FIELD_SORT_ORDER).values_list('id', 'common_name'))
    all_tractors = list(Tractor.objects.order_by('manufacturer', 'model')
                        .values_list('id', 'pk', 'manufacturer', 'model'))

    species_ids = [sid for sid, _ in all_species]
    species_names = [name for _, name in all_species]

    tractor_ids = [tid for tid, _, _, _ in all_tractors]
    tractor_labels = [
        f'{mfr} {mdl}'.strip() for _, _, mfr, mdl in all_tractors
    ]

    columns = (
        [ROW_ID, VERSION, S.COL_DATE, S.COL_COMPRESA, S.COL_PARCEL,
         S.COL_CANTIERE, S.COL_CREW, S.COL_VDP, S.COL_PRODUCT,
         S.COL_QUINTALS, S.COL_VOLUME_M3, S.COL_NOTE, S.COL_EXTRA_NOTE]
        + species_names
        + tractor_labels
        + [f'{n} %' for n in species_names]
        + [f'{l} %' for l in tractor_labels]
    )

    # Prefetch junction tables into dicts keyed by harvest_id.
    sp_map: dict[int, dict[int, int]] = {}
    for hs in HarvestSpecies.objects.all().values_list('harvest_id', FIELD_SPECIES_ID, 'percent'):
        sp_map.setdefault(hs[0], {})[hs[1]] = hs[2]

    tr_map: dict[int, dict[int, int]] = {}
    for ht in HarvestTractor.objects.all().values_list('harvest_id', 'tractor_id', 'percent'):
        tr_map.setdefault(ht[0], {})[ht[1]] = ht[2]

    rows = []
    ops = (Harvest.objects
           .select_related('parcel__region', 'crew', 'product')
           .order_by('-date', 'id'))
    for op in ops.iterator():
        mass_q = float(op.mass_q)
        sp_pcts = sp_map.get(op.id, {})
        tr_pcts = tr_map.get(op.id, {})

        # Quintal columns = total * percent / 100, rounded to 2 decimals.
        sp_quintals = [round(mass_q * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
        tr_quintals = [round(mass_q * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

        sp_pct_vals = [sp_pcts.get(sid, 0) for sid in species_ids]
        tr_pct_vals = [tr_pcts.get(tid, 0) for tid in tractor_ids]

        row = (
            [op.id, op.version, op.date.isoformat(),
             op.parcel.region.name, op.parcel.name,
             op.harvest_plan_item_id if op.harvest_plan_item_id else '',
             op.crew.name, op.record1, op.product.name, mass_q,
             float(op.volume_m3),
             render_flag_note(op.damaged, op.unhealthy, op.psr), op.note]
            + sp_quintals + tr_quintals
            + sp_pct_vals + tr_pct_vals
        )
        rows.append(row)

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('prelievi'))
    print(f'prelievi.json.gz: {len(rows)} rows, {len(columns)} columns')


def build_harvest_record(op) -> list:
    """Build a single prelievi digest row.  Used by save views for cache sync.

    The caller must ensure *op* has ``parcel.region``, ``crew``, and
    ``product`` loaded (via ``select_related``).
    """
    from apps.base.models import Species, Tractor
    from apps.prelievi.models import HarvestSpecies, HarvestTractor

    species_ids = list(Species.objects.order_by(FIELD_SORT_ORDER).values_list('id', flat=True))
    tractor_ids = list(Tractor.objects.order_by('manufacturer', 'model').values_list('id', flat=True))

    sp_pcts = dict(HarvestSpecies.objects.filter(harvest=op).values_list(FIELD_SPECIES_ID, 'percent'))
    tr_pcts = dict(HarvestTractor.objects.filter(harvest=op).values_list('tractor_id', 'percent'))

    mass_q = float(op.mass_q)
    sp_quintals = [round(mass_q * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
    tr_quintals = [round(mass_q * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

    return (
        [op.id, op.version, op.date.isoformat(),
         op.parcel.region.name, op.parcel.name,
         op.harvest_plan_item_id if op.harvest_plan_item_id else '',
         op.crew.name, op.record1, op.product.name, mass_q,
         float(op.volume_m3),
         render_flag_note(op.damaged, op.unhealthy, op.psr), op.note]
        + sp_quintals + tr_quintals
        + [sp_pcts.get(sid, 0) for sid in species_ids]
        + [tr_pcts.get(tid, 0) for tid in tractor_ids]
    )


# ---------------------------------------------------------------------------
# Parcels digest
# ---------------------------------------------------------------------------

def generate_parcels() -> None:
    from apps.base.models import Parcel

    columns = [ROW_ID, S.COL_COMPRESA, S.COL_PARCEL, S.COL_CLASS,
               S.COL_AREA_HA, S.COL_AVE_AGE, S.COL_LOCATION,
               S.COL_ALT_MIN, S.COL_ALT_MAX,
               S.COL_ASPECT, S.COL_GRADE_PCT]
    rows = []
    for p in Parcel.objects.select_related('region', 'eclass').order_by('region__name', 'name'):
        rows.append([
            p.id, p.region.name, p.name, p.eclass.name, float(p.area_ha),
            p.ave_age, p.location_name, p.altitude_min_m, p.altitude_max_m,
            p.aspect, p.grade_pct,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('parcels'))
    print(f'parcels.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Crews digest
# ---------------------------------------------------------------------------

def generate_crews() -> None:
    from apps.base.models import Crew

    columns = [ROW_ID, S.COL_NAME, S.COL_NOTE, S.COL_ACTIVE]
    rows = []
    for c in Crew.objects.order_by('name'):
        rows.append([c.id, c.name, c.notes, c.active])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('crews'))
    print(f'crews.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Species digest
# ---------------------------------------------------------------------------

def generate_species() -> None:
    """Lookup digest used by JS forms (V/m preview) and by other digest
    generators that need per-species density."""
    from apps.base.models import Species

    columns = [ROW_ID, VERSION, S.COL_NAME, S.COL_LATIN_NAME,
               S.COL_DENSITY, S.COL_SORT_ORDER, S.COL_ACTIVE]
    rows = []
    for sp in Species.objects.order_by(FIELD_SORT_ORDER):
        rows.append([
            sp.id, sp.version, sp.common_name, sp.latin_name,
            float(sp.density), sp.sort_order, sp.active,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest(FIELD_SPECIES))
    print(f'species.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Parcel-year production digest
# ---------------------------------------------------------------------------

def generate_parcel_year_production() -> None:
    """SELECT region, parcel, year, SUM(mass_q), SUM(volume_m3)
    GROUP BY region, parcel, year."""
    from apps.prelievi.models import Harvest

    columns = [S.COL_COMPRESA, S.COL_PARCEL, S.COL_YEAR,
               S.COL_QUINTALS, S.COL_VOLUME_M3]
    qs = (Harvest.objects
          .values('parcel__region__name', 'parcel__name', 'date__year')
          .annotate(total_q=Sum('mass_q'), total_v=Sum(FIELD_VOLUME_M3))
          .order_by('parcel__region__name', 'parcel__name', 'date__year'))

    rows = []
    for r in qs:
        rows.append([
            r['parcel__region__name'], r['parcel__name'],
            r['date__year'], float(r['total_q']), float(r['total_v']),
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('parcel_year_production'))
    print(f'parcel_year_production.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Audit digest
# ---------------------------------------------------------------------------

_ACTION_MAP = {'+': S.ACTION_INSERT, '~': S.ACTION_UPDATE, '-': S.ACTION_DELETE}


def generate_audit() -> None:
    """Audit log from django-simple-history across all tracked models."""
    from apps.base.models import Crew, Species, Tractor, User
    from apps.prelievi.models import Harvest

    configs = [
        (Harvest.history, S.TABLE_HARVEST, {
            'date': S.COL_DATE, 'parcel_id': S.COL_PARCEL,
            'crew_id': S.COL_CREW, 'product_id': S.COL_PRODUCT,
            'mass_q': S.COL_QUINTALS, FIELD_VOLUME_M3: S.COL_VOLUME_M3,
            'record1': S.COL_VDP,
            'record2': S.COL_PROT,
            'damaged': S.FLAG_DAMAGED, 'unhealthy': S.FLAG_UNHEALTHY,
            'psr': S.FLAG_PSR, 'note': S.COL_EXTRA_NOTE,
            'harvest_plan_item_id': S.COL_HARVEST_PLAN,
        }),
        (User.history, S.TABLE_USER, {
            'username': S.LABEL_USERNAME, 'role': S.LABEL_ROLE,
            'is_active': S.COL_ACTIVE,
        }),
        (Crew.history, S.TABLE_CREW, {
            'name': S.LABEL_NAME, 'notes': S.LABEL_NOTES,
            'active': S.COL_ACTIVE,
        }),
        (Tractor.history, S.TABLE_TRACTOR, {
            'manufacturer': S.LABEL_MANUFACTURER,
            'model': S.LABEL_MODEL, 'year': S.LABEL_YEAR,
            'active': S.COL_ACTIVE,
        }),
        (Species.history, S.TABLE_SPECIES, {
            'common_name': S.LABEL_NAME,
            'latin_name': S.LABEL_LATIN_NAME,
            'density': S.LABEL_DENSITY,
            'active': S.COL_ACTIVE,
        }),
    ]

    rows = []
    row_id = 0
    for manager, table_label, field_labels in configs:
        row_id = _audit_rows(manager, table_label, field_labels, rows, row_id)

    rows.sort(key=lambda r: r[1], reverse=True)

    columns = [ROW_ID, S.COL_TIMESTAMP, S.COL_USER, S.COL_TABLE,
               S.COL_ACTION, S.COL_OLD_VALUE, S.COL_NEW_VALUE]
    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('audit'))
    print(f'audit.json.gz: {len(rows)} rows')


def _audit_rows(manager, table_label, field_labels, rows, row_id):
    """Append audit rows for one historical model.  Returns updated row_id."""
    prev_map: dict[int, object] = {}

    for entry in manager.select_related('history_user').order_by('history_date', 'history_id'):
        pk = entry.id
        ht = entry.history_type
        user = ''
        if entry.history_user:
            u = entry.history_user
            user = f'{u.first_name} {u.last_name}'.strip() or u.username

        ts = entry.history_date.strftime('%Y-%m-%d %H:%M')
        action = _ACTION_MAP.get(ht, ht)

        if ht == '+':
            before, after = '', _format_fields(entry, field_labels)
        elif ht == '~' and pk in prev_map:
            before, after = _format_diff(prev_map[pk], entry, field_labels)
        elif ht == '-':
            before, after = _format_fields(entry, field_labels), ''
        else:
            before, after = '', _format_fields(entry, field_labels)

        row_id += 1
        rows.append([row_id, ts, user, table_label, action, before, after])
        prev_map[pk] = entry

    return row_id


def _format_fields(entry, field_labels: dict) -> str:
    """Format all tracked fields as 'label: value; ...'."""
    parts = []
    for field, label in field_labels.items():
        val = getattr(entry, field, None)
        if val is not None and val != '':
            parts.append(f'{label}: {val}')
    return '; '.join(parts)


def _format_diff(prev, current, field_labels: dict) -> tuple[str, str]:
    """Format only the changed fields as before/after strings."""
    before_parts, after_parts = [], []
    for field, label in field_labels.items():
        old = getattr(prev, field, None)
        new = getattr(current, field, None)
        if old != new:
            before_parts.append(f'{label}: {old if old is not None else ""}')
            after_parts.append(f'{label}: {new if new is not None else ""}')
    return '; '.join(before_parts), '; '.join(after_parts)


# ---------------------------------------------------------------------------
# Campionamenti digests
# ---------------------------------------------------------------------------

GRID_COLUMNS = [ROW_ID, VERSION, S.COL_NAME, S.COL_DESCRIPTION,
                S.COL_N_AREAS, S.COL_REGIONS, S.COL_N_SURVEYS,
                S.COL_LAST_UPDATE]


def build_grid_record(g) -> list:
    """Build one row of the `grids` digest from a SampleGrid instance.

    Shared between the generator and write views so the column shape is
    locked in one place (see CLAUDE.md §"Optimistic table updates").
    """
    from apps.base.models import SampleArea, Survey

    areas = SampleArea.objects.filter(sample_grid=g) \
                              .select_related('parcel__region')
    n_aree = areas.count()
    comprese = sorted({sa.parcel.region.name for sa in areas})
    n_rilev = Survey.objects.filter(sample_grid=g).count()
    last_area = areas.order_by('-modified_at').values_list(
        'modified_at', flat=True).first()
    last_updated = max(filter(None, [g.modified_at, last_area]))
    return [
        g.id, g.version, g.name, g.description, n_aree,
        ', '.join(comprese), n_rilev,
        last_updated.isoformat() if last_updated else '',
    ]


def generate_grids() -> None:
    """List of sample grids, with sample-area counts and region coverage.

    Drives Section 1 (Griglie) pulldown and summary line.
    """
    from apps.base.models import SampleGrid

    rows = [build_grid_record(g)
            for g in SampleGrid.objects.order_by('-modified_at')]
    _write_gzip_json(
        {'columns': GRID_COLUMNS, 'rows': rows}, _dest('grids'),
    )
    print(f'grids.json.gz: {len(rows)} rows')


SURVEY_COLUMNS = [ROW_ID, VERSION, S.COL_NAME, S.COL_DESCRIPTION,
                  S.COL_GRID, S.COL_HARVEST_PLAN,
                  S.COL_N_AREAS_VISITED, S.COL_N_AREAS_TOTAL,
                  S.COL_DATE_FIRST, S.COL_DATE_LAST]


def build_survey_record(s) -> list:
    """Build one row of the `surveys` digest.

    Re-computes the per-survey aggregates (n_visited, first/last date,
    n_total).  Cheap (one Sample query + one SampleArea count) compared
    to a full digest regen.  Used by save views after a tree or sample
    write changes any of those values.
    """
    from django.db.models import Count, Max, Min
    from apps.base.models import Sample, SampleArea

    agg = Sample.objects.filter(survey=s).aggregate(
        n_visited=Count('sample_area', distinct=True),
        first_date=Min('date'),
        last_date=Max('date'),
    )
    n_total = SampleArea.objects.filter(sample_grid_id=s.sample_grid_id).count()
    return [
        s.id, s.version, s.name, s.description,
        s.sample_grid_id,
        s.harvest_plan_id if s.harvest_plan_id else '',
        agg['n_visited'] or 0,
        n_total,
        agg[FIELD_FIRST_DATE].isoformat() if agg[FIELD_FIRST_DATE] else '',
        agg[FIELD_LAST_DATE].isoformat() if agg[FIELD_LAST_DATE] else '',
    ]


def generate_surveys() -> None:
    """List of surveys with completeness counts and date range.

    Drives Section 2 (Rilevamenti) pulldown.
    """
    from django.db.models import Count, Max, Min
    from apps.base.models import SampleArea, Survey

    # Same shape as build_survey_record but in one query for the
    # whole-table generator path (build_survey_record's per-row
    # aggregates would N+1).
    totals_by_grid = dict(
        SampleArea.objects
            .values('sample_grid_id')
            .annotate(n=Count('id'))
            .values_list('sample_grid_id', 'n'),
    )
    rows = []
    qs = Survey.objects.select_related('sample_grid', 'harvest_plan') \
                       .annotate(
                           n_visited=Count('sample__sample_area', distinct=True),
                           first_date=Min('sample__date'),
                           last_date=Max('sample__date'),
                       ) \
                       .order_by('-last_date', '-created_at')
    for s in qs:
        rows.append([
            s.id, s.version, s.name, s.description,
            s.sample_grid_id,
            s.harvest_plan_id if s.harvest_plan_id else '',
            s.n_visited or 0,
            totals_by_grid.get(s.sample_grid_id, 0),
            s.first_date.isoformat() if s.first_date else '',
            s.last_date.isoformat() if s.last_date else '',
        ])

    _write_gzip_json(
        {'columns': SURVEY_COLUMNS, 'rows': rows}, _dest('surveys'),
    )
    print(f'surveys.json.gz: {len(rows)} rows')


SAMPLE_AREA_COLUMNS = [ROW_ID, VERSION, S.COL_GRID, S.COL_COMPRESA,
                       S.COL_PARCEL, S.COL_NUMBER, S.COL_LAT, S.COL_LON,
                       S.COL_QUOTA, S.COL_RAGGIO, S.COL_NOTE]


def build_sample_area_record(sa) -> list:
    """Build one row of the `sample_areas` digest.  Caller must
    pre-load `parcel.region` (select_related).
    """
    return [
        sa.id, sa.version, sa.sample_grid_id,
        sa.parcel.region.name, sa.parcel.name, sa.number,
        sa.lat, sa.lon, sa.altitude_m, sa.r_m, sa.note,
    ]


def generate_sample_areas() -> None:
    """All sample-area rows across all grids.

    Drives Section 1 + Section 2 maps (filtered client-side by grid).
    """
    from apps.base.models import SampleArea

    rows = [
        build_sample_area_record(sa)
        for sa in (SampleArea.objects
                   .select_related('parcel__region', 'sample_grid')
                   .order_by('sample_grid__name',
                             'parcel__region__name',
                             'parcel__name', FIELD_NUMBER))
    ]
    _write_gzip_json(
        {'columns': SAMPLE_AREA_COLUMNS, 'rows': rows},
        _dest('sample_areas'),
    )
    print(f'sample_areas.json.gz: {len(rows)} rows')


SAMPLE_COLUMNS = [ROW_ID, VERSION, S.COL_SURVEY, S.COL_SAMPLE_AREA,
                  S.COL_DATE, S.COL_N_TREES]


def build_sample_record(s, n_alberi: int | None = None) -> list:
    """Build one row of the `samples` digest.

    Pass `n_alberi` when the caller already knows the count to avoid a
    re-query; otherwise we compute it from the TreeSample table.
    """
    from apps.base.models import TreeSample

    if n_alberi is None:
        n_alberi = TreeSample.objects.filter(sample=s).count()
    return [
        s.id, s.version, s.survey_id, s.sample_area_id,
        s.date.isoformat(), n_alberi or 0,
    ]


def generate_samples() -> None:
    """All sample visits with materialized tree counts.

    Drives Section 2 hover tooltips ("N. alberi") and visited-vs-unvisited
    map coloring (presence of a row = visited).
    """
    from django.db.models import Count
    from apps.base.models import Sample

    rows = []
    qs = Sample.objects.annotate(n_alberi=Count('treesample')) \
                       .order_by(FIELD_SURVEY_ID, FIELD_SAMPLE_AREA_ID)
    for s in qs:
        rows.append(build_sample_record(s, n_alberi=s.n_alberi or 0))

    _write_gzip_json(
        {'columns': SAMPLE_COLUMNS, 'rows': rows}, _dest('samples'),
    )
    print(f'samples.json.gz: {len(rows)} rows')


SAMPLED_TREE_COLUMNS = [ROW_ID, VERSION, S.COL_SAMPLE_AREA,
                        S.COL_SAMPLE_DATE, S.COL_COMPRESA, S.COL_PARCEL,
                        S.COL_AREA_NUM, S.COL_TREE_NUM,
                        S.COL_SPECIES, S.COL_PRODUCT, S.COL_POLLONE,
                        S.COL_MATRICINA, S.COL_D_CM, S.COL_H_M, S.COL_L10_MM,
                        S.COL_V_M3, S.COL_MASS_Q,
                        S.COL_PAI, S.COL_LAT, S.COL_LON]


def build_tree_sample_record(ts) -> list:
    """Build one row of the `sampled_trees_<survey>` digest.

    Caller must pre-load `sample.sample_area.parcel.region` and
    `tree.species` (select_related).
    """
    tree = ts.tree
    sa = ts.sample.sample_area
    return [
        ts.id, ts.version, sa.id, ts.sample.date.isoformat(),
        sa.parcel.region.name, sa.parcel.name, sa.number,
        ts.number, tree.species.common_name,
        'ceduo' if tree.coppice else 'fustaia',
        ts.shoot, ts.standard,
        ts.d_cm, float(ts.h_m), ts.l10_mm,
        float(ts.volume_m3) if ts.volume_m3 is not None else None,
        float(ts.mass_q) if ts.mass_q is not None else None,
        tree.preserved,
        tree.lat if tree.lat is not None else sa.lat,
        tree.lon if tree.lon is not None else sa.lon,
    ]


def generate_sampled_trees_for_survey(survey_id: int) -> None:
    """Per-survey tree-sample digest.  Lazy on Section 3 / Bosco overlay.

    Filename: sampled_trees_<survey_id>.json.gz.  Invalidated by
    tree_sample writes whose sample.survey_id matches.
    """
    from apps.base.models import TreeSample

    qs = (TreeSample.objects
          .filter(sample__survey_id=survey_id)
          .select_related('sample', 'sample__sample_area__parcel__region',
                          'tree__species', 'tree__parcel')
          .order_by('sample__sample_area__parcel__region__name',
                    'sample__sample_area__parcel__name',
                    'sample__sample_area__number', FIELD_NUMBER, FIELD_SHOOT))
    rows = [build_tree_sample_record(ts) for ts in qs]
    _write_gzip_json(
        {'columns': SAMPLED_TREE_COLUMNS, 'rows': rows},
        _dest(f'sampled_trees_{survey_id}'),
    )
    print(f'sampled_trees_{survey_id}.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Piano di taglio digests
# ---------------------------------------------------------------------------

HARVEST_PLAN_COLUMNS = [ROW_ID, VERSION, S.COL_NAME, S.COL_DESCRIPTION,
                       S.COL_YEAR_START, S.COL_YEAR_END]


def build_harvest_plan_record(hp) -> list:
    """Build one row of the `harvest_plans` digest."""
    return [
        hp.id, hp.version, hp.name, hp.description,
        hp.year_start, hp.year_end,
    ]


def generate_harvest_plans() -> None:
    """All harvest plans.  Drives the plan-selector pulldown at the top
    of the Piano di taglio page.
    """
    from apps.base.models import HarvestPlan

    rows = [build_harvest_plan_record(hp)
            for hp in HarvestPlan.objects.order_by('-year_start', 'id')]
    _write_gzip_json(
        {'columns': HARVEST_PLAN_COLUMNS, 'rows': rows},
        _dest('harvest_plans'),
    )
    print(f'harvest_plans.json.gz: {len(rows)} rows')


HARVEST_PLAN_ITEM_COLUMNS = [
    ROW_ID, VERSION, S.COL_HARVEST_PLAN,
    S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL,
    S.COL_COMPRESA, S.COL_PARCEL, S.COL_TYPE, S.COL_STATE, S.COL_NOTE,
    S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED, S.COL_VOLUME_ACTUAL,
    S.COL_INTERVENTION_AREA_HA, S.COL_PARCEL_AREA_HA, S.COL_TURNO_A,
    S.COL_EXTRA_NOTE,
]


def _hpi_type(item) -> str:
    """`Tipo` value for the calendar Tipo column.  Empty for region-wide
    items (no Eclass to derive from).
    """
    if item.parcel_id is None:
        return ''
    return S.TYPE_CEDUO if item.parcel.eclass.coppice else S.TYPE_FUSTAIA


def _hpi_turno(item) -> int | str:
    """Coppice rotation interval (years) read via ParcelPlanDetail.

    Returns '' for fustaia items or coppice items without a linked
    ParcelPlanDetail row.  The (plan, parcel) → harvest_detail link is
    optional (only coppice parcels get one — see piano-di-taglio.md
    "Import flow").
    """
    if item.parcel_id is None or not item.parcel.eclass.coppice:
        return ''
    from apps.base.models import ParcelPlanDetail
    ppd = (ParcelPlanDetail.objects
           .filter(harvest_plan_id=item.harvest_plan_id, parcel_id=item.parcel_id)
           .select_related('harvest_detail')
           .first())
    if ppd is None or ppd.harvest_detail.interval is None:
        return ''
    return ppd.harvest_detail.interval


def build_harvest_plan_item_record(item) -> list:
    """Build one row of the `harvest_plan_items` digest.

    Caller must pre-load `parcel.region`, `parcel.eclass`, and `region`
    (the FK directly on the item).  Reads `state` via
    `get_state_display()` for the human-readable label.
    """
    if item.parcel_id is not None:
        compresa = item.parcel.region.name
        particella = item.parcel.name
        parcel_area = float(item.parcel.area_ha)
    else:
        compresa = item.region.name
        particella = S.LABEL_ALL_PARCELS
        parcel_area = ''

    return [
        item.id, item.version, item.harvest_plan_id,
        item.year_planned,
        item.date_actual.year if item.date_actual else '',
        compresa, particella, _hpi_type(item),
        item.get_state_display(),
        render_flag_note(item.damaged, item.unhealthy, item.psr),
        float(item.volume_planned_m3) if item.volume_planned_m3 is not None else '',
        float(item.volume_marked_m3) if item.volume_marked_m3 is not None else '',
        float(item.volume_actual_m3),
        float(item.intervention_area_ha) if item.intervention_area_ha is not None else '',
        parcel_area,
        _hpi_turno(item),
        item.note,
    ]


def generate_harvest_plan_items() -> None:
    """All HarvestPlanItem rows across all plans.

    Client filters by selected plan (`p=N` URL param) and by Tipo to
    drive the fustaia vs ceduo calendar sections.  Per-item lazy
    `mark_trees_<id>` digests carry the per-item tree_marks.
    """
    from apps.base.models import HarvestPlanItem

    qs = (HarvestPlanItem.objects
          .select_related('parcel__region', 'parcel__eclass', 'region',
                          'harvest_plan')
          .order_by('harvest_plan_id', 'year_planned', 'id'))
    rows = [build_harvest_plan_item_record(it) for it in qs]
    _write_gzip_json(
        {'columns': HARVEST_PLAN_ITEM_COLUMNS, 'rows': rows},
        _dest('harvest_plan_items'),
    )
    print(f'harvest_plan_items.json.gz: {len(rows)} rows')


TREE_HEIGHT_REGRESSION_COLUMNS = [
    ROW_ID, S.COL_HARVEST_PLAN, S.COL_COMPRESA, S.COL_SPECIES,
    S.COL_FUNCTION, S.COL_A, S.COL_B, S.COL_R2, S.COL_N_REGRESSION,
]


def build_tree_height_regression_record(thr) -> list:
    """Build one row of the `tree_height_regressions` digest."""
    return [
        thr.id, thr.harvest_plan_id,
        thr.region.name, thr.species.common_name,
        thr.function, float(thr.a), float(thr.b),
        float(thr.r2), thr.n,
    ]


def generate_tree_height_regressions() -> None:
    """Per-(plan, region, species) ipsometric regression coefficients.

    Consumed JS-side by the Nuovo albero martellato modal to auto-fill
    `h_m` from `d_cm`.  Filtered client-side by active plan.
    """
    from apps.base.models import TreeHeightRegression

    qs = (TreeHeightRegression.objects
          .select_related('region', 'species')
          .order_by('harvest_plan_id', 'region__name', 'species__common_name'))
    rows = [build_tree_height_regression_record(thr) for thr in qs]
    _write_gzip_json(
        {'columns': TREE_HEIGHT_REGRESSION_COLUMNS, 'rows': rows},
        _dest('tree_height_regressions'),
    )
    print(f'tree_height_regressions.json.gz: {len(rows)} rows')


MARK_TREE_COLUMNS = [ROW_ID, VERSION, S.COL_DATE, S.COL_NUMERO,
                     S.COL_SPECIES, S.COL_D_CM, S.COL_H_M, S.COL_H_MEASURED,
                     S.COL_V_M3, S.COL_MASS_Q,
                     S.COL_LAT, S.COL_LON, S.COL_OPERATOR]


def build_tree_mark_record(tm, numero: int) -> list:
    """Build one row of a `mark_trees_<item_id>` digest.

    `numero` is the 1-based display index within the item, computed by
    the caller (not stored on tree_mark).  Caller must pre-load
    `tree.species`.
    """
    return [
        tm.id, tm.version, tm.date.isoformat(), numero,
        tm.tree.species.common_name,
        tm.d_cm, float(tm.h_m), tm.h_measured,
        float(tm.volume_m3), float(tm.mass_q),
        tm.lat, tm.lon, tm.operator,
    ]


def generate_mark_trees_for_item(item_id: int) -> None:
    """Per-harvest_plan_item tree_mark digest.

    Filename: `mark_trees_<item_id>.json.gz`.  Lazy-loaded by the
    View/Edit-item modal's Martellate section.  Pattern mirrors
    `sampled_trees_<survey_id>` in campionamenti.

    Sort: Data desc, then id (tie-break).  `Numero` is a 1-based
    sequence computed at generation time.
    """
    from apps.base.models import TreeMark

    qs = (TreeMark.objects
          .filter(harvest_plan_item_id=item_id)
          .select_related('tree__species')
          .order_by('-date', 'id'))
    rows = []
    for i, tm in enumerate(qs, start=1):
        rows.append(build_tree_mark_record(tm, numero=i))
    _write_gzip_json(
        {'columns': MARK_TREE_COLUMNS, 'rows': rows},
        _dest(f'mark_trees_{item_id}'),
    )
    print(f'mark_trees_{item_id}.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, callable] = {
    'prelievi': generate_prelievi,
    'parcels': generate_parcels,
    'crews': generate_crews,
    FIELD_SPECIES: generate_species,
    'parcel_year_production': generate_parcel_year_production,
    'audit': generate_audit,
    'grids': generate_grids,
    'surveys': generate_surveys,
    'sample_areas': generate_sample_areas,
    'samples': generate_samples,
    'harvest_plans': generate_harvest_plans,
    'harvest_plan_items': generate_harvest_plan_items,
    'tree_height_regressions': generate_tree_height_regressions,
}


def mark_all_stale() -> None:
    """Flag every digest as stale.

    Used by the bulk ETL importers, which bypass the views that
    normally call `mark_stale()` on writes.  The next read of each
    digest will trigger lazy regeneration.  Also covers existing
    `sampled_trees_<survey_id>` rows in DigestStatus so per-survey
    digests get refreshed on next read.
    """
    DigestStatus.objects.update(stale=True)
    for name in _GENERATORS:
        DigestStatus.objects.update_or_create(
            name=name, defaults={'stale': True},
        )


def generate_all() -> None:
    """Regenerate every digest (used by `make digest`).

    Also produces per-survey `sampled_trees_<id>.json` for every
    existing Survey and per-item `mark_trees_<id>.json` for every
    HarvestPlanItem that has at least one TreeMark, so the dev-time
    digest directory matches what the serving layer would lazily
    generate.
    """
    from apps.base.models import Survey, TreeMark

    for name, gen in _GENERATORS.items():
        gen()
    for survey_id in Survey.objects.values_list('id', flat=True):
        generate_sampled_trees_for_survey(survey_id)
    item_ids = (TreeMark.objects
                .values_list('harvest_plan_item_id', flat=True)
                .distinct())
    for item_id in item_ids:
        generate_mark_trees_for_item(item_id)
