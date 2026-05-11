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

from apps.base.models import DigestStatus
from config import strings as S


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


def _resolve_generator(name: str):
    """Look up a digest generator, including dynamic per-entity names.

    Supports static names registered in `_GENERATORS` plus the
    dynamic `sampled_trees_<survey_id>` pattern used by Campionamenti.
    Returns None for unknown names.
    """
    if name in _GENERATORS:
        return _GENERATORS[name]
    if name.startswith('sampled_trees_'):
        try:
            survey_id = int(name[len('sampled_trees_'):])
        except ValueError:
            return None
        return lambda: generate_sampled_trees_for_survey(survey_id)
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
    all_species = list(Species.objects.order_by('sort_order').values_list('id', 'common_name'))
    all_tractors = list(Tractor.objects.order_by('manufacturer', 'model')
                        .values_list('id', 'pk', 'manufacturer', 'model'))

    species_ids = [sid for sid, _ in all_species]
    species_names = [name for _, name in all_species]

    tractor_ids = [tid for tid, _, _, _ in all_tractors]
    tractor_labels = [
        f'{mfr} {mdl}'.strip() for _, _, mfr, mdl in all_tractors
    ]

    columns = (
        ['row_id', 'version', 'Data', 'Compresa', 'Particella', 'Squadra',
         'VDP', 'Tipo', 'Q.li', 'Volume (m³)', 'Note', 'Altre note']
        + species_names
        + tractor_labels
        + [f'{n} %' for n in species_names]
        + [f'{l} %' for l in tractor_labels]
    )

    # Prefetch junction tables into dicts keyed by harvest_id.
    sp_map: dict[int, dict[int, int]] = {}
    for hs in HarvestSpecies.objects.all().values_list('harvest_id', 'species_id', 'percent'):
        sp_map.setdefault(hs[0], {})[hs[1]] = hs[2]

    tr_map: dict[int, dict[int, int]] = {}
    for ht in HarvestTractor.objects.all().values_list('harvest_id', 'tractor_id', 'percent'):
        tr_map.setdefault(ht[0], {})[ht[1]] = ht[2]

    rows = []
    ops = (Harvest.objects
           .select_related('parcel__region', 'crew', 'note', 'product')
           .order_by('-date', 'id'))
    for op in ops.iterator():
        quintals = float(op.quintals)
        sp_pcts = sp_map.get(op.id, {})
        tr_pcts = tr_map.get(op.id, {})

        # Quintal columns = total * percent / 100, rounded to 2 decimals.
        sp_quintals = [round(quintals * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
        tr_quintals = [round(quintals * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

        sp_pct_vals = [sp_pcts.get(sid, 0) for sid in species_ids]
        tr_pct_vals = [tr_pcts.get(tid, 0) for tid in tractor_ids]

        row = (
            [op.id, op.version, op.date.isoformat(),
             op.parcel.region.name, op.parcel.name,
             op.crew.name, op.record1, op.product.name, quintals,
             float(op.volume_m3),
             op.note.name if op.note else '', op.extra_note]
            + sp_quintals + tr_quintals
            + sp_pct_vals + tr_pct_vals
        )
        rows.append(row)

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('prelievi'))
    print(f'prelievi.json.gz: {len(rows)} rows, {len(columns)} columns')


def build_harvest_record(op) -> list:
    """Build a single prelievi digest row.  Used by save views for cache sync.

    The caller must ensure *op* has ``parcel.region``, ``crew``, ``note``,
    and ``product`` loaded (via ``select_related``).
    """
    from apps.base.models import Species, Tractor
    from apps.prelievi.models import HarvestSpecies, HarvestTractor

    species_ids = list(Species.objects.order_by('sort_order').values_list('id', flat=True))
    tractor_ids = list(Tractor.objects.order_by('manufacturer', 'model').values_list('id', flat=True))

    sp_pcts = dict(HarvestSpecies.objects.filter(harvest=op).values_list('species_id', 'percent'))
    tr_pcts = dict(HarvestTractor.objects.filter(harvest=op).values_list('tractor_id', 'percent'))

    quintals = float(op.quintals)
    sp_quintals = [round(quintals * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
    tr_quintals = [round(quintals * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

    return (
        [op.id, op.version, op.date.isoformat(),
         op.parcel.region.name, op.parcel.name,
         op.crew.name, op.record1, op.product.name, quintals,
         float(op.volume_m3),
         op.note.name if op.note else '', op.extra_note]
        + sp_quintals + tr_quintals
        + [sp_pcts.get(sid, 0) for sid in species_ids]
        + [tr_pcts.get(tid, 0) for tid in tractor_ids]
    )


# ---------------------------------------------------------------------------
# Parcels digest
# ---------------------------------------------------------------------------

def generate_parcels() -> None:
    from apps.base.models import Parcel

    columns = ['row_id', 'Compresa', 'Particella', 'Classe', 'Area (ha)',
               'Età media', 'Località', 'Alt. min', 'Alt. max',
               'Esposizione', 'Pendenza %']
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

    columns = ['row_id', 'Nome', 'Note', 'Attiva']
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

    columns = ['row_id', 'version', 'Nome', 'Nome latino', 'Densità (q/m³)',
               'Sort order', 'Attiva']
    rows = []
    for sp in Species.objects.order_by('sort_order'):
        rows.append([
            sp.id, sp.version, sp.common_name, sp.latin_name,
            float(sp.density), sp.sort_order, sp.active,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('species'))
    print(f'species.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Parcel-year production digest
# ---------------------------------------------------------------------------

def generate_parcel_year_production() -> None:
    """SELECT region, parcel, year, SUM(quintals), SUM(volume_m3)
    GROUP BY region, parcel, year."""
    from apps.prelievi.models import Harvest

    columns = ['Compresa', 'Particella', 'Anno', 'Q.li', 'Volume (m³)']
    qs = (Harvest.objects
          .values('parcel__region__name', 'parcel__name', 'date__year')
          .annotate(total_q=Sum('quintals'), total_v=Sum('volume_m3'))
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
            'quintals': S.COL_QUINTALS, 'volume_m3': S.COL_VOLUME_M3,
            'record1': S.COL_VDP,
            'record2': S.COL_PROT, 'note_id': S.COL_NOTE,
            'extra_note': S.COL_EXTRA_NOTE,
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

    columns = ['row_id', S.COL_TIMESTAMP, S.COL_USER, S.COL_TABLE,
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

def generate_grids() -> None:
    """List of sample grids, with sample-area counts and region coverage.

    Drives Section 1 (Griglie) pulldown and summary line.
    """
    from apps.base.models import SampleArea, SampleGrid, Survey

    columns = ['row_id', 'version', 'Nome', 'Descrizione', 'N. aree',
               'Comprese', 'N. rilevamenti', 'Ultimo aggiornamento']
    rows = []
    for g in SampleGrid.objects.order_by('-modified_at'):
        areas = SampleArea.objects.filter(sample_grid=g) \
                                  .select_related('parcel__region')
        n_aree = areas.count()
        comprese = sorted({sa.parcel.region.name for sa in areas})
        n_rilev = Survey.objects.filter(sample_grid=g).count()
        last_area = areas.order_by('-modified_at').values_list(
            'modified_at', flat=True).first()
        last_updated = max(filter(None, [g.modified_at, last_area]))
        rows.append([
            g.id, g.version, g.name, g.description, n_aree,
            ', '.join(comprese), n_rilev,
            last_updated.isoformat() if last_updated else '',
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('grids'))
    print(f'grids.json.gz: {len(rows)} rows')


def generate_surveys() -> None:
    """List of surveys with completeness counts and date range.

    Drives Section 2 (Rilevamenti) pulldown.
    """
    from django.db.models import Count, Max, Min
    from apps.base.models import Sample, SampleArea, Survey

    columns = ['row_id', 'version', 'Nome', 'Descrizione', 'Griglia',
               'Piano di taglio', 'N. aree visitate', 'N. aree totali',
               'Data primo', 'Data ultimo']

    # Pre-compute n_total per grid (count of SampleAreas in that grid).
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

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('surveys'))
    print(f'surveys.json.gz: {len(rows)} rows')


def generate_sample_areas() -> None:
    """All sample-area rows across all grids.

    Drives Section 1 + Section 2 maps (filtered client-side by grid).
    """
    from apps.base.models import SampleArea

    columns = ['row_id', 'version', 'Griglia', 'Compresa', 'Particella',
               'Numero', 'Lat', 'Lng', 'Quota', 'Raggio', 'Note']
    rows = []
    for sa in SampleArea.objects.select_related('parcel__region', 'sample_grid') \
                                .order_by('sample_grid__name',
                                          'parcel__region__name',
                                          'parcel__name', 'number'):
        rows.append([
            sa.id, sa.version, sa.sample_grid_id,
            sa.parcel.region.name, sa.parcel.name, sa.number,
            sa.lat, sa.lng, sa.altitude_m, sa.r_m, sa.note,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('sample_areas'))
    print(f'sample_areas.json.gz: {len(rows)} rows')


def generate_samples() -> None:
    """All sample visits with materialized tree counts.

    Drives Section 2 hover tooltips ("N. alberi") and visited-vs-unvisited
    map coloring (presence of a row = visited).
    """
    from django.db.models import Count
    from apps.base.models import Sample

    columns = ['row_id', 'version', 'Survey', 'Sample area', 'Data',
               'N. alberi']
    rows = []
    qs = Sample.objects.annotate(n_alberi=Count('treesample')) \
                       .order_by('survey_id', 'sample_area_id')
    for s in qs:
        rows.append([
            s.id, s.version, s.survey_id, s.sample_area_id,
            s.date.isoformat(), s.n_alberi or 0,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('samples'))
    print(f'samples.json.gz: {len(rows)} rows')


def generate_sampled_trees_for_survey(survey_id: int) -> None:
    """Per-survey tree-sample digest.  Lazy on Section 3 / Bosco overlay.

    Filename: sampled_trees_<survey_id>.json.gz.  Invalidated by
    tree_sample writes whose sample.survey_id matches.
    """
    from apps.base.models import TreeSample

    columns = ['row_id', 'version', 'Sample area', 'Data campione',
               'Compresa', 'Particella', 'N. area', 'N. albero',
               'Specie', 'Tipo', 'Pollone', 'Matricina',
               'D (cm)', 'h (m)', 'L10 (mm)', 'V (m³)', 'm (q)',
               'PAI', 'Lat', 'Lng']
    rows = []
    qs = (TreeSample.objects
          .filter(sample__survey_id=survey_id)
          .select_related('sample', 'sample__sample_area__parcel__region',
                          'tree__species', 'tree__parcel')
          .order_by('sample__sample_area__parcel__region__name',
                    'sample__sample_area__parcel__name',
                    'sample__sample_area__number', 'number', 'shoot'))
    for ts in qs:
        tree = ts.tree
        sa = ts.sample.sample_area
        rows.append([
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
            tree.lng if tree.lng is not None else sa.lng,
        ])

    _write_gzip_json(
        {'columns': columns, 'rows': rows},
        _dest(f'sampled_trees_{survey_id}'),
    )
    print(f'sampled_trees_{survey_id}.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, callable] = {
    'prelievi': generate_prelievi,
    'parcels': generate_parcels,
    'crews': generate_crews,
    'species': generate_species,
    'parcel_year_production': generate_parcel_year_production,
    'audit': generate_audit,
    'grids': generate_grids,
    'surveys': generate_surveys,
    'sample_areas': generate_sample_areas,
    'samples': generate_samples,
}


def generate_all() -> None:
    """Regenerate every digest (used by `make digest`).

    Also produces per-survey `sampled_trees_<id>.json` for every
    existing Survey, so the dev-time digest directory matches what the
    serving layer would lazily generate.
    """
    from apps.base.models import Survey

    for name, gen in _GENERATORS.items():
        gen()
    for survey_id in Survey.objects.values_list('id', flat=True):
        generate_sampled_trees_for_survey(survey_id)
