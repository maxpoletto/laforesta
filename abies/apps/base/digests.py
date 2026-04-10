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


def regenerate_if_stale(name: str) -> Path:
    """Return the path to *name*'s digest, regenerating first if stale."""
    dest = _dest(name)
    status, _ = DigestStatus.objects.get_or_create(name=name)
    if status.stale or not dest.exists():
        _GENERATORS[name]()
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

    Columns: row_id, Data, Compresa, Particella, Squadra, VDP, Q.li, Note,
    Altre note, then one quintal column per species (alphabetical), then one
    quintal column per tractor (alphabetical).  Also carries percentage values
    for form pre-population (suffixed with " %").
    """
    from apps.base.models import Species, Tractor
    from apps.prelievi.models import HarvestOp, HarvestSpecies, HarvestTractor

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
        ['row_id', 'Data', 'Compresa', 'Particella', 'Squadra', 'VDP',
         'Q.li', 'Note', 'Altre note']
        + species_names
        + tractor_labels
        + [f'{n} %' for n in species_names]
        + [f'{l} %' for l in tractor_labels]
    )

    # Prefetch junction tables into dicts keyed by harvest_op_id.
    sp_map: dict[int, dict[int, int]] = {}
    for hs in HarvestSpecies.objects.all().values_list('harvest_op_id', 'species_id', 'percent'):
        sp_map.setdefault(hs[0], {})[hs[1]] = hs[2]

    tr_map: dict[int, dict[int, int]] = {}
    for ht in HarvestTractor.objects.all().values_list('harvest_op_id', 'tractor_id', 'percent'):
        tr_map.setdefault(ht[0], {})[ht[1]] = ht[2]

    rows = []
    ops = (HarvestOp.objects
           .select_related('parcel__region', 'crew', 'note')
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
            [op.id, op.date.isoformat(), op.parcel.region.name, op.parcel.name,
             op.crew.name, op.record1, quintals,
             op.note.name if op.note else '', op.extra_note]
            + sp_quintals + tr_quintals
            + sp_pct_vals + tr_pct_vals
        )
        rows.append(row)

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('prelievi'))
    print(f'prelievi.json.gz: {len(rows)} rows, {len(columns)} columns')


def build_harvest_record(op) -> list:
    """Build a single prelievi digest row.  Used by save views for cache sync.

    The caller must ensure *op* has ``parcel.region``, ``crew``, and ``note``
    loaded (via ``select_related``).
    """
    from apps.base.models import Species, Tractor
    from apps.prelievi.models import HarvestSpecies, HarvestTractor

    species_ids = list(Species.objects.order_by('sort_order').values_list('id', flat=True))
    tractor_ids = list(Tractor.objects.order_by('manufacturer', 'model').values_list('id', flat=True))

    sp_pcts = dict(HarvestSpecies.objects.filter(harvest_op=op).values_list('species_id', 'percent'))
    tr_pcts = dict(HarvestTractor.objects.filter(harvest_op=op).values_list('tractor_id', 'percent'))

    quintals = float(op.quintals)
    sp_quintals = [round(quintals * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
    tr_quintals = [round(quintals * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

    return (
        [op.id, op.date.isoformat(), op.parcel.region.name, op.parcel.name,
         op.crew.name, op.record1, quintals,
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
# Parcel-year production digest
# ---------------------------------------------------------------------------

def generate_parcel_year_production() -> None:
    """SELECT region, parcel, year, SUM(quintals) GROUP BY region, parcel, year."""
    from apps.prelievi.models import HarvestOp

    columns = ['Compresa', 'Particella', 'Anno', 'Q.li']
    qs = (HarvestOp.objects
          .values('parcel__region__name', 'parcel__name', 'date__year')
          .annotate(total=Sum('quintals'))
          .order_by('parcel__region__name', 'parcel__name', 'date__year'))

    rows = []
    for r in qs:
        rows.append([
            r['parcel__region__name'], r['parcel__name'],
            r['date__year'], float(r['total']),
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('parcel_year_production'))
    print(f'parcel_year_production.json.gz: {len(rows)} rows')


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, callable] = {
    'prelievi': generate_prelievi,
    'parcels': generate_parcels,
    'crews': generate_crews,
    'parcel_year_production': generate_parcel_year_production,
}


def generate_all() -> None:
    """Regenerate every digest (used by `make digest`)."""
    for name, gen in _GENERATORS.items():
        gen()
