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
import logging
import math
import os
import tempfile
from pathlib import Path

from django.conf import settings
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models import F, Sum
from django.utils import timezone

from apps.base.http import CACHE_NO_STORE, conditional_file_response
from apps.base.models import DigestStatus, render_flag_note
from apps.base.selectors import (
    active_or_default_harvest_plan, active_or_default_survey_ids,
    height_plot_survey_ids,
)
from config import strings as S
from config.constants import (
    COL_COPPICE, COL_PARCEL_ID, COL_REGION_ID, COL_SPECIES_ID,
    COL_SURVEY_ID, COL_TREE_ID, DIGEST_FUTURE_PRODUCTION, DIGEST_HYPSO_PARAMS,
    DIGEST_PARCELS,
    DIGEST_PARCEL_DENDROMETRY, DIGEST_PARCEL_DENDROMETRY_POINTS,
    DIGEST_PRESERVED_TREES, FIELD_FIRST_DATE, FIELD_LAST_DATE, FIELD_NUMBER,
    FIELD_SAMPLE_AREA_ID, FIELD_SHOOT, FIELD_SORT_ORDER, FIELD_SPECIES,
    FIELD_SPECIES_ID, FIELD_SURVEY_ID, FIELD_VOLUME_M3, M2_PER_HA,
    ROW_ID, VERSION,
)

_UNSET = object()
logger = logging.getLogger(__name__)


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
    """Mark one or more digests as needing regeneration.

    Bumps `dirty_seq` as well as setting `stale`, so a regeneration already
    in flight cannot clear the flag this mark sets: its compare-and-swap is
    keyed on the `dirty_seq` it snapshotted before generating (see
    `regenerate_if_stale`)."""
    for name in names:
        updated = DigestStatus.objects.filter(name=name).update(
            stale=True, dirty_seq=F('dirty_seq') + 1,
        )
        if not updated:
            _status, created = DigestStatus.objects.get_or_create(
                name=name, defaults={'stale': True, 'dirty_seq': 1},
            )
            if not created:
                DigestStatus.objects.filter(name=name).update(
                    stale=True, dirty_seq=F('dirty_seq') + 1,
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
        from apps.base.models import Survey
        if not Survey.objects.filter(pk=survey_id).exists():
            return None
        return lambda: generate_sampled_trees_for_survey(survey_id)
    if name.startswith(_DYNAMIC_PREFIX_MARK_TREES):
        try:
            item_id = int(name[len(_DYNAMIC_PREFIX_MARK_TREES):])
        except ValueError:
            return None
        from apps.base.models import HarvestPlanItem
        if not HarvestPlanItem.objects.filter(pk=item_id).exists():
            return None
        return lambda: generate_mark_trees_for_item(item_id)
    return None


def regenerate_if_stale(name: str) -> Path:
    """Return the path to *name*'s digest, regenerating first if stale."""
    dest = _dest(name)
    gen = _resolve_generator(name)
    if gen is None:
        raise ValueError(f'unknown digest: {name!r}')
    status, _ = DigestStatus.objects.get_or_create(name=name)
    if status.stale or not dest.exists():
        seq = status.dirty_seq
        gen()
        # Compare-and-swap on the token snapshotted *before* generating: a
        # write that marked the digest stale while we were generating bumped
        # dirty_seq, so this filter misses and the digest stays stale,
        # forcing the next read to regenerate against the newer data.  The
        # file was renamed into place atomically before this clear, so a
        # crash here over-reports staleness but never serves a stale digest.
        DigestStatus.objects.filter(name=name, dirty_seq=seq).update(stale=False)
    return dest


def serve_digest(request, name: str):
    """Serve a named digest with conditional GET and lazy regeneration.

    ``no_store`` keeps the browser's HTTP cache out of the loop.  The app
    caches digests itself (in-memory, see ``cache.js``) and revalidates
    with an explicit ``If-Modified-Since``.  Without it, a ``Last-Modified``
    response with no ``Cache-Control`` is eligible for the browser's
    *heuristic* cache: after a write the in-memory cache is patched but a
    reload would serve the browser's stale copy without ever revalidating.
    The conditional GET below still answers 304 for unchanged digests, so
    suppressing the browser cache costs no bandwidth.
    """
    return conditional_file_response(
        request,
        regenerate_if_stale(name),
        content_type='application/json',
        content_encoding='gzip',
        cache_control=CACHE_NO_STORE,
    )


# ---------------------------------------------------------------------------
# Prelievi digest
# ---------------------------------------------------------------------------

def prelievi_species_cols():
    """Return (major_ids, major_names, minor_ids, other_id) for prelievi.

    Major species get their own columns; minor species are aggregated
    into an "Other" column in both the table and the input form.
    """
    from apps.base.models import Species

    all_sp = list(
        Species.objects.order_by(FIELD_SORT_ORDER)
        .values_list('id', 'common_name', 'minor')
    )
    major = [(sid, name) for sid, name, minor in all_sp if not minor]
    minor_ids = frozenset(sid for sid, _, minor in all_sp if minor)
    other_id = next(sid for sid, name in major if name == S.SPECIES_OTHER)
    return (
        [sid for sid, _ in major],
        [name for _, name in major],
        minor_ids,
        other_id,
    )


def aggregate_sp_pcts(sp_pcts, minor_ids, other_id):
    """Fold minor-species percentages into the Other bucket."""
    agg = {}
    for sid, pct in sp_pcts.items():
        target = other_id if sid in minor_ids else sid
        agg[target] = agg.get(target, 0) + pct
    return agg


def generate_prelievi() -> None:
    """De-normalized harvest table for the Prelievi sortable-table.

    Columns: row_id, version, Region id, Parcel id, Data, Compresa, Particella,
    Squadra, VDP, Tipo, Q.li, Volume (m³), Note, Altre note, then one quintal
    column per major species (sort_order), then one quintal column per tractor
    (alphabetical).  Minor species are aggregated into the Other (Altro)
    column.  Also carries percentage values for form pre-population (suffixed
    with " %").
    """
    from apps.base.models import Tractor
    from apps.prelievi.models import Harvest, HarvestSpecies, HarvestTractor

    species_ids, species_names, minor_ids, other_id = prelievi_species_cols()

    all_tractors = list(Tractor.objects.order_by('name', 'manufacturer', 'model', 'id'))

    tractor_ids = [tr.id for tr in all_tractors]
    tractor_labels = [tr.display_name for tr in all_tractors]

    columns = (
        [ROW_ID, VERSION, COL_REGION_ID, COL_PARCEL_ID,
         S.COL_DATE, S.COL_REGION, S.COL_PARCEL,
         S.COL_WORKSITE, S.COL_CREW, S.COL_VDP, S.COL_PRODUCT,
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
           .select_related('parcel__region', 'region', 'crew', 'product')
           .order_by('-date', 'id'))
    for op in ops.iterator():
        rows.append(build_harvest_record(
            op,
            species_ids=species_ids,
            tractor_ids=tractor_ids,
            sp_pcts=aggregate_sp_pcts(sp_map.get(op.id, {}), minor_ids, other_id),
            tr_pcts=tr_map.get(op.id, {}),
        ))

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('prelievi'))
    logger.info('prelievi.json.gz: %s rows, %s columns', len(rows), len(columns))


def build_harvest_record(
        op, *, species_ids=None, tractor_ids=None, sp_pcts=None, tr_pcts=None,
) -> list:
    """Build a single prelievi digest row.  Used by save views for cache sync.

    The caller must ensure *op* has ``parcel__region``, ``region``, ``crew``,
    and ``product`` loaded (via ``select_related``).  For parcel-level harvests
    ``region`` may be None; for region-wide harvests ``parcel`` may be None.
    Full-digest generation can pass precomputed column IDs and percentage maps
    to avoid per-row queries; write views can omit them and let this helper
    fetch just the one row's related percentages.
    """
    from apps.base.models import Tractor
    from apps.prelievi.models import HarvestSpecies, HarvestTractor

    if species_ids is None or sp_pcts is None:
        default_species_ids, _, minor_ids, other_id = prelievi_species_cols()
        if species_ids is None:
            species_ids = default_species_ids
    if tractor_ids is None:
        tractor_ids = list(
            Tractor.objects.order_by('name', 'manufacturer', 'model', 'id')
            .values_list('id', flat=True)
        )

    if sp_pcts is None:
        raw_sp = dict(
            HarvestSpecies.objects.filter(harvest=op)
            .values_list(FIELD_SPECIES_ID, 'percent')
        )
        sp_pcts = aggregate_sp_pcts(raw_sp, minor_ids, other_id)
    if tr_pcts is None:
        tr_pcts = dict(
            HarvestTractor.objects.filter(harvest=op)
            .values_list('tractor_id', 'percent')
        )

    mass_q = float(op.mass_q)
    sp_quintals = [round(mass_q * sp_pcts.get(sid, 0) / 100, 2) for sid in species_ids]
    tr_quintals = [round(mass_q * tr_pcts.get(tid, 0) / 100, 2) for tid in tractor_ids]

    # For region-wide harvests parcel_id is NULL; fall back to the region FK.
    region_id = op.parcel.region_id if op.parcel_id else op.region_id
    region_name = op.parcel.region.name if op.parcel_id else op.region.name
    parcel_name = op.parcel.name if op.parcel_id else S.PARCEL_WHOLE_REGION_MARK

    return (
        [op.id, op.version, region_id, op.parcel_id,
         op.date.isoformat(), region_name, parcel_name,
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

PARCEL_COLUMNS = [
    ROW_ID, VERSION, COL_REGION_ID, S.COL_REGION, S.COL_PARCEL, S.COL_CLASS,
    COL_COPPICE, S.COL_AREA_HA, S.COL_AREA_CAD_HA, S.COL_AVE_AGE,
    S.COL_LOCATION,
    S.COL_ALT_MIN, S.COL_ALT_MAX, S.COL_ASPECT, S.COL_GRADE_PCT,
    S.COL_TYPE, S.COL_DESC_VEG, S.COL_DESC_GEO,
]


def build_parcel_record(p) -> list:
    """Build one row of the `parcels` digest.

    `Parcel` does not use TimestampedModel/history, but it still carries
    `version` so metadata edits can use the standard optimistic-lock contract.
    """
    # The geometric/cadastral area columns intentionally share the model
    # value until a separate cadastral source is available.
    return [
        p.id, p.version, p.region_id, p.region.name, p.name, p.eclass.name,
        p.eclass.coppice, float(p.area_ha), float(p.area_ha), p.ave_age,
        p.location_name,
        p.altitude_min_m, p.altitude_max_m, p.aspect, p.grade_pct,
        S.TYPE_COPPICE if p.eclass.coppice else S.TYPE_HIGHFOREST,
        p.desc_veg, p.desc_geo,
    ]


def generate_parcels() -> None:
    from apps.base.models import Parcel

    rows = [
        build_parcel_record(p)
        for p in Parcel.objects.select_related('region', 'eclass')
                       .order_by('region__name', 'name')
    ]

    _write_gzip_json({'columns': PARCEL_COLUMNS, 'rows': rows}, _dest(DIGEST_PARCELS))
    logger.info('%s.json.gz: %s rows', DIGEST_PARCELS, len(rows))


# ---------------------------------------------------------------------------
# Species digest
# ---------------------------------------------------------------------------

def generate_species() -> None:
    """Lookup digest used by JS forms (V/m preview) and by other digest
    generators that need per-species density."""
    from apps.base.models import Species

    columns = [ROW_ID, VERSION, S.COL_NAME, S.COL_LATIN_NAME,
               S.COL_DENSITY, S.COL_PRESSLER, S.COL_SORT_ORDER, S.COL_ACTIVE]
    rows = []
    for sp in Species.objects.order_by(FIELD_SORT_ORDER):
        rows.append([
            sp.id, sp.version, sp.common_name, sp.latin_name,
            float(sp.density), float(sp.pressler_default), sp.sort_order, sp.active,
        ])

    _write_gzip_json({'columns': columns, 'rows': rows}, _dest(FIELD_SPECIES))
    logger.info('species.json.gz: %s rows', len(rows))


# ---------------------------------------------------------------------------
# Audit digest
# ---------------------------------------------------------------------------

_ACTION_MAP = {'+': S.AUDIT_INSERT, '~': S.AUDIT_UPDATE, '-': S.AUDIT_DELETE}


def _tracked_models() -> set:
    """Every model registered with django-simple-history (i.e. auditable).

    Detected structurally — a tracked model exposes a `HistoryManager` on
    its `history` attribute — so the audit can never silently drift from the
    set of `HistoricalRecords()` declarations.
    """
    from django.apps import apps
    from simple_history.manager import HistoryManager
    return {m for m in apps.get_models()
            if isinstance(getattr(m, 'history', None), HistoryManager)}


def _audit_configs() -> list:
    """`(model, table_label, {field: label})` for every audited model.

    Every model carrying `HistoricalRecords()` MUST appear here so its
    writes surface in the Controllo log; `generate_audit()` asserts this
    against `_tracked_models()`, and the contract is locked by
    `test_audit_covers_all_tracked_models`.  To stop auditing a model,
    remove its `HistoricalRecords()` (as done for Sample/Tree/TreeSample/
    TreeMark/TreePreserved, whose bulk imports would swamp the log) — do not
    just drop it here.

    Field maps are selective: only domain-meaningful fields are shown.
    """
    from apps.base.models import (
        Crew, HarvestPlan, HarvestPlanItem, HypsoParam, HypsoParamSet,
        Parcel, SampleArea, SampleGrid, Species, Survey, Tractor, User,
    )
    from apps.ipso.models import IpsoUpload
    from apps.mannesi.models import ProductionCredit, WorkHour
    from apps.prelievi.models import Harvest

    return [
        (Harvest, S.TABLE_HARVEST, {
            'date': S.COL_DATE, 'region_id': S.COL_REGION, 'parcel_id': S.COL_PARCEL,
            'crew_id': S.COL_CREW, 'product_id': S.COL_PRODUCT,
            'mass_q': S.COL_QUINTALS, FIELD_VOLUME_M3: S.COL_VOLUME_M3,
            'record1': S.COL_VDP,
            'record2': S.COL_PROT,
            'damaged': S.FLAG_DAMAGED, 'unhealthy': S.FLAG_UNHEALTHY,
            'psr': S.FLAG_PSR, 'note': S.COL_EXTRA_NOTE,
            'harvest_plan_item_id': S.COL_HARVEST_PLAN,
        }),
        (User, S.TABLE_USER, {
            'username': S.LABEL_USERNAME, 'role': S.LABEL_ROLE,
            'landing_page': S.LABEL_LANDING_PAGE,
            'is_active': S.COL_ACTIVE,
        }),
        (Crew, S.TABLE_CREW, {
            'name': S.LABEL_NAME, 'notes': S.LABEL_NOTES,
            'active': S.COL_ACTIVE,
        }),
        (Tractor, S.TABLE_TRACTOR, {
            'name': S.LABEL_TRACTOR_NAME,
            'manufacturer': S.LABEL_MANUFACTURER,
            'model': S.LABEL_MODEL, 'year': S.COL_YEAR,
            'active': S.COL_ACTIVE,
        }),
        (Species, S.TABLE_SPECIES, {
            'common_name': S.LABEL_NAME,
            'latin_name': S.COL_LATIN_NAME,
            'density': S.LABEL_DENSITY,
            'active': S.COL_ACTIVE,
        }),
        (HarvestPlan, S.TABLE_HARVEST_PLAN, {
            'name': S.LABEL_NAME, 'year_start': S.COL_YEAR_START,
            'year_end': S.COL_YEAR_END, 'description': S.COL_DESCRIPTION,
            'active': S.COL_ACTIVE,
        }),
        (HarvestPlanItem, S.TABLE_HARVEST_PLAN_ITEM, {
            'harvest_plan_id': S.COL_HARVEST_PLAN,
            'region_id': S.COL_REGION, 'parcel_id': S.COL_PARCEL,
            'state': S.COL_STATE, 'year_planned': S.COL_YEAR_PLANNED,
            'volume_planned_m3': S.COL_VOLUME_PLANNED,
            'intervention_area_ha': S.COL_INTERVENTION_AREA_HA,
            'damaged': S.FLAG_DAMAGED, 'unhealthy': S.FLAG_UNHEALTHY,
            'psr': S.FLAG_PSR, 'note': S.COL_NOTE,
        }),
        (Parcel, S.TABLE_PARCEL, {
            'name': S.COL_PARCEL, 'region_id': S.COL_REGION,
            'eclass_id': S.COL_CLASS, 'area_ha': S.COL_AREA_HA,
            'ave_age': S.COL_AVE_AGE, 'location_name': S.COL_LOCATION,
            'altitude_min_m': S.COL_ALT_MIN,
            'altitude_max_m': S.COL_ALT_MAX, 'aspect': S.COL_ASPECT,
            'grade_pct': S.COL_GRADE_PCT, 'desc_veg': S.COL_DESC_VEG,
            'desc_geo': S.COL_DESC_GEO,
        }),
        (SampleGrid, S.TABLE_SAMPLE_GRID, {
            'name': S.LABEL_NAME, 'description': S.COL_DESCRIPTION,
        }),
        (SampleArea, S.TABLE_SAMPLE_AREA, {
            'sample_grid_id': S.COL_GRID, 'number': S.COL_NUMBER,
            'parcel_id': S.COL_PARCEL, 'lat': S.COL_LAT, 'lon': S.COL_LON,
            'altitude_m': S.COL_ALT, 'r_m': S.COL_RADIUS,
            'note': S.COL_NOTE,
        }),
        (Survey, S.TABLE_SURVEY, {
            'name': S.LABEL_NAME, 'sample_grid_id': S.COL_GRID,
            'description': S.COL_DESCRIPTION, 'active': S.COL_ACTIVE,
        }),
        (HypsoParamSet, S.TABLE_HYPSO_PARAM_SET, {
            'source': S.COL_HYPSO_SOURCE, 'min_n': S.COL_MIN_N,
            'use_for_height_plots': S.COL_USE_FOR_HEIGHT_PLOTS,
            'superseded_at': S.COL_SUPERSEDED_AT,
        }),
        (HypsoParam, S.TABLE_HYPSO_PARAM, {
            'param_set_id': S.TABLE_HYPSO_PARAM_SET,
            'region_id': S.COL_REGION, 'species_id': S.COL_SPECIES,
            'func': S.COL_FUNCTION, 'a': S.COL_A, 'b': S.COL_B,
            'n': S.COL_N_REGRESSION, 'r2': S.COL_R2,
        }),
        (WorkHour, S.TABLE_MANNESI_HOURS, {
            'date': S.COL_DATE, 'crew_id': S.COL_CREW,
            'hours': S.COL_HOURS, 'note': S.COL_NOTE,
        }),
        (ProductionCredit, S.TABLE_MANNESI_CREDIT, {
            'date': S.COL_DATE, 'crew_id': S.COL_CREW,
            'mass_q': S.COL_CREDITS_Q, 'note': S.COL_NOTE,
        }),
        (IpsoUpload, S.TABLE_IPSO_UPLOAD, {
            'session_id': S.IPSO_COL_SESSION,
            'mode': S.IPSO_COL_MODE,
            'operator_name': S.IPSO_COL_OPERATOR,
            'record_count': S.IPSO_COL_RECORDS,
            'record_date': S.COL_DATE,
            'state': S.IPSO_COL_STATE,
            'error_summary': S.IPSO_COL_ERROR,
            'target_type': S.IPSO_COL_TARGET,
            'target_id': S.IPSO_COL_TARGET,
            'imported_at': S.IPSO_COL_IMPORTED,
            'imported_by_id': S.IPSO_COL_IMPORTED_BY,
        }),
    ]


def generate_audit() -> None:
    """Audit log from django-simple-history across all tracked models."""
    configs = _audit_configs()

    missing = _tracked_models() - {model for model, _, _ in configs}
    if missing:
        names = ', '.join(sorted(m.__name__ for m in missing))
        raise RuntimeError(
            f'History-tracked models absent from the audit config: {names}. '
            f'Add them to _audit_configs(), or remove their '
            f'HistoricalRecords() if they should not be audited.'
        )

    rows = []
    row_id = 0
    for model, table_label, field_labels in configs:
        row_id = _audit_rows(model, model.history, table_label, field_labels,
                             rows, row_id)

    rows.sort(key=lambda r: r[1], reverse=True)

    columns = [ROW_ID, S.COL_TIMESTAMP, S.COL_USER, S.COL_TABLE,
               S.COL_ACTION, S.COL_OLD_VALUE, S.COL_NEW_VALUE]
    _write_gzip_json({'columns': columns, 'rows': rows}, _dest('audit'))
    logger.info('audit.json.gz: %s rows', len(rows))


def _audit_rows(model, manager, table_label, field_labels, rows, row_id):
    """Append audit rows for one historical model.  Returns updated row_id."""
    prev_map: dict[int, object] = {}
    value_cache: dict[tuple[object, object], object | None] = {}

    for entry in manager.select_related('history_user').order_by('history_date', 'history_id'):
        pk = entry.id
        ht = entry.history_type
        user = ''
        if entry.history_user:
            u = entry.history_user
            user = f'{u.first_name} {u.last_name}'.strip() or u.username

        ts = timezone.localtime(entry.history_date).strftime('%Y-%m-%d %H:%M')
        action = _ACTION_MAP.get(ht, ht)

        if ht == '+':
            row = ('', _format_fields(model, entry, field_labels, value_cache))
        elif ht == '~' and pk in prev_map:
            row = _format_update(
                model, prev_map[pk], entry, field_labels, value_cache,
            )
        elif ht == '-':
            row = (_format_fields(model, entry, field_labels, value_cache), '')
        else:
            row = ('', _format_fields(model, entry, field_labels, value_cache))

        if row is None:
            prev_map[pk] = entry
            continue
        before, after = row

        row_id += 1
        rows.append([row_id, ts, user, table_label, action, before, after])
        prev_map[pk] = entry

    return row_id


def _format_fields(model, entry, field_labels: dict, value_cache: dict) -> str:
    """Format all tracked fields as 'label: value; ...'."""
    parts = []
    for field, label in field_labels.items():
        val = getattr(entry, field, None)
        if val is not None and val != '':
            rendered = _format_audit_value(model, field, val, value_cache)
            parts.append(f'{label}: {rendered}')
    return '; '.join(parts)


def _format_update(
        model, prev, current, field_labels: dict, value_cache: dict,
) -> tuple[str, str] | None:
    """Format updates as full before/after records, or skip noise-only writes."""
    before = _format_fields(model, prev, field_labels, value_cache)
    after = _format_fields(model, current, field_labels, value_cache)
    password_changed = _user_password_changed(model, prev, current)

    if before != after:
        if password_changed:
            after = _append_audit_note(after, S.PASSWORD_CHANGED)
        return before, after
    if password_changed:
        return '', S.PASSWORD_CHANGED
    return None


def _user_password_changed(model, prev, current) -> bool:
    if model._meta.label_lower != 'base.user':
        return False
    return getattr(prev, 'password', None) != getattr(current, 'password', None)


def _append_audit_note(text: str, note: str) -> str:
    if not text:
        return note
    return f'{text}; {note}'


def _model_field(model, field_name: str):
    """Return the model field represented by an audit field name."""
    try:
        return model._meta.get_field(field_name)
    except FieldDoesNotExist:
        if field_name.endswith('_id'):
            try:
                return model._meta.get_field(field_name[:-3])
            except FieldDoesNotExist:
                return None
        return None


def _format_audit_value(model, field_name: str, value, value_cache: dict) -> str:
    """Render audit values as operator-readable labels where possible."""
    if value is None or value == '':
        return ''

    field = _model_field(model, field_name)
    if isinstance(field, models.ForeignKey):
        return _format_audit_fk(field.remote_field.model, value, value_cache)
    if field is not None and field.choices:
        choices = dict(field.flatchoices)
        return str(choices.get(value, value))
    if isinstance(field, models.BooleanField):
        return 'sì' if value else 'no'
    return str(value)


def _format_audit_fk(related_model, pk, value_cache: dict) -> str:
    cache_key = (related_model, pk)
    if cache_key not in value_cache:
        value_cache[cache_key] = related_model.objects.filter(pk=pk).first()
    obj = value_cache[cache_key]
    if obj is None:
        return f'#{pk}'
    if hasattr(obj, 'username'):
        full_name = f'{obj.first_name} {obj.last_name}'.strip()
        return full_name or obj.username
    return str(obj)


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
    logger.info('grids.json.gz: %s rows', len(rows))


SURVEY_COLUMNS = [ROW_ID, VERSION, S.COL_NAME, S.COL_DESCRIPTION,
                  S.COL_GRID, S.COL_N_AREAS_VISITED,
                  S.COL_N_AREAS_TOTAL, S.COL_DATE_FIRST,
                  S.COL_DATE_LAST, S.COL_ACTIVE]


def build_survey_record(
        s, *, n_visited=_UNSET, n_total=_UNSET,
        first_date=_UNSET, last_date=_UNSET,
) -> list:
    """Build one row of the `surveys` digest.

    Write paths omit aggregate arguments and recompute values for one survey.
    Full-digest generation passes precomputed annotations/counts so the row
    shape stays centralized without reintroducing N+1 aggregate queries.
    """
    from django.db.models import Count, Max, Min
    from apps.base.models import Sample, SampleArea

    if n_visited is _UNSET or first_date is _UNSET or last_date is _UNSET:
        visited_expr = 'sample_area' if s.sample_grid_id is not None else 'id'
        agg = Sample.objects.filter(survey=s).aggregate(
            n_visited=Count(visited_expr, distinct=True),
            first_date=Min('date'),
            last_date=Max('date'),
        )
        if n_visited is _UNSET:
            n_visited = agg['n_visited'] or 0
        if first_date is _UNSET:
            first_date = agg[FIELD_FIRST_DATE]
        if last_date is _UNSET:
            last_date = agg[FIELD_LAST_DATE]
    if n_total is _UNSET:
        n_total = (
            SampleArea.objects.filter(sample_grid_id=s.sample_grid_id).count()
            if s.sample_grid_id is not None else 0
        )

    return [
        s.id, s.version, s.name, s.description,
        s.sample_grid_id,
        n_visited or 0,
        n_total,
        first_date.isoformat() if first_date else '',
        last_date.isoformat() if last_date else '',
        s.active,
    ]


def generate_surveys() -> None:
    """List of surveys with completeness counts and date range.

    Drives Section 2 (Rilevamenti) pulldown.
    """
    from django.db.models import Count, Max, Min
    from apps.base.models import SampleArea, Survey

    totals_by_grid = dict(
        SampleArea.objects
            .values('sample_grid_id')
            .annotate(n=Count('id'))
            .values_list('sample_grid_id', 'n'),
    )
    rows = []
    qs = Survey.objects.select_related('sample_grid') \
                       .annotate(
                           n_visited=Count('sample__sample_area', distinct=True),
                           n_samples=Count('sample', distinct=True),
                           first_date=Min('sample__date'),
                           last_date=Max('sample__date'),
                       ) \
                       .order_by('-last_date', '-created_at')
    for s in qs:
        structured = s.sample_grid_id is not None
        n_visited = s.n_visited if structured else s.n_samples
        n_total = totals_by_grid.get(s.sample_grid_id, 0) if structured else 0
        rows.append(build_survey_record(
            s,
            n_visited=n_visited or 0,
            n_total=n_total,
            first_date=s.first_date,
            last_date=s.last_date,
        ))

    _write_gzip_json(
        {'columns': SURVEY_COLUMNS, 'rows': rows}, _dest('surveys'),
    )
    logger.info('surveys.json.gz: %s rows', len(rows))


SAMPLE_AREA_COLUMNS = [ROW_ID, VERSION, S.COL_GRID, S.COL_REGION,
                       S.COL_PARCEL, S.COL_NUMBER, S.COL_LAT, S.COL_LON,
                       S.COL_ALT, S.COL_RADIUS, S.COL_NOTE]


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
    logger.info('sample_areas.json.gz: %s rows', len(rows))


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
    logger.info('samples.json.gz: %s rows', len(rows))


SAMPLED_TREE_COLUMNS = [ROW_ID, VERSION, S.COL_SAMPLE_AREA,
                        S.COL_SAMPLE_DATE, S.COL_REGION, S.COL_PARCEL,
                        S.COL_AREA_NUM, S.COL_TREE_NUM,
                        S.COL_SPECIES, S.COL_PRODUCT, COL_COPPICE,
                        S.COL_COPPICE_SHOOT,
                        S.COL_COPPICE_STD, S.COL_D_CM, S.COL_H_M, S.COL_L10_MM,
                        S.COL_PRESSLER, S.COL_V_M3, S.COL_MASS_Q,
                        S.COL_PRESERVED, S.COL_LAT, S.COL_LON]


def build_tree_sample_record(ts) -> list:
    """Build one row of the `sampled_trees_<survey>` digest.

    Structured samples use their sample area's parcel and area coordinates.
    Unstructured samples have no sample area, so the digest falls back to the
    tree's parcel and tree coordinates.  Caller must pre-load the relevant
    parcel/species relations.
    """
    tree = ts.tree
    sa = ts.sample.sample_area
    if sa is None:
        parcel = tree.parcel
        area_id = None
        area_number = ''
        lat = tree.lat
        lon = tree.lon
    else:
        parcel = sa.parcel
        area_id = sa.id
        area_number = sa.number
        lat = tree.lat if tree.lat is not None else sa.lat
        lon = tree.lon if tree.lon is not None else sa.lon
    return [
        ts.id, ts.version, area_id, ts.sample.date.isoformat(),
        parcel.region.name, parcel.name, area_number,
        ts.number, tree.species.common_name,
        S.TYPE_COPPICE if tree.coppice else S.TYPE_HIGHFOREST,
        tree.coppice,
        ts.shoot, ts.standard,
        ts.d_cm, float(ts.h_m), ts.l10_mm, float(ts.pressler_coeff),
        float(ts.volume_m3) if ts.volume_m3 is not None else None,
        float(ts.mass_q) if ts.mass_q is not None else None,
        tree.preserved,
        lat, lon,
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
                          'tree__species', 'tree__parcel__region')
          .order_by('sample__sample_area__parcel__region__name',
                    'tree__parcel__region__name',
                    'sample__sample_area__parcel__name',
                    'tree__parcel__name',
                    'sample__sample_area__number', FIELD_NUMBER, FIELD_SHOOT))
    rows = [build_tree_sample_record(ts) for ts in qs]
    _write_gzip_json(
        {'columns': SAMPLED_TREE_COLUMNS, 'rows': rows},
        _dest(f'sampled_trees_{survey_id}'),
    )
    logger.info('sampled_trees_%s.json.gz: %s rows', survey_id, len(rows))


# ---------------------------------------------------------------------------
# Piano di taglio digests
# ---------------------------------------------------------------------------

HARVEST_PLAN_COLUMNS = [ROW_ID, VERSION, S.COL_NAME, S.COL_DESCRIPTION,
                       S.COL_YEAR_START, S.COL_YEAR_END, S.COL_ACTIVE]


def build_harvest_plan_record(hp) -> list:
    """Build one row of the `harvest_plans` digest."""
    return [
        hp.id, hp.version, hp.name, hp.description,
        hp.year_start, hp.year_end, hp.active,
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
    logger.info('harvest_plans.json.gz: %s rows', len(rows))


HARVEST_PLAN_ITEM_COLUMNS = [
    ROW_ID, VERSION, S.COL_HARVEST_PLAN,
    S.COL_YEAR_PLANNED, S.COL_YEAR_ACTUAL,
    S.COL_REGION, S.COL_PARCEL, S.COL_PARCEL_AREA_HA,
    S.COL_TYPE, COL_COPPICE, S.COL_STATE, S.COL_NOTE,
    S.COL_VOLUME_PLANNED, S.COL_VOLUME_MARKED, S.COL_VOLUME_ACTUAL,
    S.COL_INTERVENTION_AREA_HA, S.COL_PERIOD_Y, S.COL_EXTRA_NOTE,
]


def _hpi_type(item) -> str:
    """`Tipo` value for the calendar Tipo column.  Empty for region-wide
    items (no Eclass to derive from).
    """
    is_coppice = _hpi_coppice(item)
    if is_coppice is None:
        return ''
    return S.TYPE_COPPICE if is_coppice else S.TYPE_HIGHFOREST


def _hpi_coppice(item) -> bool | None:
    """Stable item-kind flag.  None means whole-region, not parcel-backed."""
    if item.parcel_id is None:
        return None
    return bool(item.parcel.eclass.coppice)


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


def _region_area_by_id(region_ids=None) -> dict[int, float]:
    from apps.base.models import Parcel
    qs = Parcel.objects
    if region_ids is not None:
        qs = qs.filter(region_id__in=region_ids)
    return {
        row['region_id']: float(row['area'])
        for row in qs.values('region_id').annotate(area=Sum('area_ha'))
        if row['area'] is not None
    }


def build_harvest_plan_item_record(item, region_area_by_id: dict | None = None) -> list:
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
        # Whole-region marker — also the CSV round-trip representation.
        particella = S.PARCEL_WHOLE_REGION_MARK
        if region_area_by_id is None:
            region_area_by_id = _region_area_by_id([item.region_id])
        parcel_area = region_area_by_id.get(item.region_id, '')

    return [
        item.id, item.version, item.harvest_plan_id,
        item.year_planned,
        item.date_actual.year if item.date_actual else '',
        compresa, particella, parcel_area, _hpi_type(item), _hpi_coppice(item),
        item.get_state_display(),
        render_flag_note(item.damaged, item.unhealthy, item.psr),
        float(item.volume_planned_m3) if item.volume_planned_m3 is not None else '',
        float(item.volume_marked_m3) if item.volume_marked_m3 is not None else '',
        float(item.volume_actual_m3),
        float(item.intervention_area_ha) if item.intervention_area_ha is not None else '',
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
    region_area_by_id = _region_area_by_id()
    rows = [build_harvest_plan_item_record(it, region_area_by_id) for it in qs]
    _write_gzip_json(
        {'columns': HARVEST_PLAN_ITEM_COLUMNS, 'rows': rows},
        _dest('harvest_plan_items'),
    )
    logger.info('harvest_plan_items.json.gz: %s rows', len(rows))


HYPSO_PARAM_COLUMNS = [
    ROW_ID, S.COL_REGION, S.COL_SPECIES, S.COL_FUNCTION,
    S.COL_A, S.COL_B, S.COL_N_REGRESSION, S.COL_R2,
]


def hypso_param_row(row_id, region_name, species_name, func, a, b, n, r2) -> list:
    """The `hypso_params` row shape (note: n before r2).

    Shared by the digest generator and the compute-candidate preview so the
    two never drift; `id` is None for an unsaved candidate row.
    """
    return [row_id, region_name, species_name, func,
            float(a), float(b), n, float(r2)]


def build_hypso_param_record(p) -> list:
    """Build one row of the `hypso_params` digest from a HypsoParam."""
    return hypso_param_row(
        p.id, p.region.name, p.species.common_name, p.func,
        p.a, p.b, p.n, p.r2,
    )


def generate_hypso_params() -> None:
    """The active hypsometric parameter set's (region, species) coefficients.

    Consumed by the Impostazioni settings table and, JS-side, by the Nuovo
    albero martellato modal to auto-fill `h_m` from `d_cm`.  Only the active
    set (superseded_at IS NULL) is served; empty if there is none.
    """
    from apps.base.models import HypsoParam, HypsoParamSet

    active = HypsoParamSet.objects.active().first()
    if active is None:
        rows = []
    else:
        qs = (HypsoParam.objects
              .filter(param_set=active)
              .select_related('region', 'species')
              .order_by('region__name', 'species__common_name'))
        rows = [build_hypso_param_record(p) for p in qs]
    _write_gzip_json(
        {'columns': HYPSO_PARAM_COLUMNS, 'rows': rows},
        _dest(DIGEST_HYPSO_PARAMS),
    )
    logger.info('%s.json.gz: %s rows', DIGEST_HYPSO_PARAMS, len(rows))


MARK_TREE_COLUMNS = [ROW_ID, VERSION, S.COL_DATE, S.COL_NUMBER,
                     S.COL_SPECIES, S.COL_D_CM, S.COL_H_M, S.COL_H_MEASURED,
                     S.COL_V_M3, S.COL_MASS_Q,
                     S.COL_LAT, S.COL_LON, S.COL_OPERATOR]


def build_tree_mark_record(tm) -> list:
    """Build one row of a `mark_trees_<item_id>` digest.

    Caller must pre-load `tree.species`.
    """
    return [
        tm.id, tm.version, tm.date.isoformat(), tm.number,
        tm.tree.species.common_name,
        tm.d_cm, float(tm.h_m), tm.h_measured,
        float(tm.volume_m3) if tm.volume_m3 is not None else None,
        float(tm.mass_q) if tm.mass_q is not None else None,
        tm.lat, tm.lon, tm.operator,
    ]


def generate_mark_trees_for_item(item_id: int) -> None:
    """Per-harvest_plan_item tree_mark digest.

    Filename: `mark_trees_<item_id>.json.gz`.  Lazy-loaded by the
    View/Edit-item modal's Martellate section.  Pattern mirrors
    `sampled_trees_<survey_id>` in campionamenti.

    Sort: Numero ascending.
    """
    from apps.base.models import TreeMark

    qs = (TreeMark.objects
          .filter(harvest_plan_item_id=item_id)
          .select_related('tree__species')
          .order_by(FIELD_NUMBER))
    rows = [build_tree_mark_record(tm) for tm in qs]
    _write_gzip_json(
        {'columns': MARK_TREE_COLUMNS, 'rows': rows},
        _dest(f'mark_trees_{item_id}'),
    )
    logger.info('mark_trees_%s.json.gz: %s rows', item_id, len(rows))


# ---------------------------------------------------------------------------
# Bosco digests
# ---------------------------------------------------------------------------

PRESERVED_TREE_COLUMNS = [
    ROW_ID, VERSION, COL_TREE_ID, COL_PARCEL_ID, COL_SPECIES_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_SPECIES, S.COL_NUMBER, S.COL_DATE,
    S.COL_ESTIMATED_BIRTH_YEAR, S.COL_D_CM, S.COL_H_M, S.COL_H_MEASURED,
    S.COL_LAT, S.COL_LON, S.COL_NOTE,
]


def build_preserved_tree_record(pai) -> list:
    """Build one row of the `preserved_trees` digest.

    Caller must pre-load `parcel.region` and `tree.species`.
    """
    tree = pai.tree
    return [
        pai.id, pai.version, tree.id, pai.parcel_id, tree.species_id,
        pai.parcel.region.name, pai.parcel.name, tree.species.common_name,
        pai.number, pai.date.isoformat() if pai.date else '',
        tree.estimated_birth_year if tree.estimated_birth_year is not None else '',
        pai.d_cm if pai.d_cm is not None else '',
        float(pai.h_m) if pai.h_m is not None else '',
        pai.h_measured, pai.lat, pai.lon, pai.note,
    ]


def generate_preserved_trees() -> None:
    from apps.base.models import TreePreserved

    rows = [
        build_preserved_tree_record(pai)
        for pai in (TreePreserved.objects
                    .filter(tree__preserved=True)
                    .select_related('parcel__region', 'tree__species')
                    .order_by('parcel__region__name', 'parcel__name',
                              'number', 'id'))
    ]
    _write_gzip_json(
        {'columns': PRESERVED_TREE_COLUMNS, 'rows': rows},
        _dest(DIGEST_PRESERVED_TREES),
    )
    logger.info('%s.json.gz: %s rows', DIGEST_PRESERVED_TREES, len(rows))


FUTURE_PRODUCTION_COLUMNS = [
    ROW_ID, VERSION, S.COL_HARVEST_PLAN, COL_PARCEL_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_YEAR_PLANNED, S.COL_VOLUME_PLANNED,
]


def build_future_production_record(item) -> list:
    """Build one row of the `future_production` digest."""
    return [
        item.id, item.version, item.harvest_plan_id, item.parcel_id,
        item.parcel.region.name, item.parcel.name, item.year_planned,
        float(item.volume_planned_m3) if item.volume_planned_m3 is not None else None,
    ]


def generate_future_production() -> None:
    from apps.base.models import HarvestPlanItem

    plan = active_or_default_harvest_plan()
    qs = HarvestPlanItem.objects.none()
    if plan is not None:
        qs = (HarvestPlanItem.objects
              .filter(harvest_plan=plan, parcel__isnull=False,
                      parcel__eclass__coppice=False,
                      volume_planned_m3__isnull=False)
              .select_related('parcel__region', 'parcel__eclass')
              .order_by('parcel__region__name', 'parcel__name', 'year_planned', 'id'))
    rows = [build_future_production_record(item) for item in qs]
    _write_gzip_json(
        {'columns': FUTURE_PRODUCTION_COLUMNS, 'rows': rows},
        _dest(DIGEST_FUTURE_PRODUCTION),
    )
    logger.info('%s.json.gz: %s rows', DIGEST_FUTURE_PRODUCTION, len(rows))


DENDROMETRY_COLUMNS = [
    ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_SPECIES_ID,
    S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY, S.COL_SAMPLE_AREA_HA, S.COL_SPECIES,
    S.COL_DIAM_CLASS_CM, S.COL_N_TREES, S.COL_VOLUME_M3,
    S.COL_BASAL_AREA_M2, S.COL_AVG_H_M, S.COL_INCREMENT_PCT,
]

DENDROMETRY_POINT_COLUMNS = [
    ROW_ID, COL_PARCEL_ID, COL_SURVEY_ID, COL_TREE_ID,
    COL_SPECIES_ID, S.COL_REGION, S.COL_PARCEL, S.COL_SURVEY,
    S.COL_SPECIES, S.COL_D_CM, S.COL_H_M,
]


def diameter_class_cm(d_cm: int) -> int:
    """5 cm diameter class centered on multiples of 5.

    Integer diameters 18..22 map to class 20; 23..27 map to class 25.
    """
    return int((int(d_cm) + 2) // 5 * 5)


def basal_area_m2(d_cm: int) -> float:
    radius_m = float(d_cm) / 200.0
    return math.pi * radius_m * radius_m


def annual_increment_pct(d_cm: int, l10_mm: int, pressler_coeff) -> float | None:
    """Annual Pressler volume-growth percentage from outer-ten-rings width.

    L10 is a radial ten-year measurement in mm. Annual diameter growth as a
    percentage of current diameter is ``2 * L10_mm / D_cm``; the Pressler
    coefficient converts that to estimated volume increment.

    `l10_mm = 0` means no core was measured, so it contributes no increment
    observation rather than a zero-growth observation.
    """
    if not d_cm or not l10_mm:
        return None
    return float(pressler_coeff) * 2.0 * float(l10_mm) / float(d_cm)


def _dendrometry_queryset(survey_ids=None):
    from apps.base.models import TreeSample

    if survey_ids is None:
        survey_ids = active_or_default_survey_ids()
    if not survey_ids:
        return TreeSample.objects.none()
    return (TreeSample.objects
            .filter(
                sample__survey_id__in=survey_ids,
                sample__sample_area_id__isnull=False,
            )
            .select_related('sample__survey', 'sample__sample_area__parcel__region',
                            'tree', 'tree__species')
            .order_by('sample__sample_area__parcel__region__name',
                      'sample__sample_area__parcel__name',
                      'sample__survey__name', 'tree__species__common_name',
                      'd_cm', 'id'))


def sample_area_ha(sample_area) -> float:
    return math.pi * float(sample_area.r_m) ** 2 / M2_PER_HA


def _dendrometry_sample_area_coverage() -> dict[tuple[int, int], float]:
    from apps.base.models import Sample

    survey_ids = active_or_default_survey_ids()
    if not survey_ids:
        return {}
    coverage = {}
    qs = (Sample.objects
          .filter(survey_id__in=survey_ids, sample_area_id__isnull=False)
          .select_related('sample_area', 'sample_area__parcel'))
    for sample in qs:
        key = (sample.sample_area.parcel_id, sample.survey_id)
        coverage[key] = coverage.get(key, 0.0) + sample_area_ha(sample.sample_area)
    return coverage


def generate_parcel_dendrometry() -> None:
    coverage = _dendrometry_sample_area_coverage()
    groups = {}
    for ts in _dendrometry_queryset():
        parcel = ts.sample.sample_area.parcel
        survey = ts.sample.survey
        species = ts.tree.species
        key = (parcel.id, survey.id, species.id, diameter_class_cm(ts.d_cm))
        group = groups.setdefault(key, {
            'parcel': parcel, 'survey': survey, 'species': species,
            'class_cm': key[3], 'n': 0, 'volume': 0.0, 'basal': 0.0,
            'height': 0.0, 'increments': [],
        })
        group['n'] += 1
        if ts.volume_m3 is not None:
            group['volume'] += float(ts.volume_m3)
        group['basal'] += basal_area_m2(ts.d_cm)
        group['height'] += float(ts.h_m)
        inc = annual_increment_pct(ts.d_cm, ts.l10_mm, ts.pressler_coeff)
        if inc is not None:
            group['increments'].append(inc)

    rows = []
    for row_id, group in enumerate(sorted(
            groups.values(),
            key=lambda g: (g['parcel'].region.name, g['parcel'].name,
                           g['survey'].name, g['species'].common_name,
                           g['class_cm'])), 1):
        parcel = group['parcel']
        survey = group['survey']
        species = group['species']
        increments = group['increments']
        rows.append([
            row_id, parcel.id, survey.id, species.id,
            parcel.region.name, parcel.name, survey.name,
            round(coverage.get((parcel.id, survey.id), 0.0), 6),
            species.common_name, group['class_cm'], group['n'], round(group['volume'], 4),
            round(group['basal'], 6), round(group['height'] / group['n'], 4),
            round(sum(increments) / len(increments), 4) if increments else None,
        ])

    _write_gzip_json(
        {'columns': DENDROMETRY_COLUMNS, 'rows': rows},
        _dest(DIGEST_PARCEL_DENDROMETRY),
    )
    logger.info('%s.json.gz: %s rows', DIGEST_PARCEL_DENDROMETRY, len(rows))


def generate_parcel_dendrometry_points() -> None:
    rows = []
    for ts in _dendrometry_queryset(height_plot_survey_ids()):
        parcel = ts.sample.sample_area.parcel
        survey = ts.sample.survey
        species = ts.tree.species
        rows.append([
            ts.id, parcel.id, survey.id, ts.tree_id, species.id,
            parcel.region.name, parcel.name, survey.name, species.common_name,
            ts.d_cm, float(ts.h_m),
        ])
    _write_gzip_json(
        {'columns': DENDROMETRY_POINT_COLUMNS, 'rows': rows},
        _dest(DIGEST_PARCEL_DENDROMETRY_POINTS),
    )
    logger.info('%s.json.gz: %s rows', DIGEST_PARCEL_DENDROMETRY_POINTS, len(rows))


# ---------------------------------------------------------------------------
# Generator registry
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, callable] = {
    'prelievi': generate_prelievi,
    DIGEST_PARCELS: generate_parcels,
    FIELD_SPECIES: generate_species,
    'audit': generate_audit,
    'grids': generate_grids,
    'surveys': generate_surveys,
    'sample_areas': generate_sample_areas,
    'samples': generate_samples,
    'harvest_plans': generate_harvest_plans,
    'harvest_plan_items': generate_harvest_plan_items,
    DIGEST_FUTURE_PRODUCTION: generate_future_production,
    DIGEST_PARCEL_DENDROMETRY: generate_parcel_dendrometry,
    DIGEST_PARCEL_DENDROMETRY_POINTS: generate_parcel_dendrometry_points,
    DIGEST_PRESERVED_TREES: generate_preserved_trees,
    DIGEST_HYPSO_PARAMS: generate_hypso_params,
}


def mark_all_stale() -> None:
    """Flag every digest as stale.

    Used by the bulk ETL importers, which bypass the views that
    normally call `mark_stale()` on writes.  The next read of each
    digest will trigger lazy regeneration.  Also covers existing
    `sampled_trees_<survey_id>` rows in DigestStatus so per-survey
    digests get refreshed on next read.
    """
    DigestStatus.objects.update(stale=True, dirty_seq=F('dirty_seq') + 1)
    for name in _GENERATORS:
        DigestStatus.objects.get_or_create(
            name=name, defaults={'stale': True, 'dirty_seq': 1},
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
