"""Unified species editor: typed equation input and atomic configuration writes."""
import hashlib
import json
from types import SimpleNamespace

from django.db import transaction
from django.template.loader import render_to_string

from apps.base import hypsometry
from apps.base.digests import build_species_record, mark_stale, mark_stale_prefixes
from apps.base.equations import coefficients_for_species, lock_species
from apps.base.idempotency import begin_nonce_write
from apps.base.models import HypsoParam, Region, Species, VolumeEquation
from apps.base.numparse import int_or_none, parse_decimal
from apps.base.responses import conflict_response, row_patch, submitted_version, success_response
from apps.base.species import PROTECTED_SPECIES_NAMES
from apps.base.validation import InputValidationError, decimal_field_value, validation_error
from config import strings as S
from config.constants import (
    BOSCO_SPECIES_DIGESTS, DIGEST_HYPSO_PARAMS, DIGEST_PARCEL_DENDROMETRY_POINTS,
    DIGEST_PREFIX_MARK_TREES, DIGEST_PREFIX_SAMPLED_TREES, PRESSLER_DEFAULT,
)

DEPENDENTS = ('species', 'prelievi', DIGEST_HYPSO_PARAMS,
              DIGEST_PARCEL_DENDROMETRY_POINTS, *BOSCO_SPECIES_DIGESTS)
PREFIXES = (DIGEST_PREFIX_MARK_TREES, DIGEST_PREFIX_SAMPLED_TREES)


def equation_state():
    """A configuration snapshot, independent of display/cache generation time."""
    active = hypsometry.active_set()
    payload = {
        'species': list(Species.objects.order_by('pk').values('id', 'version', 'common_name')),
        'regions': list(Region.objects.order_by('pk').values('id', 'name')),
        'volume': list(VolumeEquation.objects.order_by('pk').values()),
        'height_set': list(type(active).objects.filter(pk=active.pk).values()) if active else None,
        'height_surveys': list(active.surveys.order_by('pk').values_list('pk', flat=True)) if active else [],
        'height': list(HypsoParam.objects.filter(param_set=active).order_by('pk').values()) if active else [],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def editor_context(obj):
    state = equation_state()
    current = {p.region_id: p for p in HypsoParam.objects.filter(
        param_set__superseded_at__isnull=True, species=obj,
    ).select_related('origin_set').prefetch_related('origin_set__surveys')} if obj else {}
    return {
        'obj': obj, 'pressler_default': PRESSLER_DEFAULT,
        'protected_species_names': PROTECTED_SPECIES_NAMES,
        'equation_state': state,
        'height_regions': [{'region': r, 'equation': current.get(r.pk)}
                           for r in Region.objects.order_by('name')],
        'volume': VolumeEquation.objects.filter(species=obj).first() if obj else None,
        'volume_choices': Species.objects.filter(volume_equation__isnull=False)
                           .exclude(pk=obj.pk if obj else None).order_by('common_name'),
    }


def editor_html(request, obj):
    return render_to_string('impostazioni/_species_form.html', editor_context(obj), request=request)


def number(value, model, field, *, optional=False):
    if value in (None, '') and optional:
        return None
    parsed = parse_decimal(value)
    parsed = decimal_field_value(parsed, model, field) if parsed is not None else None
    if parsed is None:
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    return parsed


def volume_values(raw):
    if not isinstance(raw, dict):
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    mode = raw.get('mode')
    if mode == 'none':
        return None
    source = raw.get('source') or ''
    if not isinstance(source, str):
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    if mode == 'proxy':
        target = int_or_none(raw.get('proxy_species_id'))
        if not target:
            raise InputValidationError(S.ERR_EQUATION_REFERENCE)
        return dict(proxy_species_id=target, b0=None, b1=None, b2=None, source=source.strip())
    if mode != 'own':
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    return dict(proxy_species_id=None, source=source.strip(), **{
        key: number(raw.get(key), 'base.VolumeEquation', key, optional=key == 'b2')
        for key in ('b0', 'b1', 'b2')
    })


def validate_volume_graph(species_id, values):
    validate_volume_graphs({species_id: values})


def validate_volume_graphs(updates):
    equations = {p.species_id: p for p in VolumeEquation.objects.all()}
    for species_id, values in updates.items():
        if values is None:
            equations.pop(species_id, None)
        else:
            equations[species_id] = SimpleNamespace(**values)
    if any(coefficients_for_species(pk, equations) is None for pk in equations):
        raise InputValidationError(S.ERR_EQUATION_REFERENCE)


def persist_volume_equation(species, values):
    equation = VolumeEquation.objects.filter(species=species).first()
    if values is None:
        if equation:
            equation.delete()
    elif equation is None:
        VolumeEquation.objects.create(species=species, **values)
    elif any(getattr(equation, key) != value for key, value in values.items()):
        for key, value in values.items():
            setattr(equation, key, value)
        equation.version += 1
        equation.save()


def height_changes(species, raw):
    if not isinstance(raw, list):
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    regions = {r.pk: r for r in Region.objects.all()}
    current = {p.region_id: p for p in HypsoParam.objects.filter(
        param_set__superseded_at__isnull=True, species=species,
    )} if species else {}
    rows, removed, seen = [], [], set()
    for entry in raw:
        if not isinstance(entry, dict) or type(entry.get('enabled')) is not bool:
            raise InputValidationError(S.ERR_EQUATION_INVALID)
        region_id = int_or_none(entry.get('region_id'))
        if region_id not in regions or region_id in seen:
            raise InputValidationError(S.ERR_EQUATION_INVALID)
        seen.add(region_id)
        previous = current.get(region_id)
        if not entry['enabled']:
            if previous:
                removed.append((region_id, species.pk))
            continue
        a, b = (number(entry.get(key), 'base.HypsoParam', key) for key in ('a', 'b'))
        r2 = number(entry.get('r2'), 'base.HypsoParam', 'r2', optional=True)
        raw_n = entry.get('n')
        n = int_or_none(raw_n) if raw_n not in (None, '') else None
        if ((raw_n not in (None, '') and (n is None or n < 1))
                or (r2 is not None and not 0 <= r2 <= 1)):
            raise InputValidationError(S.ERR_EQUATION_INVALID)
        if previous and (previous.a, previous.b, previous.r2, previous.n) == (a, b, r2, n):
            continue
        rows.append(hypsometry.ParamRow(regions[region_id], species, a, b, r2, n))
    return rows, removed


def invalidate_equations():
    mark_stale(*dict.fromkeys(DEPENDENTS), 'audit')
    mark_stale_prefixes(*PREFIXES)


def save_configuration(request, body, metadata):
    begin_nonce_write(request)
    with transaction.atomic():
        species = lock_species()
        row_id = int_or_none(body.get('row_id'))
        obj = next((s for s in species if s.pk == row_id), None)
        if row_id and obj is None:
            from django.http import Http404
            raise Http404
        if (body.get('equation_state') != equation_state()
                or (obj and obj.version != submitted_version(body))):
            return conflict_response(
                data_id='species', row_id=obj.pk if obj else 0,
                record=build_species_record(obj) if obj else [],
                html=editor_html(request, obj),
            )
        if Species.objects.filter(common_name__iexact=metadata['common_name']).exclude(pk=row_id).exists():
            return validation_error([S.ERR_SPECIES_NAME_DUPLICATE])
        values = volume_values(body['volume_equation']) if 'volume_equation' in body else ...
        if values is not ...:
            validate_volume_graph(obj.pk if obj else 0, values)
        rows, removed = height_changes(obj, body['height_equations']) if 'height_equations' in body else ([], [])
        if obj is None:
            obj = Species.objects.create(**metadata)
        else:
            for key, value in metadata.items():
                setattr(obj, key, value)
            obj.version += 1
            obj.save()
        if values is not ...:
            persist_volume_equation(obj, values)
        if rows or removed:
            for row in rows:
                row.species = obj
            hypsometry.merge_active_set(rows, source='manual', remove_keys=removed)
        invalidate_equations()
        return success_response(request, body, data_id='species', row_id=obj.pk,
            patches=[row_patch('species', obj.pk, build_species_record(obj))],
            invalidates={'data_ids': [key for key in dict.fromkeys(DEPENDENTS) if key != 'species'],
                         'prefixes': list(PREFIXES)})
