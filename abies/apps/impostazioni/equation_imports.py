"""Read-only CSV previews and signed, atomic confirmation of equation upserts."""
from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.core import signing
from django.db import transaction
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base import csv_io, hypsometry
from apps.base.auth import require_writer
from apps.base.digests import build_species_record
from apps.base.equations import lock_species
from apps.base.idempotency import begin_nonce_write
from apps.base.models import HypsoParam, Region, Species, VolumeEquation
from apps.base.numparse import int_or_none
from apps.base.responses import parse_json_body, row_patch, success_response, validation_error
from apps.base.validation import InputValidationError
from apps.impostazioni.species_equations import (
    DEPENDENTS, PREFIXES, equation_state, invalidate_equations, number,
    persist_volume_equation, validate_volume_graphs, volume_values,
)
from config import strings as S

SALT = 'species-equations.import.v1'
PREVIEW_MAX_AGE = 15 * 60


def _resolve(name, options, mapping, index):
    if str(index) in mapping:
        pk = int_or_none(mapping[str(index)])
        return next((obj for obj in options if obj.pk == pk), None)
    matches = [obj for obj in options if obj.name.casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else None


def _preview_rows(request, body):
    kind = body.get('kind')
    if kind not in ('height', 'volume'):
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    file = csv_io.json_file_bytes(body, 'file')
    if file is None:
        raise InputValidationError(S.ERR_CSV_FILE_REQUIRED)
    reader = csv_io.read(file)
    state = equation_state()
    species = list(Species.objects.order_by('common_name'))
    # A common resolver accepts name-bearing reference objects at this boundary.
    for sp in species:
        sp.name = sp.common_name
    regions = list(Region.objects.order_by('name'))
    mappings = [body.get(key, {}) for key in ('species_map', 'region_map', 'proxy_map')]
    if any(not isinstance(mapping, dict) for mapping in mappings):
        raise InputValidationError(S.ERR_EQUATION_INVALID)
    records, errors, rows, missing = [], [], [], []
    seen = set()
    for index, raw in enumerate(reader, 2):
        if any(value is not None and not isinstance(value, str) for value in raw.values()):
            raise InputValidationError(S.ERR_EQUATION_INVALID)
        row = {(key or '').strip().casefold(): (value or '').strip() for key, value in raw.items()}
        name = row.get(S.CSV_COL_SPECIES.casefold()) or row.get('specie') or row.get('species') or ''
        region_name = row.get(S.CSV_COL_REGION.casefold(), '')
        proxy_name = row.get(S.CSV_COL_EQUATION_REFERENCE.casefold()) or row.get('proxy_species') or ''
        sp = _resolve(name, species, mappings[0], index)
        region = _resolve(region_name, regions, mappings[1], index) if kind == 'height' else None
        proxy = _resolve(proxy_name, species, mappings[2], index) if proxy_name else None
        unresolved = {'index': index, 'name': name, 'region_name': region_name,
                      'proxy_name': proxy_name, 'species': sp, 'region': region, 'proxy': proxy,
                      'needs_region': kind == 'height' and region is None}
        issue = None
        if sp is None or (kind == 'height' and region is None) or (proxy_name and proxy is None):
            missing.append(unresolved)
            issue = S.ERR_EQUATION_MATCH
        old, new = '', ''
        try:
            key = (sp.pk if sp else None, region.pk if region else None)
            if sp and key in seen:
                raise InputValidationError(S.ERR_EQUATION_DUPLICATE)
            seen.add(key)
            if kind == 'height':
                if row.get('funzione', 'ln') != 'ln':
                    raise InputValidationError(S.ERR_CSV_FUNCTION_INVALID.format(index, row.get('funzione')))
                values = {}
                for field in ('a', 'b', 'r2'):
                    parsed = reader.decimal(row.get(field)) if row.get(field) else None
                    if row.get(field) and parsed is None:
                        raise InputValidationError(S.ERR_EQUATION_INVALID)
                    values[field] = number(str(parsed) if parsed is not None else None,
                        'base.HypsoParam', field, optional=field == 'r2')
                n = reader.integer(row.get('n')) if row.get('n') else None
                if (row.get('n') and (n is None or n < 1)) or (values['r2'] is not None and not 0 <= values['r2'] <= 1):
                    raise InputValidationError(S.ERR_EQUATION_INVALID)
                values['n'] = n
                previous = HypsoParam.objects.filter(param_set__superseded_at__isnull=True,
                    species=sp, region=region).first() if sp and region else None
                old = f'a={previous.a}; b={previous.b}; n={previous.n}; r²={previous.r2}' if previous else '—'
                new = f"a={values['a']}; b={values['b']}; n={n}; r²={values['r2']}"
            else:
                raw_values = {'mode': 'proxy' if proxy_name else 'own',
                              'proxy_species_id': proxy.pk if proxy else None,
                              'source': row.get(S.CSV_COL_EQUATION_SOURCE.casefold()) or row.get('source') or ''}
                if proxy_name and any(row.get(key) for key in ('b0', 'b1', 'b2')):
                    raise InputValidationError(S.ERR_EQUATION_INVALID)
                for key in ('b0', 'b1', 'b2'):
                    parsed = reader.decimal(row.get(key)) if row.get(key) else None
                    if row.get(key) and parsed is None:
                        raise InputValidationError(S.ERR_EQUATION_INVALID)
                    raw_values[key] = str(parsed) if parsed is not None else ''
                values = volume_values(raw_values) if not proxy_name or proxy else raw_values
                previous = VolumeEquation.objects.filter(species=sp).first() if sp else None
                old = _volume_label(previous) if previous else '—'
                new = _volume_label(values)
            if not issue:
                records.append({'species_id': sp.pk, 'region_id': region.pk if region else None,
                                'values': values})
        except InputValidationError as exc:
            issue = str(exc)
        if issue:
            errors.append(S.ERR_EQUATION_ROW.format(index, issue))
        rows.append([index, name, sp.common_name if sp else '—', region.name if region else region_name,
                     old, new, issue or S.EQUATION_UPDATE])
    if not rows:
        errors.append(S.ERR_CSV_EMPTY)
    if kind == 'volume' and not errors:
        try:
            validate_volume_graphs({r['species_id']: r['values'] for r in records})
        except InputValidationError as exc:
            errors.append(str(exc))
    token = ''
    if state != equation_state():
        raise InputValidationError(S.ERR_EQUATION_PREVIEW_STALE)
    if not errors:
        serializable = [{**record, 'values': {key: str(value) if isinstance(value, Decimal) else value
                         for key, value in record['values'].items()}} for record in records]
        token = signing.dumps({'user_id': request.user.pk, 'kind': kind,
            'state': state, 'records': serializable}, salt=SALT, compress=True)
    data = {'token': token, 'errors': errors, 'columns': [
        'row_id', S.EQUATION_INCOMING, S.EQUATION_MATCHED, S.COL_REGION,
        S.EQUATION_PREVIOUS, S.EQUATION_PROPOSED, S.EQUATION_RESULT], 'rows': rows}
    data['html'] = render_to_string('impostazioni/_equation_import_preview.html', {
        **data, 'missing': missing, 'species': species, 'regions': regions,
    }, request=request)
    return data


def _volume_label(equation):
    get = equation.get if isinstance(equation, dict) else lambda key: getattr(equation, key)
    if get('proxy_species_id'):
        sp = Species.objects.filter(pk=get('proxy_species_id')).first()
        return f'{S.EQUATION_REFERENCE}: {sp.common_name if sp else get("proxy_species_id")}'
    return '; '.join(f'{key}={get(key)}' for key in ('b0', 'b1', 'b2')) + f'; {S.EQUATION_SOURCE}: {get("source")}'


@login_required
@require_writer
@require_POST
def preview_import(request):
    body, error = parse_json_body(request)
    if error:
        return error
    try:
        return JsonResponse(_preview_rows(request, body))
    except (csv_io.CsvError, InputValidationError) as exc:
        return validation_error([str(exc)])


@login_required
@require_writer
@require_POST
def confirm_import(request):
    body, error = parse_json_body(request)
    if error:
        return error
    if not isinstance(body.get('token'), str):
        return validation_error([S.ERR_EQUATION_PREVIEW_EXPIRED])
    try:
        payload = signing.loads(body.get('token', ''), salt=SALT, max_age=PREVIEW_MAX_AGE)
        if payload['user_id'] != request.user.pk:
            raise signing.BadSignature()
    except (signing.BadSignature, TypeError, ValueError, KeyError):
        return validation_error([S.ERR_EQUATION_PREVIEW_EXPIRED])
    begin_nonce_write(request)
    with transaction.atomic():
        species = {sp.pk: sp for sp in lock_species()}
        if payload['state'] != equation_state():
            return JsonResponse({'status': 'conflict', 'message': S.ERR_EQUATION_PREVIEW_STALE}, status=400)
        if payload['kind'] == 'height':
            regions = {region.pk: region for region in Region.objects.all()}
            rows = [hypsometry.ParamRow(regions[r['region_id']], species[r['species_id']],
                Decimal(r['values']['a']), Decimal(r['values']['b']),
                Decimal(r['values']['r2']) if r['values']['r2'] is not None else None,
                r['values']['n']) for r in payload['records']]
            hypsometry.merge_active_set(rows, source='imported')
        else:
            updates = {}
            for row in payload['records']:
                values = row['values']
                updates[row['species_id']] = volume_values({**values,
                    'mode': 'proxy' if values.get('proxy_species_id') else 'own'})
            validate_volume_graphs(updates)
            for pk, values in updates.items():
                persist_volume_equation(species[pk], values)
        invalidate_equations()
        return success_response(request, body,
            patches=[row_patch('species', sp.pk, build_species_record(sp)) for sp in species.values()],
            invalidates={'data_ids': list(dict.fromkeys(key for key in DEPENDENTS if key != 'species')),
                         'prefixes': list(PREFIXES)}, extra={'message': S.EQUATION_IMPORT_SAVED})


@login_required
@require_writer
def export_volume(request):
    delimiter, decimal = csv_io.export_format()
    buf, writer = csv_io.csv_buffer(delimiter)
    writer.writerow([S.CSV_COL_SPECIES.lower(), 'b0', 'b1', 'b2',
                     S.CSV_COL_EQUATION_SOURCE, S.CSV_COL_EQUATION_REFERENCE])
    for equation in VolumeEquation.objects.select_related('species', 'proxy_species').order_by('species__common_name'):
        writer.writerow([equation.species.common_name,
            *[csv_io.format_decimal(getattr(equation, key), decimal) if getattr(equation, key) is not None else ''
              for key in ('b0', 'b1', 'b2')], equation.source,
            equation.proxy_species.common_name if equation.proxy_species_id else ''])
    response = HttpResponse(buf.getvalue(), content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="{S.CSV_FILE_VOLUME_EQUATIONS}"'
    response['Cache-Control'] = 'no-store'
    return response
