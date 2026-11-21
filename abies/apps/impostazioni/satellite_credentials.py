"""Admin-only Copernicus credential settings."""
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import F
from django.http import JsonResponse
from django.views.decorators.cache import never_cache
from django.views.decorators.debug import sensitive_variables
from django.views.decorators.http import require_http_methods

from apps.base.auth import require_admin
from apps.base.digests import mark_stale
from apps.base.models import Role, SatelliteCredentials, TerrainRefresh, User
from apps.base.responses import parse_json_body, validation_error
from apps.bosco.satellite_credentials import decrypt_secret, encrypt_secret
from config import strings as S
from config.constants import MESSAGE, VERSION


def credential_state(row):
    return {'client_id': row.client_id if row else '', 'configured': bool(decrypt_secret(row)),
            VERSION: row.version if row else 0}


@sensitive_variables()
def submitted_pair(body, row):
    client_id, secret = body.get('client_id'), body.get('client_secret')
    if not isinstance(client_id, str) or not isinstance(secret, str):
        return None
    client_id, secret = client_id.strip(), secret.strip()
    if client_id and len(client_id) <= 255 and len(secret) <= 4096:
        if not secret and row and client_id == row.client_id:
            secret = decrypt_secret(row)
        if secret:
            return client_id, secret
    return None


@login_required
@require_admin
@never_cache
@require_http_methods(['GET', 'POST'])
@sensitive_variables()
def credentials(request):
    rows = SatelliteCredentials.objects.filter(pk=1)
    if request.method == 'GET':
        return JsonResponse(credential_state(rows.first()))
    body, error = parse_json_body(request)
    if error:
        return error
    action = body.get('action')
    if action == 'test':
        from apps.bosco.satellite import SatelliteAuthError, SatelliteError, get_access_token

        pair = submitted_pair(body, rows.first())
        if pair is None:
            return validation_error([S.SATELLITE_CREDENTIALS_REQUIRED])
        try:
            get_access_token(pair)
        except SatelliteAuthError:
            return validation_error([S.SATELLITE_CREDENTIALS_REJECTED])
        except SatelliteError:
            return JsonResponse({MESSAGE: S.SATELLITE_CREDENTIALS_TEST_FAILED}, status=502)
        return JsonResponse({MESSAGE: S.SATELLITE_CREDENTIALS_TESTED})

    with transaction.atomic():
        rows.update(version=F('version'))
        if not User.objects.filter(pk=request.user.pk, is_active=True, role=Role.ADMIN).exists():
            return JsonResponse({MESSAGE: S.ERR_FORBIDDEN}, status=403)
        current = rows.first()
        if type(body.get(VERSION)) is not int or body[VERSION] != (current.version if current else 0):
            return JsonResponse({**credential_state(current), MESSAGE: S.SATELLITE_CREDENTIALS_STALE}, status=409)
        if action == 'remove':
            if current and current.client_id:
                current.client_id = current.encrypted_secret = ''
                current.version += 1
                current.save(update_fields=['client_id', 'encrypted_secret', 'version'])
                mark_stale('audit')
            return JsonResponse({**credential_state(current), MESSAGE: S.SATELLITE_CREDENTIALS_SAVED})
        pair = submitted_pair(body, current) if action == 'save' else None
        if pair is None:
            return validation_error([S.SATELLITE_CREDENTIALS_REQUIRED])
        row = current or SatelliteCredentials()
        row.client_id, secret = pair
        row.encrypted_secret = encrypt_secret(secret)
        if current:
            row.version += 1
        row.save()
        TerrainRefresh.objects.filter(state=TerrainRefresh.State.WAITING).update(
            state=TerrainRefresh.State.QUEUED, retry_at=None,
        )
        mark_stale('audit')
    return JsonResponse({**credential_state(row), MESSAGE: S.SATELLITE_CREDENTIALS_SAVED})
