"""Impostazioni (settings) views: password, crews, tractors, species, users."""

import json
from decimal import Decimal, InvalidOperation

from allauth.account.models import EmailAddress
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.http import require_POST

from apps.base.auth import require_admin, require_writer
from apps.base.digests import mark_stale
from apps.base.middleware import save_nonce
from apps.base.models import Crew, LoginMethod, Role, Species, Tractor, User
from config import strings as S


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------

@login_required
@require_POST
def password_view(request):
    body = json.loads(request.body)
    pw1 = body.get('password1', '')
    pw2 = body.get('password2', '')
    if not pw1:
        return _error(S.ERR_PASSWORD_REQUIRED)
    if pw1 != pw2:
        return _error(S.PASSWORD_MISMATCH)
    try:
        validate_password(pw1, request.user)
    except ValidationError as e:
        return _error(' '.join(e.messages))
    request.user.set_password(pw1)
    request.user.save()
    update_session_auth_hash(request, request.user)
    return JsonResponse({S.MESSAGE: S.PASSWORD_CHANGED})


# ---------------------------------------------------------------------------
# Crews
# ---------------------------------------------------------------------------

CREW_COLS = [S.ROW_ID, S.LABEL_NAME, S.LABEL_NOTES, S.COL_ACTIVE]


def _crew_row(c):
    return [c.id, c.name, c.notes, c.active]


@login_required
@require_writer
def crews_data(request):
    return _list(Crew, CREW_COLS, _crew_row)


@login_required
@require_writer
def crews_form(request, obj_id=None):
    return _form('impostazioni/_crew_form.html', Crew, obj_id, request)


@login_required
@require_writer
@require_POST
def crews_save(request):
    body = json.loads(request.body)
    parsed = {
        S.FIELD_NAME: body.get(S.FIELD_NAME, '').strip(),
        S.FIELD_NOTES: body.get(S.FIELD_NOTES, ''),
        S.FIELD_ACTIVE: body.get(S.FIELD_ACTIVE) == 'true',
    }
    if not parsed[S.FIELD_NAME]:
        return _error(S.ERR_NAME_REQUIRED)
    obj, err = _save(Crew, body, parsed)
    if err:
        return err
    mark_stale('audit')
    return _saved(obj, _crew_row, body, request)


# ---------------------------------------------------------------------------
# Tractors
# ---------------------------------------------------------------------------

TRACTOR_COLS = [S.ROW_ID, S.LABEL_MANUFACTURER, S.LABEL_MODEL,
                S.LABEL_YEAR, S.COL_ACTIVE]


def _tractor_row(t):
    return [t.id, t.manufacturer, t.model, t.year or '', t.active]


@login_required
@require_writer
def tractors_data(request):
    return _list(Tractor, TRACTOR_COLS, _tractor_row)


@login_required
@require_writer
def tractors_form(request, obj_id=None):
    return _form('impostazioni/_tractor_form.html', Tractor, obj_id, request)


@login_required
@require_writer
@require_POST
def tractors_save(request):
    body = json.loads(request.body)
    year = body.get(S.FIELD_YEAR, '')
    parsed = {
        S.FIELD_MANUFACTURER: body.get(S.FIELD_MANUFACTURER, '').strip(),
        S.FIELD_MODEL: body.get(S.FIELD_MODEL, '').strip(),
        S.FIELD_YEAR: int(year) if year else None,
        S.FIELD_ACTIVE: body.get(S.FIELD_ACTIVE) == 'true',
    }
    if not parsed[S.FIELD_MANUFACTURER]:
        return _error(S.ERR_NAME_REQUIRED)
    obj, err = _save(Tractor, body, parsed)
    if err:
        return err
    mark_stale('audit')
    return _saved(obj, _tractor_row, body, request)


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

SPECIES_COLS = [S.ROW_ID, S.LABEL_NAME, S.LABEL_LATIN_NAME,
                S.LABEL_DENSITY, S.COL_ACTIVE]


def _species_row(s):
    return [s.id, s.common_name, s.latin_name, float(s.density), s.active]


@login_required
@require_writer
def species_data(request):
    return _list(Species, SPECIES_COLS, _species_row)


@login_required
@require_writer
def species_form(request, obj_id=None):
    return _form('impostazioni/_species_form.html', Species, obj_id, request)


@login_required
@require_writer
@require_POST
def species_save(request):
    body = json.loads(request.body)
    try:
        density = Decimal(str(body.get(S.FIELD_DENSITY, '0') or '0'))
    except InvalidOperation:
        return _error(S.ERR_DENSITY_INVALID)
    if density <= 0:
        return _error(S.ERR_DENSITY_INVALID)
    parsed = {
        S.FIELD_COMMON_NAME: body.get(S.FIELD_COMMON_NAME, '').strip(),
        S.FIELD_LATIN_NAME: body.get(S.FIELD_LATIN_NAME, '').strip(),
        S.FIELD_DENSITY: density,
        S.FIELD_ACTIVE: body.get(S.FIELD_ACTIVE) == 'true',
    }
    if not parsed[S.FIELD_COMMON_NAME]:
        return _error(S.ERR_NAME_REQUIRED)
    obj, err = _save(Species, body, parsed)
    if err:
        return err
    # species.json is consumed by V/m preview forms; bust on edits.
    mark_stale('audit', S.FIELD_SPECIES)
    return _saved(obj, _species_row, body, request)


# ---------------------------------------------------------------------------
# Users (admin only)
# ---------------------------------------------------------------------------

USER_COLS = [S.ROW_ID, S.LABEL_FIRST_NAME, S.LABEL_LAST_NAME, S.LABEL_USERNAME,
             S.LABEL_EMAIL, S.LABEL_LOGIN_METHOD, S.LABEL_CREATED_AT, S.COL_ACTIVE]

ROLE_LABELS = [
    (Role.ADMIN, S.ROLE_ADMIN),
    (Role.WRITER, S.ROLE_WRITER),
    (Role.READER, S.ROLE_READER),
]


def _user_row(u):
    return [u.id, u.first_name, u.last_name, u.username, u.email,
            u.login_method, u.date_joined.strftime('%Y-%m-%d'), u.is_active]


@login_required
@require_admin
def users_data(request):
    return _list(User, USER_COLS, _user_row)


@login_required
@require_admin
def users_form(request, obj_id=None):
    obj = User.objects.get(id=obj_id) if obj_id else None
    html = render_to_string('impostazioni/_user_form.html', {
        'obj': obj,
        'roles': ROLE_LABELS,
        'login_methods': LoginMethod.choices,
    }, request=request)
    return JsonResponse({S.HTML: html})


@login_required
@require_admin
@require_POST
def users_save(request):
    body = json.loads(request.body)
    row_id = body.get(S.ROW_ID)
    row_id = int(row_id) if row_id else None

    email = body.get(S.FIELD_EMAIL, '').strip()
    if not email:
        return _error(S.ERR_EMAIL_REQUIRED)

    login_method = body.get(S.FIELD_LOGIN_METHOD, LoginMethod.PASSWORD)

    # OAuth users are matched by email; we only need a unique username for
    # Django's bookkeeping, so reuse the email rather than asking the admin.
    if login_method == LoginMethod.OAUTH:
        username = email
    else:
        username = body.get(S.FIELD_USERNAME, '').strip()
        if not username:
            return _error(S.ERR_USERNAME_REQUIRED)
    role = body.get(S.FIELD_ROLE, Role.READER)
    active = body.get(S.FIELD_IS_ACTIVE) == 'true'
    first_name = body.get(S.FIELD_FIRST_NAME, '').strip()
    last_name = body.get(S.FIELD_LAST_NAME, '').strip()

    pw1 = body.get('password1', '')
    pw2 = body.get('password2', '')

    if row_id:
        user = User.objects.get(id=row_id)
        user.username = username
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.role = role
        user.login_method = login_method
        user.is_active = active
        if pw1:
            err = _validate_password(pw1, pw2, user)
            if err:
                return err
            user.set_password(pw1)
        user.save()
    else:
        if login_method == LoginMethod.PASSWORD:
            if not pw1:
                return _error(S.ERR_PASSWORD_REQUIRED)
            err = _validate_password(pw1, pw2)
            if err:
                return err
        user = User.objects.create_user(
            username=username, email=email, password=pw1 or None,
            first_name=first_name, last_name=last_name,
            role=role, login_method=login_method, is_active=active,
        )
        if login_method != LoginMethod.PASSWORD:
            user.set_unusable_password()
            user.save()

    # allauth matches incoming OAuth logins against verified EmailAddress
    # rows, not against user.email.  Keep those rows in sync with the user.
    _sync_email_address(user)

    mark_stale('audit')

    nonce = body.get(S.FIELD_NONCE)
    response_data = {S.ROW_ID: user.id, S.RECORD: _user_row(user)}
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sync_email_address(user):
    """Keep allauth's EmailAddress table aligned with user.email.

    OAuth users get exactly one verified primary row matching user.email.
    Password users get none (we don't use email auth for them).
    """
    EmailAddress.objects.filter(user=user).delete()
    if user.login_method == LoginMethod.OAUTH and user.email:
        EmailAddress.objects.create(
            user=user, email=user.email, verified=True, primary=True,
        )


def _list(model, columns, row_fn):
    rows = [row_fn(obj) for obj in model.objects.order_by('pk')]
    return JsonResponse({S.COLUMNS: columns, S.ROWS: rows})


def _form(template, model, obj_id, request):
    obj = model.objects.get(id=obj_id) if obj_id else None
    html = render_to_string(template, {'obj': obj}, request=request)
    return JsonResponse({S.HTML: html})


def _save(model, body, parsed):
    """Create or update a TimestampedModel with optimistic locking."""
    row_id = body.get(S.ROW_ID)
    row_id = int(row_id) if row_id else None

    with transaction.atomic():
        if row_id:
            version = int(body.get(S.VERSION, 0))
            obj = model.objects.select_for_update().get(id=row_id)
            if obj.version != version:
                return None, JsonResponse({
                    S.STATUS: S.STATUS_CONFLICT, S.MESSAGE: S.ERROR_CONFLICT,
                }, status=400)
            for field, value in parsed.items():
                setattr(obj, field, value)
            obj.version += 1
            obj.save()
        else:
            obj = model.objects.create(**parsed)

    return obj, None


def _saved(obj, row_fn, body, request):
    nonce = body.get(S.FIELD_NONCE)
    response_data = {S.ROW_ID: obj.id, S.RECORD: row_fn(obj)}
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


def _error(message):
    return JsonResponse({S.STATUS: S.STATUS_VALIDATION_ERROR, S.MESSAGE: message}, status=400)


def _validate_password(pw1, pw2, user=None):
    if pw1 != pw2:
        return _error(S.PASSWORD_MISMATCH)
    try:
        validate_password(pw1, user)
    except ValidationError as e:
        return _error(' '.join(e.messages))
    return None
