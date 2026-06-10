"""Impostazioni (settings) views: password, crews, tractors, species, users."""

from allauth.account.models import EmailAddress
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.base import csv_io, hypsometry
from apps.base.auth import require_admin, require_writer
from apps.base.numparse import parse_decimal
from apps.base.digests import (
    HYPSO_PARAM_COLUMNS, build_harvest_plan_record, build_survey_record,
    hypso_param_row, mark_stale, serve_digest,
)
from apps.base.responses import (
    parse_json_body, row_patch, save_model_response, success_response,
    validation_error,
)
from apps.base.models import (
    Crew, HarvestPlan, HYPSO_FUNC_LN, HypsoParam, HypsoParamSource,
    LoginMethod, Role, Species, Survey, Tractor, TreeSample, User,
)
from config import strings as S
from config.constants import (
    COLUMNS, DIGEST_FUTURE_PRODUCTION, DIGEST_HYPSO_PARAMS,
    DIGEST_PARCEL_DENDROMETRY, FIELD_ACTIVE, FIELD_COMMON_NAME,
    FIELD_CREATED_AT, FIELD_DENSITY, FIELD_EMAIL, FIELD_FILE, FIELD_FIRST_NAME,
    FIELD_HARVEST_PLAN_ID, FIELD_IS_ACTIVE, FIELD_LAST_NAME,
    FIELD_LATIN_NAME, FIELD_LOGIN_METHOD, FIELD_MANUFACTURER, FIELD_MIN_N,
    FIELD_MINOR, FIELD_MODEL, FIELD_NAME,
    FIELD_NOTES, FIELD_PASSWORD1, FIELD_PASSWORD2, FIELD_ROLE,
    FIELD_SOURCE, FIELD_SPECIES, FIELD_SURVEY_IDS, FIELD_SURVEYS,
    FIELD_USERNAME, FIELD_YEAR,
    HTML, MESSAGE, ROWS, ROW_ID, VERSION, is_truthy,
)


# ---------------------------------------------------------------------------
# Password change
# ---------------------------------------------------------------------------

@login_required
@require_POST
def password_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    if request.user.login_method != LoginMethod.PASSWORD:
        return _error(S.ERR_FORBIDDEN)
    pw1 = body.get(FIELD_PASSWORD1, '')
    pw2 = body.get(FIELD_PASSWORD2, '')
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
    return JsonResponse({MESSAGE: S.PASSWORD_CHANGED})


# ---------------------------------------------------------------------------
# Crews
# ---------------------------------------------------------------------------

CREW_COLS = [ROW_ID, S.LABEL_NAME, S.LABEL_NOTES, S.COL_ACTIVE]


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
    body, error = parse_json_body(request)
    if error:
        return error
    parsed = {
        FIELD_NAME: body.get(FIELD_NAME, '').strip(),
        FIELD_NOTES: body.get(FIELD_NOTES, ''),
        FIELD_ACTIVE: is_truthy(body.get(FIELD_ACTIVE)),
    }
    if not parsed[FIELD_NAME]:
        return _error(S.ERR_NAME_REQUIRED)
    # crew.name is a value column in the prelievi digest.
    return save_model_response(
        request, body, model=Crew, data_id='crews', values=parsed,
        row_fn=_crew_row, stale=('prelievi', 'audit'),
    )


# ---------------------------------------------------------------------------
# Tractors
# ---------------------------------------------------------------------------

TRACTOR_COLS = [ROW_ID, S.LABEL_MANUFACTURER, S.LABEL_MODEL,
                S.COL_YEAR, S.COL_ACTIVE]


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
    body, error = parse_json_body(request)
    if error:
        return error
    year = body.get(FIELD_YEAR, '')
    parsed = {
        FIELD_MANUFACTURER: body.get(FIELD_MANUFACTURER, '').strip(),
        FIELD_MODEL: body.get(FIELD_MODEL, '').strip(),
        FIELD_YEAR: int(year) if year else None,
        FIELD_ACTIVE: is_truthy(body.get(FIELD_ACTIVE)),
    }
    if not parsed[FIELD_MANUFACTURER]:
        return _error(S.ERR_NAME_REQUIRED)
    # Tractor labels are columns in the prelievi digest.
    return save_model_response(
        request, body, model=Tractor, data_id='tractors', values=parsed,
        row_fn=_tractor_row, stale=('prelievi', 'audit'),
    )


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

SPECIES_COLS = [ROW_ID, S.LABEL_NAME, S.COL_LATIN_NAME,
                S.LABEL_DENSITY, S.COL_MINOR, S.COL_ACTIVE]


def _species_row(s):
    return [s.id, s.common_name, s.latin_name, float(s.density),
            s.minor, s.active]


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
    body, error = parse_json_body(request)
    if error:
        return error
    density = parse_decimal(body.get(FIELD_DENSITY))
    if density is None or density <= 0:
        return _error(S.ERR_DENSITY_INVALID)
    parsed = {
        FIELD_COMMON_NAME: body.get(FIELD_COMMON_NAME, '').strip(),
        FIELD_LATIN_NAME: body.get(FIELD_LATIN_NAME, '').strip(),
        FIELD_DENSITY: density,
        FIELD_MINOR: is_truthy(body.get(FIELD_MINOR)),
        FIELD_ACTIVE: is_truthy(body.get(FIELD_ACTIVE)),
    }
    if not parsed[FIELD_COMMON_NAME]:
        return _error(S.ERR_NAME_REQUIRED)
    # The "Altro" species backs the minor-aggregation column, so it must
    # itself stay major; otherwise prelievi generation has no bucket to fold
    # minor species into.
    if parsed[FIELD_MINOR] and parsed[FIELD_COMMON_NAME] == S.SPECIES_OTHER:
        return _error(S.ERR_OTHER_NOT_MINOR.format(S.SPECIES_OTHER))
    # species.minor / common_name / sort_order define the prelievi column
    # set; species.json is also consumed by V/m preview forms.
    return save_model_response(
        request, body, model=Species, data_id=FIELD_SPECIES, values=parsed,
        row_fn=_species_row, stale=('prelievi', 'audit', FIELD_SPECIES),
    )



# ---------------------------------------------------------------------------
# Bosco source settings (writer+)
# ---------------------------------------------------------------------------

BOSCO_DIGESTS = (DIGEST_FUTURE_PRODUCTION, DIGEST_PARCEL_DENDROMETRY)


@login_required
@require_writer
def future_production_data(request):
    active = HarvestPlan.objects.filter(active=True).order_by('-year_end', 'id').first()
    default = active or _default_future_plan()
    return JsonResponse({
        'active_id': default.id if default else None,
        'plans': [
            {
                'id': p.id,
                'name': p.name,
                'year_start': p.year_start,
                'year_end': p.year_end,
                'active': bool(default and p.id == default.id),
            }
            for p in HarvestPlan.objects.order_by('-year_start', 'name')
        ],
    })


@login_required
@require_writer
@require_POST
def future_production_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    plan_id = body.get(FIELD_HARVEST_PLAN_ID)
    try:
        plan_id = int(plan_id)
    except (TypeError, ValueError):
        return _error(S.ERR_FUTURE_PLAN_REQUIRED)
    if not HarvestPlan.objects.filter(id=plan_id).exists():
        return _error(S.ERR_PLAN_NOT_FOUND)

    changed = []
    with transaction.atomic():
        active_qs = HarvestPlan.objects.select_for_update().filter(active=True)
        for plan in active_qs.exclude(id=plan_id):
            plan.active = False
            plan.version += 1
            plan.save()
            changed.append(plan)
        selected = HarvestPlan.objects.select_for_update().get(id=plan_id)
        if not selected.active:
            selected.active = True
            selected.version += 1
            selected.save()
            changed.append(selected)
        mark_stale(*BOSCO_DIGESTS, 'harvest_plans', 'audit')
    return success_response(
        request, body,
        patches=[
            row_patch('harvest_plans', plan.id, build_harvest_plan_record(plan))
            for plan in changed
        ],
        extra={MESSAGE: S.FUTURE_PRODUCTION_SAVED},
    )


@login_required
@require_writer
def dendrometry_data(request):
    active_ids = list(
        Survey.objects.filter(active=True).order_by('name').values_list('id', flat=True)
    )
    if not active_ids:
        first = Survey.objects.order_by('name').first()
        active_ids = [first.id] if first else []
    counts = _dendrometry_counts(active_ids)
    active = set(active_ids)
    return JsonResponse({
        'active_ids': active_ids,
        'counts': counts,
        'surveys': [
            {'id': s.id, 'name': s.name, 'active': s.id in active}
            for s in Survey.objects.order_by('name')
        ],
    })


@login_required
@require_writer
@require_POST
def dendrometry_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    raw_ids = body.get(FIELD_SURVEY_IDS)
    if not isinstance(raw_ids, list):
        return _error(S.ERR_DENDROMETRY_SURVEYS_REQUIRED)
    try:
        survey_ids = [int(s) for s in raw_ids]
    except (TypeError, ValueError):
        return _error(S.ERR_DENDROMETRY_SURVEYS_REQUIRED)
    if Survey.objects.exists() and not survey_ids:
        return _error(S.ERR_DENDROMETRY_SURVEYS_REQUIRED)
    if Survey.objects.filter(id__in=survey_ids).count() != len(set(survey_ids)):
        return _error(S.ERR_CSV_SURVEY_REQUIRED)

    desired = set(survey_ids)
    changed = []
    with transaction.atomic():
        qs = Survey.objects.select_for_update().filter(Q(active=True) | Q(id__in=desired))
        for survey in qs:
            active = survey.id in desired
            if survey.active == active:
                continue
            survey.active = active
            survey.version += 1
            survey.save()
            changed.append(survey)
        mark_stale(*BOSCO_DIGESTS, 'surveys', 'audit')
    return success_response(
        request, body,
        patches=[
            row_patch('surveys', survey.id, build_survey_record(survey))
            for survey in changed
        ],
        extra={MESSAGE: S.DENDROMETRY_SAVED},
    )


def _default_future_plan():
    year = timezone.localdate().year
    return (HarvestPlan.objects
            .filter(year_start__lte=year, year_end__gte=year)
            .order_by('-year_end', 'id')
            .first())


def _dendrometry_counts(survey_ids):
    if not survey_ids:
        return {'trees': 0, 'regions': 0, 'parcels': 0}
    qs = TreeSample.objects.filter(sample__survey_id__in=survey_ids)
    return {
        'trees': qs.count(),
        'regions': qs.values('sample__sample_area__parcel__region_id').distinct().count(),
        'parcels': qs.values('sample__sample_area__parcel_id').distinct().count(),
    }

# ---------------------------------------------------------------------------
# Users (admin only)
# ---------------------------------------------------------------------------

USER_COLS = [ROW_ID, S.LABEL_FIRST_NAME, S.LABEL_LAST_NAME, S.LABEL_USERNAME,
             S.LABEL_EMAIL, S.LABEL_LOGIN_METHOD, S.LABEL_CREATED_AT, S.COL_ACTIVE]

ROLE_LABELS = [
    (Role.ADMIN, S.LABEL_ROLE_ADMIN),
    (Role.WRITER, S.LABEL_ROLE_WRITER),
    (Role.READER, S.LABEL_ROLE_READER),
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
    return JsonResponse({HTML: html})


@login_required
@require_admin
@require_POST
def users_save(request):
    body, error = parse_json_body(request)
    if error:
        return error
    row_id = body.get(ROW_ID)
    row_id = int(row_id) if row_id else None

    email = body.get(FIELD_EMAIL, '').strip()
    if not email:
        return _error(S.ERR_EMAIL_REQUIRED)

    login_method = body.get(FIELD_LOGIN_METHOD, LoginMethod.PASSWORD)

    # OAuth users are matched by email; we only need a unique username for
    # Django's bookkeeping, so reuse the email rather than asking the admin.
    if login_method == LoginMethod.OAUTH:
        username = email
    else:
        username = body.get(FIELD_USERNAME, '').strip()
        if not username:
            return _error(S.ERR_USERNAME_REQUIRED)
    role = body.get(FIELD_ROLE, Role.READER)
    active = is_truthy(body.get(FIELD_IS_ACTIVE))
    first_name = body.get(FIELD_FIRST_NAME, '').strip()
    last_name = body.get(FIELD_LAST_NAME, '').strip()

    pw1 = body.get(FIELD_PASSWORD1, '')
    pw2 = body.get(FIELD_PASSWORD2, '')

    if row_id:
        user = User.objects.get(id=row_id)
        user.username = username
        user.email = email
        user.first_name = first_name
        user.last_name = last_name
        user.role = role
        user.login_method = login_method
        user.is_active = active
        if login_method == LoginMethod.OAUTH:
            user.set_unusable_password()
        elif pw1:
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

    return success_response(
        request, body, data_id='users', row_id=user.id,
        patches=[row_patch('users', user.id, _user_row(user))],
    )


# ---------------------------------------------------------------------------
# Hypsometric parameters (writer+).  Managed as a whole set, not row-by-row:
# the table is read-only; compute / import / clear replace the active set,
# archiving the prior one (see docs/hypsometry.md).
# ---------------------------------------------------------------------------

@login_required
def hypso_params_data(request):
    # Any authenticated role: the Piano di taglio mark form also reads this
    # digest to auto-fill h.  Mutating endpoints below stay writer-only.
    return serve_digest(request, DIGEST_HYPSO_PARAMS)


@login_required
@require_writer
def hypso_params_active_set(request):
    """Active-set metadata for the description panel (source=None if none)."""
    s = hypsometry.active_set()
    if s is None:
        return JsonResponse({FIELD_SOURCE: None})
    return JsonResponse({
        FIELD_SOURCE: s.source,
        FIELD_CREATED_AT: s.created_at.date().isoformat(),
        FIELD_MIN_N: s.min_n,
        FIELD_SURVEYS: list(
            s.surveys.order_by(FIELD_NAME).values_list(FIELD_NAME, flat=True)
        ),
    })


def _candidate_payload(rows):
    return {COLUMNS: HYPSO_PARAM_COLUMNS, ROWS: [
        hypso_param_row(None, r.region.name, r.species.common_name,
                        HYPSO_FUNC_LN, r.a, r.b, r.n, r.r2)
        for r in rows
    ]}


def _parse_compute_body(body):
    """Validate {min_n, survey_ids}; returns (survey_ids, min_n, error)."""
    raw_ids = body.get(FIELD_SURVEY_IDS)
    if not isinstance(raw_ids, list) or not raw_ids:
        return None, None, _error(S.ERR_HYPSO_SURVEYS_REQUIRED)
    try:
        survey_ids = [int(s) for s in raw_ids]
        min_n = int(body.get(FIELD_MIN_N))
    except (TypeError, ValueError):
        return None, None, _error(S.ERR_MIN_N_INVALID)
    if min_n < 1:
        return None, None, _error(S.ERR_MIN_N_INVALID)
    return survey_ids, min_n, None


@login_required
@require_writer
@require_POST
def hypso_params_compute(request):
    body, error = parse_json_body(request)
    if error:
        return error
    survey_ids, min_n, err = _parse_compute_body(body)
    if err:
        return err
    rows = hypsometry.compute_params(survey_ids, min_n)
    return JsonResponse(_candidate_payload(rows))


@login_required
@require_writer
@require_POST
def hypso_params_accept(request):
    body, error = parse_json_body(request)
    if error:
        return error
    survey_ids, min_n, err = _parse_compute_body(body)
    if err:
        return err
    rows = hypsometry.compute_params(survey_ids, min_n)
    hypsometry.replace_active_set(
        rows, source=HypsoParamSource.COMPUTED, min_n=min_n,
        survey_ids=survey_ids,
    )
    mark_stale(DIGEST_HYPSO_PARAMS, 'audit')
    return success_response(request, body, extra={MESSAGE: S.HYPSO_SAVED})


@login_required
@require_writer
@require_POST
def hypso_params_import(request):
    body, error = parse_json_body(request)
    if error:
        return error
    try:
        file = csv_io.json_file_bytes(body, FIELD_FILE)
    except csv_io.CsvError as e:
        return _error(str(e))
    if file is None:
        return _error(S.ERR_CSV_FILE_REQUIRED)
    rows, errors = hypsometry.parse_param_csv(file)
    if errors:
        return _error('\n'.join(errors))
    if not rows:
        return _error(S.ERR_CSV_EMPTY)
    hypsometry.replace_active_set(
        rows, source=HypsoParamSource.IMPORTED, min_n=None, survey_ids=[],
    )
    mark_stale(DIGEST_HYPSO_PARAMS, 'audit')
    return success_response(request, body, extra={MESSAGE: S.HYPSO_SAVED})


@login_required
@require_writer
def hypso_params_export(request):
    s = hypsometry.active_set()
    params = []
    if s is not None:
        params = (HypsoParam.objects
                  .filter(param_set=s)
                  .select_related('region', 'species')
                  .order_by('region__name', 'species__common_name'))
    return _hypso_export_response(params)


@login_required
@require_writer
@require_POST
def hypso_params_clear(request):
    body, error = parse_json_body(request)
    if error:
        return error
    hypsometry.clear_active_set()
    mark_stale(DIGEST_HYPSO_PARAMS, 'audit')
    return success_response(request, body, extra={MESSAGE: S.HYPSO_CLEARED})


def _hypso_export_response(params):
    """Stream the active set as CSV in the active locale's format (lowercase
    header; ``;``+``,`` for Italian, ``,``+``.`` otherwise).

    Column order matches the settings table (..., a, b, n, r2); consumers
    read by header name, so the order is for human readability only.
    """
    delimiter, decimal = csv_io.export_format()
    buf, writer = csv_io.csv_buffer(delimiter)
    writer.writerow([
        S.CSV_COL_COMPRESA.lower(), S.CSV_COL_GENERE.lower(),
        S.CSV_COL_FUNZIONE, S.CSV_COL_A, S.CSV_COL_B,
        S.CSV_COL_N_REGRESSION, S.CSV_COL_R2,
    ])
    for p in params:
        writer.writerow([
            p.region.name, p.species.common_name, p.func,
            csv_io.format_decimal(p.a, decimal),
            csv_io.format_decimal(p.b, decimal),
            p.n,
            csv_io.format_decimal(p.r2, decimal),
        ])
    resp = HttpResponse(buf.getvalue(), content_type='text/csv; charset=utf-8')
    resp['Content-Disposition'] = (
        f'attachment; filename="{S.CSV_FILE_REGRESSION}"'
    )
    return resp


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
    return JsonResponse({COLUMNS: columns, ROWS: rows})


def _form(template, model, obj_id, request):
    obj = model.objects.get(id=obj_id) if obj_id else None
    html = render_to_string(template, {'obj': obj}, request=request)
    return JsonResponse({HTML: html})



def _error(message):
    return validation_error([message])


def _validate_password(pw1, pw2, user=None):
    if pw1 != pw2:
        return _error(S.PASSWORD_MISMATCH)
    try:
        validate_password(pw1, user)
    except ValidationError as e:
        return _error(' '.join(e.messages))
    return None
