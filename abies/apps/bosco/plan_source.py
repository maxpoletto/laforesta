"""Shared harvest-plan source for Bosco's planned-harvest maps."""

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from apps.base.auth import require_writer
from apps.base.digests import build_harvest_plan_record, mark_stale
from apps.base.idempotency import begin_nonce_write
from apps.base.models import HarvestPlan
from apps.base.responses import parse_json_body, row_patch, success_response, validation_error
from apps.base.selectors import active_or_default_harvest_plan
from config import strings as S
from config.constants import (
    DIGEST_FUTURE_PRODUCTION, FIELD_ACTIVE, FIELD_ACTIVE_ID, FIELD_DATA_IDS,
    FIELD_HARVEST_PLAN_ID, FIELD_ID, FIELD_NAME, FIELD_PLANS,
    FIELD_YEAR_START, FIELD_YEAR_END, MESSAGE,
)


def plan_error(message):
    return validation_error([message])


@login_required
def data_view(request):
    default = active_or_default_harvest_plan()
    return JsonResponse({
        FIELD_ACTIVE_ID: default.id if default else None,
        FIELD_PLANS: [
            {
                FIELD_ID: p.id,
                FIELD_NAME: p.name,
                FIELD_YEAR_START: p.year_start,
                FIELD_YEAR_END: p.year_end,
                FIELD_ACTIVE: bool(default and p.id == default.id),
            }
            for p in HarvestPlan.objects.order_by('-year_start', 'name')
        ],
    })


@login_required
@require_writer
@require_POST
def save_view(request):
    body, error = parse_json_body(request)
    if error:
        return error
    plan_id = body.get(FIELD_HARVEST_PLAN_ID)
    try:
        plan_id = int(plan_id)
    except (TypeError, ValueError):
        return plan_error(S.ERR_FUTURE_PLAN_REQUIRED)
    if not HarvestPlan.objects.filter(id=plan_id).exists():
        return plan_error(S.ERR_PLAN_NOT_FOUND)

    changed = []
    begin_nonce_write(request)
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
        if changed:
            mark_stale(DIGEST_FUTURE_PRODUCTION, 'harvest_plans', 'audit')
    return success_response(
        request, body,
        patches=[
            row_patch('harvest_plans', plan.id, build_harvest_plan_record(plan))
            for plan in changed
        ],
        invalidates={FIELD_DATA_IDS: [DIGEST_FUTURE_PRODUCTION]} if changed else None,
        extra={MESSAGE: S.BOSCO_PLAN_SOURCE_SAVED},
    )
