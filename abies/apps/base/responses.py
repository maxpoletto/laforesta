"""Shared JSON response shapes for data-entry endpoints (the POST/response
protocol in docs/data-architecture.md).
"""

import json

from django.db import transaction
from django.http import JsonResponse

from apps.base.digests import mark_stale
from apps.base.middleware import save_nonce
from apps.base.numparse import int_or_none
from config import strings as S
from config.constants import (
    DATA_ID, DELETES, FIELD_ERRORS, FIELD_NONCE, HTML, MESSAGE, PATCHES,
    RECORD, ROW_ID, STATUS, STATUS_CONFLICT, STATUS_NOT_FOUND,
    STATUS_VALIDATION_ERROR, VERSION,
)


def row_patch(data_id: str, row_id: int, record: list) -> dict:
    """One digest row update in the generic optimistic-update envelope."""
    return {DATA_ID: data_id, ROW_ID: row_id, RECORD: record}


def row_patches(data_id: str, records: list[list]) -> list[dict]:
    """N digest row updates. Each record carries its row id at index 0."""
    return [row_patch(data_id, r[0], r) for r in records]


def row_delete(data_id: str, row_id: int) -> dict:
    """One digest row removal in the generic optimistic-update envelope."""
    return {DATA_ID: data_id, ROW_ID: row_id}


def save_model_response(
        request,
        body: dict,
        *,
        model,
        data_id: str,
        values: dict,
        row_fn,
        stale: tuple[str, ...] = (),
        row_id: int | None = None,
        unique_field: str | None = None,
        unique_value=None,
        unique_error: str | None = None,
        unique_case_insensitive: bool = False,
        conflict_fn=None,
        extra_patches=None,
        extra: dict | None = None,
) -> JsonResponse:
    """Create/update a small TimestampedModel and return the standard write
    envelope.

    This is intentionally narrow: callers still parse and validate their own
    domain fields.  The helper owns only the repeated optimistic-lock, optional
    unique-field check, stale marking, and row-patch response plumbing.
    """
    row_id = row_id if row_id is not None else _row_id_from_body(body)

    if unique_error and unique_field and unique_value is not None:
        lookup = f'{unique_field}__iexact' if unique_case_insensitive else unique_field
        dup = model.objects.filter(**{lookup: unique_value})
        if row_id is not None:
            dup = dup.exclude(id=row_id)
        if dup.exists():
            return validation_error([unique_error])

    with transaction.atomic():
        if row_id is not None:
            obj = model.objects.select_for_update().filter(id=row_id).first()
            if obj is None:
                return JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
            if obj.version != submitted_version(body):
                if conflict_fn:
                    return conflict_fn(obj)
                return conflict_response(
                    data_id=data_id, row_id=obj.id, record=row_fn(obj),
                )
            for field, value in values.items():
                setattr(obj, field, value)
            obj.version += 1
            obj.save()
        else:
            obj = model.objects.create(**values)
        if stale:
            mark_stale(*stale)

    patches = [row_patch(data_id, obj.id, row_fn(obj))]
    if extra_patches:
        patches.extend(extra_patches(obj) if callable(extra_patches) else extra_patches)
    return success_response(
        request, body, data_id=data_id, row_id=obj.id, patches=patches,
        extra=extra,
    )


def submitted_version(body: dict) -> int:
    """Optimistic-lock version from a JSON body; invalid/missing is stale."""
    return int_or_none(body.get(VERSION)) or 0


def _row_id_from_body(body: dict) -> int | None:
    row_id = body.get(ROW_ID)
    return int(row_id) if row_id not in (None, '') else None


def success_response(
        request,
        body: dict | None,
        *,
        data_id: str | None = None,
        row_id: int | None = None,
        patches: list[dict] | None = None,
        deletes: list[dict] | None = None,
        extra: dict | None = None,
) -> JsonResponse:
    """HTTP 200 write response with generic cache changes and nonce save.

    Row payloads are accepted only through ``patches``/``deletes``.  The
    top-level ``data_id``/``row_id`` keys remain as lightweight identifiers
    for callers that need to select or navigate to the affected entity.
    """
    response_data: dict = {}
    if data_id is not None:
        response_data[DATA_ID] = data_id
    if row_id is not None:
        response_data[ROW_ID] = row_id
    if extra:
        response_data.update(extra)

    if patches:
        response_data[PATCHES] = list(patches)
    if deletes:
        response_data[DELETES] = list(deletes)

    nonce = body.get(FIELD_NONCE) if body else None
    if nonce:
        save_nonce(nonce, request.user, response_data)
    return JsonResponse(response_data)


def conflict_response(
        *,
        data_id: str,
        row_id: int,
        record: list,
        html: str = '',
        message: str = S.ERROR_CONFLICT,
        extra: dict | None = None,
) -> JsonResponse:
    """HTTP 400 optimistic-lock conflict with current row data."""
    response_data = {
        STATUS: STATUS_CONFLICT,
        MESSAGE: message,
        DATA_ID: data_id,
        ROW_ID: row_id,
        RECORD: record,
        PATCHES: [row_patch(data_id, row_id, record)],
    }
    if html:
        response_data[HTML] = html
    if extra:
        response_data.update(extra)
    return JsonResponse(response_data, status=400)


def _validation_response(message: str, errors: list, html: str = '') -> JsonResponse:
    return JsonResponse({
        STATUS: STATUS_VALIDATION_ERROR,
        MESSAGE: message,
        FIELD_ERRORS: errors,
        HTML: html,
    }, status=400)


def validation_error(errors: list, html: str = '') -> JsonResponse:
    """HTTP 400 validation response: every error joined into the message, the
    full list in ``field_errors``, optional replacement form HTML.
    """
    return _validation_response(' '.join(errors), errors, html)


def parse_json_body(request) -> tuple[dict | None, JsonResponse | None]:
    """Parse a JSON object request body, or return a 400 validation response."""
    try:
        body = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return None, validation_error([S.ERR_JSON_INVALID])
    if not isinstance(body, dict):
        return None, validation_error([S.ERR_JSON_INVALID])
    return body, None


def csv_error_list(errors: list) -> JsonResponse:
    """HTTP 400 validation response for a CSV import: the first per-row error as
    the message, the full list in ``field_errors``."""
    return _validation_response(errors[0] if errors else '', errors)
