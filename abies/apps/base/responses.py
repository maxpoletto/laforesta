"""Shared JSON response shapes for data-entry endpoints (the POST/response
protocol in docs/data-architecture.md).
"""

from django.http import JsonResponse

from apps.base.middleware import save_nonce
from config import strings as S
from config.constants import (
    DATA_ID, DELETES, FIELD_ERRORS, FIELD_NONCE, HTML, MESSAGE, PATCHES,
    RECORD, ROW_ID, STATUS, STATUS_CONFLICT,
    STATUS_VALIDATION_ERROR,
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


def csv_error_list(errors: list) -> JsonResponse:
    """HTTP 400 validation response for a CSV import: the first per-row error as
    the message, the full list in ``field_errors``."""
    return _validation_response(errors[0] if errors else '', errors)
