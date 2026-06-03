"""Shared JSON response shapes for data-entry endpoints (the POST/response
protocol in docs/data-architecture.md).
"""

from django.http import JsonResponse

from config.constants import (
    FIELD_ERRORS, HTML, MESSAGE, STATUS, STATUS_VALIDATION_ERROR,
)


def _validation_response(message: str, errors: list) -> JsonResponse:
    return JsonResponse({
        STATUS: STATUS_VALIDATION_ERROR,
        MESSAGE: message,
        FIELD_ERRORS: errors,
        HTML: '',
    }, status=400)


def validation_error(errors: list) -> JsonResponse:
    """HTTP 400 validation response: every error joined into the message, the
    full list in ``field_errors``, no form HTML."""
    return _validation_response(' '.join(errors), errors)


def csv_error_list(errors: list) -> JsonResponse:
    """HTTP 400 validation response for a CSV import: the first per-row error as
    the message, the full list in ``field_errors``."""
    return _validation_response(errors[0] if errors else '', errors)
