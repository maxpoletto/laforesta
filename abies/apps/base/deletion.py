"""Shared parsing and write-boundary checks for single and selected deletes."""
from django.db import connection
from django.db.models import F
from django.http import JsonResponse

from apps.base.numparse import int_or_none
from apps.base.responses import check_delete_version, validation_error
from apps.base.validation import json_fields_valid
from config import strings as S
from config.constants import ROW_ID, ROWS, STATUS, STATUS_NOT_FOUND


def parse_delete_rows(body, *, row_id=None):
    """Normalize the single-row body or an explicit nonempty selection.

    Each row carries its own cached edit token. Never interpret an omitted or
    empty selection as all rows, and never silently coerce floats or booleans.
    """
    if ROWS in body:
        rows = body[ROWS]
        if row_id is not None or ROW_ID in body or not isinstance(rows, list) or not rows:
            return None, validation_error([S.ERR_DELETE_SELECTION_INVALID])
    else:
        rows = [{**body, ROW_ID: row_id if row_id is not None else body.get(ROW_ID)}]
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not json_fields_valid(row):
            return None, validation_error([S.ERR_DELETE_SELECTION_INVALID])
        pk = int_or_none(row.get(ROW_ID))
        if pk is None or pk <= 0 or pk in result:
            message = (S.ERR_DELETE_SELECTION_INVALID if ROWS in body
                       else S.ERR_ROW_ID_INVALID)
            return None, validation_error([message])
        result[pk] = row
    return result, None


def lock_delete_rows(queryset, rows, *, data_id, record_fn):
    """Claim and check every row before the caller deletes anything.

    The caller owns transaction.atomic(), parent locks, domain protections,
    dirtiness and the success envelope. A missing/out-of-scope row fails the
    entire selection. The no-op UPDATE serializes SQLite even without a nonce;
    PostgreSQL uses deterministic row locks instead.
    """
    queryset = queryset.filter(pk__in=rows).order_by('pk')
    if not connection.features.has_select_for_update:
        queryset.update(version=F('version'))
    objects = list(queryset.select_for_update(of=('self',)))
    if len(objects) != len(rows):
        return None, JsonResponse({STATUS: STATUS_NOT_FOUND}, status=404)
    for obj in objects:
        key = data_id(obj) if callable(data_id) else data_id
        error = check_delete_version(obj, rows[obj.pk], data_id=key, record_fn=record_fn)
        if error:
            return None, error
    return objects, None
