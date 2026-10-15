"""Request-owned write transactions, claimed just before domain mutations.

Parsing stays in autocommit. The first begin_nonce_write() claims the nonce
and opens the outer transaction; middleware commits it only after the view
has constructed and recorded its success response. Existing atomic blocks
remain nested savepoints, not early commits of the domain data.
"""

import hashlib
import json
import re
from datetime import timedelta

from django.db import IntegrityError, connection, transaction
from django.http import JsonResponse
from django.utils import timezone

from apps.base.models import UsedNonce
from config import strings as S
from config.constants import (
    FIELD_NONCE, MESSAGE, STATUS, STATUS_CONFLICT, STATUS_VALIDATION_ERROR,
)


API_PREFIX = '/api/'
NONCE_TTL = timedelta(hours=24)
_STATE_ATTRIBUTE = '_abies_nonce_write'
_WRITE_COMMANDS = {'INSERT', 'UPDATE', 'DELETE', 'REPLACE'}
_CTE_WRITE = re.compile(r'\b(?:INSERT|UPDATE|DELETE|REPLACE)\b', re.IGNORECASE)


class NonceReplay(Exception):
    """Internal short circuit handled by NonceMiddleware.process_exception."""

    def __init__(self, response):
        self.response = response


def prepare_nonce_write(request):
    """Return request state, an invalid-nonce response, or None when unused."""
    if request.method != 'POST' or not request.path_info.startswith(API_PREFIX):
        return None
    # The public mobile upload protocol uses its own session UUID/checksum
    # receipts, not the authenticated web forms' nonce protocol.
    if request.path_info == '/api/ipso/uploads/':
        return None
    user = getattr(request, 'user', None)
    if user is None or not user.is_authenticated:
        return None
    if request.content_type in ('multipart/form-data', 'application/x-www-form-urlencoded'):
        body = request.POST
    else:
        # JSON views also accept JSON sent with a nonstandard Content-Type.
        # Such requests must not bypass their normal idempotency guarantees.
        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
    if not isinstance(body, dict) or FIELD_NONCE not in body:
        return None
    nonce = body[FIELD_NONCE]
    if nonce is None or nonce == '':
        return None
    if not isinstance(nonce, str) or not nonce.strip() or len(nonce) > 64:
        return JsonResponse({
            STATUS: STATUS_VALIDATION_ERROR, MESSAGE: S.ERR_NONCE_INVALID,
        }, status=400)
    return NonceWrite(request, nonce, _request_hash(request, body))


def _request_hash(request, body):
    if hasattr(body, 'getlist'):
        fields = {key: body.getlist(key) for key in body if key != 'csrfmiddlewaretoken'}
        files = {}
        for key in request.FILES:
            files[key] = []
            for uploaded in request.FILES.getlist(key):
                checksum = hashlib.sha256()
                try:
                    for chunk in uploaded.chunks():
                        checksum.update(chunk)
                finally:
                    uploaded.seek(0)
                files[key].append([
                    uploaded.name, uploaded.content_type, uploaded.size, checksum.hexdigest(),
                ])
        payload = ['form', fields, files]
    else:
        payload = ['json', body]
    checksum = hashlib.sha256()
    identity = [request.method, request.get_full_path(), payload]
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':')).iterencode(identity):
        checksum.update(chunk.encode('utf-8'))
    return checksum.hexdigest()


def begin_nonce_write(request):
    """Call after parsing, before the first write or mutating import helper.

    With no nonce this is a no-op and existing transaction boundaries apply.
    With a nonce the transaction remains owned by middleware through response
    construction and receipt persistence. A concurrent replay short-circuits
    the view before it can repeat the mutation.
    """
    state = getattr(request, _STATE_ATTRIBUTE, None)
    if state is not None:
        state.begin()


def complete_nonce_write(request, response_data):
    state = getattr(request, _STATE_ATTRIBUTE, None)
    if state is None:
        return False
    if state.receipt is None:
        raise RuntimeError('Call begin_nonce_write() before writing domain data')
    state.receipt.response_json = json.dumps(response_data)
    state.receipt.save(update_fields=['response_json'])
    state.completed = True
    return True


def on_nonce_rollback(request, callback):
    """Compensate an unavoidable file rewrite if this request rolls back.

    Prefer transaction.on_commit() for destructive file operations. Rewrites
    needed to construct the success response can instead register restoration
    here, before touching the files. This does not make files crash-atomic.
    """
    state = getattr(request, _STATE_ATTRIBUTE, None)
    if state is not None:
        state.rollback_callbacks.append(callback)


def after_nonce_rollback(request, callback):
    """Queue error diagnostics outside the rolled-back domain transaction.

    Return False when the request is not nonce-managed; the caller then keeps
    its existing diagnostic-save behavior. Never use this for domain writes.
    """
    state = getattr(request, _STATE_ATTRIBUTE, None)
    if state is None:
        return False
    state.after_rollback_callbacks.append(callback)
    return True


class NonceWrite:
    def __init__(self, request, nonce, request_hash):
        self.request = request
        self.nonce = nonce
        self.request_hash = request_hash
        self.atomic = None
        self.receipt = None
        self.completed = False
        self.replayed = False
        self.committed = False
        self.rollback_callbacks = []
        self.after_rollback_callbacks = []

    def cached_response(self):
        used = UsedNonce.objects.filter(
            nonce=self.nonce, user=self.request.user,
            created_at__gte=timezone.now() - NONCE_TTL,
        ).first()
        return self._response(used) if used is not None else None

    def _response(self, used):
        # Preserve pre-upgrade receipts: their original request cannot be
        # reconstructed. They expire normally rather than allowing duplicates.
        if used.request_hash and used.request_hash != self.request_hash:
            return JsonResponse({
                STATUS: STATUS_CONFLICT, MESSAGE: S.ERR_NONCE_REUSED,
            }, status=409)
        return JsonResponse(json.loads(used.response_json))

    def begin(self):
        if self.atomic is not None:
            return
        atomic = transaction.atomic()
        atomic.__enter__()
        self.atomic = atomic
        transaction.on_commit(self._mark_committed)
        try:
            # Claim as the first write, before reading a transaction snapshot.
            # The unique constraint serializes competing claims. The inner
            # savepoint lets us inspect the winning receipt after a conflict.
            with transaction.atomic():
                self.receipt = UsedNonce.objects.create(
                    nonce=self.nonce, user=self.request.user,
                    request_hash=self.request_hash, response_json='',
                )
        except IntegrityError:
            used = UsedNonce.objects.get(nonce=self.nonce, user=self.request.user)
            if used.created_at < timezone.now() - NONCE_TTL:
                used.delete()
                self.receipt = UsedNonce.objects.create(
                    nonce=self.nonce, user=self.request.user,
                    request_hash=self.request_hash, response_json='',
                )
            else:
                self.replayed = True
                raise NonceReplay(self._response(used)) from None

    def check_write_boundary(self, execute, sql, params, many, context):
        command = sql.lstrip().split(None, 1)[0].upper()
        writes = command in _WRITE_COMMANDS or (
            command == 'WITH' and _CTE_WRITE.search(sql)
        )
        if writes and self.atomic is None:
            raise RuntimeError('Call begin_nonce_write() before writing domain data')
        return execute(sql, params, many, context)

    def _mark_committed(self):
        self.committed = True

    def _restore_files(self):
        callbacks, self.rollback_callbacks = self.rollback_callbacks, []
        for callback in reversed(callbacks):
            callback()

    def finish(self, success):
        if self.atomic is None:
            self._save_error_diagnostics()
            return
        atomic, self.atomic = self.atomic, None
        complete = success and self.completed
        if not complete:
            transaction.set_rollback(True)
            try:
                # Restore while still holding the write lock, before a retry
                # can use or rewrite the same files.
                self._restore_files()
            finally:
                atomic.__exit__(None, None, None)
            self._save_error_diagnostics()
        else:
            try:
                atomic.__exit__(None, None, None)
            except BaseException:
                # A failed COMMIT is different from a failing on_commit hook:
                # never undo files after the database has actually committed.
                if not self.committed:
                    self._restore_files()
                raise
        if success and not self.completed and not self.replayed:
            raise RuntimeError('Nonce writes must return success_response()')

    def _save_error_diagnostics(self):
        callbacks, self.after_rollback_callbacks = self.after_rollback_callbacks, []
        for callback in callbacks:
            callback()


def run_nonce_write(request, state, get_response):
    cached = state.cached_response()
    if cached is not None:
        return cached
    setattr(request, _STATE_ATTRIBUTE, state)
    try:
        with connection.execute_wrapper(state.check_write_boundary):
            response = get_response(request)
        # Django can convert a view exception to an HTTP error before returning
        # through middleware. Error responses must roll back too.
        state.finish(200 <= response.status_code < 300)
        return response
    except NonceReplay as replay:
        state.finish(False)
        return replay.response
    except BaseException:
        state.finish(False)
        raise
    finally:
        delattr(request, _STATE_ATTRIBUTE)
