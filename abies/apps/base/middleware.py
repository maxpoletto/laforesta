"""Request/response middleware: CSP, nonce dedup, rate limiting."""

import json
import time
from collections import defaultdict

from django.conf import settings
from django.db import IntegrityError, transaction
from django.http import JsonResponse
from django.utils import timezone

from apps.base.models import UsedNonce
from apps.base.runtime_snapshot import runtime_snapshot_lock
from apps.base.validation import InputValidationError, validation_error
from apps.base.idempotency import (
    API_PREFIX, NONCE_TTL, NonceReplay, complete_nonce_write,
    prepare_nonce_write, run_nonce_write,
)
from config import strings as S
from config.constants import (
    MESSAGE, STATUS, STATUS_RATE_LIMITED,
)


# ---------------------------------------------------------------------------
# Content-Security-Policy
# ---------------------------------------------------------------------------

# Inline styles allowed because vendored libraries (Leaflet) may inject
# <style> elements or set style attributes dynamically.
#
# img-src whitelist covers Leaflet tile providers used by `map-common.js`:
#   - tile.openstreetmap.org           (OSM basemap)
#   - server.arcgisonline.com          (Esri World Imagery satellite)
#   - tile.opentopomap.org             (OpenTopoMap)
# Any future tile source added to BASEMAPS in map-common.js needs a
# matching entry here. `blob:` is required for local image previews before
# observation photos are uploaded.
TILE_SOURCES = [
    "https://*.tile.openstreetmap.org",
    "https://server.arcgisonline.com",
    "https://*.tile.opentopomap.org",
]

CSP_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    f"img-src 'self' data: blob: {' '.join(TILE_SOURCES)}",
    "font-src 'self'",
    "connect-src 'self'",
    "base-uri 'self'",
    "object-src 'none'",
])


class CSPMiddleware:
    """Add Content-Security-Policy header to every response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response['Content-Security-Policy'] = CSP_POLICY
        return response


# ---------------------------------------------------------------------------
# Idempotency nonce
# ---------------------------------------------------------------------------

class NonceMiddleware:
    """Return cached response for replayed idempotency nonces.

    Parsing precedes the write transaction. Views call begin_nonce_write()
    before mutating data and success_response() to record the receipt; this
    middleware owns the outer transaction until both have succeeded.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        state = prepare_nonce_write(request)
        if state is None:
            return self.get_response(request)
        if isinstance(state, JsonResponse):
            return state
        return run_nonce_write(request, state, self.get_response)

    def process_exception(self, request, exception):
        if isinstance(exception, NonceReplay):
            return exception.response
        return None


def save_nonce(nonce, user, response_data, *, request=None):
    """Record a used nonce with its success response for idempotency replay."""
    if request is not None and complete_nonce_write(request, response_data):
        UsedNonce.objects.filter(created_at__lt=timezone.now() - NONCE_TTL).delete()
        return
    UsedNonce.objects.filter(created_at__lt=timezone.now() - NONCE_TTL).delete()
    try:
        with transaction.atomic():
            UsedNonce.objects.create(
                nonce=nonce,
                user=user,
                response_json=json.dumps(response_data),
            )
    except IntegrityError:
        pass


# ---------------------------------------------------------------------------
# Runtime snapshots
# ---------------------------------------------------------------------------


class RuntimeSnapshotMiddleware:
    """Keep media-changing requests, including commit hooks, outside snapshots.

    Shared locks permit ordinary concurrent writes. Backup takes an exclusive
    lock only while copying the database, primary photos and staged inbox.
    This middleware must wrap NonceMiddleware's transaction and commit hooks.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        touches_media = (
            request.path_info in {
                '/api/bosco/observations/save/', '/api/bosco/observations/delete/',
            }
            or request.path_info.startswith('/api/ipso/uploads/')
        )
        if touches_media and request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            with runtime_snapshot_lock(settings.RUNTIME_SNAPSHOT_LOCK_PATH):
                return self.get_response(request)
        return self.get_response(request)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class InputValidationMiddleware:
    """Translate only explicitly detected invalid values into the API envelope.

    Runs inside NonceMiddleware, so a converted 400 also rolls back its claim
    and mutation. Arithmetic/programming exceptions are not broadly swallowed.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_exception(self, request, exception):
        if request.path_info.startswith(API_PREFIX) and isinstance(exception, InputValidationError):
            return validation_error([str(exception)])
        return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

SAFE_METHODS = frozenset({'GET', 'HEAD', 'OPTIONS'})


class RateLimitMiddleware:
    """Per-user rate limiting on API endpoints.

    Uses separate safe-read and mutation quotas with in-memory tracking --
    suitable for single-process deployments. Resets on server restart
    (acceptable for soft protection).
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self._requests = defaultdict(list)

    def __call__(self, request):
        if not request.path.startswith(API_PREFIX):
            return self.get_response(request)
        if not hasattr(request, 'user') or not request.user.is_authenticated:
            return self.get_response(request)

        bucket = 'read' if request.method in SAFE_METHODS else 'write'
        limit = (
            settings.API_READ_RATE_LIMIT if bucket == 'read'
            else settings.API_WRITE_RATE_LIMIT
        )
        key = (request.user.pk, bucket)
        now = time.monotonic()
        cutoff = now - settings.API_RATE_WINDOW_S

        # Reads and writes have separate quotas: detail-page fan-out must not
        # consume the smaller quota protecting data-entry endpoints.
        timestamps = [t for t in self._requests[key] if t > cutoff]
        self._requests[key] = timestamps

        if len(timestamps) >= limit:
            return JsonResponse(
                {STATUS: STATUS_RATE_LIMITED, MESSAGE: S.ERROR_RATE_LIMIT},
                status=429,
            )

        timestamps.append(now)
        return self.get_response(request)
