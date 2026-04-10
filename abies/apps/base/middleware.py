"""Request/response middleware: CSP, nonce dedup, rate limiting."""

import json
import time
from collections import defaultdict

from django.http import JsonResponse

from apps.base.models import UsedNonce
from config import strings as S


# ---------------------------------------------------------------------------
# Content-Security-Policy
# ---------------------------------------------------------------------------

# Inline styles allowed because vendored libraries (Leaflet) may inject
# <style> elements or set style attributes dynamically.
CSP_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
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

    Only activates on JSON POST requests that include a ``nonce`` field.
    Views are responsible for calling :func:`save_nonce` after a successful
    write so that future replays get the original response.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method != 'POST' or request.content_type != 'application/json':
            return self.get_response(request)

        try:
            body = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self.get_response(request)

        nonce = body.get('nonce') if isinstance(body, dict) else None
        if not nonce:
            return self.get_response(request)

        try:
            used = UsedNonce.objects.get(nonce=nonce)
            return JsonResponse(json.loads(used.response_json))
        except UsedNonce.DoesNotExist:
            return self.get_response(request)


def save_nonce(nonce, user, response_data):
    """Record a used nonce with its success response for idempotency replay."""
    UsedNonce.objects.create(
        nonce=nonce,
        user=user,
        response_json=json.dumps(response_data),
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

RATE_LIMIT = 60        # max requests per window
RATE_WINDOW_S = 60     # window size in seconds
API_PREFIX = '/abies/api/'


class RateLimitMiddleware:
    """Per-user rate limiting on API endpoints.

    Uses in-memory tracking -- suitable for single-process deployments.
    Resets on server restart (acceptable for soft protection).
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self._requests = defaultdict(list)

    def __call__(self, request):
        if not request.path.startswith(API_PREFIX):
            return self.get_response(request)
        if not hasattr(request, 'user') or not request.user.is_authenticated:
            return self.get_response(request)

        user_id = request.user.pk
        now = time.monotonic()
        cutoff = now - RATE_WINDOW_S

        # Prune expired timestamps and check limit.
        timestamps = [t for t in self._requests[user_id] if t > cutoff]
        self._requests[user_id] = timestamps

        if len(timestamps) >= RATE_LIMIT:
            return JsonResponse(
                {'status': 'rate_limited', 'message': S.ERROR_RATE_LIMIT},
                status=429,
            )

        timestamps.append(now)
        return self.get_response(request)
