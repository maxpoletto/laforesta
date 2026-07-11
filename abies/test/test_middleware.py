"""Tests for middleware: CSP, nonce dedup, rate limiting."""

import json
from datetime import timedelta

import pytest
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.test import Client, RequestFactory

from apps.base.middleware import (
    CSPMiddleware, NonceMiddleware, RateLimitMiddleware, save_nonce,
)
from apps.base.models import UsedNonce
from config import strings as S
from config.constants import (
    DATA_ID, FIELD_DATE, FIELD_NONCE, RECORD, ROW_ID, STATUS,
    STATUS_RATE_LIMITED,
)


# -- Helpers ----------------------------------------------------------------

def ok_response(request):
    return HttpResponse('OK')


def json_ok_response(request):
    return JsonResponse({'result': 'ok'})


# -- CSP -------------------------------------------------------------------

class TestCSPMiddleware:
    def test_header_on_html_page(self, db):
        resp = Client().get('/login/')
        csp = resp['Content-Security-Policy']
        assert "script-src 'self'" in csp
        assert "default-src 'self'" in csp
        assert "base-uri 'self'" in csp
        assert "object-src 'none'" in csp

    def test_blocks_inline_scripts(self, db):
        resp = Client().get('/login/')
        csp = resp['Content-Security-Policy']
        assert "'unsafe-inline'" not in csp.split('script-src')[1].split(';')[0]

    def test_header_on_arbitrary_view(self):
        mw = CSPMiddleware(ok_response)
        resp = mw(RequestFactory().get('/anything/'))
        assert 'Content-Security-Policy' in resp


# -- Nonce -----------------------------------------------------------------

class TestNonceMiddleware:
    def test_replay_returns_cached_response(self, admin_user):
        cached = {DATA_ID: 'prelievi', ROW_ID: 42, RECORD: [42, '2024-01-01']}
        save_nonce('nonce-aaa', admin_user, cached)

        request = RequestFactory().post(
            '/api/prelievi/save/',
            data=json.dumps({FIELD_NONCE: 'nonce-aaa', FIELD_DATE: '2024-01-01'}),
            content_type='application/json',
        )
        request.user = admin_user
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == cached

    def test_fresh_nonce_passes_through(self, db):
        request = RequestFactory().post(
            '/api/prelievi/save/',
            data=json.dumps({FIELD_NONCE: 'nonce-bbb'}),
            content_type='application/json',
        )
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == {'result': 'ok'}

    def test_replay_is_scoped_to_user(self, admin_user, writer_user):
        cached = {DATA_ID: 'prelievi', ROW_ID: 42, RECORD: [42, '2024-01-01']}
        save_nonce('nonce-scoped', admin_user, cached)

        request = RequestFactory().post(
            '/api/prelievi/save/',
            data=json.dumps({FIELD_NONCE: 'nonce-scoped'}),
            content_type='application/json',
        )
        request.user = writer_user
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == {'result': 'ok'}

    def test_no_nonce_field_passes_through(self, db):
        request = RequestFactory().post(
            '/api/prelievi/save/',
            data=json.dumps({FIELD_DATE: '2024-01-01'}),
            content_type='application/json',
        )
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == {'result': 'ok'}

    def test_non_json_post_passes_through(self, db):
        request = RequestFactory().post(
            '/api/prelievi/save/',
            data={FIELD_NONCE: 'nonce-ccc'},
        )
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == {'result': 'ok'}

    def test_get_request_passes_through(self):
        request = RequestFactory().get('/api/prelievi/data/')
        resp = NonceMiddleware(ok_response)(request)
        assert resp.content == b'OK'

    def test_malformed_json_passes_through(self, db):
        request = RequestFactory().post(
            '/api/prelievi/save/',
            data=b'{bad json',
            content_type='application/json',
        )
        resp = NonceMiddleware(json_ok_response)(request)
        assert json.loads(resp.content) == {'result': 'ok'}

    def test_save_nonce_creates_record(self, admin_user):
        save_nonce('nonce-ddd', admin_user, {ROW_ID: 1})
        used = UsedNonce.objects.get(nonce='nonce-ddd')
        assert json.loads(used.response_json) == {ROW_ID: 1}
        assert used.user == admin_user

    def test_save_nonce_prunes_old_records(self, admin_user):
        old = UsedNonce.objects.create(
            nonce='nonce-old', user=admin_user, response_json='{}',
        )
        UsedNonce.objects.filter(pk=old.pk).update(
            created_at=timezone.now() - timedelta(hours=25),
        )
        recent = UsedNonce.objects.create(
            nonce='nonce-recent', user=admin_user, response_json='{}',
        )

        save_nonce('nonce-new', admin_user, {ROW_ID: 1})

        assert not UsedNonce.objects.filter(pk=old.pk).exists()
        assert UsedNonce.objects.filter(pk=recent.pk).exists()
        assert UsedNonce.objects.filter(nonce='nonce-new').exists()

    def test_save_nonce_ignores_duplicate_nonce(self, admin_user):
        save_nonce('nonce-dupe', admin_user, {ROW_ID: 1})
        save_nonce('nonce-dupe', admin_user, {ROW_ID: 2})

        used = UsedNonce.objects.get(nonce='nonce-dupe', user=admin_user)
        assert json.loads(used.response_json) == {ROW_ID: 1}


# -- Rate limiting ---------------------------------------------------------


def test_rate_limit_middleware_wraps_nonce_replay():
    middleware = list(settings.MIDDLEWARE)
    assert middleware.index('apps.base.middleware.RateLimitMiddleware') < \
        middleware.index('apps.base.middleware.NonceMiddleware')


class TestRateLimitMiddleware:
    def test_under_limit_passes(self, admin_user):
        mw = RateLimitMiddleware(ok_response)
        for _ in range(5):
            request = RequestFactory().get('/api/prelievi/data/')
            request.user = admin_user
            assert mw(request).status_code == 200

    def test_over_limit_returns_429(self, admin_user):
        mw = RateLimitMiddleware(ok_response)
        for _ in range(60):
            request = RequestFactory().get('/api/prelievi/data/')
            request.user = admin_user
            mw(request)

        request = RequestFactory().get('/api/prelievi/data/')
        request.user = admin_user
        resp = mw(request)
        assert resp.status_code == 429
        assert json.loads(resp.content)[STATUS] == STATUS_RATE_LIMITED

    def test_non_api_path_not_limited(self, admin_user):
        mw = RateLimitMiddleware(ok_response)
        for _ in range(100):
            request = RequestFactory().get('/prelievi')
            request.user = admin_user
            assert mw(request).status_code == 200

    def test_unauthenticated_not_limited(self):
        mw = RateLimitMiddleware(ok_response)
        for _ in range(100):
            request = RequestFactory().get('/api/prelievi/data/')
            request.user = AnonymousUser()
            assert mw(request).status_code == 200

    def test_different_users_have_separate_limits(self, admin_user, writer_user):
        mw = RateLimitMiddleware(ok_response)

        # Exhaust admin's quota.
        for _ in range(60):
            request = RequestFactory().get('/api/prelievi/data/')
            request.user = admin_user
            mw(request)

        # Writer should still pass.
        request = RequestFactory().get('/api/prelievi/data/')
        request.user = writer_user
        assert mw(request).status_code == 200

        # Admin should be blocked.
        request = RequestFactory().get('/api/prelievi/data/')
        request.user = admin_user
        assert mw(request).status_code == 429
