"""Tests for auth adapter."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps.base.auth import (
    NoSignupAdapter, WhitelistSocialAdapter, require_admin, require_writer,
)
from apps.base.models import LoginMethod, Role, User
from config import strings as S
from config.constants import MESSAGE


class TestNoSignupAdapter:
    def test_signup_disabled(self, db):
        adapter = NoSignupAdapter()
        assert adapter.is_open_for_signup(request=None) is False


def _ok_view(_request):
    return SimpleNamespace(status_code=200)


def _json_body(response):
    return json.loads(response.content.decode())


class TestRoleDecorators:
    def test_require_writer_rejects_reader(self):
        request = SimpleNamespace(user=SimpleNamespace(can_modify=False))

        response = require_writer(_ok_view)(request)

        assert response.status_code == 403
        assert _json_body(response) == {MESSAGE: S.ERR_FORBIDDEN}

    def test_require_writer_allows_writer(self):
        request = SimpleNamespace(user=SimpleNamespace(can_modify=True))

        response = require_writer(_ok_view)(request)

        assert response.status_code == 200

    def test_require_admin_rejects_non_admin(self):
        request = SimpleNamespace(user=SimpleNamespace(role=Role.WRITER))

        response = require_admin(_ok_view)(request)

        assert response.status_code == 403
        assert _json_body(response) == {MESSAGE: S.ERR_FORBIDDEN}

    def test_require_admin_allows_admin(self):
        request = SimpleNamespace(user=SimpleNamespace(role=Role.ADMIN))

        response = require_admin(_ok_view)(request)

        assert response.status_code == 200


def _sociallogin(email, *, is_existing=False):
    return SimpleNamespace(
        is_existing=is_existing,
        user=SimpleNamespace(email=email),
        connect=Mock(),
    )


def _oauth_user(email, **kwargs):
    return User.objects.create_user(
        username=kwargs.pop('username', email or 'oauth-user'),
        email=email,
        login_method=kwargs.pop('login_method', LoginMethod.OAUTH),
        **kwargs,
    )


@pytest.mark.django_db
class TestWhitelistSocialAdapter:
    def test_existing_social_login_is_left_alone(self):
        sociallogin = _sociallogin('person@example.com', is_existing=True)

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_not_called()

    def test_connects_matching_active_oauth_user(self):
        user = _oauth_user('person@example.com')
        sociallogin = _sociallogin('person@example.com')

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_called_once_with(None, user)

    def test_no_match_does_not_connect(self):
        sociallogin = _sociallogin('missing@example.com')

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_not_called()

    def test_inactive_user_does_not_connect(self):
        _oauth_user('person@example.com', is_active=False)
        sociallogin = _sociallogin('person@example.com')

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_not_called()

    def test_password_method_user_does_not_connect(self):
        _oauth_user(
            'person@example.com',
            login_method=LoginMethod.PASSWORD,
            password='testpass123!',
        )
        sociallogin = _sociallogin('person@example.com')

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_not_called()

    @pytest.mark.parametrize('email', ['', None])
    def test_blank_email_does_not_connect(self, email):
        sociallogin = _sociallogin(email)

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_not_called()

    def test_email_match_is_case_insensitive(self):
        user = _oauth_user('Person@Example.COM')
        sociallogin = _sociallogin('PERSON@example.com')

        WhitelistSocialAdapter().pre_social_login(request=None, sociallogin=sociallogin)

        sociallogin.connect.assert_called_once_with(None, user)
