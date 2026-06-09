"""Tests for auth adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from apps.base.auth import NoSignupAdapter, WhitelistSocialAdapter
from apps.base.models import LoginMethod, User


class TestNoSignupAdapter:
    def test_signup_disabled(self, db):
        adapter = NoSignupAdapter()
        assert adapter.is_open_for_signup(request=None) is False


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
