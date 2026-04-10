"""Tests for auth adapter."""

import pytest

from apps.base.auth import NoSignupAdapter


class TestNoSignupAdapter:
    def test_signup_disabled(self, db):
        adapter = NoSignupAdapter()
        assert adapter.is_open_for_signup(request=None) is False
