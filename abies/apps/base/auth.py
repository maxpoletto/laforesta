"""Authentication helpers."""

from allauth.account.adapter import DefaultAccountAdapter


class NoSignupAdapter(DefaultAccountAdapter):
    """Users are created by admins only — no self-registration."""

    def is_open_for_signup(self, request):
        return False
