"""Authentication helpers."""

from functools import wraps

from allauth.account.adapter import DefaultAccountAdapter
from allauth.core.exceptions import ImmediateHttpResponse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect

from apps.base.models import LoginMethod, Role, User
from apps.base.oauth import microsoft_oauth_app
from config import strings as S
from config.constants import MESSAGE


class NoSignupAdapter(DefaultAccountAdapter):
    """Users are created by admins only — no self-registration."""

    def is_open_for_signup(self, request):
        return False

    def get_login_redirect_url(self, request):
        from apps.base.landing import user_landing_page
        return user_landing_page(request.user)


class WhitelistSocialAdapter(DefaultSocialAccountAdapter):
    """Connect incoming OAuth logins to pre-whitelisted users by email.

    Because every account is admin-provisioned, we trust the provider's
    asserted identity and match by user.email.  This bypasses allauth's
    default requirement that the incoming email be flagged verified —
    Microsoft's Graph API does not assert verification, so the default
    flow would never match.
    """

    def list_apps(self, request, provider=None, client_id=None):
        app = microsoft_oauth_app()
        if (app is not None and provider in (None, app.provider)
                and client_id in (None, app.client_id)):
            return [app]
        return []

    def get_app(self, request, provider, client_id=None):
        # Both the login and callback views select their app through this hook.
        apps = self.list_apps(request, provider=provider, client_id=client_id)
        if not apps:
            _deny_oauth_login(request)
        return apps[0]

    def pre_social_login(self, request, sociallogin):
        # Existing-link/email lookup replaces sociallogin.user with the local
        # user before this hook. Only email_addresses still describes the
        # incoming provider identity (Microsoft supplies one primary address).
        email = (
            (sociallogin.email_addresses[0].email or '').strip()
            if sociallogin.email_addresses else ''
        )
        is_existing = sociallogin.is_existing
        candidates = User.objects.filter(
            email__iexact=email, login_method=LoginMethod.OAUTH, is_active=True,
        ).exclude(email='')
        if is_existing:
            # A link cannot bypass policy or move to another whitelisted user.
            candidates = candidates.filter(pk=sociallogin.user.pk)
        user = candidates.order_by('id').first()
        if user is None:
            _deny_oauth_login(request)
        if is_existing:
            sociallogin.user = user
        else:
            sociallogin.connect(request, user)


def _deny_oauth_login(request):
    messages.error(request, S.ERR_OAUTH_LOGIN_NOT_ALLOWED)
    raise ImmediateHttpResponse(redirect('login'))


def require_writer(view):
    """Decorator: 403 for users below writer role."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not request.user.can_modify:
            return JsonResponse({MESSAGE: S.ERR_FORBIDDEN}, status=403)
        return view(request, *args, **kwargs)
    return wrapper


def require_admin(view):
    """Decorator: 403 for non-admin users."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.user.role != Role.ADMIN:
            return JsonResponse({MESSAGE: S.ERR_FORBIDDEN}, status=403)
        return view(request, *args, **kwargs)
    return wrapper
