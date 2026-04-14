"""Authentication helpers."""

from functools import wraps

from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.http import JsonResponse

from config import strings as S


class NoSignupAdapter(DefaultAccountAdapter):
    """Users are created by admins only — no self-registration."""

    def is_open_for_signup(self, request):
        return False


class WhitelistSocialAdapter(DefaultSocialAccountAdapter):
    """Connect incoming OAuth logins to pre-whitelisted users by email.

    Because every account is admin-provisioned, we trust the provider's
    asserted identity and match by user.email.  This bypasses allauth's
    default requirement that the incoming email be flagged verified —
    Microsoft's Graph API does not assert verification, so the default
    flow would never match.
    """

    def pre_social_login(self, request, sociallogin):
        if sociallogin.is_existing:
            return
        email = (sociallogin.user.email or '').lower()
        if not email:
            return
        from apps.base.models import LoginMethod, User
        try:
            user = User.objects.get(
                email__iexact=email,
                login_method=LoginMethod.OAUTH,
                is_active=True,
            )
        except User.DoesNotExist:
            return
        sociallogin.connect(request, user)


def require_writer(view):
    """Decorator: 403 for users below writer role."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.user.role not in ('admin', 'writer'):
            return JsonResponse({'message': S.ERR_FORBIDDEN}, status=403)
        return view(request, *args, **kwargs)
    return wrapper


def require_admin(view):
    """Decorator: 403 for non-admin users."""
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if request.user.role != 'admin':
            return JsonResponse({'message': S.ERR_FORBIDDEN}, status=403)
        return view(request, *args, **kwargs)
    return wrapper
