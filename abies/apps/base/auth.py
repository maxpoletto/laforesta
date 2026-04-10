"""Authentication helpers."""

from functools import wraps

from allauth.account.adapter import DefaultAccountAdapter
from django.http import JsonResponse

from config import strings as S


class NoSignupAdapter(DefaultAccountAdapter):
    """Users are created by admins only — no self-registration."""

    def is_open_for_signup(self, request):
        return False


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
