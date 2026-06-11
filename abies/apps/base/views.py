"""Base views — login, logout, shell, geo file serving."""

from pathlib import Path

from axes.decorators import axes_dispatch
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.utils.http import (
    url_has_allowed_host_and_scheme,
)
from django.views import View

from apps.base.http import CACHE_NO_CACHE, conditional_file_response
from apps.base.models import LoginMethod
from config import strings as S

# Whitelist of geo files we serve via `geo_view`.  Anything else 404s,
# even if it exists under `settings.GEO_DIR`.
ALLOWED_GEO_FILES = {
    'particelle.geojson',
    'terreni.geojson',
}


@login_required
def shell_view(request: HttpRequest) -> HttpResponse:
    """The long-lived SPA shell.  All post-login navigation happens here."""
    return render(request, 'base/shell.html')


@method_decorator(axes_dispatch, name='dispatch')
class LoginView(View):
    """Password login.  OAuth is handled by allauth's own URLs."""

    def get(self, request: HttpRequest) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)
        return render(request, 'base/login.html', {'next': request.GET.get('next', '')})

    def post(self, request: HttpRequest) -> HttpResponse:
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None and user.login_method == LoginMethod.PASSWORD:
            login(request, user)
            next_url = request.POST.get('next') or settings.LOGIN_REDIRECT_URL
            if not url_has_allowed_host_and_scheme(
                    next_url, allowed_hosts={request.get_host()},
                    require_https=request.is_secure()):
                next_url = settings.LOGIN_REDIRECT_URL
            return redirect(next_url)
        return render(request, 'base/login.html', {
            'error_message': S.ERR_LOGIN_INVALID,
            'next': request.POST.get('next', ''),
        }, status=400)


def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect('login')


@login_required
def geo_view(request: HttpRequest, filename: str) -> HttpResponse:
    """Serve a whitelisted geo file with conditional-GET.

    The geo directory is bind-mounted on prod and lives at
    `settings.GEO_DIR` in dev.  We don't go through Django's
    static files because geo data is data, not asset.
    """
    if filename not in ALLOWED_GEO_FILES:
        raise Http404
    path = Path(settings.GEO_DIR) / filename
    if not path.is_file():
        raise Http404
    # Geo data is large and effectively immutable, but can change on deploy:
    # `no-cache` lets the browser keep a copy yet revalidate via
    # If-Modified-Since (304, no re-transfer), so it never serves stale data.
    return conditional_file_response(
        request,
        path,
        content_type='application/geo+json',
        cache_control=CACHE_NO_CACHE,
    )
