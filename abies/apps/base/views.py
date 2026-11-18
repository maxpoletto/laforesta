"""Base views — login, logout, shell, geo file serving."""

from axes.decorators import axes_dispatch
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.utils.http import (
    url_has_allowed_host_and_scheme,
)
from django.views import View
from django.views.decorators.http import require_POST

from apps.base.digests import serve_digest
from apps.base.landing import user_landing_page
from apps.base.map_data import map_response
from apps.base.models import LoginMethod
from apps.base.oauth import microsoft_oauth_app
from apps.ipso.models import IpsoUpload, IpsoUploadState
from config import strings as S
from config.constants import FIELD_SPECIES

@login_required
def home_view(request: HttpRequest) -> HttpResponse:
    return redirect(user_landing_page(request.user))


@login_required
def shell_view(request: HttpRequest) -> HttpResponse:
    """The long-lived SPA shell.  All post-login navigation happens here."""
    if request.path.startswith('/campionamenti'):
        suffix = request.path.removeprefix('/campionamenti')
        target = f'/rilevamenti{suffix}'
        if request.META.get('QUERY_STRING'):
            target = f'{target}?{request.META["QUERY_STRING"]}'
        return redirect(target)
    return render(request, 'base/shell.html', {
        'ipso_upload_pending_count': IpsoUpload.objects.filter(
            state=IpsoUploadState.RECEIVED,
        ).count(),
    })


@method_decorator(axes_dispatch, name='dispatch')
class LoginView(View):
    """Password login.  OAuth is handled by allauth's own URLs."""

    def get(self, request: HttpRequest) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect(user_landing_page(request.user))
        return render(request, 'base/login.html', _login_context(
            request.GET.get('next', ''),
        ))

    def post(self, request: HttpRequest) -> HttpResponse:
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None and user.login_method == LoginMethod.PASSWORD:
            login(request, user)
            next_url = request.POST.get('next') or user_landing_page(user)
            if not url_has_allowed_host_and_scheme(
                    next_url, allowed_hosts={request.get_host()},
                    require_https=request.is_secure()):
                next_url = user_landing_page(user)
            return redirect(next_url)
        return render(request, 'base/login.html', _login_context(
            request.POST.get('next', ''),
            error_message=S.ERR_LOGIN_INVALID,
        ), status=400)


def _login_context(next_url: str, *, error_message: str = '') -> dict:
    return {
        'error_message': error_message,
        'microsoft_oauth_configured': microsoft_oauth_app() is not None,
        'next': next_url,
    }


@require_POST
def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect('login')


@login_required
def species_data(request: HttpRequest) -> HttpResponse:
    return serve_digest(request, FIELD_SPECIES)


@login_required
def geo_view(request: HttpRequest, filename: str) -> HttpResponse:
    if filename not in {'map.geojson', 'terreni.geojson', 'roads.geojson'}:
        raise Http404
    return map_response(request, roads=filename == 'roads.geojson')
