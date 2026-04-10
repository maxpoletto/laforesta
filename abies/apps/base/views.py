"""Base views — login, logout, shell."""

from axes.decorators import axes_dispatch
from django.contrib.auth import authenticate, login, logout
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils.decorators import method_decorator
from django.views import View


@method_decorator(axes_dispatch, name='dispatch')
class LoginView(View):
    """Password login.  OAuth is handled by allauth's own URLs."""

    def get(self, request: HttpRequest) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect('shell')
        return render(request, 'base/login.html', {'next': request.GET.get('next', '')})

    def post(self, request: HttpRequest) -> HttpResponse:
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.POST.get('next') or '/abies/prelievi'
            return redirect(next_url)
        return render(request, 'base/login.html', {
            'error_message': 'Nome utente o password non validi.',
            'next': request.POST.get('next', ''),
        }, status=400)


def logout_view(request: HttpRequest) -> HttpResponse:
    logout(request)
    return redirect('login')
