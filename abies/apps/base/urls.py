"""Base URL patterns — auth + shell."""

from django.urls import path, re_path

from apps.base import views

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    # The shell catches all domain paths for client-side routing.
    re_path(r'^(?:bosco|prelievi|controllo|impostazioni)(?:/.*)?$',
            views.shell_view, name='shell'),
]
