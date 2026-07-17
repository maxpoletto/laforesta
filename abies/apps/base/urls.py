"""Base URL patterns — auth + shell + geo data."""

from django.urls import path, re_path

from apps.base import views

urlpatterns = [
    path('login/', views.LoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('api/species/data/', views.species_data, name='species-data'),
    path('api/geo/<str:filename>', views.geo_view, name='geo'),
    # The shell catches all domain paths for client-side routing.
    re_path(r'^(?:bosco|piano-di-taglio|rilevamenti|campionamenti|squadre|prelievi|importazione|controllo|impostazioni)(?:/.*)?$',
            views.shell_view, name='shell'),
]
