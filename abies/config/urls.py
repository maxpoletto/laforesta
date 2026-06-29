"""Abies URL configuration.

The app is served at the root of its own subdomain (abies.laforesta.it /
abies-dev.laforesta.it). No URL prefix is needed -- the subdomain itself
scopes the deployment.
"""

from django.contrib import admin
from django.urls import include, path

from apps.base import views as base_views

urlpatterns = [
    # `/` resolves to the user's configured landing page.
    path('', base_views.home_view, name='home'),
    path('admin/', admin.site.urls),
    path('ipso/', include('apps.ipso.urls')),
    path('api/bosco/', include('apps.bosco.urls')),
    path('api/squadre/', include('apps.mannesi.urls')),
    path('api/prelievi/', include('apps.prelievi.urls')),
    path('api/campionamenti/', include('apps.campionamenti.urls')),
    path('api/piano-di-taglio/', include('apps.piano_di_taglio.urls')),
    path('api/controllo/', include('apps.controllo.urls')),
    path('api/impostazioni/', include('apps.impostazioni.urls')),
    path('api/ipso/', include('apps.ipso.api_urls')),
    path('accounts/', include('allauth.urls')),
    path('', include('apps.base.urls')),
]
