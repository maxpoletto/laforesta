"""Abies URL configuration."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('abies/admin/', admin.site.urls),
    path('abies/api/prelievi/', include('apps.prelievi.urls')),
    path('abies/api/controllo/', include('apps.controllo.urls')),
    path('abies/api/impostazioni/', include('apps.impostazioni.urls')),
    path('abies/', include('apps.base.urls')),
    path('abies/accounts/', include('allauth.urls')),
]
