"""Abies URL configuration."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('abies/admin/', admin.site.urls),
    path('abies/api/prelievi/', include('apps.prelievi.urls')),
    path('abies/', include('apps.base.urls')),
    path('abies/accounts/', include('allauth.urls')),
]
