"""Abies URL configuration."""

from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('abies/admin/', admin.site.urls),
]
