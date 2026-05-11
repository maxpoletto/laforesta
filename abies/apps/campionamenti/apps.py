"""Campionamenti app configuration."""

from django.apps import AppConfig

from config import strings as S


class CampionamentiConfig(AppConfig):
    name = 'apps.campionamenti'
    verbose_name = S.APP_CAMPIONAMENTI
