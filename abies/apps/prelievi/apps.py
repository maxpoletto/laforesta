from django.apps import AppConfig

from config import strings as S


class PrelieviConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.prelievi'
    verbose_name = S.APP_PRELIEVI
