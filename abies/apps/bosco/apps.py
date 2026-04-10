from django.apps import AppConfig

from config import strings as S


class BoscoConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.bosco'
    verbose_name = S.APP_BOSCO
