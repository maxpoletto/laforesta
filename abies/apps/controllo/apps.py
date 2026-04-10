from django.apps import AppConfig

from config import strings as S


class ControlloConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.controllo'
    verbose_name = S.APP_CONTROLLO
