from django.apps import AppConfig

from config import strings as S


class MannesiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.mannesi'
    verbose_name = S.APP_SQUADRE
