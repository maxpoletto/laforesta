from django.apps import AppConfig

from config import strings as S


class ImpostazioniConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.impostazioni'
    verbose_name = S.APP_IMPOSTAZIONI
