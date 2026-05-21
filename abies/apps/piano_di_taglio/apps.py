"""Piano di taglio app configuration."""

from django.apps import AppConfig

from config import strings as S


class PianoDiTaglioConfig(AppConfig):
    name = 'apps.piano_di_taglio'
    verbose_name = S.APP_PIANO_DI_TAGLIO
