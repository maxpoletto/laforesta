from django.apps import AppConfig

from config import strings as S


class BaseConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.base'
    verbose_name = S.APP_BASE

    def ready(self):
        from django.db.backends.signals import connection_created
        connection_created.connect(_set_sqlite_pragmas)


def _set_sqlite_pragmas(sender, connection, **kwargs):
    """Enable WAL mode and tuned sync for SQLite connections."""
    if connection.vendor == 'sqlite':
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA synchronous=NORMAL;')
