"""Template context processors for the shared shell."""

from django.conf import settings

from config import strings as S


def strings(request):
    """Expose the active-locale string module (`S`) and `LANGUAGE_CODE` to
    every template.

    Templates write `{{ S.CANCEL }}` for UI text (source of truth in
    `config/strings_it.py`).  The shell sets `<html lang="{{ LANGUAGE_CODE }}">`
    so client JS can read the active locale via `document.documentElement.lang`
    for number formatting (see CLAUDE.md §"Number formatting").
    """
    return {
        'S': S,
        'LANGUAGE_CODE': settings.LANGUAGE_CODE,
        'ABIES_APP_NAME': settings.ABIES_APP_NAME,
        'ABIES_BRAND_FAVICON_STATIC': settings.ABIES_BRAND_FAVICON_STATIC,
        'ABIES_BRAND_LOGO_STATIC': settings.ABIES_BRAND_LOGO_STATIC,
        'ABIES_BRAND_NAME': settings.ABIES_BRAND_NAME,
        'ABIES_INSTANCE': settings.ABIES_INSTANCE,
        'ABIES_SITE_TITLE': settings.ABIES_SITE_TITLE,
    }
