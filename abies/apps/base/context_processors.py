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
    return {'S': S, 'LANGUAGE_CODE': settings.LANGUAGE_CODE}
