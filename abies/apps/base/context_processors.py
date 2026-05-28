"""Template context processors for the shared shell."""

from config import strings as S


def strings(request):
    """Expose the active-locale string module to every template as `S`.

    Templates can then write `{{ S.CANCEL }}` etc. instead of inlining
    Italian literals — the source of truth stays in `config/strings_it.py`.
    """
    return {'S': S}
