"""Template localization convention tests."""

from pathlib import Path


def test_regular_templates_use_active_locale_suffix():
    offenders = []
    for path in Path('apps').glob('*/templates/**/*.html'):
        if path.is_symlink():
            continue
        if not path.name.endswith('_it.html'):
            offenders.append(str(path))

    assert offenders == []
