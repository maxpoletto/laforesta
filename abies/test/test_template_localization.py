"""Template localization convention tests."""

from pathlib import Path


def _localized_templates():
    return sorted(Path('apps').glob('*/templates/**/*_it.html'))


def test_regular_templates_use_active_locale_suffix():
    offenders = []
    for path in Path('apps').glob('*/templates/**/*.html'):
        if path.is_symlink():
            continue
        if not path.name.endswith('_it.html'):
            offenders.append(str(path))

    assert offenders == []


def test_active_locale_templates_have_materialized_aliases():
    offenders = []
    suffix = '_it.html'
    for source in _localized_templates():
        link = source.with_name(f'{source.name[:-len(suffix)]}.html')
        if not link.is_symlink() or link.resolve() != source.resolve():
            offenders.append(f'{link} -> {source.name}')

    assert offenders == []
