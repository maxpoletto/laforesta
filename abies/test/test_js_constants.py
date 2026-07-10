"""Static contracts for the shared browser constants module."""

import posixpath
import re
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CONSTANTS_PATH = ROOT / 'apps/base/static/base/js/constants.js'
CONSTANTS_STATIC_PATH = 'base/js/constants.js'
IMPORT_RE = re.compile(
    r"import\s*\{(?P<names>[^}]*)\}\s*from\s*"
    r"['\"](?P<module>[^'\"]+)['\"]",
    re.DOTALL,
)
EXPORT_RE = re.compile(r'^export const ([A-Za-z0-9_]+)', re.MULTILINE)


def test_shared_constants_named_imports_are_exported():
    exports = set(EXPORT_RE.findall(CONSTANTS_PATH.read_text()))
    missing = []

    for suffix in ('*.js', '*.mjs'):
        for source_path in (ROOT / 'apps').rglob(suffix):
            source = source_path.read_text()
            for match in IMPORT_RE.finditer(source):
                module = match.group('module')
                if not module.startswith('.'):
                    continue
                parts = source_path.parts
                static_index = parts.index('static')
                source_static_path = PurePosixPath(*parts[static_index + 1:])
                imported_path = posixpath.normpath(
                    str(source_static_path.parent / module)
                )
                if imported_path != CONSTANTS_STATIC_PATH:
                    continue
                for item in match.group('names').split(','):
                    name = item.strip().split(' as ', 1)[0].strip()
                    if name and name not in exports:
                        missing.append(
                            f'{source_path.relative_to(ROOT)} imports {name}'
                        )

    assert not missing, '\n'.join(missing)
