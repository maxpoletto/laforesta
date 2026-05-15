#!/usr/bin/env python3
"""Materialize unsuffixed template symlinks for the active language.

Usage: link-templates.py <suffix>     # e.g. _it, _en

For every `<name><suffix>.html` under `apps/*/templates/`, (re)create a
relative symlink `<name>.html` → `<name><suffix>.html`.  Files without
the active suffix are left alone (templates already in English, test
fixtures, etc.).

Mirrors the install-time language-selection pattern of
`config/strings.py` and `apps/base/static/base/js/strings.js`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_GLOB = 'apps/*/templates'


def main(suffix: str) -> int:
    if not suffix.startswith('_'):
        print(f"suffix must start with '_' (got {suffix!r})", file=sys.stderr)
        return 1

    count = 0
    for tmpl_root in ROOT.glob(TEMPLATES_GLOB):
        for src in tmpl_root.rglob(f'*{suffix}.html'):
            link = src.with_name(src.name[: -len(f'{suffix}.html')] + '.html')
            target = src.name  # relative target = sibling filename
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(target, link)
            count += 1
    print(f"link-templates: materialized {count} symlinks for suffix {suffix!r}")
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
