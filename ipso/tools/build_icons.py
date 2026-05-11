#!/usr/bin/env python3
"""Generate ipso/img/ from the canonical La Foresta artwork.

Sources (paths resolved relative to this script):
- ../../logo/logo-grande.png  (1440x1440 RGBA) -> 192x192 and 512x512 PNG
- ../../bosco/a/f.gif         (16x16)  -> copied verbatim (favicon)
- ../../bosco/a/l.gif         (128x128) -> copied verbatim (mid-size icon)

The PNGs feed the PWA manifest's install icons (Android prefers 192 and
512). The GIFs cover the in-page <link rel="icon"> and the small-size
manifest entries.

The output directory is named `img/` (not the more obvious `icons/`)
because the default Apache `mods-enabled/alias.conf` ships a global
`Alias /icons/ "/usr/share/apache2/icons/"` that intercepts any
request for `/icons/*` before it can reach our DocumentRoot.

Idempotent. Safe to re-run.
"""

import shutil
import sys
from pathlib import Path

from PIL import Image


PNG_SIZES = (192, 512)


def main() -> int:
    here = Path(__file__).resolve().parent
    ipso_root = here.parent
    repo_root = ipso_root.parent

    src_logo = repo_root / 'logo' / 'logo-grande.png'
    src_fgif = repo_root / 'bosco' / 'a' / 'f.gif'
    src_lgif = repo_root / 'bosco' / 'a' / 'l.gif'

    for p in (src_logo, src_fgif, src_lgif):
        if not p.is_file():
            print(f'missing source file: {p}', file=sys.stderr)
            return 1

    img_dir = ipso_root / 'img'
    img_dir.mkdir(exist_ok=True)

    # GIFs: copy verbatim. shutil.copy preserves the file as-is, which is
    # exactly what we want for these pre-rendered assets.
    for src in (src_fgif, src_lgif):
        dst = img_dir / src.name
        shutil.copy(src, dst)
        print(f'wrote {dst.relative_to(ipso_root)} ({dst.stat().st_size} bytes)', file=sys.stderr)

    # PNGs: downsample logo-grande.png to 192 and 512. LANCZOS gives the
    # cleanest reduction from a 1440px source. The source already has an
    # alpha channel, so the result has transparent corners on Android.
    with Image.open(src_logo) as im:
        # Pillow yells if the source isn't RGBA; convert defensively.
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        for size in PNG_SIZES:
            out = img_dir / f'icon-{size}.png'
            resized = im.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(out, format='PNG', optimize=True)
            print(
                f'wrote {out.relative_to(ipso_root)} '
                f'({size}x{size}, {out.stat().st_size} bytes)',
                file=sys.stderr,
            )

    return 0


if __name__ == '__main__':
    sys.exit(main())
