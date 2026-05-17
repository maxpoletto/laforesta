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

# Maskable icon: Android adaptive-icon launchers apply a circular,
# squircle, or rounded-square mask to the install icon. The 80% rule
# (W3C Maskable icon spec) says: everything important must live inside
# the inner 80% of the canvas. Our logo has a green border touching the
# canvas edge and "LA FORESTA" text near the bottom — both would get
# cropped if we used the raw logo as-is. So we render the logo into the
# inner 80% of a 512x512 white canvas, leaving 10% safe padding on each
# side that the mask can chew through without hitting brand content.
MASKABLE_SIZE = 512
MASKABLE_INNER = int(MASKABLE_SIZE * 0.8)   # 410 px
MASKABLE_OFFSET = (MASKABLE_SIZE - MASKABLE_INNER) // 2  # 51 px


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

    img_dir = ipso_root / 'src' / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)

    # GIFs: copy verbatim. shutil.copy preserves the file as-is, which is
    # exactly what we want for these pre-rendered assets.
    for src in (src_fgif, src_lgif):
        dst = img_dir / src.name
        shutil.copy(src, dst)
        print(f'wrote {dst.relative_to(ipso_root)} ({dst.stat().st_size} bytes)', file=sys.stderr)

    # PNGs: downsample logo-grande.png to 192 and 512 (purpose=any), plus
    # one 512 maskable variant with padding for adaptive icon contexts.
    # LANCZOS gives the cleanest reduction from a 1440px source.
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

        # Maskable: white canvas + logo at 80% size, centered. The mask
        # only ever crops into the white padding; brand content (green
        # border, trees, text) stays intact under any adaptive shape.
        canvas = Image.new(
            'RGBA', (MASKABLE_SIZE, MASKABLE_SIZE), (255, 255, 255, 255),
        )
        inner = im.resize(
            (MASKABLE_INNER, MASKABLE_INNER), Image.Resampling.LANCZOS,
        )
        # paste with alpha mask so any transparent pixels in the logo
        # come through to white, not to the (also white) canvas — net
        # behaviour is identical here but keeps the call generic.
        canvas.paste(inner, (MASKABLE_OFFSET, MASKABLE_OFFSET), inner)
        out = img_dir / 'icon-512-maskable.png'
        canvas.save(out, format='PNG', optimize=True)
        print(
            f'wrote {out.relative_to(ipso_root)} '
            f'(maskable {MASKABLE_SIZE}x{MASKABLE_SIZE}, '
            f'{out.stat().st_size} bytes)',
            file=sys.stderr,
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
