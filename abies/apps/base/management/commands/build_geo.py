"""Build terreni.geojson in GEO_DIR and enrich parcel polygons from the DB."""

import json
from pathlib import Path
from typing import NamedTuple

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.base.models import Parcel


class GeoBuildStats(NamedTuple):
    matched: int
    db_parcels_without_geometry: int


# Mirrors apps/base/static/base/js/geo.js parcelNames(): terreni.geojson stores
# the region in `properties.layer` and the parcel name after the first `-` in
# `properties.name`.
def _parcel_name(feature):
    props = feature.get('properties') or {}
    region = props.get('layer') or ''
    full_name = props.get('name') or ''
    dash = full_name.find('-')
    parcel = full_name[dash + 1:] if dash >= 0 else full_name
    return region, parcel


def enrich_terreni_geojson(src_path, dst_path):
    """Write terreni.geojson with static parcel metadata merged in.

    The source geometry remains authoritative.  We only add locale-neutral data
    that is as static as the geometry itself: currently parcel forest type.
    """
    by_name = {
        (p.region.name, p.name): p.eclass.coppice
        for p in Parcel.objects.select_related('region', 'eclass')
    }
    with open(src_path, encoding='utf-8') as f:
        data = json.load(f)

    matched_keys = set()
    for feature in data.get('features') or []:
        key = _parcel_name(feature)
        if key not in by_name:
            continue
        props = feature.setdefault('properties', {})
        props['coppice'] = by_name[key]
        matched_keys.add(key)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_name(f'.{dst_path.name}.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')
    tmp_path.replace(dst_path)
    return GeoBuildStats(
        matched=len(matched_keys),
        db_parcels_without_geometry=len(set(by_name) - matched_keys),
    )


class Command(BaseCommand):
    help = "Build geo files in GEO_DIR, enriching terreni.geojson from parcels."

    def add_arguments(self, parser):
        parser.add_argument(
            'data_dir', type=Path,
            help="Directory containing terreni.geojson.",
        )
        parser.add_argument(
            '--output-dir', type=Path, default=settings.GEO_DIR,
            help="Destination directory (defaults to settings.GEO_DIR).",
        )

    def handle(self, *args, data_dir, output_dir, **options):
        if not data_dir.is_dir():
            raise CommandError(f'{data_dir} is not a directory')
        terreni = data_dir / 'terreni.geojson'
        if not terreni.is_file():
            raise CommandError(f'{terreni} not found')

        stats = enrich_terreni_geojson(terreni, output_dir / 'terreni.geojson')
        self.stdout.write(
            f'Geo: enriched terreni.geojson '
            f'({stats.matched} parcels matched, '
            f'{stats.db_parcels_without_geometry} DB parcels without geometry)'
        )
