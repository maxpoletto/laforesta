"""Initialize printable-map context from OSM and a Copernicus surface model."""
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.bosco.map_context import OVERPASS_URL, build_region
from apps.bosco.map_context_data import MapContextError


class Command(BaseCommand):
    help = 'Build local roads, water and DSM contour vectors for printable Bosco maps.'

    def add_arguments(self, parser):
        parser.add_argument('--geojson', type=Path, default=Path(settings.GEO_DIR) / 'terreni.geojson')
        parser.add_argument('--output-dir', type=Path, default=Path(settings.MAP_CONTEXT_DIR))
        parser.add_argument('--region', action='append', default=[])
        parser.add_argument('--buffer-m', type=float, default=500)
        parser.add_argument('--contour-interval', type=int, default=20)
        parser.add_argument('--force', action='store_true', help='Refresh an existing matching build.')
        parser.add_argument('--osm-file', type=Path, help='Use a local Overpass JSON response.')
        parser.add_argument('--dem', type=Path, help='Use a local WGS84 elevation GeoTIFF.')
        parser.add_argument('--overpass-url', default=OVERPASS_URL)

    def handle(self, **options):
        try:
            geojson = json.loads(options['geojson'].read_text())
            regions = options['region'] or sorted({f['properties']['layer'] for f in geojson['features']})
            for region in regions:
                self.stdout.write(f'{region}: preparing roads, water and contours…', ending='\n')
                payload, changed = build_region(
                    geojson, region, options['output_dir'], buffer_m=options['buffer_m'],
                    interval=options['contour_interval'], force=options['force'],
                    osm_file=options['osm_file'], dem=options['dem'], endpoint=options['overpass_url'],
                )
                self.stdout.write(f'{region}: {"built" if changed else "kept"} {len(payload["features"])} vector features')
        except (MapContextError, OSError, ValueError, KeyError) as exc:
            raise CommandError(str(exc)) from exc
