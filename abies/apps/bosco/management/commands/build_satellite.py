"""Build Sentinel-2 satellite data for Bosco."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.bosco import satellite


class Command(BaseCommand):
    help = "Build Sentinel-2 satellite rasters, manifests, and time series."

    def add_arguments(self, parser):
        subparsers = parser.add_subparsers(dest='action', required=True)
        for action in ('dates', 'fetch', 'precompute', 'all'):
            subparser = subparsers.add_parser(action)
            self._add_common_args(subparser)
            if action in {'dates', 'all'}:
                self._add_date_args(subparser)
            if action in {'fetch', 'all'}:
                subparser.add_argument(
                    '--date', action='append', default=[],
                    help="Date to fetch (YYYY-MM-DD). Repeatable. "
                         "Defaults to each region's dates.txt.",
                )
                subparser.add_argument(
                    '--force', action='store_true',
                    help="Refetch dates whose GeoTIFF set is already complete.",
                )

    def _add_common_args(self, parser):
        parser.add_argument(
            '--geojson', type=Path,
            default=Path(settings.GEO_DIR) / 'terreni.geojson',
            help="Parcel GeoJSON (default: settings.GEO_DIR/terreni.geojson).",
        )
        parser.add_argument(
            '--output-dir', type=Path, default=Path(settings.SATELLITE_DIR),
            help="Satellite data root (default: settings.SATELLITE_DIR).",
        )
        parser.add_argument(
            '--region', action='append', default=[],
            help="Region/layer to build. Repeatable. Defaults to all GeoJSON layers.",
        )

    def _add_date_args(self, parser):
        parser.add_argument('--summer-months', default='6,7')
        parser.add_argument('--winter-months', default='1,2')
        parser.add_argument('--max-cloud-summer', type=float, default=1.0)
        parser.add_argument('--max-cloud-winter', type=float, default=10.0)
        parser.add_argument('--year-start', type=int, default=satellite.FIRST_YEAR)
        parser.add_argument('--year-end', type=int, default=None)

    def handle(self, *args, **options):
        try:
            geojson = options['geojson']
            output_dir = options['output_dir']
            if not geojson.is_file():
                raise satellite.SatelliteError(f'{geojson} not found')
            regions = options['region'] or satellite.regions_from_geojson(geojson)
            if not regions:
                raise satellite.SatelliteError(f'no regions found in {geojson}')

            action = options['action']
            if action in {'dates', 'all'}:
                self._build_dates(geojson, output_dir, regions, options)
            if action in {'fetch', 'all'}:
                self._fetch(geojson, output_dir, regions, options)
            if action in {'precompute', 'all'}:
                self._precompute(geojson, output_dir, regions)
        except satellite.SatelliteError as exc:
            raise CommandError(str(exc)) from exc

    def _build_dates(self, geojson, output_dir, regions, options):
        windows = [
            (
                'summer',
                satellite.parse_months(options['summer_months']),
                options['max_cloud_summer'],
            ),
            (
                'winter',
                satellite.parse_months(options['winter_months']),
                options['max_cloud_winter'],
            ),
        ]
        for region in regions:
            region_dir = output_dir / region
            all_dates = set()
            for label, months, max_cloud in windows:
                dates = satellite.find_region_dates(
                    geojson, region, months, max_cloud,
                    year_start=options['year_start'],
                    year_end=options['year_end'],
                )
                satellite.write_dates_file(region_dir / f'dates-{label}.txt', dates)
                all_dates.update(dates)
                self.stdout.write(
                    f'{region}: wrote {len(dates)} {label} date(s)'
                )
            satellite.write_dates_file(region_dir / 'dates.txt', sorted(all_dates))
            self.stdout.write(f'{region}: wrote {len(all_dates)} total date(s)')

    def _fetch(self, geojson, output_dir, regions, options):
        for region in regions:
            region_dir = output_dir / region
            dates = options['date'] or satellite.read_dates_file(region_dir / 'dates.txt')
            fetched, skipped = satellite.fetch_region(
                geojson, region_dir, region, dates, force=options['force'],
                log=self.stdout.write,
            )
            self.stdout.write(
                f'{region}: fetched {fetched} date(s); skipped {skipped}'
            )

    def _precompute(self, geojson, output_dir, regions):
        for region in regions:
            region_dir = output_dir / region
            count = satellite.precompute_region(geojson, region_dir, region)
            self.stdout.write(f'{region}: precomputed {count} date(s)')
