"""Load an initial parcel map, linking cataloged parcels where available."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import DEFAULT_DB_ALIAS

from apps.base.map_data import map_geojson
from apps.base.map_import_v1 import import_initial_map, initial_map_path
from apps.base.models import MapFeature, Parcel


class Command(BaseCommand):
    help = 'Import an initial map.geojson (legacy name: terreni.geojson) into SQLite.'

    def add_arguments(self, parser):
        parser.add_argument('data_dir', type=Path, help='GeoJSON file or canonical directory.')
        parser.add_argument('--output-dir', type=Path,
                            help='Also export the enriched terreni.geojson for legacy tools.')

    def handle(self, *args, data_dir, output_dir, **options):
        path = Path(data_dir)
        try:
            if path.is_dir():
                path = initial_map_path(path)
                if path is None:
                    self.stdout.write('No initial map supplied.')
                    return
            imported = import_initial_map(apps, DEFAULT_DB_ALIAS, path)
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=output_dir, delete=False, mode='w', encoding='utf-8') as stream:
                    temporary = Path(stream.name)
                    try:
                        json.dump(map_geojson(), stream, ensure_ascii=False, separators=(',', ':'))
                        stream.flush()
                        temporary.replace(output_dir / 'terreni.geojson')
                    finally:
                        temporary.unlink(missing_ok=True)
        except (ValueError, OSError, OverflowError) as exc:
            raise CommandError(str(exc)) from exc
        matched = MapFeature.objects.exclude(parcel=None).order_by().values('parcel').distinct().count()
        missing = Parcel.objects.filter(map_features__isnull=True).count()
        status = 'map imported' if imported else 'map already imported'
        self.stdout.write(f'{path}: {status} ({matched} parcels matched, '
                          f'{missing} DB parcels without geometry)')
