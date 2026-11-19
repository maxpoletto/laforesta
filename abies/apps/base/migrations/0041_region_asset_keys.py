"""Give each compresa one directory key and upgrade its generated map files."""
import gzip
import json
import os
import re
import tempfile
from pathlib import Path
from uuid import UUID

from django.conf import settings
from django.db import migrations, models
from django.db.migrations.exceptions import IrreversibleError

from apps.base.map_keys import initial_map_key


def replace_metadata(path, convert, *, compressed=False):
    if path.is_symlink():
        raise ValueError(f'Refusing to rewrite a symlink: {path}')
    if not path.exists():
        return
    with (gzip.open(path, 'rt', encoding='utf-8') if compressed else path.open(encoding='utf-8')) as stream:
        payload = json.load(stream)
    if not convert(payload):
        return
    # Publish the small metadata file atomically; a retry can see either version.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        with (gzip.open(tmp, 'wt', encoding='utf-8') if compressed else tmp.open('w', encoding='utf-8')) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        tmp.chmod(path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def convert_series(payload, old):
    version = payload.get('schema_version', 1)
    if version == 2:
        return False
    if version != 1:
        raise ValueError(f'Unknown satellite summary version {version!r} for {old!r}')
    prefix = payload.pop('region', old) + '-'
    parcels = payload.get('parcels', [])
    names = [name.removeprefix(prefix) for name in parcels]
    values = payload.get('means', {}).get('parcels', {})
    means = {name.removeprefix(prefix): value for name, value in values.items()}
    if len(names) != len(set(names)) or len(means) != len(values):
        raise ValueError(f'Ambiguous satellite parcel labels for {old!r}')
    if 'parcels' in payload:
        payload['parcels'] = names  # Mask pixel values are indices into this ordered list.
    if 'parcels' in payload.get('means', {}):
        payload['means']['parcels'] = means
    payload['schema_version'] = 2
    return True


def convert_context(payload, old, key):
    if 'region_key' in payload:
        return False
    identity = payload.pop('region_directory', payload.get('region'))
    # Preserve the reader's stale-map check if the old snapshot belonged elsewhere.
    payload['region_key'] = key if identity == old else None
    return True


def is_asset_directory(path, satellite):
    if not satellite:
        return (path / 'context.json.gz').is_file()
    files = {'manifest.json', 'timeseries.json', 'parcel-mask.tif',
             'dates.txt', 'dates-summer.txt', 'dates-winter.txt'}
    return any(entry.name in files or re.fullmatch(r'\d{4}-\d{2}-\d{2}', entry.name)
               for entry in path.iterdir())


def directory_uuid(name):
    try:
        key = UUID(name)
        return key if str(key) == name else None
    except ValueError:
        return None


def move_assets(root, known, *, satellite):
    if root is None:
        return
    root = Path(root)
    try:
        children = list(root.iterdir())
    except FileNotFoundError:
        if root.is_symlink():
            raise ValueError(f'Generated-map directory is unavailable: {root}')
        return
    moves = dict(known)
    for path in children:
        if path.name in moves or path.is_symlink() or not path.is_dir():
            continue
        if directory_uuid(path.name):
            continue  # Already in the new layout (possibly after an interrupted migration).
        if is_asset_directory(path, satellite):
            # make dev and file-only builds can have imagery before the CSV catalog is loaded.
            moves[path.name] = str(initial_map_key(path.name))
    for old, key in moves.items():
        if not old or Path(old).name != old or old in ('.', '..'):
            continue  # Such a display name never owned a direct child of the root.
        source, target = root / old, root / key
        if source.is_symlink() or target.is_symlink():
            raise ValueError(f'Refusing to migrate a symlink: {source} -> {target}')
        if not source.exists():
            continue
        if not source.is_dir():
            raise ValueError(f'Expected a region directory: {source}')
        if target != source:
            if target.exists():
                raise ValueError(f'Cannot migrate {source}: {target} already exists')
            if key in moves:
                raise ValueError(f'Cannot migrate {source}: {target} belongs to another compresa')
        if satellite:
            replace_metadata(source / 'timeseries.json', lambda data: convert_series(data, old))
        else:
            replace_metadata(source / 'context.json.gz', lambda data: convert_context(data, old, key), compressed=True)
        if source != target:
            source.rename(target)


def migrate_assets(apps, schema_editor):
    alias = schema_editor.connection.alias
    regions = apps.get_model('base', 'Region').objects.using(alias)
    keys = {}
    for row in regions.all():
        old = row.map_directory or row.name
        if old in keys:
            raise ValueError(f'Multiple comprese claim the same map directory: {old}')
        key = directory_uuid(row.map_directory) or initial_map_key(old)
        keys[old] = str(key)
        regions.filter(pk=row.pk).update(map_key=key)
    for root, satellite in ((settings.SATELLITE_DIR, True), (settings.MAP_CONTEXT_DIR, False)):
        move_assets(root, keys, satellite=satellite)


def reverse_without_assets(apps, schema_editor):
    # File-only builds may predate their catalog, so their old directory names
    # cannot be reconstructed from the database.
    for root, satellite in ((settings.SATELLITE_DIR, True), (settings.MAP_CONTEXT_DIR, False)):
        if root and Path(root).is_dir() and any(
                path.is_dir() and is_asset_directory(path, satellite) for path in Path(root).iterdir()):
            raise IrreversibleError('Restore the pre-upgrade database and generated-map directories together.')
    regions = apps.get_model('base', 'Region').objects.using(schema_editor.connection.alias)
    for pk, key in regions.values_list('pk', 'map_key'):
        regions.filter(pk=pk).update(map_directory=str(key))


class Migration(migrations.Migration):
    dependencies = [('base', '0040_editable_regions')]
    operations = [
        migrations.AddField('region', 'map_key', models.UUIDField(null=True, editable=False)),
        migrations.RunPython(migrate_assets, reverse_without_assets),
        migrations.AlterField('region', 'map_key', models.UUIDField(unique=True, editable=False)),
        migrations.RemoveField('region', 'map_directory'),
    ]
