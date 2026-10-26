"""A database snapshot must restore every photo it references, despite live edits."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import runpy
import sqlite3
import tarfile
from threading import Event

import pytest

from test.image_helpers import image_bytes

SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'backup'


@pytest.fixture
def backup():
    return runpy.run_path(str(SCRIPT))


def photo_data(tmp_path):
    data = tmp_path / 'data'
    path = data / 'observation-media' / '1' / 'photo.jpg'
    path.parent.mkdir(parents=True)
    content = image_bytes()
    path.write_bytes(content)
    with sqlite3.connect(data / 'db.sqlite3') as db:
        db.execute('CREATE TABLE base_observationphoto (id INTEGER PRIMARY KEY, file_path TEXT, checksum TEXT, size_bytes INTEGER)')
        db.execute('INSERT INTO base_observationphoto VALUES (1, ?, ?, ?)',
                   ('1/photo.jpg', hashlib.sha256(content).hexdigest(), len(content)))
    return data, path, content


def create(backup, data, tmp_path):
    directory = tmp_path / 'backup'
    directory.mkdir(exist_ok=True)
    return backup['create_backup'](
        data_dir=data, backup_dir=directory, instance='test', reason='scheduled',
        now=datetime(2026, 10, 26, tzinfo=timezone.utc), include_ipso_inbox=True,
    )


def test_live_delete_after_database_snapshot_keeps_photo_in_backup(backup, tmp_path, monkeypatch):
    data, path, content = photo_data(tmp_path)
    finished = Event()
    original_copy = backup['copy_sqlite_database']
    original_tar_open = tarfile.open
    futures = []

    def delete_photo():
        # This is the public lock protocol used by media-mutating requests.
        with (data / '.runtime-snapshot.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            with sqlite3.connect(data / 'db.sqlite3') as db:
                db.execute('DELETE FROM base_observationphoto')
            path.unlink()
        finished.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        def snapshot_then_delete(source, target):
            result = original_copy(source, target)
            futures.append(pool.submit(delete_photo))
            finished.wait(0.1)
            return result

        def archive_after_snapshot(*args, **kwargs):
            mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
            if mode == 'w:gz':
                assert finished.wait(5), 'media writes remained blocked during compression'
            return original_tar_open(*args, **kwargs)

        monkeypatch.setitem(backup['create_backup'].__globals__, 'copy_sqlite_database', snapshot_then_delete)
        monkeypatch.setattr(tarfile, 'open', archive_after_snapshot)
        archive = create(backup, data, tmp_path)
        for future in futures:
            future.result(timeout=5)

    assert not path.exists()
    with tarfile.open(archive) as tar:
        assert tar.extractfile('observation-media/1/photo.jpg').read() == content
        snapshot = tmp_path / 'snapshot.sqlite3'
        snapshot.write_bytes(tar.extractfile('db.sqlite3').read())
    with sqlite3.connect(snapshot) as db:
        assert db.execute('SELECT COUNT(*) FROM base_observationphoto').fetchone() == (1,)


@pytest.mark.parametrize('problem', ['missing', 'corrupt'])
def test_incomplete_photo_snapshot_is_never_published(backup, tmp_path, problem):
    data, path, content = photo_data(tmp_path)
    if problem == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'x' * len(content))
    with pytest.raises(SystemExit, match='photo'):
        create(backup, data, tmp_path)
    assert not list((tmp_path / 'backup').glob('*.tar.gz'))


def test_restore_validates_photos_before_exposing_destination(backup, tmp_path):
    data, path, content = photo_data(tmp_path)
    archive = tmp_path / 'incomplete.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(data / 'db.sqlite3', arcname='db.sqlite3')
        metadata = json.dumps({'format': 1}).encode()
        member = tarfile.TarInfo('manifest.json')
        member.size = len(metadata)
        tar.addfile(member, io.BytesIO(metadata))
    destination = tmp_path / 'restore'
    with pytest.raises(SystemExit, match='photo'):
        backup['restore_backup'](archive, destination)
    assert not destination.exists()


def test_backup_refuses_to_overwrite_a_completed_archive(backup, tmp_path):
    data, _, _ = photo_data(tmp_path)
    archive = create(backup, data, tmp_path)
    original = archive.read_bytes()
    with pytest.raises(SystemExit, match='exists'):
        create(backup, data, tmp_path)
    assert archive.read_bytes() == original
