import io
import json
import sqlite3
import subprocess
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'bin' / 'backup'


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def _make_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('create table sample(id integer primary key, name text not null)')
    conn.execute('insert into sample(name) values (?)', ('abies',))
    conn.commit()
    conn.close()


def test_backup_creates_compressed_sqlite_and_ipso_bundle(tmp_path):
    data_dir = tmp_path / 'data'
    backup_dir = tmp_path / 'backup'
    _make_db(data_dir / 'db.sqlite3')
    inbox_file = data_dir / 'ipso-inbox' / '2026' / '06' / 'session-1' / 'upload.json'
    inbox_file.parent.mkdir(parents=True)
    inbox_file.write_text('{"ok": true}\n', encoding='utf-8')
    tmp_file = inbox_file.with_name('upload.json.tmp')
    tmp_file.write_text('partial', encoding='utf-8')

    _run(
        '--data-dir', str(data_dir),
        '--backup-dir', str(backup_dir),
        '--instance', 'prod',
        '--reason', 'scheduled',
        '--now', '20260625T210000Z',
    )

    archives = list(backup_dir.glob('abies-prod-20260625T210000Z-scheduled.tar.gz'))
    assert len(archives) == 1
    archive = archives[0]

    extracted_db = tmp_path / 'restore.sqlite3'
    with tarfile.open(archive, 'r:gz') as tar:
        names = set(tar.getnames())
        assert 'manifest.json' in names
        assert 'db.sqlite3' in names
        assert 'ipso-inbox/2026/06/session-1/upload.json' in names
        assert 'ipso-inbox/2026/06/session-1/upload.json.tmp' not in names
        manifest = json.load(tar.extractfile('manifest.json'))
        assert manifest['sqlite']['integrity_check'] == 'ok'
        assert manifest['contents'] == ['db.sqlite3', 'ipso-inbox']
        extracted_db.write_bytes(tar.extractfile('db.sqlite3').read())

    conn = sqlite3.connect(extracted_db)
    try:
        assert conn.execute('select name from sample').fetchone() == ('abies',)
        assert conn.execute('pragma integrity_check').fetchone() == ('ok',)
    finally:
        conn.close()


def test_restore_extracts_backup_to_empty_destination(tmp_path):
    data_dir = tmp_path / 'data'
    backup_dir = tmp_path / 'backup'
    _make_db(data_dir / 'db.sqlite3')
    inbox_file = data_dir / 'ipso-inbox' / '2026' / '06' / 'session-1' / 'upload.json'
    inbox_file.parent.mkdir(parents=True)
    inbox_file.write_text('{"ok": true}\n', encoding='utf-8')

    _run(
        '--data-dir', str(data_dir),
        '--backup-dir', str(backup_dir),
        '--instance', 'prod',
        '--now', '20260625T210000Z',
        '--no-prune',
    )
    archive = backup_dir / 'abies-prod-20260625T210000Z-scheduled.tar.gz'
    restore_dir = tmp_path / 'restore'

    _run('--restore', str(archive), '--dest', str(restore_dir))

    assert (restore_dir / 'manifest.json').is_file()
    assert (restore_dir / 'ipso-inbox' / '2026' / '06' / 'session-1' / 'upload.json').read_text(
        encoding='utf-8',
    ) == '{"ok": true}\n'
    conn = sqlite3.connect(restore_dir / 'db.sqlite3')
    try:
        assert conn.execute('select name from sample').fetchone() == ('abies',)
        assert conn.execute('pragma integrity_check').fetchone() == ('ok',)
    finally:
        conn.close()


def test_restore_rejects_unsafe_archive_member(tmp_path):
    archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        payload = b'bad'
        info = tarfile.TarInfo('../evil')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), '--restore', str(archive), '--dest', str(tmp_path / 'restore')],
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert 'unsafe backup member path' in proc.stderr
    assert not (tmp_path / 'evil').exists()


def test_prune_keeps_recent_saturdays_and_every_fourth_old_saturday(tmp_path):
    backup_dir = tmp_path / 'backup'
    backup_dir.mkdir()
    now = datetime(2026, 6, 25, 21, 0, tzinfo=timezone.utc)

    recent = now - timedelta(days=2)
    old_weekday = now - timedelta(days=20)
    old_saturday = _previous_saturday(now - timedelta(days=20))
    very_old_keep = _previous_saturday_with_iso_week(now - timedelta(days=220), 0)
    very_old_prune = _previous_saturday_with_iso_week(now - timedelta(days=220), 1)

    for dt in [recent, old_weekday, old_saturday, very_old_keep, very_old_prune]:
        _touch_archive(backup_dir, 'prod', dt)
    other_instance = _touch_archive(backup_dir, 'dev', old_weekday)

    _run(
        '--backup-dir', str(backup_dir),
        '--instance', 'prod',
        '--now', '20260625T210000Z',
        '--prune-only',
    )

    assert _archive_path(backup_dir, 'prod', recent).exists()
    assert not _archive_path(backup_dir, 'prod', old_weekday).exists()
    assert _archive_path(backup_dir, 'prod', old_saturday).exists()
    assert _archive_path(backup_dir, 'prod', very_old_keep).exists()
    assert not _archive_path(backup_dir, 'prod', very_old_prune).exists()
    assert other_instance.exists()


def _touch_archive(backup_dir, instance, dt):
    path = _archive_path(backup_dir, instance, dt)
    path.write_text('placeholder', encoding='utf-8')
    return path


def _archive_path(backup_dir, instance, dt):
    return backup_dir / f'abies-{instance}-{dt:%Y%m%dT%H%M%SZ}-scheduled.tar.gz'


def _previous_saturday(dt):
    while dt.weekday() != 5:
        dt -= timedelta(days=1)
    return dt


def _previous_saturday_with_iso_week(dt, modulo):
    while True:
        dt = _previous_saturday(dt)
        if dt.isocalendar().week % 4 == modulo:
            return dt
        dt -= timedelta(days=1)
