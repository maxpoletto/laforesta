"""Carry the destination's encrypted credentials through an instance mirror."""
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory


TABLE = 'base_satellitecredentials'
COLUMNS = 'singleton_id, version, created_at, modified_at, client_id, encrypted_secret'


def has_credentials_table(connection):
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE,),
    ).fetchone()


def prepare(target, incoming, stash):
    if target.is_file():
        with closing(sqlite3.connect(target)) as connection:
            row = (connection.execute(f'SELECT {COLUMNS} FROM {TABLE} WHERE singleton_id=1').fetchone()
                   if has_credentials_table(connection) else None)
        if row is not None:
            with TemporaryDirectory(dir=stash.parent) as directory:
                staged = Path(directory) / stash.name
                with staged.open('w') as stream:
                    json.dump(row, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                staged.chmod(0o600)
                staged.replace(stash)
    with closing(sqlite3.connect(incoming)) as connection, connection:
        if has_credentials_table(connection):
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute(f'DELETE FROM {TABLE}')


def restore(target, stash):
    if not stash.exists():
        return
    with closing(sqlite3.connect(target)) as connection, connection:
        connection.execute(f'INSERT OR REPLACE INTO {TABLE} ({COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)',
                           json.loads(stash.read_text()))
    stash.unlink()


if __name__ == '__main__':
    action, *paths = sys.argv[1:]
    {'prepare': prepare, 'restore': restore}[action](*map(Path, paths))
