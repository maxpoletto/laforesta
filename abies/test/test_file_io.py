"""Directory durability must close its descriptor on success and failure."""
import os
from pathlib import Path

import pytest

from apps.base.file_io import fsync_directory


@pytest.mark.parametrize('fail', [False, True])
def test_directory_sync_closes_descriptor(monkeypatch, fail):
    calls = []
    def opened(path, flags):
        calls.append(('open', path, flags))
        return 42
    def synced(fd):
        calls.append(('sync', fd))
        if fail:
            raise OSError('Interrupted directory sync')
    monkeypatch.setattr(os, 'open', opened)
    monkeypatch.setattr(os, 'fsync', synced)
    monkeypatch.setattr(os, 'close', lambda fd: calls.append(('close', fd)))
    if fail:
        with pytest.raises(OSError, match='Interrupted directory sync'):
            fsync_directory(Path('/tmp'))
    else:
        fsync_directory(Path('/tmp'))
    assert calls == [('open', Path('/tmp'), os.O_RDONLY), ('sync', 42), ('close', 42)]
