"""Cross-process coordination for database/filesystem snapshots (no Django)."""

from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def runtime_snapshot_lock(path, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
