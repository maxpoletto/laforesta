"""Small durability primitives; callers retain publication/rollback ownership."""
import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
