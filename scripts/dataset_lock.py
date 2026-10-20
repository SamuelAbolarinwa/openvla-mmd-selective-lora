"""Serialise RLDS statistics discovery/generation across training processes."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def dataset_statistics_lock(dataset_root):
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.vla-statistics.lock').open('a') as handle:
        print('Waiting for dataset statistics lock...', flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
