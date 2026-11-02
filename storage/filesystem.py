"""The filesystem storage backend: one NumPy archive (``.npz``) per batch.

Batches land under ``<base_path>/<YYYYMMDD>/`` — the folder name is the
day and nothing else, so every run of a given day shares that folder
and batch numbering continues across runs (determined at open time from
the highest batch file already on disk). A model service ingesting a
day folder thus gets every batch of that day, in write order.

Each file stores the batch as column arrays: timestamp, source, reward,
step_count, terminated, plus one ``obs_<key>`` array per observation
key (stacked when the shapes allow, else an object array — such files
need ``np.load(..., allow_pickle=True)``). Unlike flat JSON, the integer
dtypes of the MiniGrid images survive the round trip verbatim, and
image-sized payloads stay compact.

Writes go through a temp file and an atomic rename, so a training job
that scans the folder mid-run only ever sees complete batch files.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np

from storage.base import StorageBackend

DAY_STAMP = "%Y%m%d"


def _obs_arrays(batch) -> dict[str, np.ndarray]:
    """One array per observation key, stacked when the shapes allow.

    Keys whose values don't stack (e.g. a mental map that is None on
    some steps of the batch) fall back to an object array; reading such
    an array requires np.load(..., allow_pickle=True).
    """
    keys: list[str] = []
    for record in batch:
        for key in record.obs:
            if key not in keys:
                keys.append(key)
    arrays = {}
    for key in keys:
        values = [record.obs.get(key) for record in batch]
        try:
            arrays[f"obs_{key}"] = np.stack(values)
        except (ValueError, TypeError):
            fallback = np.empty(len(values), dtype=object)
            fallback[:] = values
            arrays[f"obs_{key}"] = fallback
    return arrays


class FilesystemBackend(StorageBackend):
    """Record batches as .npz files under a day-stamped directory.

    The directory is created lazily, on the first batch, so a run that
    produced no observations leaves nothing behind.
    """

    name = "filesystem"

    def __init__(self, batch_size: int, base_path: str):
        super().__init__(batch_size=batch_size, base_path=base_path)
        self._day_dir = None  # resolved lazily, on the first batch
        self._next_index = 0  # continues the day's numbering across runs

    def open(self) -> None:
        if self._day_dir is None:
            day = datetime.now().strftime(DAY_STAMP)
            self._day_dir = Path(self.base_path) / day
            self._day_dir.mkdir(parents=True, exist_ok=True)
            # Same-day reruns append to the folder: resume after the
            # highest batch index already on disk.
            for path in self._day_dir.glob("batch_*.npz"):
                self._next_index = max(self._next_index, int(path.stem.split("_")[1]) + 1)

    def write_batch(self, batch) -> None:
        self.open()  # lazy: the first batch creates the day directory
        path = self._day_dir / f"batch_{self._next_index:05d}.npz"
        self._next_index += 1
        columns = {
            "timestamp": np.array([r.timestamp for r in batch], dtype=np.float64),
            "source": np.array([r.source for r in batch]),
            "reward": np.array([r.reward for r in batch], dtype=np.float64),
            "step_count": np.array([r.step_count for r in batch], dtype=np.int64),
            "terminated": np.array([r.terminated for r in batch], dtype=bool),
            **_obs_arrays(batch),
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **columns)  # file handle: no '.npz' appended to tmp
        os.replace(tmp, path)  # atomic: readers never see a partial batch

    def describe(self) -> str:
        return (
            f"storage: filesystem — batches of {self.batch_size} records "
            f"as .npz under {self.base_path}/<date>/"
        )


def get_backend(batch_size: int, base_path: str) -> FilesystemBackend:
    """Factory the backend registry (storage.load_backend) looks for."""
    return FilesystemBackend(batch_size=batch_size, base_path=base_path)
