"""The storage backend contract: where record batches go and how.

Users keep interacting with the env, and the resulting observations
should outlive the process for later model training. The backend owns
the destination (filesystem today, S3 tomorrow); the base class owns
the batching: records accumulate through add(), a batch is handed to
write_batch() at every batch_size records — and at every episode
boundary, so a batch never spans two episodes — with any tail flushed
on close() so a run is never more than one batch short on disk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from records import ObsRecord


class StorageBackend:
    """Base class every storage backend extends.

    Subclasses plug in a destination by implementing open,
    write_batch and describe; batching is handled here once and for
    all.
    """

    name: str = "abstract"

    def __init__(self, batch_size: int, base_path: str):
        self.batch_size = batch_size
        self.base_path = base_path
        self._pending: list[ObsRecord] = []
        # Counters for the end-of-run summary.
        self.batch_count = 0
        self.record_count = 0

    # ---- batching (shared by every backend) ----

    def add(self, record: ObsRecord) -> None:
        """Sink one record; write a batch every batch_size records.

        A full batch is written as soon as it is; a terminated record
        flushes the batch it ends, and a reset closes the previous
        episode's batch before joining the fresh one — so a batch never
        mixes two episodes.
        """
        if record.source == "reset" and self._pending:
            self.flush()
        self._pending.append(record)
        if len(self._pending) >= self.batch_size or record.terminated:
            self.flush()

    def flush(self) -> None:
        """Write the pending records as one (possibly partial) batch."""
        if self._pending:
            self.write_batch(self._pending)
            self.batch_count += 1
            self.record_count += len(self._pending)
            self._pending = []

    def close(self) -> int:
        """Flush the partial tail batch and return the total batch count."""
        self.flush()
        return self.batch_count

    # ---- destination (per backend) ----

    def open(self) -> None:
        """Prepare the destination before the first batch is written."""
        raise NotImplementedError

    def write_batch(self, _batch: list[ObsRecord]) -> None:
        """Persist one batch of records (full or episode-boundary tail)."""
        raise NotImplementedError

    def describe(self) -> str:
        """One-line description for the startup log."""
        raise NotImplementedError
