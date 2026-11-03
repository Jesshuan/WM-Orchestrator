"""The shared observation record, imported by the client loop and by the
storage plugins (which persist batches of these to disk)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObsRecord:
    """One observation in the shared queue, tagged with its provenance."""

    obs: dict  # plugin.unpack_obs output; the keys are env-specific
    source: str  # "agent", "user" or "reset"
    reward: float = 0.0
    terminated: bool = False  # episode-ending step; storage flushes on it
    step_count: int = -1
    timestamp: float = field(default_factory=time.time)
