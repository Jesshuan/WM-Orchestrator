"""The orchestrator's global config: one yaml file, loaded once, typed.

Every component (the async act client now, the orchestrator later) reads
the same config.yaml, so hyper-parameters like the observation-queue size
live in a single place. Unknown yaml keys are ignored; missing ones fall
back to the defaults below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Default destination for record batches: <project root>/records —
# anchored to the repo, not the working directory.
DEFAULT_RECORDS_PATH = str(Path(__file__).with_name("records"))


@dataclass(frozen=True)
class OrchestratorConfig:
    """Typed view of config.yaml."""

    # Size x of the shared observation queue: only the x most recent
    # observations (agent, user or reset) are kept in memory.
    obs_buffer_size: int = 512

    # Record storage: every ``record_batch_size`` records pushed to the
    # obs buffer, a batch of that size is written by storage backend
    # ``storage_mode`` under ``records_path``.
    storage_mode: str = "filesystem"
    record_batch_size: int = 64
    records_path: str = DEFAULT_RECORDS_PATH

    # Think time of the placeholder policy, in seconds.
    agent_think_time: float = 0.001

    # Server the act client connects to.
    host: str = "127.0.0.1"
    port: int = 50051

    # Env plugin to load (env_plugins/<name>.py).
    env: str = "minigrid"


DEFAULTS = OrchestratorConfig()


def load_config(path: str | Path | None = None) -> OrchestratorConfig:
    """Overlay config.yaml (or ``path``) on the defaults above."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    data: dict = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    orchestrator = data.get("orchestrator") or {}
    agent = data.get("agent") or {}
    server = data.get("server") or {}
    env = data.get("env") or {}
    storage = data.get("storage") or {}
    return OrchestratorConfig(
        obs_buffer_size=int(orchestrator.get("obs_buffer_size", DEFAULTS.obs_buffer_size)),
        storage_mode=str(storage.get("mode", DEFAULTS.storage_mode)),
        record_batch_size=int(storage.get("batch_size", DEFAULTS.record_batch_size)),
        records_path=str(storage.get("path", DEFAULTS.records_path)),
        agent_think_time=float(agent.get("think_time", DEFAULTS.agent_think_time)),
        host=str(server.get("host", DEFAULTS.host)),
        port=int(server.get("port", DEFAULTS.port)),
        env=str(env.get("name", DEFAULTS.env)),
    )
