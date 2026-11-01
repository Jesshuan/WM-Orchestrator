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


@dataclass(frozen=True)
class OrchestratorConfig:
    """Typed view of config.yaml."""

    # Size x of the shared observation queue: only the x most recent
    # observations (agent, user or reset) are kept in memory.
    obs_buffer_size: int = 512

    # Think time of the placeholder policy, in seconds.
    agent_think_time: float = 0.001


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
    return OrchestratorConfig(
        obs_buffer_size=int(orchestrator.get("obs_buffer_size", DEFAULTS.obs_buffer_size)),
        agent_think_time=float(agent.get("think_time", DEFAULTS.agent_think_time)),
    )
