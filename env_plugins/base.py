"""The per-env plugin contract for async_act_client.

An EnvPlugin owns everything that varies by env: the transport (channel,
stub and the Reset/Step requests over its own pb2 protocol), the
on-screen key mapping, the agent's action space, and observation
unpacking/display. The core loop in async_act_client only ever calls
connect/reset/step/close and reads the generic outcomes below — it knows
nothing about any particular proto.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResetOutcome:
    """What a reset produced, stripped of any env-specific protocol."""

    obs: dict  # plain, env-specific keys chosen by the plugin
    step_count: int = -1


@dataclass(frozen=True)
class StepOutcome:
    """What a step produced, stripped of any env-specific protocol."""

    obs: dict
    reward: float = 0.0
    terminated: bool = False
    step_count: int = -1


class EnvPlugin:
    """Base class every env plugin extends."""

    # Help block printed once at startup (replaces KEYS_HELP).
    keys_help: str = ""

    # Terminal key -> action value understood by step().
    key_to_action: dict = {}

    # What the placeholder agent may pick.
    agent_actions: list = []

    # Transport exception types that mean "server error" in the core loop.
    connection_errors: tuple = ()

    # ---- transport: each env speaks its own protocol behind these ----

    async def connect(self, host: str, port: int) -> None:
        """Open the connection to the env server."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the connection."""
        raise NotImplementedError

    async def reset(self) -> ResetOutcome:
        """Reset the env; returns the first observation."""
        raise NotImplementedError

    async def step(self, action) -> StepOutcome:
        """Apply one action; returns the resulting observation."""
        raise NotImplementedError

    def action_name(self, action) -> str:
        """Human-readable name of an action, for the step log lines."""
        raise NotImplementedError

    def error_message(self, exc: Exception) -> str:
        """One-line human-readable summary of a transport error."""
        raise NotImplementedError

    # ---- observation presentation ----

    def describe_obs(self, obs: dict) -> str:
        """One-line description appended to step/reset messages."""
        raise NotImplementedError

    def print_payload(self, obs: dict) -> None:
        """Multi-line debug dump of the full unpacked payload."""
        raise NotImplementedError
