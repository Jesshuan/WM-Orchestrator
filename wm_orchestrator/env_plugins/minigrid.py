"""The minigrid plugin: transport (MinigridEnv gRPC service), key mapping,
agent action space, obs unpack/display."""

import grpc

from grpc_api import minigrid_api_pb2 as pb2
from grpc_api import minigrid_api_pb2_grpc as pb2_grpc
from grpc_api.obs_codec import unpack_obs as _unpack_obs

from .base import EnvPlugin, ResetOutcome, StepOutcome


class MinigridPlugin(EnvPlugin):
    """Everything the client needs to speak the MiniGrid env."""

    connection_errors = (grpc.aio.AioRpcError,)

    keys_help = """\
async client keys:  a/d/w = left/right/forward   space = toggle
                    p = pickup   o = drop        e/return = done
                    r = reset    c = hand control agent <-> user
                    q = quit"""

    # Terminal keys, mirroring manual_control's physical controls.
    key_to_action: dict[str, int] = {
        "a": pb2.Action.LEFT,
        "d": pb2.Action.RIGHT,
        "w": pb2.Action.FORWARD,
        " ": pb2.Action.TOGGLE,
        "p": pb2.Action.PICKUP,
        "o": pb2.Action.DROP,
        "e": pb2.Action.DONE
    }

    # What the placeholder agent may pick: every action of key_to_action.
    # The agent may end the episode too (DONE is part of its action space);
    # resets, on the other hand, stay the user's alone.
    agent_actions: list[int] = list(key_to_action.values())

    def __init__(self):
        self._channel = None
        self._stub = None

    # ---- transport (MinigridEnv gRPC service, minigrid_api_pb2) ----

    async def connect(self, host: str, port: int) -> None:
        self._channel = grpc.aio.insecure_channel(f"{host}:{port}")
        self._stub = pb2_grpc.MinigridEnvStub(self._channel)

    async def close(self) -> None:
        if self._channel is not None:
            await self._channel.close()

    async def reset(self) -> ResetOutcome:
        response = await self._stub.Reset(pb2.ResetRequest())
        return ResetOutcome(
            obs=_unpack_obs(response.observation),
            step_count=response.step_count,
        )

    async def step(self, action) -> StepOutcome:
        response = await self._stub.Step(pb2.StepRequest(action=action))
        return StepOutcome(
            obs=_unpack_obs(response.observation),
            reward=response.reward,
            terminated=response.terminated,
            step_count=response.step_count,
        )

    def action_name(self, action) -> str:
        return pb2.Action.Name(action)

    def error_message(self, exc: Exception) -> str:
        return f"{exc.code()} {exc.details()}"

    # ---- observation presentation ----

    def describe_obs(self, obs: dict) -> str:
        """One-line description of an unpacked minigrid observation."""
        image = obs["image"]
        line = (
            f"obs: numpy array shape={image.shape} dtype={image.dtype} "
            f"values={image.min()}..{image.max()}"
        )
        mental = obs["mental_map"]
        if mental is not None:
            line += (
                f" | mental map: {mental.shape[0]}x{mental.shape[1]} grayscale "
                f"values={mental.min()}..{mental.max()}"
            )
        return line

    def print_payload(self, obs: dict) -> None:
        """Print the full unpacked observation payload received over the wire.

        Debug mode only: the image array in full, plus direction/mission, and
        the mental map as an NxN grid of grayscale values (255 = agent,
        100 = goal). Same as grpc_api/act_client.py --debug.
        """
        image = obs["image"]
        print(f"  image: shape={image.shape} dtype={image.dtype}")
        print("  image values:")
        for line in str(image).splitlines():
            print(f"    {line}")
        if obs["direction"] is not None:
            print(f"  direction: {obs['direction']}")
        if obs["mission"]:
            print(f"  mission: {obs['mission']!r}")
        mental = obs["mental_map"]
        if mental is not None:
            print(f"  mental map ({mental.shape[0]}x{mental.shape[1]}):")
            for row in mental:
                print("    " + " ".join(f"{v:3d}" for v in row))


def get_plugin() -> EnvPlugin:
    """Factory the plugin registry (env_plugins.load_plugin) looks for."""
    return MinigridPlugin()
