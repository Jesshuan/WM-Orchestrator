"""Asynchronous acting gRPC client for the Minigrid API — agent/user shared control.

Drives the same single env as act_client.py, but the event loop never
blocks: a thread reads keypresses and hands them to asyncio, while the
agent — for now a placeholder policy (a tiny sleep plus a random action;
a real agent service will replace it) — issues its own steps. Both
sources step the same stub, and every observation produced, by the agent,
the user or a reset, lands in one bounded in-memory queue that keeps only
the x most recent ones (x = orchestrator config's obs_buffer_size).

Control starts with the agent. Pressing 'c' suspends the agent and hands
the manual controls of act_client.py (a/d/w/space/p/o/e, r, q) to the
user; pressing 'c' again releases control and the agent resumes. Movement
keys pressed while the agent is in control are simply ignored.

Run from this directory, inside the env that has grpc_api installed:
    python async_act_client.py [--host HOST] [--port PORT] [--config CONFIG]
    python async_act_client.py --debug
        (--debug prints the full observation payload received per response,
        like grpc_api/act_client.py's debug mode)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import grpc

from grpc_api import minigrid_api_pb2 as pb2
from grpc_api import minigrid_api_pb2_grpc as pb2_grpc
from grpc_api.obs_codec import unpack_obs

from orchestrator_config import OrchestratorConfig, load_config

logger = logging.getLogger("async_act_client")

KEYS_HELP = """\
async client keys:  a/d/w = left/right/forward   space = toggle
                    p = pickup   o = drop        e/return = done
                    r = reset    c = hand control agent <-> user
                    q = quit"""

# What the placeholder agent may pick. The agent has restricted rights:
# DONE (ending the episode) and resets are the user's alone — the agent
# can never terminate the game, so DONE stays out of its action space.
AGENT_ACTIONS: list[int] = [
    pb2.Action.LEFT,
    pb2.Action.RIGHT,
    pb2.Action.FORWARD,
    pb2.Action.PICKUP,
    pb2.Action.DROP,
    pb2.Action.TOGGLE,
]

# Terminal keys, mirroring manual_control's physical controls.
KEY_TO_ACTION: dict[str, int] = {
    "a": pb2.Action.LEFT,
    "d": pb2.Action.RIGHT,
    "w": pb2.Action.FORWARD,
    " ": pb2.Action.TOGGLE,
    "p": pb2.Action.PICKUP,
    "o": pb2.Action.DROP,
    "e": pb2.Action.DONE,
    # alias of the older manual_control binding, for muscle memory
    "\r": pb2.Action.DONE,
}

QUIT_KEYS = ("q", "Q", "\x03", "\x04")  # q, Ctrl-C, Ctrl-D


def _read_key() -> str:
    """Read one keypress from the terminal in raw mode."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _describe_obs(obs_decoded: dict) -> str:
    """Rebuild the numpy obs and describe it (see obs_codec.unpack_obs)."""
    
    image = obs_decoded["image"]
    line = (
        f"obs: numpy array shape={image.shape} dtype={image.dtype} "
        f"values={image.min()}..{image.max()}"
    )
    mental = obs_decoded["mental_map"]
    if mental is not None:
        line += (
            f" | mental map: {mental.shape[0]}x{mental.shape[1]} grayscale "
            f"values={mental.min()}..{mental.max()}"
        )
    return line


class Mode(Enum):
    """Who is allowed to send Step commands."""

    AGENT = auto()
    USER = auto()


@dataclass(frozen=True)
class ObsRecord:
    """One observation in the shared queue, tagged with its provenance."""

    obs: dict  # unpack_obs output: image / direction / mission / mental_map
    source: str  # "agent", "user" or "reset"
    reward: float = 0.0
    step_count: int = -1
    timestamp: float = field(default_factory=time.time)


class ObsBuffer:
    """The shared observation queue: a sliding window of the x last obs."""

    def __init__(self, maxlen: int):
        self._queue: deque[ObsRecord] = deque(maxlen=maxlen)

    def push(self, record: ObsRecord) -> None:
        """Append one observation; the oldest is dropped once x is reached."""
        self._queue.append(record)

    def __len__(self) -> int:
        return len(self._queue)

    def snapshot(self) -> list[ObsRecord]:
        """Oldest-to-newest copy, for later consumers (trainer, replay...)."""
        return list(self._queue)


async def agent_act(obs: dict, cfg: OrchestratorConfig) -> int:
    """Placeholder policy — the future agent service plugs in right here.

    Thinks for a fixed instant and picks a random action, using the last
    observation it received (`obs`).
    """
    await asyncio.sleep(cfg.agent_think_time)
    return random.choice(AGENT_ACTIONS)


def _print_payload(obs_decoded: dict) -> None:
    """Print the full unpacked observation payload received over the wire.

    Debug mode only: the image array in full, plus direction/mission, and
    the mental map as an NxN grid of grayscale values (255 = agent,
    100 = goal). Same as grpc_api/act_client.py --debug.
    """
    image = obs_decoded["image"]
    print(f"  image: shape={image.shape} dtype={image.dtype}")
    print("  image values:")
    for line in str(image).splitlines():
        print(f"    {line}")
    if obs_decoded["direction"] is not None:
        print(f"  direction: {obs_decoded['direction']}")
    if obs_decoded["mission"]:
        print(f"  mission: {obs_decoded['mission']!r}")
    mental = obs_decoded["mental_map"]
    if mental is not None:
        print(f"  mental map ({mental.shape[0]}x{mental.shape[1]}):")
        for row in mental:
            print("    " + " ".join(f"{v:3d}" for v in row))


def _spawn_key_reader(queue: asyncio.Queue[str]) -> threading.Thread:
    """Read keypresses in a daemon thread; deliver them to the loop's queue.

    The thread runs until the process exits; keys pressed while the loop
    is busy simply wait in the asyncio queue.
    """
    loop = asyncio.get_running_loop()

    def _read_loop() -> None:
        while True:
            key = _read_key()
            loop.call_soon_threadsafe(queue.put_nowait, key)

    thread = threading.Thread(target=_read_loop, name="key-reader", daemon=True)
    thread.start()
    return thread


async def run(args: argparse.Namespace) -> int:
    """Act over gRPC, either the agent or the user, into one obs queue."""
    if not sys.stdin.isatty():
        print("act mode needs an interactive terminal", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    obs_buffer = ObsBuffer(cfg.obs_buffer_size)
    print(f"observation queue keeps the last {cfg.obs_buffer_size} in memory")
    print(KEYS_HELP)

    channel = grpc.aio.insecure_channel(f"{args.host}:{args.port}")
    stub = pb2_grpc.MinigridEnvStub(channel)

    keys: asyncio.Queue[str] = asyncio.Queue()
    _spawn_key_reader(keys)

    mode = Mode.AGENT
    obs: dict

    async def on_key(key: str) -> bool:
        """Handle one keypress; return False to stop the loop."""
        nonlocal mode, obs
        if key in QUIT_KEYS:
            return False
        if key == "c":
            if mode is Mode.AGENT:
                mode = Mode.USER
                print("user takes control — agent suspended (c to release)")
            else:
                mode = Mode.AGENT
                print("agent back in control (c to take over)")
            return True
        if key == "r":
            response = await stub.Reset(pb2.ResetRequest())
            obs = unpack_obs(response.observation)
            obs_buffer.push(ObsRecord(obs=obs, source="reset", step_count=response.step_count))
            print(f"reset! {_describe_obs(obs)}")
            if debug:
                _print_payload(obs)
            return True
        if mode is Mode.AGENT:
            return True  # movement keys only matter when the user is in control
        action = KEY_TO_ACTION.get(key)
        if action is None:
            return True
        response = await stub.Step(pb2.StepRequest(action=action))
        obs = unpack_obs(response.observation)
        obs_buffer.push(
            ObsRecord(obs=obs, source="user", reward=response.reward, step_count=response.step_count)
        )
        print(
            f"[user] key={key} action={pb2.Action.Name(action)} "
            f"reward={response.reward:+.2f} terminated={response.terminated} "
            f"{_describe_obs(obs)}"
        )
        if debug:
            print("observation payload:")
            _print_payload(obs)
        return True

    running = True
    debug = args.debug
    try:
        response = await stub.Reset(pb2.ResetRequest())
        obs = unpack_obs(response.observation)
        obs_buffer.push(ObsRecord(obs=obs, source="reset", step_count=response.step_count))
        print(f"reset: {_describe_obs(obs)}")
        if debug:
            _print_payload(obs)

        while running:
            # First drain everything already pressed (toggles, resets, quit).
            while not keys.empty():
                running = await on_key(keys.get_nowait())
                if not running:
                    break
            if not running:
                break

            if mode is Mode.USER:
                # Block on the next keypress: the user decides the pace.
                running = await on_key(await keys.get())
                continue

            # Agent mode: one step, then re-check the keys right away so a
            # 'c' pressed mid-step is honored at most one step late.
            action = await agent_act(obs, cfg)
            response = await stub.Step(pb2.StepRequest(action=action))
            obs = unpack_obs(response.observation)
            obs_buffer.push(
                ObsRecord(
                    obs=obs, source="agent", reward=response.reward, step_count=response.step_count
                )
            )
            print(
                f"[agent] action={pb2.Action.Name(action)} "
                f"reward={response.reward:+.2f} terminated={response.terminated} "
                f"{_describe_obs(obs)}"
            )
            if debug:
                print("observation payload:")
                _print_payload(obs)

    except grpc.aio.AioRpcError as exc:
        print(f"server error: {exc.code()} {exc.details()}", file=sys.stderr)
    finally:
        await channel.close()
        print(f"bye — {len(obs_buffer)} observations kept (capacity {cfg.obs_buffer_size})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server address")
    parser.add_argument("--port", type=int, default=50051, help="server port")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="orchestrator yaml config (default: config.yaml next to this file)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print the full observation payload received per response",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
