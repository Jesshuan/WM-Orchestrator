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

Run from the project root, inside the env that has grpc_api installed:
    python scripts/run_client.py [--host HOST] [--port PORT] [--env ENV]
                                 [--config CONFIG]
    python scripts/run_client.py --debug
        (--debug prints the full observation payload received per response,
        like grpc_api/act_client.py's debug mode)

The client is env-agnostic: everything env-specific — the transport
(channel, stub and the Reset/Step requests over the env's own pb2
protocol), the key mapping, the agent action space, and observation
unpacking/display — lives in wm_orchestrator/env_plugins/<name>.py, chosen by env.name
in config.yaml; --host/--port/--env override the yaml for a single run.
The core loop below only calls env_plugin.connect/reset/step/close and logs
plain outcomes, so another env just ships another plugin.

For later model training, the observations also outlive the process:
every record pushed to the queue feeds a storage backend as well, which
writes a batch to disk each time record_batch_size records have
accumulated — and at every episode boundary, so a batch never spans two
episodes (storage.* keys in config.yaml; filesystem by default, one
day-stamped folder of .npz batch files). Backends come from
wm_orchestrator/storage/<mode>.py, the same plugin shape as env_plugins.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import threading
from collections import deque
from enum import Enum, auto

from wm_orchestrator.config import OrchestratorConfig, load_config
from wm_orchestrator.records import ObsRecord
from wm_orchestrator.env_plugins import load_plugin
from wm_orchestrator.env_plugins.base import EnvPlugin
from wm_orchestrator.storage import load_backend
from wm_orchestrator.storage.base import StorageBackend

logger = logging.getLogger("wm_orchestrator.client")

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


class Mode(Enum):
    """Who is allowed to send Step commands."""

    AGENT = auto()
    USER = auto()


class ObsBuffer:
    """The shared observation queue: a sliding window of the x last obs.

    Each push also feeds ``sink`` (a storage backend) when one is
    attached, so the records outlive the in-memory window on disk.
    """

    def __init__(self, maxlen: int, sink: StorageBackend | None = None):
        self._queue: deque[ObsRecord] = deque(maxlen=maxlen)
        self._sink = sink

    def push(self, record: ObsRecord) -> None:
        """Append one observation; the oldest is dropped once x is reached."""
        self._queue.append(record)
        if self._sink is not None:
            self._sink.add(record)

    def __len__(self) -> int:
        return len(self._queue)

    def snapshot(self) -> list[ObsRecord]:
        """Oldest-to-newest copy, for later consumers (trainer, replay...)."""
        return list(self._queue)


async def agent_act(obs: dict, cfg: OrchestratorConfig, env_plugin: EnvPlugin) -> int:
    """Placeholder policy — the future agent service plugs in right here.

    Thinks for a fixed instant and picks a random action, using the last
    observation it received (`obs`).
    """
    await asyncio.sleep(cfg.agent_think_time)
    return random.choice(env_plugin.agent_actions)


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
    env_plugin = load_plugin(args.env if args.env is not None else cfg.env)
    host = args.host if args.host is not None else cfg.host
    port = args.port if args.port is not None else cfg.port
    storage = load_backend(
        cfg.storage_mode,
        batch_size=cfg.record_batch_size,
        base_path=cfg.records_path,
    )
    obs_buffer = ObsBuffer(cfg.obs_buffer_size, sink=storage)
    print(f"observation queue keeps the last {cfg.obs_buffer_size} in memory")
    print(storage.describe())
    if cfg.obs_buffer_size % cfg.record_batch_size:
        print(
            f"warning: batch_size {cfg.record_batch_size} does not divide "
            f"obs_buffer_size {cfg.obs_buffer_size} — batches still flush "
            f"every {cfg.record_batch_size} records, independent of the window",
            file=sys.stderr,
        )

    await env_plugin.connect(host, port)
    storage.open()

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
            outcome = await env_plugin.reset()
            obs = outcome.obs
            obs_buffer.push(ObsRecord(obs=obs, source="reset", step_count=outcome.step_count))
            print(f"reset! {env_plugin.describe_obs(obs)}")
            if debug:
                env_plugin.print_payload(obs)
            return True
        if mode is Mode.AGENT:
            return True  # movement keys only matter when the user is in control
        action = env_plugin.key_to_action.get(key)
        if action is None:
            return True
        outcome = await env_plugin.step(action)
        obs = outcome.obs
        obs_buffer.push(
            ObsRecord(
                obs=obs,
                source="user",
                reward=outcome.reward,
                terminated=outcome.terminated,
                step_count=outcome.step_count,
            )
        )
        print(
            f"[user] key={key} action={env_plugin.action_name(action)} "
            f"reward={outcome.reward:+.2f} terminated={outcome.terminated} "
            f"{env_plugin.describe_obs(obs)}"
        )
        if debug:
            print("observation payload:")
            env_plugin.print_payload(obs)
        return True

    running = True
    debug = args.debug
    try:
        outcome = await env_plugin.reset()
        obs = outcome.obs
        obs_buffer.push(ObsRecord(obs=obs, source="reset", step_count=outcome.step_count))
        print(f"reset: {env_plugin.describe_obs(obs)}")
        if debug:
            env_plugin.print_payload(obs)

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
            action = await agent_act(obs, cfg, env_plugin)
            outcome = await env_plugin.step(action)
            obs = outcome.obs
            obs_buffer.push(
                ObsRecord(
                    obs=obs,
                    source="agent",
                    reward=outcome.reward,
                    terminated=outcome.terminated,
                    step_count=outcome.step_count,
                )
            )
            print(
                f"[agent] action={env_plugin.action_name(action)} "
                f"reward={outcome.reward:+.2f} terminated={outcome.terminated} "
                f"{env_plugin.describe_obs(obs)}"
            )
            if debug:
                print("observation payload:")
                env_plugin.print_payload(obs)

    except env_plugin.connection_errors as exc:
        print(f"server error: {env_plugin.error_message(exc)}", file=sys.stderr)
    finally:
        await env_plugin.close()
        batches = storage.close()
        print(
            f"bye — {len(obs_buffer)} observations kept (capacity {cfg.obs_buffer_size}), "
            f"{storage.record_count} written to {cfg.records_path} in {batches} batch(es)"
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="server address (default: server.host from config.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="server port (default: server.port from config.yaml)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="env plugin to load (default: env.name from config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="orchestrator yaml config (default: config.yaml at the project root)",
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
