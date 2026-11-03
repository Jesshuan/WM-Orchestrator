"""Non-interactive smoke test for wm_orchestrator.client against a live server.

Patches the key reader to feed a scripted sequence:
  'c' -> user takes control, 'w'/' '/'p' -> user steps, 'c' -> agent resumes,
then lets the agent act for a couple of seconds before cancelling.

--port 50061 (here) overrides server.port from config.yaml; everything
else (host, env plugin) comes from the yaml defaults.
"""

import asyncio
import sys
import time
from pathlib import Path

# Make the package importable when run in-place, without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wm_orchestrator.client as ac

scripted = iter(["c", "w", " ", "p", "r", "c"])


def fake_read_key() -> str:
    try:
        return next(scripted)
    except StopIteration:
        time.sleep(3600)  # no more scripted keys; just idle forever
        return ""


ac._read_key = fake_read_key
sys.stdin.isatty = lambda: True


async def main() -> None:
    args = ac.parse_args(["--port", "50061", "--debug"])
    task = asyncio.create_task(ac.run(args))
    await asyncio.sleep(4.0)  # user steps, a user reset, then some agent steps
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


asyncio.run(main())
