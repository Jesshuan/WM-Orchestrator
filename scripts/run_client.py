"""Thin launcher for the async act client (the code lives in
wm_orchestrator.client; this script only exists to be run in-place,
without installing the package)."""

import sys
from pathlib import Path

# Make the package importable when run in-place (python scripts/run_client.py),
# without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wm_orchestrator.client import main

if __name__ == "__main__":
    sys.exit(main())
