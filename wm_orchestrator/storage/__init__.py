"""Storage backend registry: wm_orchestrator/storage/<name>.py must define get_backend()."""

from __future__ import annotations

import importlib


def load_backend(name: str, batch_size: int, base_path: str):
    """Import the storage backend ``name``, return its configured StorageBackend."""
    module_name = f"wm_orchestrator.storage.{name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"unknown storage mode {name!r}: no plugin module "
            f"'{module_name}.py' (add one with a get_backend() -> "
            f"StorageBackend factory)"
        ) from exc
    factory = getattr(module, "get_backend", None)
    if factory is None:
        raise ValueError(f"backend {module_name!r} defines no get_backend() factory")
    return factory(batch_size=batch_size, base_path=base_path)
