"""Env plugin registry: env_plugins/<name>.py must define get_plugin()."""

import importlib


def load_plugin(name: str):
    """Import plugins.<name> and return its EnvPlugin instance."""
    module_name = f"env_plugins.{name}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"unknown env {name!r}: no plugin module '{module_name}.py' "
            f"(add one with a get_plugin() -> EnvPlugin factory)"
        ) from exc
    factory = getattr(module, "get_plugin", None)
    if factory is None:
        raise ValueError(f"plugin {module_name!r} defines no get_plugin() factory")
    return factory()
