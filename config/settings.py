"""Project path constants and safe environment-variable helpers.

Nothing in this module loads secrets into source control: values are read
from the process environment at call time only, never written to disk, and
never bundled with a default that looks like a real credential.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DOCS_DIR: Final[Path] = PROJECT_ROOT / "docs"
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
SCRIPTS_DIR: Final[Path] = PROJECT_ROOT / "scripts"
TESTS_DIR: Final[Path] = PROJECT_ROOT / "tests"

MODEL_CONFIG_PATH: Final[Path] = CONFIG_DIR / "model_config.yaml"


def get_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, returning ``default`` if it is unset."""
    return os.environ.get(name, default)


def get_env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable (``1``/``true``/``yes``/``on``)."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int | None = None) -> int | None:
    """Read an integer environment variable, falling back to ``default`` on absence."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable '{name}' is not a valid integer.") from exc


def require_env(name: str) -> str:
    """Read a required environment variable, raising a clear error if unset.

    The error message never includes the value of any other environment
    variable, so it is safe to surface directly to a user or log.
    """
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Required environment variable '{name}' is not set.")
    return value
