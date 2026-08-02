"""Project-wide constants that do not belong in the model configuration schema."""

from __future__ import annotations

from typing import Final

DEFAULT_ENCODING: Final[str] = "utf-8"
MIN_SUPPORTED_PYTHON: Final[tuple[int, int]] = (3, 11)
CONFIG_SCHEMA_VERSION: Final[int] = 1

# Planned training-scale checkpoints, see docs/roadmap.md. Names only -- no
# architecture for these stages is implemented in Phase 1.
MODEL_STAGE_NAMES: Final[tuple[str, ...]] = (
    "GenPy-Nano",
    "GenPy-Tiny",
    "GenPy-Mini",
)
