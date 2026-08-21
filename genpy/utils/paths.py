"""Stable project paths independent of the current working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INSTRUCTION_DATA_DIR = DATA_DIR / "instruction"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"


def ensure_runtime_directories() -> None:
    """Create runtime directories used by later checkpoints."""
    for directory in (
        CONFIG_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, INSTRUCTION_DATA_DIR,
        CHECKPOINT_DIR, LOG_DIR, REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
