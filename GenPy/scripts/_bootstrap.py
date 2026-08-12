"""Make direct ``python scripts/<command>.py`` execution import the project."""

import sys
from pathlib import Path


def ensure_project_root() -> None:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
