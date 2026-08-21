from pathlib import Path

import subprocess
import sys


def test_training_smoke_cli() -> None:
    result = subprocess.run([sys.executable, "scripts/run_training_smoke.py"], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path("reports/checkpoint_6_smoke_report.json").is_file()
