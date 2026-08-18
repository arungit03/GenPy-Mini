import json

import numpy as np
import pytest

from scripts.train import main


def test_train_cli_dry_run_validates_without_optimizer_step(tmp_path, monkeypatch, capsys):
    data = tmp_path / "train.bin"
    np.arange(32, dtype=np.uint16).tofile(data)
    (tmp_path / "train_metadata.json").write_text(json.dumps({"dtype": "uint16", "token_count": 32, "vocab_size": 32000}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train.py", "--train-data", str(data), "--max-steps", "1", "--dry-run", "--device", "cpu"])
    assert main() == 0
    assert "Dry run passed" in capsys.readouterr().out


def test_train_cli_rejects_nonpositive_stop_after_steps(tmp_path, monkeypatch):
    data = tmp_path / "train.bin"
    np.arange(32, dtype=np.uint16).tofile(data)
    (tmp_path / "train_metadata.json").write_text(json.dumps({"dtype": "uint16", "token_count": 32, "vocab_size": 32000}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train-data",
            str(data),
            "--max-steps",
            "20",
            "--stop-after-steps",
            "0",
            "--dry-run",
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="stop-after-steps"):
        main()
