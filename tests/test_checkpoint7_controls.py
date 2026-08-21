import json
import subprocess
import sys
from pathlib import Path

from training_helpers import tiny_engine


def test_session_limit_preserves_global_target_and_resume(tmp_path) -> None:
    engine = tiny_engine(tmp_path, steps=3)
    first = engine.run(session_steps=1)
    assert first.global_step == 1
    assert engine.scheduler.total_steps == 3
    assert engine.last_run_status == "SESSION_COMPLETE"
    resumed = tiny_engine(tmp_path / "resume", steps=3)
    resumed.resume(tmp_path / "run" / "checkpoints" / "step_000000000001")
    final = resumed.run(session_steps=2)
    assert final.global_step == 3
    assert resumed.scheduler.total_steps == 3
    assert resumed.last_run_status == "TRAINING_COMPLETE"


def test_checkpoint_transport_package_and_safe_restore(tmp_path) -> None:
    engine = tiny_engine(tmp_path, steps=1)
    engine.train_optimizer_step()
    engine.save_checkpoint()
    archive = tmp_path / "checkpoint.tar"
    package = subprocess.run([
        sys.executable, "scripts/package_latest_checkpoint.py", "--run-dir", str(tmp_path / "run"), "--output", str(archive),
    ], capture_output=True, text=True)
    assert package.returncode == 0, package.stderr
    restored_dir = tmp_path / "restored"
    restore = subprocess.run([
        sys.executable, "scripts/restore_checkpoint_archive.py", "--archive", str(archive), "--sha256", str(archive) + ".sha256", "--run-dir", str(restored_dir),
    ], capture_output=True, text=True)
    assert restore.returncode == 0, restore.stderr
    pointer = json.loads((restored_dir / "checkpoints/latest.json").read_text(encoding="utf-8"))
    assert (restored_dir / "checkpoints" / pointer["checkpoint"] / "COMPLETE").is_file()


def test_progress_inspector_reads_engine_metrics(tmp_path) -> None:
    engine = tiny_engine(tmp_path, steps=1)
    engine.run()
    result = subprocess.run([
        sys.executable, "scripts/inspect_pretraining_progress.py", "--metrics", str(tmp_path / "run/logs/training_metrics.jsonl"),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["current_step"] == 1
    assert summary["latest_validation_loss"] is not None


def test_production_config_has_fixed_budget() -> None:
    from genpy.training.config import load_training_config

    config = load_training_config("configs/pretrain_200m_kaggle.yaml")
    assert config.training.max_steps == 1980
    assert config.training.max_tokens is None
    assert config.training.device == "cuda"
    assert config.training.precision == "bf16"
    assert config.scheduler.warmup_steps == 100


def test_checkpoint6_finalizer_requires_real_gpu_evidence(tmp_path) -> None:
    from finalize_training_reports import verify_gpu_evidence

    evidence = tmp_path / "gpu.json"
    evidence.write_text(json.dumps({"model_parameters": 201560832, "device": "cuda:0", "precision": "bf16", "production_optimizer_steps": 4, "production_checkpoint_written": True, "full_regression_tests": 74}), encoding="utf-8")
    passed, details = verify_gpu_evidence(evidence)
    assert passed and all(details["checks"].values())
