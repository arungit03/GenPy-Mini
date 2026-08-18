"""Atomic, resumable checkpoint management for trusted local files."""

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from genpy.training.state import TrainingState


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a trusted local checkpoint artifact."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(self, directory: Path | str, keep_last: int = 3) -> None:
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.latest_pointer = self.directory / "latest.json"

    def save(self, *, model, optimizer, scheduler, precision, state: TrainingState, sampler=None, model_config=None, training_config=None, max_steps=None, data_metadata=None, provenance=None) -> Path:
        filename = f"step_{state.global_step:08d}.pt"
        final_path = self.directory / filename
        temporary = self.directory / f".{filename}.{os.getpid()}.tmp"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "precision": precision.state_dict(),
            "training_state": state.state_dict(),
            "sampler": None if sampler is None else sampler.state_dict(),
            "model_config": None if model_config is None else asdict(model_config),
            "training_config": None if training_config is None else asdict(training_config),
            "max_steps": max_steps,
            "data_metadata": data_metadata,
            "provenance": provenance,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        try:
            torch.save(payload, temporary)
            with temporary.open("ab") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, final_path)
            pointer_temp = self.directory / f".latest.{os.getpid()}.tmp"
            pointer_temp.write_text(json.dumps({"checkpoint": filename}), encoding="utf-8")
            os.replace(pointer_temp, self.latest_pointer)
            self._retain()
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        return final_path

    def initialize_from_checkpoint(self, path: Path | str, *, model, model_config=None) -> dict:
        """Load only model weights for a new continuation phase.

        This deliberately does not touch the optimizer, scheduler, scaler,
        sampler, RNG state, or TrainingState of the new run.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        if checkpoint_path.is_symlink():
            raise ValueError("checkpoint symlinks are not accepted")
        checkpoint_path = checkpoint_path.resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if model_config is not None and payload.get("model_config") != asdict(model_config):
            raise ValueError("checkpoint model configuration is incompatible")
        try:
            model.load_state_dict(payload["model"])
        except RuntimeError as error:
            raise ValueError("checkpoint model weights are incompatible with the target architecture") from error
        source_state = payload.get("training_state") or {}
        return {
            "initialization_mode": "weights_only",
            "source_checkpoint_path": str(checkpoint_path),
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_global_step": int(source_state.get("global_step", 0)),
            "source_tokens_seen": int(source_state.get("tokens_seen", 0)),
            "optimizer_restored": False,
            "scheduler_restored": False,
            "scaler_restored": False,
            "sampler_restored": False,
        }

    def _retain(self) -> None:
        files = sorted(self.directory.glob("step_*.pt"), key=lambda path: path.stat().st_mtime)
        latest = self.latest_path()
        for path in files[:-self.keep_last]:
            if latest is None or path != latest:
                path.unlink(missing_ok=True)

    def latest_path(self) -> Path | None:
        if not self.latest_pointer.is_file():
            return None
        data = json.loads(self.latest_pointer.read_text(encoding="utf-8"))
        path = self.directory / data["checkpoint"]
        return path if path.is_file() else None

    def load(self, path: Path | str | None = None, *, model, optimizer, scheduler, precision, state: TrainingState, sampler=None, model_config=None, training_config=None, expected_max_steps=None) -> dict:
        checkpoint_path = self.latest_path() if path is None else Path(path)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError("No usable checkpoint was found")
        # Checkpoints are trusted local artifacts; never accept remote paths.
        if checkpoint_path.is_symlink():
            raise ValueError("checkpoint symlinks are not accepted")
        checkpoint_path = checkpoint_path.resolve()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if expected_max_steps is not None and payload.get("max_steps") != expected_max_steps:
            raise ValueError(
                "checkpoint max_steps is incompatible with the requested full training horizon"
            )
        if model_config is not None and payload.get("model_config") != asdict(model_config):
            raise ValueError("checkpoint model configuration is incompatible")
        if training_config is not None:
            saved = payload.get("training_config") or {}
            for key in ("sequence_length", "micro_batch_size", "gradient_accumulation_steps"):
                if saved.get(key) != getattr(training_config, key):
                    raise ValueError(f"checkpoint training configuration is incompatible for {key}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        precision.load_state_dict(payload["precision"])
        restored = TrainingState.from_state_dict(payload["training_state"])
        state.__dict__.update(restored.__dict__)
        if sampler is not None and payload.get("sampler") is not None:
            sampler.load_state_dict(payload["sampler"])
        rng = payload.get("rng", {})
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        return payload
