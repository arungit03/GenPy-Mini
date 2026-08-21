"""Atomic, compatibility-checked training checkpoints and RNG state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from .state import TrainingState


def capture_rng_state(batcher=None) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if batcher is not None:
        state["batcher"] = batcher.state_dict()
    return state


def restore_rng_state(state: dict, batcher=None) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if batcher is not None and state.get("batcher") is not None:
        batcher.load_state_dict(state["batcher"])


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointManager:
    def __init__(self, root: str | Path, keep_last: int = 3) -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True); self.keep_last = keep_last

    def valid_checkpoints(self) -> list[Path]:
        return sorted((path for path in self.root.glob("step_*") if (path / "COMPLETE").is_file()), key=lambda p: p.name)

    def latest(self) -> Path | None:
        pointer = self.root / "latest.json"
        if pointer.is_file():
            try:
                candidate = self.root / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
                if (candidate / "COMPLETE").is_file():
                    return candidate
            except (KeyError, json.JSONDecodeError, OSError):
                pass
        valid = self.valid_checkpoints()
        return valid[-1] if valid else None

    def save(self, model, optimizer, scheduler, precision_manager, state: TrainingState, batcher, metadata: dict) -> Path:
        name = f"step_{state.global_step:012d}"
        temporary = self.root / f".{name}.tmp-{uuid.uuid4().hex}"
        final = self.root / name
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            torch.save(model.state_dict(), temporary / "model.pt")
            if optimizer is not None: torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            if scheduler is not None: torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            if precision_manager is not None: torch.save(precision_manager.state_dict(), temporary / "scaler.pt")
            (temporary / "trainer_state.json").write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
            (temporary / "rng_state.pt").write_bytes(b"")
            torch.save(capture_rng_state(batcher), temporary / "rng_state.pt")
            metadata = {"schema_version": 1, "global_step": state.global_step, "tokens_seen": state.tokens_seen, "created_at": time.time(), **metadata}
            (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            essential = ["model.pt", "trainer_state.json", "metadata.json", "rng_state.pt"]
            if not all((temporary / filename).is_file() for filename in essential):
                raise RuntimeError("checkpoint essential-file verification failed")
            (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
            if final.exists(): shutil.rmtree(final)
            os.replace(temporary, final)
            pointer_tmp = self.root / "latest.json.tmp"
            pointer_tmp.write_text(json.dumps({"checkpoint": name}, indent=2) + "\n", encoding="utf-8")
            os.replace(pointer_tmp, self.root / "latest.json")
            self._rotate()
            return final
        finally:
            if temporary.exists(): shutil.rmtree(temporary, ignore_errors=True)

    def _rotate(self) -> None:
        valid = self.valid_checkpoints()
        for path in valid[:-self.keep_last]:
            shutil.rmtree(path)

    def load(self, path: str | Path, model, optimizer=None, scheduler=None, precision_manager=None, state: TrainingState | None = None, batcher=None, expected_metadata: dict | None = None) -> TrainingState:
        path = Path(path)
        if not (path / "COMPLETE").is_file(): raise ValueError(f"invalid incomplete checkpoint: {path}")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for key, value in (expected_metadata or {}).items():
            if metadata.get(key) != value: raise ValueError(f"checkpoint compatibility mismatch for {key}")
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=False))
        if optimizer is not None and (path / "optimizer.pt").is_file():
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
            # Load safely on CPU, then move tensor state to the active model
            # device before the next optimizer update (notably for CUDA).
            parameter_device = next(model.parameters()).device
            for optimizer_state in optimizer.state.values():
                for key, value in list(optimizer_state.items()):
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(parameter_device)
        if scheduler is not None and (path / "scheduler.pt").is_file(): scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
        if precision_manager is not None and (path / "scaler.pt").is_file(): precision_manager.load_state_dict(torch.load(path / "scaler.pt", map_location="cpu", weights_only=False))
        restored = TrainingState.from_dict(json.loads((path / "trainer_state.json").read_text(encoding="utf-8")))
        restore_rng_state(torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False), batcher)
        return restored
