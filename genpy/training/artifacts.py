"""Model-only artifact writing and reload verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from .checkpoint import _hash_file


def verify_model(model, expected_parameters: int = 201560832) -> dict:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    tied = model.lm_head.weight is model.token_embedding.weight
    return {"forward_ready": True, "finite_parameters": finite, "parameter_count": parameters, "parameter_count_pass": parameters == expected_parameters, "weight_tying": tied}


def save_model_artifact(model, output_dir: str | Path, model_config: str | Path, tokenizer_manifest: str | Path, training_summary: dict, expected_parameters: int = 201560832) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_model(model, expected_parameters=expected_parameters)
    if not verification["finite_parameters"] or not verification["parameter_count_pass"] or not verification["weight_tying"]:
        raise ValueError(f"model verification failed: {verification}")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, output_dir / "model.pt")
    shutil.copy2(model_config, output_dir / "model_config.yaml")
    shutil.copy2(tokenizer_manifest, output_dir / "tokenizer_manifest.json")
    (output_dir / "training_summary.json").write_text(json.dumps({**training_summary, "verification": verification}, indent=2) + "\n", encoding="utf-8")
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.name != "SHA256SUMS.txt" and path.is_file():
            lines.append(f"{_hash_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir / "model.pt"


def verify_saved_model_artifact(model, artifact_dir: str | Path, expected_parameters: int = 201560832) -> dict:
    artifact_dir = Path(artifact_dir)
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False))
    result = verify_model(model, expected_parameters=expected_parameters)
    result["save_reload_pass"] = result["finite_parameters"] and result["parameter_count_pass"] and result["weight_tying"]
    return result
