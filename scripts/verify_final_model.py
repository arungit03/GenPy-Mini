"""Inference-only final model verification; performs no optimizer update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.training.artifacts import verify_saved_model_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="artifacts/models/genpy200m_pretrain_v1")
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(config.model)
    artifact = ROOT / args.artifact
    checks = verify_saved_model_artifact(model, artifact)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros((1, config.model.max_seq_len), dtype=torch.long))
    checks.update({"forward_pass": True, "finite_logits": bool(torch.isfinite(logits).all()), "context_length": logits.shape[1] == 1024})
    checks["pass"] = all(checks.values())
    print(json.dumps(checks, indent=2))
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
