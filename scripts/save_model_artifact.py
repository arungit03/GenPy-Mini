"""Save and verify model-only weights from a complete checkpoint."""

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
from genpy.training.artifacts import save_model_artifact, verify_saved_model_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/models/genpy200m_pretrain_v1")
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--data", default="data/tokenized/genpy-32k/TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()
    config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(config.model)
    checkpoint = ROOT / args.checkpoint
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("checkpoint is incomplete")
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False))
    summary = json.loads((ROOT / args.summary).read_text(encoding="utf-8")) if args.summary else {}
    output = save_model_artifact(model, ROOT / args.output, ROOT / args.model_config, ROOT / args.data, summary)
    print(f"Saved model-only artifact: {output}")
    print(json.dumps(verify_saved_model_artifact(model, output.parent), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
