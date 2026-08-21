"""Export a model-only GenPy-Instruct artifact; never includes optimizer state."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.data.io import sha256_file
from genpy.model import GenPyForCausalLM
from genpy.training.artifacts import save_model_artifact, verify_saved_model_artifact
from genpy.training.checkpoint import _hash_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/genpy-200m-instruct-v1")
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--sft-manifest", default="data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json")
    parser.add_argument("--base-model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--evaluation", default="reports/checkpoint_8_final_eval.json")
    args = parser.parse_args()
    config_path = ROOT / args.model_config
    config = load_config(config_path)
    checkpoint = ROOT / args.checkpoint
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.pt"
    model = GenPyForCausalLM(config.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    output = ROOT / args.output
    summary = {"artifact_name": "GenPy-200M-Instruct-v1", "base_model_sha256": sha256_file(ROOT / args.base_model) if (ROOT / args.base_model).is_file() else None, "sft_manifest_sha256": sha256_file(ROOT / args.sft_manifest) if (ROOT / args.sft_manifest).is_file() else None, "evaluation_report": args.evaluation, "provenance": {"pretrained_weights_used": False, "initialized_from_checkpoint_7": True}}
    save_model_artifact(model, output, config_path, ROOT / args.tokenizer / "TOKENIZER_MANIFEST.json", summary)
    tokenizer_dir = ROOT / args.tokenizer
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy2(tokenizer_dir / name, output / name)
    if (ROOT / args.evaluation).is_file():
        shutil.copy2(ROOT / args.evaluation, output / "evaluation_report.json")
    if (ROOT / args.sft_manifest).is_file():
        shutil.copy2(ROOT / args.sft_manifest, output / "sft_manifest.json")
    (output / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{_hash_file(path)}  {path.name}" for path in sorted(output.iterdir()) if path.name != "SHA256SUMS.txt" and path.is_file()) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "model_sha256": sha256_file(output / "model.pt"), "verification": verify_saved_model_artifact(model, output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
