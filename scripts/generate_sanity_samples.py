"""Inference-only qualitative generation sanity check for a trained artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.tokenizer import GenPyTokenizer


PROMPTS = [
    "Write Python code to check whether a number is even or odd.",
    "Write a Python function to reverse a string.",
    "Write Python code to find the largest number in a list.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--output", default="reports/checkpoint_7_generation_samples.txt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(config.model)
    model.load_state_dict(torch.load(ROOT / args.model, map_location="cpu", weights_only=False))
    model.eval()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    lines = ["GenPy Checkpoint 7 qualitative generation sanity samples", "Inference-only; not a benchmark evaluation.", ""]
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([ids], dtype=torch.long)
            for _ in range(args.max_new_tokens):
                logits = model(input_ids[:, -config.model.max_seq_len:])[:, -1, :]
                next_id = int(torch.argmax(logits, dim=-1).item())
                input_ids = torch.cat((input_ids, torch.tensor([[next_id]], dtype=torch.long)), dim=1)
                if next_id == tokenizer.eos_token_id:
                    break
            lines.extend([f"PROMPT: {prompt}", tokenizer.decode(input_ids[0].tolist()), ""])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
