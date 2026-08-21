"""Generate the fixed qualitative SFT sanity set using greedy decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from evaluate_coding import SANITY_PROMPTS, generate, prompt_text
from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.tokenizer import GenPyTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="reports/checkpoint_8_sft_samples.txt")
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(config.model)
    model.load_state_dict(torch.load(ROOT / args.model, map_location="cpu", weights_only=False))
    model.eval()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    lines = ["GenPy Checkpoint 8 SFT samples", "Greedy decoding; qualitative only.", ""]
    for prompt in SANITY_PROMPTS:
        generated, eos, tokens = generate(model, tokenizer, prompt_text({"instruction": prompt}), args.max_new_tokens, torch.device("cpu"))
        lines.extend([f"PROMPT: {prompt}", f"EOS: {eos} | TOKENS: {tokens}", generated, ""])
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
