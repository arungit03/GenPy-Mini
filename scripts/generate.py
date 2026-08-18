"""Generate text from a GenPy checkpoint."""

import argparse
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from genpy.config import load_model_config
from genpy.inference import GenerationConfig, generate, load_checkpoint_weights
from genpy.model import GenPyForCausalLM
from genpy.tokenizer import GenPyTokenizer


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--allow-special-tokens",
        action="store_true",
        help="allow PAD, BOS, and UNK to be sampled",
    )
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    tokenizer = GenPyTokenizer.from_file(args.tokenizer)
    device = _device(args.device)
    model = GenPyForCausalLM(model_config)
    payload = load_checkpoint_weights(model, args.checkpoint, model_config)
    model.to(device)
    if args.precision != "fp32":
        model.to(dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.precision])
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("prompt must produce at least one token")
    forbidden = () if args.allow_special_tokens else (
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        forbidden_token_ids=forbidden,
        greedy=args.greedy,
        seed=args.seed,
        precision=args.precision,
    )
    output_ids = generate(model, torch.tensor([prompt_ids], dtype=torch.long, device=device), config)
    print(tokenizer.decode(output_ids[0].tolist()))
    state = payload.get("training_state") or {}
    print(f"checkpoint_global_step={int(state.get('global_step', 0))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
