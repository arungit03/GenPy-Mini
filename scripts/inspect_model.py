"""Inspect the canonical GenPy model architecture and parameter storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.model.utils import count_parameters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_200m.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    model = GenPyForCausalLM(config.model)
    parameters = count_parameters(model)
    print("GenPy-200M\n===========")
    print(f"\nVocabulary:      {config.model.vocab_size:,}\nContext:         {config.model.max_seq_len:,}\nLayers:          {config.model.n_layers}\nHidden:          {config.model.d_model}\nHeads:           {config.model.n_heads}\nHead dim:        {config.model.head_dim}\nFFN:             {config.model.ffn_hidden_size}")
    print("\nNormalization:   RMSNorm\nPosition:        RoPE\nActivation:      SwiGLU\nCausal:          Yes\nBias-free:       Yes\nWeight tied:     Yes")
    print(f"\nParameters:\n{parameters:,}")
    print(f"Parameter storage estimate only: FP32 {parameters * 4 / 2**20:.1f} MiB; FP16/BF16 {parameters * 2 / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
