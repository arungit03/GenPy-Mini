"""Shared local verification CLI helpers."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.config import load_config


def tiny_config():
    config = load_config(ROOT / "configs/model_200m.yaml")
    return replace(config.model, vocab_size=128, max_seq_len=32, n_layers=2, d_model=64, n_heads=4, head_dim=16, ffn_hidden_size=128)
