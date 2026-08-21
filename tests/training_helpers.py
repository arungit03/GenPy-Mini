from dataclasses import replace
from pathlib import Path

import numpy as np

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.training.config import load_training_config
from genpy.training.dataset import MemmapTokenDataset
from genpy.training.trainer import TrainingEngine


def tiny_engine(tmp_path: Path, steps: int = 2, accumulation: int = 1) -> TrainingEngine:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    train_path, validation_path = tmp_path / "train.bin", tmp_path / "validation.bin"
    (np.arange(256, dtype=np.uint16) % 64).tofile(train_path)
    (np.arange(128, dtype=np.uint16) % 64).tofile(validation_path)
    config = load_training_config("configs/training_smoke.yaml")
    config = replace(config, training=replace(config.training, max_steps=steps, sequence_length=16, gradient_accumulation_steps=accumulation), validation=replace(config.validation, batches=1))
    return TrainingEngine(model=GenPyForCausalLM(model_config, attention_backend="eager"), train_dataset=MemmapTokenDataset(train_path, 16), validation_dataset=MemmapTokenDataset(validation_path, 16), config=config, run_dir=tmp_path / "run")
