from dataclasses import replace

import numpy as np
import torch

from genpy.config import load_config
from genpy.model import GenPyForCausalLM
from genpy.training.config import load_training_config
from genpy.training.sft_dataset import SFTMemmapDataset
from genpy.training.sft_trainer import SFTTrainingEngine


def make_engine(tmp_path, steps=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_config = replace(load_config("configs/model_200m.yaml").model, vocab_size=64, max_seq_len=16, n_layers=1, d_model=16, n_heads=2, head_dim=8, ffn_hidden_size=24)
    input_values = np.tile(np.asarray([1, 5, 6, 7, 2], dtype=np.uint16), 8)
    label_values = np.tile(np.asarray([-100, -100, 6, 7, 2], dtype=np.int32), 8)
    input_values.tofile(tmp_path / "inputs.bin"); label_values.tofile(tmp_path / "labels.bin"); np.save(tmp_path / "offsets.npy", np.arange(0, 41, 5, dtype=np.int64))
    dataset = SFTMemmapDataset(tmp_path / "inputs.bin", tmp_path / "labels.bin", tmp_path / "offsets.npy", 4)
    config = load_training_config("configs/sft_200m_kaggle.yaml")
    config = replace(config, training=replace(config.training, device="cpu", precision="fp32", sequence_length=4, max_steps=steps, gradient_accumulation_steps=1), validation=replace(config.validation, batches=1), checkpoint=replace(config.checkpoint, interval_steps=100))
    torch.manual_seed(123)
    model = GenPyForCausalLM(model_config, attention_backend="eager")
    return SFTTrainingEngine(model, dataset, dataset, config, tmp_path / "run", "base-hash", "manifest-hash")


def test_sft_optimizer_starts_fresh_and_session_status(tmp_path) -> None:
    engine = make_engine(tmp_path, 2)
    assert not engine.optimizer.state
    state = engine.run(session_steps=1)
    assert state.global_step == 1 and engine.last_run_status == "SESSION_COMPLETE"
    assert engine.scheduler.total_steps == 2 and state.tokens_seen > 0
    engine.train_dataset.close()


def test_sft_resume_restores_global_step_and_scheduler_budget(tmp_path) -> None:
    engine = make_engine(tmp_path, 2)
    engine.run(session_steps=1)
    resumed = make_engine(tmp_path / "resume", 2)
    resumed.resume(tmp_path / "run" / "checkpoints" / "step_000000000001")
    state = resumed.run(session_steps=1)
    assert state.global_step == 2 and resumed.scheduler.total_steps == 2
    engine.train_dataset.close(); resumed.train_dataset.close()
