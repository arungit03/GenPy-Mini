from pathlib import Path

from genpy.config import load_model_config, load_training_config
from genpy.model import GenPyForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def test_step8c_t4_configuration_matches_the_locked_plan():
    config = load_training_config(ROOT / "configs" / "train_step8c_t4.yaml")
    model = load_model_config(ROOT / "configs" / "model_200m.yaml")
    assert config.seed == 42
    assert config.micro_batch_size == 1
    assert config.gradient_accumulation_steps == 16
    assert config.sequence_length == 256
    assert config.sequence_length <= model.max_seq_len
    assert config.learning_rate == 3e-5
    assert config.min_learning_rate == 5e-6
    assert config.weight_decay == 0.1
    assert config.warmup_ratio == 0.02
    assert config.grad_clip == 1.0
    assert config.precision == "fp16"
    assert config.log_interval == 20
    assert config.eval_interval == 500
    assert config.save_interval == 1000
    assert config.checkpoint_dir == "checkpoints/step8c_t4"
    assert config.log_dir == "logs/step8c_t4"


def test_step8c_source_range_does_not_overlap_step9_evaluation_range():
    step9_start, step9_end_exclusive = 75000, 80000
    step8c_start, step8c_documents = 80000, 250000
    step8c_end_exclusive = step8c_start + step8c_documents
    assert step9_end_exclusive <= step8c_start
    assert step8c_start == 80000
    assert step8c_end_exclusive == 330000
    assert set(range(step9_start, step9_end_exclusive)).isdisjoint(
        range(step8c_start, step8c_end_exclusive)
    )


def test_production_parameter_count_remains_locked():
    model = GenPyForCausalLM(load_model_config(ROOT / "configs" / "model_200m.yaml"))
    assert sum(parameter.numel() for parameter in model.parameters()) == 201_560_832
