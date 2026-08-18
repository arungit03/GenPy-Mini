import json
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers

from genpy.model import GenPyForCausalLM
from tests.test_model_architecture import tiny_config


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _artifacts(tmp_path):
    config = tiny_config()
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({"model": asdict(config)}), encoding="utf-8")
    model = GenPyForCausalLM(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(config),
            "training_state": {"global_step": 7},
        },
        checkpoint,
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<|pad|>": 0,
                "<|bos|>": 1,
                "<|eos|>": 2,
                "<|unk|>": 3,
                "hello": 4,
            },
            unk_token="<|unk|>",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    validation_path = tmp_path / "validation.bin"
    np.asarray(list(range(21)), dtype=np.uint16).tofile(validation_path)
    (tmp_path / "validation_metadata.json").write_text(
        json.dumps({"dtype": "uint16", "token_count": 21, "vocab_size": 256}),
        encoding="utf-8",
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model_config, checkpoint, tokenizer_path, validation_path, parameter_count


def test_generation_and_evaluation_clis_smoke(tmp_path):
    model_config, checkpoint, tokenizer, validation, parameter_count = _artifacts(tmp_path)
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/generate.py",
            "--model-config",
            str(model_config),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--prompt",
            "hello",
            "--device",
            "cpu",
            "--greedy",
            "--max-new-tokens",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "checkpoint_global_step=7" in generated.stdout

    evaluation = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--model-config",
            str(model_config),
            "--checkpoint",
            str(checkpoint),
            "--validation-data",
            str(validation),
            "--sequence-length",
            "4",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(evaluation.stdout)
    assert result["checkpoint_global_step"] == 7
    assert result["parameters"] == parameter_count
    assert result["evaluation_windows"] == 5
    assert result["evaluated_tokens"] == 20
