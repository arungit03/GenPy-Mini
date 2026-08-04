from __future__ import annotations

import json

import pytest
import torch

from genpy.model.compatibility import (
    CompatibilityError,
    load_project_state,
    save_project_state,
)
from genpy.model.config import load_model_config
from genpy.model.transformer import build_model


def test_state_save_load_equivalence_and_weight_tying(tmp_path, phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    source = build_model(config).eval()
    inputs = torch.arange(8).unsqueeze(0) % config.vocab_size
    expected = source(inputs).logits.detach()
    save_project_state(source, tmp_path)
    loaded = build_model(config).eval()
    load_project_state(loaded, tmp_path)
    torch.testing.assert_close(loaded(inputs).logits, expected)
    assert loaded.token_embedding.weight is loaded.lm_head.weight


def test_tokenizer_fingerprint_mismatch_is_rejected(tmp_path, phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    config = load_model_config(phase4_fixture["model_config"], phase4_fixture["root"])
    model = build_model(config)
    save_project_state(model, tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tokenizer_fingerprint"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CompatibilityError, match="tokenizer_fingerprint"):
        load_project_state(build_model(config), tmp_path)
