from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.training.packing import (
    PackingError,
    _pack_records,
    load_packing_config,
    prepare_packed_data,
    tokenize_record,
)


def test_long_record_continuation_and_no_target_duplication(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(phase4_fixture["tokenizer_artifact"])
    record = tokenize_record(
        tokenizer,
        "pretraining",
        {
            "code": "\n".join(f"value_{index} = {index}" for index in range(50)),
            "_input_shard": "safe",
            "_input_checksum": "0" * 64,
        },
        "full_lm",
    )
    tokens, masks, stats, _ = _pack_records(iter([record]), 16, 0, True)
    assert len(tokens) > 1
    for left, right in zip(tokens, tokens[1:], strict=False):
        assert left[-1] == right[0]
    active = sum(int(mask.sum()) for mask in masks)
    assert active == stats["active_loss_targets"]
    assert active == len(record.token_ids) - 1
    assert all(array.dtype == np.dtype("<u2") for array in tokens)


def test_assistant_only_mask_starts_at_assistant_boundary(phase4_fixture) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(phase4_fixture["tokenizer_artifact"])
    record = tokenize_record(
        tokenizer,
        "instruction",
        {
            "prompt": "Add two values.",
            "code": "def add(a, b):\n    return a + b\n",
            "_input_shard": "safe",
            "_input_checksum": "0" * 64,
        },
        "assistant_only",
    )
    assistant = record.token_ids.index(tokenizer.special_token_ids["assistant"])
    assert not any(record.target_active[:assistant])
    assert all(record.target_active[assistant:])


def test_production_packing_is_blocked_without_frozen_tokenizer() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_packing_config(root / "configs/data/packing.yaml")
    with pytest.raises(PackingError, match="frozen production tokenizer"):
        prepare_packed_data(config, dry_run=True)
