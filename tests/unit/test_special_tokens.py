from __future__ import annotations

import pytest

from genpy.tokenizer.config import SPECIAL_TOKEN_TEXT
from genpy.tokenizer.serialization import SerializationError, serialize_pretraining
from genpy.tokenizer.tokenizer import GenPyTokenizer


@pytest.mark.parametrize("token", SPECIAL_TOKEN_TEXT)
def test_every_reserved_token_collision_is_rejected(token: str) -> None:
    with pytest.raises(SerializationError, match="reserved"):
        serialize_pretraining(f"value = {token!r}\n")


def test_special_tokens_are_atomic_and_stable(trained_tokenizer_artifact) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(trained_tokenizer_artifact)
    for expected_id, token in enumerate(SPECIAL_TOKEN_TEXT):
        assert tokenizer.token_to_id(token) == expected_id
        assert tokenizer._tokenizer.encode(token).ids == [expected_id]
