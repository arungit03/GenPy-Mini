from __future__ import annotations

import pytest

from genpy.tokenizer.serialization import (
    SYSTEM_MESSAGE,
    SerializationError,
    serialize_instruction,
    serialize_pretraining,
)


def test_canonical_pretraining_serialization_preserves_python() -> None:
    code = "if True:\n\tprint(\"ok\")\n"
    result = serialize_pretraining(code)
    assert result.text == f"<|bos|><|code|>\n{code}<|end|><|eos|>\n"
    assert result.structural_tokens == 4


def test_canonical_instruction_omits_fixed_system_message() -> None:
    result = serialize_instruction("Print hello.", 'print("hello")\n', SYSTEM_MESSAGE)
    assert SYSTEM_MESSAGE not in result.text
    assert "```" not in result.text
    assert result.structural_tokens == 6


@pytest.mark.parametrize(("prompt", "code"), [("", "x = 1\n"), ("prompt", "")])
def test_empty_instruction_content_is_rejected(prompt: str, code: str) -> None:
    with pytest.raises(SerializationError):
        serialize_instruction(prompt, code)
