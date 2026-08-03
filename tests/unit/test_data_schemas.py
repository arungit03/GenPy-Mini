from __future__ import annotations

from dataclasses import replace

import pytest

from genpy.data.schemas import (
    InstructionRecord,
    Message,
    instruction_content_digest,
    stable_record_id,
)
from tests.unit._helpers import make_record


def test_pretraining_schema_and_stable_record_id() -> None:
    record = make_record("value = 3\nprint(value)\n")
    record.validate()
    assert stable_record_id("a", "b", "c") == stable_record_id("a", "b", "c")
    with pytest.raises(ValueError, match="content_sha256"):
        replace(record, content_sha256="wrong").validate()


def test_instruction_schema_accepts_one_valid_assistant() -> None:
    messages = [
        Message("system", "You are GenPy, a Python code generator."),
        Message("user", "Print a greeting."),
        Message("assistant", 'print("Hello")'),
    ]
    record = InstructionRecord(
        record_id="instruction-fixture-1",
        messages=messages,
        category="basic_io",
        difficulty="beginner",
        tests=[{"stdin": "", "stdout": "Hello\n"}],
        source_id="project-authored-fixture",
        licence_spdx="MIT",
        content_sha256=instruction_content_digest(messages),
        generation_method="human",
        problem_family="greeting-fixture",
    )
    record.validate()


def test_instruction_schema_rejects_unknown_role_and_invalid_python() -> None:
    unknown = Message("tool", "fictional")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown message role"):
        unknown.validate()
    messages = [Message("user", "Write code."), Message("assistant", "if:")]
    record = InstructionRecord(
        record_id="bad",
        messages=messages,
        category="debugging",
        difficulty="beginner",
        source_id="fixture",
        licence_spdx="MIT",
        content_sha256=instruction_content_digest(messages),
        problem_family="bad-fixture",
    )
    with pytest.raises(ValueError, match="valid Python"):
        record.validate()
