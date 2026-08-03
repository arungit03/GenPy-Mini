from __future__ import annotations

import pytest

from genpy.data.normalize import NormalizationError, normalize_python_bytes


def test_normalize_newlines_and_preserve_indentation() -> None:
    value = normalize_python_bytes(
        b"if True:\r\n    print('yes')  \r\n",
        minimum_bytes=1,
        maximum_bytes=1000,
    )
    assert value == "if True:\n    print('yes')\n"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [(b"value = '\xff'", "invalid_utf8"), (b"value = 1\0\n", "null_byte")],
)
def test_normalize_rejects_unsafe_bytes(payload: bytes, reason: str) -> None:
    with pytest.raises(NormalizationError, match=reason):
        normalize_python_bytes(payload, minimum_bytes=1, maximum_bytes=1000)
