from __future__ import annotations

import json
from pathlib import Path

from genpy.tokenizer.tokenizer import GenPyTokenizer


def test_roundtrip_indentation_quotes_unicode_and_emoji(trained_tokenizer_artifact) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(trained_tokenizer_artifact)
    samples = json.loads(
        Path("tests/fixtures/tokenizer/representative_samples.json").read_text(encoding="utf-8")
    )
    samples.extend(
        [
            "if ready:\n\tprint('tabs')\n\n",
            'text = """one\ntwo"""\n# comment\n',
            "path = 'C:/safe/example.py'\n",
        ]
    )
    for sample in samples:
        assert tokenizer.decode(tokenizer.encode_text(sample)) == sample


def test_save_and_load_encoding_equivalence(trained_tokenizer_artifact, tmp_path) -> None:  # type: ignore[no-untyped-def]
    tokenizer = GenPyTokenizer.load(trained_tokenizer_artifact)
    path = tmp_path / "copy.json"
    tokenizer.save(path)
    sample = "def identity(value):\n    return value\n"
    reloaded_ids = tokenizer._tokenizer.from_file(str(path)).encode(sample).ids
    assert reloaded_ids == tokenizer.encode_text(sample)
