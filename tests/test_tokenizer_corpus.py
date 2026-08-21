from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.corpus import format_document


def test_corpus_fields_and_boundaries() -> None:
    record = {"instruction": "Write code", "input": "n = 2", "response": "print(n)", "id": "secret", "family_id": "hidden"}
    text = format_document(record, TokenizerConfig())
    assert text == "<BOS>\n### User\nWrite code\n\nn = 2\n\n### Assistant\nprint(n)\n<EOS>"
    assert "secret" not in text and "family_id" not in text
    assert text.startswith("<BOS>") and text.endswith("<EOS>")


def test_empty_input_is_omitted() -> None:
    text = format_document({"instruction": "Explain", "input": "", "response": "return None"}, TokenizerConfig())
    assert "### User\nExplain\n\n### Assistant" in text
