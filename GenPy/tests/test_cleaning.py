from genpy.config import ProcessingConfig
from genpy.data.cleaning import assess_quality, normalize_text


CONFIG = ProcessingConfig(42, 10, 1000, True, True, True, True, True)


def test_normalization_preserves_unicode_and_paragraphs():
    text = "  café\r\n\r\nA  β  line.\x00\n\n\nCode\tkept  "
    result = normalize_text(text, CONFIG)
    assert result == "café\n\nA β line.\n\nCode\tkept"


def test_quality_rejection_reasons():
    assert assess_quality(None, 3, 10).reason == "missing_text"
    assert assess_quality(4, 3, 10).reason == "invalid_text_type"
    assert assess_quality("", 3, 10).reason == "empty_text"
    assert assess_quality("ab", 3, 10).reason == "too_short"
    assert assess_quality("abcdefghijk", 3, 10).reason == "too_long"
    assert assess_quality("good", 3, 10).accepted
