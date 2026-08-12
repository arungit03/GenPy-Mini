"""Tokenizer-independent evaluation metrics for validation text."""

from typing import Iterable, Mapping

from .tokenizer import GenPyTokenizer


CONTENT_TYPE_SAMPLES = {
    "english": "GenPy learns language from clean text.",
    "python": "def add(a, b):\n\treturn a + b",
    "c": "int main(void) { return 0; }",
    "numbers": "2026-08-12 has 3.14159 and 42%.",
    "mathematics": "α + β = γ; ∑ x² = y",
    "urls": "https://example.org/path?q=genpy",
    "tamil": "ஜென்பை ஒரு மொழி மாதிரி.",
    "hindi": "जेनपाई एक भाषा मॉडल है।",
    "emoji": "Language 🌍 works with emoji 🚀.",
    "mixed": "GenPy नमस्ते தமிழ் 中文 🌟",
}


def evaluate_texts(tokenizer: GenPyTokenizer, texts: Iterable[str]) -> dict:
    documents = characters = utf8_bytes = tokens = unknown = 0
    for text in texts:
        ids = tokenizer.encode(text)
        documents += 1
        characters += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        tokens += len(ids)
        unknown += sum(token_id == tokenizer.unk_token_id for token_id in ids)
    return {
        "documents_evaluated": documents,
        "characters_evaluated": characters,
        "utf8_bytes_evaluated": utf8_bytes,
        "tokens_produced": tokens,
        "unknown_tokens": unknown,
        "characters_per_token": characters / tokens if tokens else 0.0,
        "bytes_per_token": utf8_bytes / tokens if tokens else 0.0,
        "tokens_per_character": tokens / characters if characters else 0.0,
        "tokens_per_document": tokens / documents if documents else 0.0,
    }


def evaluate_content_types(tokenizer: GenPyTokenizer) -> Mapping[str, dict]:
    return {name: evaluate_texts(tokenizer, [text]) for name, text in CONTENT_TYPE_SAMPLES.items()}
