from pathlib import Path

from genpy.config import load_tokenizer_config
from genpy.tokenizer.corpus import CorpusReader
from genpy.tokenizer.evaluation import evaluate_content_types, evaluate_texts
from genpy.tokenizer.trainer import train_from_iterator


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_metrics_are_mathematically_consistent():
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    reader = CorpusReader(ROOT / "tests" / "fixtures", "tokenizer_corpus.jsonl.gz")
    tokenizer = train_from_iterator(config.tokenizer, reader.texts(max_documents=100), vocab_size=512)
    texts = ["abc", "café"]
    metrics = evaluate_texts(tokenizer, texts)
    assert metrics["documents_evaluated"] == 2
    assert metrics["characters_evaluated"] == sum(len(text) for text in texts)
    assert metrics["utf8_bytes_evaluated"] == sum(len(text.encode("utf-8")) for text in texts)
    assert metrics["characters_per_token"] == metrics["characters_evaluated"] / metrics["tokens_produced"]
    assert metrics["unknown_tokens"] == 0
    assert len(evaluate_content_types(tokenizer)) == 10
