from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import train_from_documents
from genpy.tokenizer.validation import contract_checks, measure_texts


def test_validation_metrics_and_contract() -> None:
    backend = train_from_documents(TokenizerConfig(vocab_size=300, min_frequency=1), ["<BOS> hello <EOS>"])
    tokenizer = GenPyTokenizer(backend, expected_vocab_size=300)
    assert contract_checks(tokenizer) == {"vocabulary_size": False, "special_token_ids": True}
    metrics = measure_texts(tokenizer, ["hello", "café"])
    assert metrics.examples == 2
    assert metrics.unknown_tokens == 0
