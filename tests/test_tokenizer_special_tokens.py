from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import train_from_documents


def test_reserved_ids_are_atomic() -> None:
    backend = train_from_documents(TokenizerConfig(vocab_size=300, min_frequency=1), ["<BOS> hello <EOS>"])
    tokenizer = GenPyTokenizer(backend, expected_vocab_size=300)
    assert tokenizer.encode("<PAD>") == [0]
    assert tokenizer.encode("<BOS>") == [1]
    assert tokenizer.encode("<EOS>") == [2]
    assert tokenizer.encode("<UNK>") == [3]
