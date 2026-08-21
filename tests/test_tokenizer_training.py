from pathlib import Path

from tokenizers import Tokenizer

from genpy.tokenizer.config import TokenizerConfig
from genpy.tokenizer.trainer import build_fresh_tokenizer, build_trainer, save_model_files, train_from_documents


def test_fresh_bpe_trains_and_reloads(tmp_path: Path) -> None:
    config = TokenizerConfig(vocab_size=300, min_frequency=1)
    documents = ["<BOS>\ndef add(a, b):\n    return a + b\n<EOS>", "<BOS>\nprint('café')\n<EOS>"]
    tokenizer = train_from_documents(config, documents)
    assert 260 <= tokenizer.get_vocab_size() <= 300
    assert tokenizer.token_to_id("<PAD>") == 0
    save_model_files(tokenizer, tmp_path)
    reloaded = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert reloaded.encode("def add(a, b):").ids == tokenizer.encode("def add(a, b):").ids


def test_training_starts_from_empty_bpe_model() -> None:
    tokenizer = build_fresh_tokenizer(TokenizerConfig(vocab_size=300, min_frequency=1))
    assert tokenizer.get_vocab_size() == 0
    assert tokenizer.token_to_id("<UNK>") is None
    assert build_trainer(TokenizerConfig(vocab_size=300, min_frequency=1)) is not None
