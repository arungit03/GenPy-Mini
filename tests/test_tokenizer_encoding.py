from pathlib import Path

from genpy.config import load_tokenizer_config
from genpy.tokenizer.corpus import CorpusReader
from genpy.tokenizer.trainer import train_from_iterator


ROOT = Path(__file__).resolve().parents[1]


def tokenizer():
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    reader = CorpusReader(ROOT / "tests" / "fixtures", "tokenizer_corpus.jsonl.gz")
    return train_from_iterator(config.tokenizer, reader.texts(max_documents=150), vocab_size=512)


def test_unicode_whitespace_and_determinism():
    tok = tokenizer()
    samples = ["English words", "café தமிழ் हिन्दी 中文 🚀", " leading  spaces\n\tcode", "def f(x):\n    return x"]
    for text in samples:
        ids = tok.encode(text)
        assert ids == tok.encode(text)
        assert tok.decode(ids) == text
        assert tok.unk_token_id not in ids


def test_explicit_boundaries_only():
    tok = tokenizer()
    plain = tok.encode("hello")
    assert tok.bos_token_id not in plain and tok.eos_token_id not in plain
    assert tok.encode("hello", add_bos=True)[0] == tok.bos_token_id
    assert tok.encode("hello", add_eos=True)[-1] == tok.eos_token_id
    assert tok.encode_document("hello")[-1] == tok.eos_token_id
    assert not tok.decode(tok.encode("Hello")).startswith(" ")
