from pathlib import Path

from genpy.config import load_tokenizer_config
from genpy.tokenizer.corpus import CorpusReader
from genpy.tokenizer.trainer import train_from_iterator


ROOT = Path(__file__).resolve().parents[1]


def test_small_tokenizer_training_save_reload(tmp_path):
    config = load_tokenizer_config(ROOT / "configs" / "tokenizer.yaml")
    reader = CorpusReader(ROOT / "tests" / "fixtures", "tokenizer_corpus.jsonl.gz")
    tokenizer = train_from_iterator(config.tokenizer, reader.texts(max_documents=100), vocab_size=512)
    assert tokenizer.vocab_size == 512
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id) == (0, 1, 2, 3)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    reloaded = tokenizer.from_file(path)
    assert reloaded.encode("café தமிழ் 🚀") == tokenizer.encode("café தமிழ் 🚀")
