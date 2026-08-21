"""Fresh Byte-Level BPE training using the local GenPy corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .config import SPECIAL_TOKENS, TokenizerConfig


def build_trainer(config: TokenizerConfig) -> BpeTrainer:
    return BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[
            AddedToken(SPECIAL_TOKENS[name], special=True, normalized=False)
            for name in ("pad", "bos", "eos", "unk")
        ],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )


def build_fresh_tokenizer(config: TokenizerConfig) -> Tokenizer:
    config.validate()
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk"]))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=config.add_prefix_space,
        trim_offsets=False,
        use_regex=True,
    )
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def train_from_documents(config: TokenizerConfig, documents: Iterable[str]) -> Tokenizer:
    tokenizer = build_fresh_tokenizer(config)
    trainer = build_trainer(config)
    tokenizer.train_from_iterator(documents, trainer=trainer)
    expected = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for token, token_id in expected.items():
        if tokenizer.token_to_id(token) != token_id:
            raise RuntimeError(f"special token ID mismatch for {token}: {tokenizer.token_to_id(token)}")
    return tokenizer


def pad_vocab_with_corpus_tokens(tokenizer: Tokenizer, documents: Iterable[str], target_size: int) -> int:
    """Complete a small-corpus BPE with deterministic tokens observed in that corpus.

    The generated corpus has fewer than 32K distinct BPE merge opportunities. This
    keeps the core BPE learned from the corpus and fills the exact model vocabulary
    with additional observed substrings, rather than fabricated or external tokens.
    """
    needed = target_size - tokenizer.get_vocab_size(with_added_tokens=True)
    if needed <= 0:
        return 0
    candidates: list[str] = []
    seen: set[str] = set(tokenizer.get_vocab())
    for document in documents:
        for length in (8, 7, 6, 5, 4, 3):
            for start in range(0, max(0, len(document) - length + 1)):
                candidate = document[start:start + length]
                if (candidate in seen or len(candidate) < 3 or not candidate.isprintable()
                        or candidate.strip() != candidate or "<" in candidate or ">" in candidate):
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                if len(candidates) >= needed:
                    break
            if len(candidates) >= needed:
                break
        if len(candidates) >= needed:
            break
    added = tokenizer.add_tokens(candidates[:needed])
    if tokenizer.get_vocab_size(with_added_tokens=True) != target_size:
        raise RuntimeError(
            f"corpus-derived vocabulary completion failed: "
            f"{tokenizer.get_vocab_size(with_added_tokens=True)} / {target_size}"
        )
    return added


def save_model_files(tokenizer: Tokenizer, output: str | Path) -> None:
    """Save the canonical tokenizer JSON plus model vocab and merge files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output / "tokenizer.json"), pretty=True)
    saved = tokenizer.model.save(str(output), "genpy")
    for path in saved:
        path = Path(path)
        if path.name.endswith("-merges.txt"):
            path.replace(output / "merges.txt")
        else:
            path.unlink()
    vocab = {token: token_id for token, token_id in tokenizer.get_vocab().items()}
    (output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
