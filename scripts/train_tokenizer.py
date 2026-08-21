"""Train and audit the production GenPy Byte-Level BPE tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tokenizers

from genpy.data.io import iter_records, sha256_file
from genpy.tokenizer.config import SPECIAL_IDS, SPECIAL_TOKENS, TokenizerConfig, config_dict, load_tokenizer_config
from genpy.tokenizer.corpus import collect_stats, iter_documents
from genpy.tokenizer.trainer import build_fresh_tokenizer, build_trainer, pad_vocab_with_corpus_tokens, save_model_files

EXPECTED_TRAIN_SHA256 = "17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44"


def digest_files(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def combined_digest(digests: dict[str, str]) -> str:
    payload = "".join(f"{name}:{digests[name]}\n" for name in sorted(digests)).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_auxiliary_files(output: Path, config: TokenizerConfig, config_path: Path, dataset: Path, stats: dict) -> None:
    write_json(output / "tokenizer_config.json", {
        **config_dict(config),
        "name": config.name,
        "version": "GenPy-Tokenizer-32K-v1" if config.vocab_size == 32000 else "GenPy-Tokenizer-smoke-v1",
    })
    write_json(output / "special_tokens_map.json", {
        "pad_token": "<PAD>", "bos_token": "<BOS>", "eos_token": "<EOS>", "unk_token": "<UNK>",
        "special_token_ids": SPECIAL_IDS,
    })
    files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"]
    hashes = digest_files(output, files)
    corpus_path = ROOT / "data/interim/tokenizer/train_corpus.jsonl"
    corpus_files = [str(dataset.relative_to(ROOT)).replace("\\", "/")]
    corpus_sha256 = sha256_file(dataset)
    if corpus_path.is_file():
        corpus_files.append(str(corpus_path.relative_to(ROOT)).replace("\\", "/"))
        corpus_sha256 = sha256_file(corpus_path)
    write_json(output / "TOKENIZER_MANIFEST.json", {
        "tokenizer_name": config.name,
        "tokenizer_version": "GenPy-Tokenizer-32K-v1" if config.vocab_size == 32000 else "GenPy-Tokenizer-smoke-v1",
        "tokenizer_type": config.tokenizer_type,
        "vocab_size": config.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": SPECIAL_IDS,
        "training_corpus_files": corpus_files,
        "training_examples": stats["training_examples"],
        "corpus_characters": stats["characters"],
        "corpus_bytes": stats["bytes"],
        "corpus_lines": stats["lines"],
        "core_bpe_vocab_size": stats.get("core_bpe_vocab_size"),
        "corpus_derived_completion_tokens": stats.get("corpus_derived_completion_tokens"),
        "training_corpus_sha256": corpus_sha256,
        "dataset_train_sha256": sha256_file(dataset),
        "expected_dataset_train_sha256": EXPECTED_TRAIN_SHA256,
        "configuration_sha256": sha256_file(config_path),
        "artifact_sha256": combined_digest(hashes),
        "python_version": platform.python_version(),
        "tokenizers_library_version": tokenizers.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained_tokenizer_used": False,
        "external_vocab_reused": False,
        "external_merge_table_reused": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--input", default="data/instruction/python/train.jsonl")
    parser.add_argument("--output", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--vocab-size", type=int)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_tokenizer_config(config_path)
    if args.vocab_size is not None:
        requested_vocab_size = args.vocab_size
        # Four specials plus the complete 256-byte alphabet require 260 IDs.
        effective_vocab_size = max(args.vocab_size, 260) if args.smoke else args.vocab_size
        config = TokenizerConfig(**{**config.__dict__, "vocab_size": effective_vocab_size})
        config.validate()
        if args.smoke and effective_vocab_size != requested_vocab_size:
            print(f"Smoke vocabulary request {requested_vocab_size} adjusted to {effective_vocab_size} for byte coverage.")
    dataset = ROOT / args.input
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    dataset_hash = sha256_file(dataset)
    if not args.smoke and dataset_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError(f"Dataset integrity: FAIL ({dataset_hash} != {EXPECTED_TRAIN_SHA256})")
    if not args.smoke and not config.train_split_only:
        raise RuntimeError("production tokenizer training must use the train split only")
    stats = collect_stats(dataset, config, skip_invalid=args.smoke).to_dict()
    print("Dataset verified" if args.smoke or dataset_hash == EXPECTED_TRAIN_SHA256 else "Dataset integrity: FAIL")
    print(f"Training examples: {stats['training_examples']:,}")
    print(f"Corpus characters: {stats['characters']:,}; bytes: {stats['bytes']:,}")
    print(f"Vocabulary target: {config.vocab_size:,}")
    if args.dry_run:
        print("Dry run: configuration and input validated; no tokenizer trained.")
        return 0

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    print("Training fresh Byte-Level BPE...")
    backend = build_fresh_tokenizer(config)
    backend.train_from_iterator(iter_documents(dataset, config, skip_invalid=args.smoke), trainer=build_trainer(config))
    core_vocab_size = backend.get_vocab_size(with_added_tokens=True)
    added_corpus_tokens = pad_vocab_with_corpus_tokens(backend, iter_documents(dataset, config, skip_invalid=args.smoke), config.vocab_size)
    if backend.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise RuntimeError(f"Vocabulary target not reached: {backend.get_vocab_size(with_added_tokens=True)} / {config.vocab_size}")
    for name, token_id in SPECIAL_IDS.items():
        if backend.token_to_id(SPECIAL_TOKENS[name]) != token_id:
            raise RuntimeError(f"Special token contract failed for {SPECIAL_TOKENS[name]}")
    print("Saving tokenizer...")
    save_model_files(backend, output)
    stats["core_bpe_vocab_size"] = core_vocab_size
    stats["corpus_derived_completion_tokens"] = added_corpus_tokens
    make_auxiliary_files(output, config, config_path, dataset, stats)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "tokenizer_corpus_stats.json", stats)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
