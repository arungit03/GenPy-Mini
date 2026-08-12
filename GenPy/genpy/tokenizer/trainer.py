"""Iterator-based tokenizer training and artifact metadata."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import tokenizers

from genpy.config import TokenizerPipelineConfig

from .builder import build_tokenizer, build_trainer
from .corpus import CorpusReader, CorpusStats, find_step2_manifest
from .tokenizer import GenPyTokenizer


def train_from_iterator(config, texts: Iterable[str], vocab_size: Optional[int] = None, show_progress: bool = False) -> GenPyTokenizer:
    raw = build_tokenizer(config, vocab_size=vocab_size)
    raw.train_from_iterator(texts, trainer=build_trainer(config, vocab_size=vocab_size, show_progress=show_progress))
    return GenPyTokenizer(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tokenizer_artifacts(
    tokenizer: GenPyTokenizer,
    config: TokenizerPipelineConfig,
    output_dir: Path,
    corpus_stats: CorpusStats,
    mode: str,
    source_manifest: Optional[Path] = None,
    shard_names=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / config.output.tokenizer_file
    tokenizer.save(tokenizer_path)
    checksum = sha256_file(tokenizer_path)
    config_path = output_dir / config.output.config_file
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "tokenizer_name": config.tokenizer.name,
        "algorithm": config.tokenizer.algorithm,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": dict(zip(("pad", "bos", "eos", "unk"), config.tokenizer.special_tokens.ordered)),
        "special_token_ids": {"pad": tokenizer.pad_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "unk": tokenizer.unk_token_id},
        "normalizer": config.tokenizer.normalizer,
        "byte_level": {"add_prefix_space": config.tokenizer.add_prefix_space, "use_regex": config.tokenizer.use_regex},
        "min_frequency": config.tokenizer.min_frequency,
        "max_token_length": config.tokenizer.max_token_length,
        "training_corpus": {
            "source_manifest": str(source_manifest) if source_manifest else None,
            "shards": sorted(shard_names or []),
            **corpus_stats.to_dict(),
        },
        "training_mode": mode,
        "tokenizers_version": tokenizers.__version__,
        "python_version": platform.python_version(),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "tokenizer_json_sha256": checksum,
    }
    manifest_path = output_dir / config.output.manifest_file
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"tokenizer_path": tokenizer_path, "config_path": config_path, "manifest_path": manifest_path, "manifest": manifest}


def train_from_corpus(
    pipeline_config: TokenizerPipelineConfig,
    input_dir: Path,
    output_dir: Path,
    mode: str = "development",
    max_documents: Optional[int] = None,
    max_bytes: Optional[int] = None,
    source_manifest: Optional[Path] = None,
    vocab_size: Optional[int] = None,
) -> dict:
    reader = CorpusReader(input_dir, pipeline_config.training_data.train_pattern, pipeline_config.training_data.text_field)
    tokenizer = train_from_iterator(pipeline_config.tokenizer, reader.texts(max_documents, max_bytes), vocab_size, pipeline_config.training.show_progress)
    manifest = find_step2_manifest(Path(pipeline_config.training_data.input_dir).parent / "manifests", source_manifest)
    result = save_tokenizer_artifacts(tokenizer, pipeline_config, output_dir, reader.stats, mode, manifest, [path.name for path in reader.shard_paths()])
    result["tokenizer"] = tokenizer
    result["corpus_stats"] = reader.stats
    return result
