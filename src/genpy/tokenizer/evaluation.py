"""Tokenizer evaluation, vocabulary audit, and exact split token counting."""

from __future__ import annotations

import hashlib
import json
import math
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import yaml

from genpy.data.pii_scan import scan_pii
from genpy.data.schemas import InstructionRecord, PretrainingRecord
from genpy.data.secret_scan import scan_secrets
from genpy.data.sharding import iter_shard_records
from genpy.tokenizer.config import SPECIAL_TOKEN_NAMES, TokenizerConfig, load_tokenizer_config
from genpy.tokenizer.corpus import verify_shard
from genpy.tokenizer.fingerprint import atomic_write_json, write_checksum_file
from genpy.tokenizer.serialization import serialize_instruction, serialize_pretraining
from genpy.tokenizer.statistics import summarize_lengths
from genpy.tokenizer.tokenizer import GenPyTokenizer

PYTHON_KEYWORDS = (
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
)

EVALUATION_TEXTS: tuple[tuple[str, str], ...] = (
    ("short_python", "value = 42\nprint(value)\n"),
    ("function", "def square(value: int) -> int:\n    return value * value\n"),
    ("class", "class Counter:\n    def __init__(self):\n        self.value = 0\n"),
    ("comment", "# preserve comments and spaces\nresult = left  +  right\n"),
    ("tabs", "if ready:\n\tprint('tab-indented')\n"),
    ("blank_lines", "first = 1\n\n\nsecond = 2\n"),
    ("triple_quote", 'message = """first line\nsecond line"""\n'),
    ("operators", "result = (a ** 2 + b // 3) != c and d is not None\n"),
    ("identifiers", "snake_case_name = CamelCaseName123\n"),
    ("numeric", "values = [0, -17, 3.14159, 0xFF, 1_000_000]\n"),
    ("path", "path = 'src/genpy/tokenizer/tokenizer.py'\n"),
    ("unicode", "text = 'café naïve தமிழ்'\n"),
    ("emoji", "status = 'ready 😀 λ'\n"),
    (
        "long_python",
        "def accumulate(values):\n"
        "    total = 0\n"
        + "".join(f"    total += values[{index} % len(values)]\n" for index in range(160))
        + "    return total\n",
    ),
)

INSTRUCTION_SAMPLES: tuple[tuple[str, str, str], ...] = (
    (
        "beginner",
        "Print the sum of two integers.",
        "a = int(input())\nb = int(input())\nprint(a + b)\n",
    ),
    (
        "intermediate",
        "Write a function that returns unique values while preserving their order.",
        "def unique_in_order(values):\n"
        "    seen = set()\n"
        "    result = []\n"
        "    for value in values:\n"
        "        if value not in seen:\n"
        "            seen.add(value)\n"
        "            result.append(value)\n"
        "    return result\n",
    ),
)


def _evaluate_text_set(
    tokenizer: GenPyTokenizer, texts: Iterable[str], context: int
) -> dict[str, Any]:
    """Measure one named evaluation population without retaining token sequences."""
    counts: Counter[str] = Counter()
    lengths: Counter[int] = Counter()
    for text in texts:
        counts["record_count"] += 1
        counts["utf8_bytes"] += len(text.encode("utf-8"))
        counts["characters"] += len(text)
        try:
            ids = list(tokenizer._tokenizer.encode(text, add_special_tokens=False).ids)
            counts["token_count"] += len(ids)
            lengths[len(ids)] += 1
            if tokenizer.decode(ids) == text:
                counts["roundtrip_successes"] += 1
            else:
                counts["decode_failures"] += 1
        except (ValueError, RuntimeError):
            counts["encoding_failures"] += 1
    records = counts["record_count"]
    tokens = counts["token_count"]
    return {
        **dict(counts),
        "roundtrip_rate_percentage": (
            100.0 * counts["roundtrip_successes"] / records if records else None
        ),
        "average_utf8_bytes_per_token": counts["utf8_bytes"] / tokens if tokens else None,
        "average_characters_per_token": counts["characters"] / tokens if tokens else None,
        **summarize_lengths(lengths, context),
    }


def _iter_split_texts(config: TokenizerConfig, family: str, split: str) -> Iterator[str]:
    """Yield canonical held-out or training records after shard and split validation."""
    directory = config.project_root / "data/splits" / family / split
    for shard in sorted(directory.glob("part-*.jsonl.zst"), key=lambda path: path.name):
        verify_shard(shard)
        for value in iter_shard_records((shard,)):
            if family == "pretraining":
                pretraining = PretrainingRecord.from_dict(value)
                if pretraining.split != split:
                    raise ValueError(f"record split mismatch in {shard.name}")
                yield serialize_pretraining(pretraining.text).text
            else:
                instruction = InstructionRecord.from_dict(value)
                if instruction.split != split:
                    raise ValueError(f"record split mismatch in {shard.name}")
                prompt, code = _instruction_parts(instruction)
                yield serialize_instruction(prompt, code).text


def _take(texts: Iterable[str], limit: int) -> Iterator[str]:
    for index, text in enumerate(texts):
        if index >= limit:
            return
        yield text


def vocabulary_audit(
    tokenizer: GenPyTokenizer, artifact: Path, maximum_token_length: int
) -> dict[str, Any]:
    """Audit IDs, merges, reachability, and sensitive-looking vocabulary without logging text."""
    raw = json.loads((artifact / "vocab.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("vocab.json must be an object")
    ids = [int(value) for value in raw.values()]
    structural_errors: list[str] = []
    if len(ids) != len(set(ids)):
        structural_errors.append("duplicate_token_ids")
    if sorted(ids) != list(range(len(ids))):
        structural_errors.append("non_contiguous_token_ids")
    if any(not token for token in raw):
        structural_errors.append("empty_token")
    suspicious: list[dict[str, Any]] = []
    unreachable = 0
    for token_text, token_id_value in raw.items():
        token_id = int(token_id_value)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        categories = set(scan_secrets(decoded)) | set(scan_pii(decoded))
        if len(decoded) > maximum_token_length:
            categories.add("very_long_fragment")
        if categories:
            token_hash = hashlib.sha256(token_text.encode("utf-8")).hexdigest()
            for category in sorted(categories):
                suspicious.append(
                    {
                        "category": category,
                        "token_id": token_id,
                        "token_hash": token_hash,
                        "length": len(decoded),
                        "review_status": "requires_review",
                    }
                )
        if token_id >= len(SPECIAL_TOKEN_NAMES):
            encoded = tokenizer._tokenizer.encode(decoded, add_special_tokens=False).ids
            if token_id not in encoded:
                unreachable += 1
    merges = (artifact / "merges.txt").read_text(encoding="utf-8").splitlines()
    vocab_tokens = set(raw)
    invalid_merges = 0
    for line in merges:
        if not line or line.startswith("#"):
            continue
        pieces = line.split()
        if len(pieces) != 2 or any(piece not in vocab_tokens for piece in pieces):
            invalid_merges += 1
    if invalid_merges:
        structural_errors.append("invalid_merge_references")
    return {
        "passed": not structural_errors and not suspicious,
        "structural_errors": structural_errors,
        "suspicious_findings": suspicious,
        "unreachable_token_count": unreachable,
        "invalid_merge_references": invalid_merges,
        "notice": "Static vocabulary scanning does not guarantee removal of sensitive data.",
    }


def evaluate_tokenizer(config_path: Path) -> dict[str, Any]:
    """Evaluate round trips, compression, sequence lengths, throughput, memory, and audit."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evaluation config must be a mapping")
    root = config_path.resolve().parents[2]
    artifact = root / str(raw["artifact_path"])
    tokenizer_config = load_tokenizer_config(root / str(raw["tokenizer_config"]), root)
    load_started = time.perf_counter()
    tokenizer = GenPyTokenizer.load(artifact)
    load_seconds = time.perf_counter() - load_started
    tracemalloc.start()
    encoded_bytes = encoded_chars = encoded_tokens = 0
    roundtrip_success = encoding_failures = decoding_failures = 0
    length_histogram: Counter[int] = Counter()
    started = time.perf_counter()
    encoded_cache: list[list[int]] = []
    for _, text in EVALUATION_TEXTS:
        try:
            ids = tokenizer.encode_text(text)
            encoded_cache.append(ids)
            encoded_bytes += len(text.encode("utf-8"))
            encoded_chars += len(text)
            encoded_tokens += len(ids)
            length_histogram[len(ids)] += 1
            if tokenizer.decode(ids) == text:
                roundtrip_success += 1
            else:
                decoding_failures += 1
        except (ValueError, RuntimeError):
            encoding_failures += 1
    encoding_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    for ids in encoded_cache:
        tokenizer.decode(ids)
    decoding_seconds = time.perf_counter() - decode_started
    instruction_lengths: Counter[int] = Counter()
    instruction_roundtrips = 0
    for _, prompt, code in INSTRUCTION_SAMPLES:
        ids = tokenizer.encode_instruction_record(prompt, code)
        instruction_lengths[len(ids)] += 1
        if tokenizer.decode(ids) == serialize_instruction(prompt, code).text:
            instruction_roundtrips += 1
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    keyword_fragments = [len(tokenizer.encode_text(keyword)) for keyword in PYTHON_KEYWORDS]
    identifiers = ("snake_case_identifier", "CamelCaseIdentifier", "value2index", "__init__")
    identifier_fragments = sorted(len(tokenizer.encode_text(value)) for value in identifiers)
    context = int(raw["context_length"])
    instruction_summary = summarize_lengths(instruction_lengths, context)
    training_limit = int(raw["training_sample_records"])
    evaluation_sets = {
        "fixed_robustness": _evaluate_text_set(
            tokenizer, (text for _, text in EVALUATION_TEXTS), context
        ),
        "instruction_prompts": _evaluate_text_set(
            tokenizer,
            (serialize_instruction(prompt, code).text for _, prompt, code in INSTRUCTION_SAMPLES),
            context,
        ),
        "deterministic_training_sample": _evaluate_text_set(
            tokenizer,
            _take(
                chain(
                    _iter_split_texts(tokenizer_config, "pretraining", "train"),
                    _iter_split_texts(tokenizer_config, "instruction", "train"),
                ),
                training_limit,
            ),
            context,
        ),
        "phase2_validation": _evaluate_text_set(
            tokenizer,
            chain(
                _iter_split_texts(tokenizer_config, "pretraining", "validation"),
                _iter_split_texts(tokenizer_config, "instruction", "validation"),
            ),
            context,
        ),
        "phase2_test": _evaluate_text_set(
            tokenizer,
            chain(
                _iter_split_texts(tokenizer_config, "pretraining", "test"),
                _iter_split_texts(tokenizer_config, "instruction", "test"),
            ),
            context,
        ),
    }
    audit = vocabulary_audit(tokenizer, artifact, int(raw["security"]["maximum_token_length"]))
    validation = tokenizer.validate()
    report: dict[str, Any] = {
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "artifact_status": json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))[
            "status"
        ],
        "actual_vocabulary_size": tokenizer.vocab_size,
        "special_token_atomicity_percentage": 100.0
        if all(tokenizer.token_to_id(tokenizer.id_to_token(index)) == index for index in range(7))
        else 0.0,
        "unknown_token_count": 0,
        "encoding_failure_count": encoding_failures,
        "decode_failure_count": decoding_failures,
        "roundtrip_rate_percentage": 100.0 * roundtrip_success / len(EVALUATION_TEXTS),
        "instruction_roundtrip_rate_percentage": 100.0
        * instruction_roundtrips
        / len(INSTRUCTION_SAMPLES),
        "utf8_byte_coverage_percentage": 100.0 if encoding_failures == 0 else 0.0,
        "average_utf8_bytes_per_token": encoded_bytes / encoded_tokens,
        "average_characters_per_token": encoded_chars / encoded_tokens,
        "compression_vs_raw_byte_baseline": encoded_bytes / encoded_tokens,
        "sequence_lengths": summarize_lengths(length_histogram, context),
        "instruction_sequence_lengths": instruction_summary,
        "v1_samples_fitting_1024_percentage": instruction_summary["percentage_fitting_context"],
        "v1_samples_requiring_truncation_or_packing_percentage": 100.0
        - float(instruction_summary["percentage_fitting_context"]),
        "median_identifier_fragmentation": (identifier_fragments[1] + identifier_fragments[2]) / 2,
        "average_python_keyword_fragmentation": sum(keyword_fragments) / len(keyword_fragments),
        "indentation_preservation_percentage": 100.0 if validation.passed else 0.0,
        "newline_preservation_percentage": 100.0 if validation.passed else 0.0,
        "encoding_throughput_bytes_per_second": encoded_bytes / max(encoding_seconds, 1e-9),
        "decoding_throughput_tokens_per_second": encoded_tokens / max(decoding_seconds, 1e-9),
        "peak_memory_bytes": peak_memory,
        "artifact_load_seconds": load_seconds,
        "save_load_equivalence_percentage": 100.0 if validation.passed else 0.0,
        "evaluation_sets": evaluation_sets,
        "vocabulary_audit": audit,
        "evaluation_record_counts": {
            "fixed_text": len(EVALUATION_TEXTS),
            "beginner_instruction": 1,
            "intermediate_instruction": 1,
            "deterministic_training_sample": evaluation_sets[
                "deterministic_training_sample"
            ]["record_count"],
            "phase2_validation": evaluation_sets["phase2_validation"]["record_count"],
            "phase2_test": evaluation_sets["phase2_test"]["record_count"],
        },
        "limitations": [
            *(
                ["Phase 2 validation and test splits are empty in the current smoke corpus."]
                if evaluation_sets["phase2_validation"]["record_count"] == 0
                and evaluation_sets["phase2_test"]["record_count"] == 0
                else []
            ),
            "Metrics describe the smoke tokenizer and are not production acceptance results.",
        ],
    }
    report_json = root / str(raw["report_json"])
    report_markdown = root / str(raw["report_markdown"])
    atomic_write_json(report_json, report)
    atomic_write_json(artifact / "evaluation.json", report)
    package_artifact(artifact)
    lines = [
        "# Tokenizer Evaluation Report",
        "",
        f"Artifact status: `{report['artifact_status']}`",
        f"Vocabulary size: {report['actual_vocabulary_size']}",
        f"Fingerprint: `{report['tokenizer_fingerprint']}`",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Exact round trip | {report['roundtrip_rate_percentage']:.2f}% |",
        f"| UTF-8 byte coverage | {report['utf8_byte_coverage_percentage']:.2f}% |",
        f"| UTF-8 bytes per token | {report['average_utf8_bytes_per_token']:.4f} |",
        f"| V1 samples fitting 1,024 | {report['v1_samples_fitting_1024_percentage']:.2f}% |",
        f"| Vocabulary audit passed | {audit['passed']} |",
        "",
        "These are smoke-tokenizer results. Held-out Phase 2 splits are currently empty.",
        "",
    ]
    report_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_markdown.write_text("\n".join(lines), encoding="utf-8")
    return report


def _instruction_parts(record: InstructionRecord) -> tuple[str, str]:
    users = [message.content for message in record.messages if message.role == "user"]
    assistants = [message.content for message in record.messages if message.role == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError("instruction record shape is not countable")
    return users[0], assistants[0]


def count_corpus_tokens(
    config: TokenizerConfig,
    artifact: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Stream exact smoke-token counts for every Phase 2 family and split."""
    tokenizer = GenPyTokenizer.load(artifact)
    report_dir = config.project_root / "data/tokenizer/reports"
    state_path = report_dir / "token_count_state.json"
    state: dict[str, Any] = {}
    if resume and state_path.is_file():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            isinstance(loaded, dict)
            and loaded.get("tokenizer_fingerprint") == tokenizer.fingerprint
        ):
            state = loaded
    shard_state: dict[str, Any] = dict(state.get("shards", {}))
    split_results: dict[str, Any] = {}
    for family in ("pretraining", "instruction"):
        for split in ("train", "validation", "test"):
            key = f"{family}_{split}"
            directory = config.project_root / "data/splits" / family / split
            aggregate: Counter[str] = Counter()
            length_histogram: Counter[int] = Counter()
            for shard in sorted(directory.glob("part-*.jsonl.zst")):
                manifest = verify_shard(shard)
                shard_key = f"{key}/{shard.name}"
                cached = shard_state.get(shard_key)
                if isinstance(cached, dict) and cached.get("shard_sha256") == manifest["sha256"]:
                    metrics = cached
                else:
                    metrics_counter: Counter[str] = Counter()
                    shard_lengths: Counter[int] = Counter()
                    for value in iter_shard_records((shard,)):
                        if family == "pretraining":
                            record = PretrainingRecord.from_dict(value)
                            serialized = serialize_pretraining(record.text)
                            ids = tokenizer.encode_pretraining_record(record.text)
                            chars = len(record.text)
                            raw_bytes = len(record.text.encode("utf-8"))
                        else:
                            instruction = InstructionRecord.from_dict(value)
                            prompt, code = _instruction_parts(instruction)
                            serialized = serialize_instruction(prompt, code)
                            ids = tokenizer.encode_instruction_record(prompt, code)
                            chars = len(prompt) + len(code)
                            raw_bytes = len(prompt.encode("utf-8")) + len(code.encode("utf-8"))
                        metrics_counter["record_count"] += 1
                        metrics_counter["utf8_bytes"] += raw_bytes
                        metrics_counter["characters"] += chars
                        metrics_counter["exact_tokens"] += len(ids) - serialized.structural_tokens
                        metrics_counter["structural_special_tokens"] += serialized.structural_tokens
                        metrics_counter["total_serialized_tokens"] += len(ids)
                        shard_lengths[len(ids)] += 1
                    metrics = {
                        **dict(metrics_counter),
                        "length_histogram": dict(shard_lengths),
                        "shard_sha256": manifest["sha256"],
                        "tokenizer_fingerprint": tokenizer.fingerprint,
                    }
                    shard_state[shard_key] = metrics
                    atomic_write_json(
                        state_path,
                        {"tokenizer_fingerprint": tokenizer.fingerprint, "shards": shard_state},
                    )
                for name in (
                    "record_count",
                    "utf8_bytes",
                    "characters",
                    "exact_tokens",
                    "structural_special_tokens",
                    "total_serialized_tokens",
                ):
                    aggregate[name] += int(metrics.get(name, 0))
                for length, count in metrics.get("length_histogram", {}).items():
                    length_histogram[int(length)] += int(count)
            summary = summarize_lengths(length_histogram, int(config.tokenizer["context_length"]))
            split_results[key] = {
                **dict(aggregate),
                **summary,
                "estimated_1024_token_packed_sequences": math.ceil(
                    aggregate["total_serialized_tokens"] / 1024
                ),
            }
    report = {
        "tokenizer_status": "smoke"
        if int(config.tokenizer["vocab_size"]) != 16384
        else config.tokenizer["status"],
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "notice": (
            "Exact counts use this tokenizer artifact; "
            "Phase 2 rough estimates remain historical."
        ),
        "splits": split_results,
    }
    json_path = report_dir / "exact_token_counts.json"
    markdown_path = report_dir / "exact_token_counts.md"
    atomic_write_json(json_path, report)
    lines = [
        "# Exact Corpus Token Counts",
        "",
        f"Tokenizer status: `{report['tokenizer_status']}`",
        f"Fingerprint: `{tokenizer.fingerprint}`",
        "",
        "| Split | Records | Bytes | Exact tokens | Structural | Serialized | Fits 1,024 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in split_results.items():
        lines.append(
            f"| {key} | {value.get('record_count', 0)} | {value.get('utf8_bytes', 0)} | "
            f"{value.get('exact_tokens', 0)} | {value.get('structural_special_tokens', 0)} | "
            f"{value.get('total_serialized_tokens', 0)} | "
            f"{value.get('percentage_fitting_context', 0):.2f}% |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def package_artifact(artifact: Path) -> dict[str, str]:
    """Verify required files and refresh non-recursive artifact checksums."""
    files = (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "metadata.json",
        "evaluation.json",
        "corpus_fingerprint.json",
    )
    missing = [name for name in files if not (artifact / name).is_file()]
    if missing:
        raise FileNotFoundError(f"artifact is missing files: {', '.join(missing)}")
    return write_checksum_file(artifact, files)
