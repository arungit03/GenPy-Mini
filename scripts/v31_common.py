"""Shared immutable-input and tokenizer helpers for Checkpoint 8-v3.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3_VERSION = "genpy-sft-v3-semantic-v1"
EXPECTED_V3_PACKAGE_SHA = "53b9a69189454067fefd64f4c9e26150ad04024934615509cbee274fd2d96b0e"
TOKENIZER_DIR = ROOT / "artifacts/tokenizer/genpy-32k"
REPORT_DIR = ROOT / "reports/checkpoint_8_v3_1"
V3_RAW = ROOT / "data/instruction/python_v3"
V3_SFT = ROOT / "data/instruction/sft_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(data: object) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def tokenizer_identity():
    manifest_path = TOKENIZER_DIR / "TOKENIZER_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = {"tokenizer_name": manifest["tokenizer_name"], "tokenizer_version": manifest["tokenizer_version"], "tokenizer_type": manifest["tokenizer_type"], "vocab_size": manifest["vocab_size"], "special_token_ids": manifest["special_token_ids"], "artifact_sha256": manifest["artifact_sha256"], "dataset_train_sha256": manifest["dataset_train_sha256"]}
    return manifest, identity, canonical_sha256(manifest), canonical_sha256(identity)


def behavior_vectors():
    from genpy.tokenizer import GenPyTokenizer
    tokenizer = GenPyTokenizer.load(TOKENIZER_DIR)
    texts = ["def solve(n):\n    return n % 2 == 0\n", "Write a Python function named solve that reverses text.", "### User\nCalculate factorial.\n\n### Assistant\n", "values = [1, 2, 3, 4]"]
    vectors = [{"text": text, "token_ids": tokenizer.encode(text)} for text in texts]
    return {"tokenizer_name": tokenizer.name, "vocab_size": tokenizer.vocab_size, "vectors": vectors}, canonical_sha256({"tokenizer_name": tokenizer.name, "vocab_size": tokenizer.vocab_size, "vectors": vectors})


def frozen_source_hashes():
    paths = {f"python_v3/{name}.jsonl": V3_RAW / f"{name}.jsonl" for name in ("train", "validation", "challenge", "sanity")}
    paths.update({f"sft_v3/{name}.jsonl": V3_SFT / f"{name}.jsonl" for name in ("train", "validation", "challenge")})
    paths["v3_package"] = ROOT / "artifacts/checkpoint_8_v3/GenPy-SFT-v3-Semantic-Pilot.zip"
    return {name: sha256(path) for name, path in paths.items()}


def write_json_and_text(json_path: Path, text_path: Path, title: str, data: dict):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path.write_text(title + "\n\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
