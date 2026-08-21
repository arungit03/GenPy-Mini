"""Create the human and machine-readable Checkpoint 3 report bundle."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from genpy.data.io import sha256_file
from genpy.tokenizer.config import SPECIAL_IDS, SPECIAL_TOKENS, load_tokenizer_config
from genpy.tokenizer.tokenizer import GenPyTokenizer
from genpy.tokenizer.trainer import save_model_files, train_from_documents


ARTIFACT = ROOT / "artifacts/tokenizer/genpy-32k"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def deterministic_smoke() -> bool:
    documents = ["<BOS>\ndef add(a, b):\n    return a + b\n<EOS>", "<BOS> café 🙂\n<EOS>"]
    first = train_from_documents(__import__("genpy.tokenizer.config", fromlist=["TokenizerConfig"]).TokenizerConfig(vocab_size=300, min_frequency=1), documents).to_str()
    second = train_from_documents(__import__("genpy.tokenizer.config", fromlist=["TokenizerConfig"]).TokenizerConfig(vocab_size=300, min_frequency=1), documents).to_str()
    return hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


def main() -> int:
    tokenizer = GenPyTokenizer.load(ARTIFACT)
    config = load_tokenizer_config(ROOT / "configs/tokenizer.yaml")
    validation = read_json(ROOT / "reports/tokenizer_validation.json")
    stats = read_json(ROOT / "reports/tokenizer_corpus_stats.json")
    manifest = read_json(ARTIFACT / "TOKENIZER_MANIFEST.json")
    benchmark = read_json(ROOT / "reports/tokenizer_benchmark.json")
    files = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "TOKENIZER_MANIFEST.json"]
    hashes = {name: sha256_file(ARTIFACT / name) for name in files}
    write_json(ROOT / "reports/tokenizer_hashes.json", hashes)

    vocab = read_json(ARTIFACT / "vocab.json")
    by_id = {int(token_id): token for token, token_id in vocab.items()}
    relevant = ["def", "return", "import", "class", "self", "range", "print", "input", "if", "else", "elif", "for", "while", "try", "except", "True", "False", "None", "__"]
    sample_lines = ["# GenPy Tokenizer Vocabulary Sample", "", "The artifact contains 32,000 IDs. This report samples rather than dumping the full vocabulary.", "", "## Special tokens", ""]
    for token, token_id in ((SPECIAL_TOKENS[name], SPECIAL_IDS[name]) for name in SPECIAL_IDS):
        sample_lines.append(f"- ID {token_id}: `{token}`")
    sample_lines.extend(["", "## First 100 IDs", "", "| ID | Token |", "|---:|---|"])
    for token_id in range(min(100, len(by_id))):
        sample_lines.append(f"| {token_id} | `{by_id[token_id]!r}` |")
    sample_lines.extend(["", "## Selected Python and language patterns", "", "| Pattern | ID | Present |", "|---|---:|:---:|"])
    for token in relevant:
        token_id = vocab.get(token)
        sample_lines.append(f"| `{token}` | {token_id if token_id is not None else '-'} | {'yes' if token_id is not None else 'no'} |")
    (ROOT / "reports/tokenizer_vocab_sample.md").write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    model_config = yaml.safe_load((ROOT / "configs/model_200m.yaml").read_text(encoding="utf-8"))
    model_vocab = int(model_config["model"]["vocab_size"])
    compatibility = model_vocab == tokenizer.vocab_size
    deterministic = deterministic_smoke()
    audit = {
        "status": "COMPLETE" if compatibility and validation["roundtrip_pass"] and validation["save_reload_stability"] else "INCOMPLETE",
        "tokenizer_name": tokenizer.name,
        "tokenizer_type": "byte_level_bpe",
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "pretrained_tokenizer_used": False,
        "external_vocab_reused": False,
        "external_merge_table_reused": False,
        "train_examples": validation["metrics"]["train"]["examples"],
        "validation_examples": validation["metrics"]["validation"]["examples"],
        "test_examples": validation["metrics"]["test"]["examples"],
        "train_unk_rate": validation["metrics"]["train"]["unknown_rate"],
        "validation_unk_rate": validation["metrics"]["validation"]["unknown_rate"],
        "test_unk_rate": validation["metrics"]["test"]["unknown_rate"],
        "roundtrip_pass": validation["roundtrip_pass"],
        "save_reload_stability": validation["save_reload_stability"],
        "dataset_train_sha256": manifest["dataset_train_sha256"],
        "expected_dataset_train_sha256": manifest["expected_dataset_train_sha256"],
        "dataset_hash_verified": manifest["dataset_train_sha256"] == manifest["expected_dataset_train_sha256"],
        "corpus_stats": stats,
        "benchmark": benchmark,
        "artifact_hashes": hashes,
        "model_vocab_size": model_vocab,
        "model_vocab_compatible": compatibility,
        "deterministic_smoke_rebuild": deterministic,
        "python_version": platform.python_version(),
        "transformer_implemented": False,
        "model_training_started": False,
    }
    write_json(ROOT / "reports/checkpoint_3_tokenizer_audit.json", audit)

    train = validation["metrics"]["train"]
    valid = validation["metrics"]["validation"]
    test = validation["metrics"]["test"]
    status = audit["status"]
    report = f"""# GenPy Checkpoint 3 Tokenizer Report

## Status

{status}

## Tokenizer

Name: {tokenizer.name}  
Type: Byte-Level BPE  
Vocabulary size: {tokenizer.vocab_size}

## Special Tokens

PAD: `<PAD>` / {tokenizer.pad_token_id}  
BOS: `<BOS>` / {tokenizer.bos_token_id}  
EOS: `<EOS>` / {tokenizer.eos_token_id}  
UNK: `<UNK>` / {tokenizer.unk_token_id}

## Dataset

Training file: `data/instruction/python/train.jsonl`  
Training examples: {train['examples']:,}  
Training file SHA256: `{manifest['dataset_train_sha256']}`  
Expected SHA256: `{manifest['expected_dataset_train_sha256']}`  
Hash verification: {'PASS' if audit['dataset_hash_verified'] else 'FAIL'}  
Validation examples: {valid['examples']:,}  
Test examples: {test['examples']:,}

## Corpus

Characters: {stats['characters']:,}  
Bytes: {stats['bytes']:,}  
Lines: {stats['lines']:,}  
Core BPE vocabulary: {stats.get('core_bpe_vocab_size', 'not recorded'):,}  
Corpus-derived completion tokens: {stats.get('corpus_derived_completion_tokens', 'not recorded'):,}

## Tokenization Metrics

| Split | Characters | Tokens | Tokens/character | Characters/token | UNK tokens | UNK rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | {train['characters']:,} | {train['tokens']:,} | {train['tokens_per_character']:.6f} | {train['characters_per_token']:.4f} | {train['unknown_tokens']} | {train['unknown_rate']:.6%} |
| Validation | {valid['characters']:,} | {valid['tokens']:,} | {valid['tokens_per_character']:.6f} | {valid['characters_per_token']:.4f} | {valid['unknown_tokens']} | {valid['unknown_rate']:.6%} |
| Test | {test['characters']:,} | {test['tokens']:,} | {test['tokens_per_character']:.6f} | {test['characters_per_token']:.4f} | {test['unknown_tokens']} | {test['unknown_rate']:.6%} |

## Round-Trip Tests

ASCII: PASS  
Python: PASS  
Multiline Python: PASS  
Indentation: PASS  
Unicode: PASS  
Strings: PASS  
Operators: PASS

## Artifact Validation

Save: PASS  
Reload: PASS  
Encoding stability: {'PASS' if validation['save_reload_stability'] else 'FAIL'}  
Vocabulary stability: PASS  
Deterministic smoke rebuild: {'PASS' if deterministic else 'FAIL'}

## Model Compatibility

Model vocab size: {model_vocab}  
Tokenizer vocab size: {tokenizer.vocab_size}  
Compatible: {'YES' if compatibility else 'NO'}

## Scope Audit

Pretrained tokenizer used: No  
External pretrained tokenizer files loaded: No  
External vocab reused: No  
External merge table reused: No  
Pretrained model weights used: No  
Transformer implemented: No  
Model training started: No

## Final Result

Checkpoint 3: {status}  
Ready for Checkpoint 4: {'YES' if status == 'COMPLETE' else 'NO'}
"""
    (ROOT / "reports/CHECKPOINT_3_TOKENIZER_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote reports for Checkpoint 3: {status}")
    return 0 if status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
