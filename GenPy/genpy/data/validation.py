"""Validation of processed JSONL.GZ shards and their document invariants."""

import gzip
import json
from pathlib import Path
from typing import Dict, Optional, Set

from .dedup import content_hash
from .schema import GenPyDocument


def validate_dataset(processed_dir: Path, manifest_path: Optional[Path] = None) -> Dict[str, object]:
    processed_dir = Path(processed_dir)
    errors = []
    files = sorted(processed_dir.glob("*.jsonl.gz"))
    seen_hashes: Set[str] = set()
    document_count = 0
    train_count = 0
    validation_count = 0
    for path in files:
        expected_split = "validation" if path.name.startswith("validation-") else "train"
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        value = json.loads(line)
                        document = GenPyDocument.from_dict(value)
                    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                        errors.append(f"{path.name}:{line_number}: invalid document: {exc}")
                        continue
                    if document.split != expected_split:
                        errors.append(f"{path.name}:{line_number}: split does not match filename")
                    if content_hash(document.text) != document.content_hash:
                        errors.append(f"{path.name}:{line_number}: content hash mismatch")
                    if document.char_count != len(document.text) or document.byte_count != len(document.text.encode("utf-8")):
                        errors.append(f"{path.name}:{line_number}: length counters mismatch")
                    if document.content_hash in seen_hashes:
                        errors.append(f"{path.name}:{line_number}: duplicate content hash")
                    seen_hashes.add(document.content_hash)
                    document_count += 1
                    if document.split == "train":
                        train_count += 1
                    elif document.split == "validation":
                        validation_count += 1
        except (OSError, EOFError, UnicodeError) as exc:
            errors.append(f"{path.name}: unable to read gzip JSONL: {exc}")
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            errors.append(f"manifest not found: {manifest_path}")
        else:
            try:
                with manifest_path.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                listed = set(manifest.get("generated_shard_filenames", []))
                actual = {path.name for path in files}
                if listed != actual:
                    errors.append("manifest shard list does not match output directory")
            except (OSError, json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"invalid manifest: {exc}")
    return {
        "valid": not errors and bool(files),
        "files": [path.name for path in files],
        "documents": document_count,
        "train_documents": train_count,
        "validation_documents": validation_count,
        "errors": errors,
    }
