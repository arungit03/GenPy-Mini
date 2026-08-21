"""Optional local CodeSearchNet Python adapter; no download is performed here."""

from pathlib import Path
from typing import Iterator

from ..io import iter_records
from ..schema import InstructionExample


def iter_python_records(path: str | Path) -> Iterator[InstructionExample]:
    """Yield only records with explicit, verified license metadata."""
    for raw in iter_records(path):
        language = str(raw.get("language", "python")).lower()
        license_name = raw.get("license") or raw.get("metadata", {}).get("license")
        query = raw.get("docstring") or raw.get("query") or raw.get("instruction")
        code = raw.get("code") or raw.get("response")
        if language != "python" or not query or not code or not license_name:
            continue
        metadata = dict(raw.get("metadata", {}))
        metadata.update({
            "repository": raw.get("repository"), "path": raw.get("path"),
            "source_url": raw.get("url") or raw.get("source_url"),
            "original_query": query, "license": license_name,
        })
        yield InstructionExample(
            id=str(raw.get("id", "")), task_type="code_generation", category="misc",
            instruction=f"Write a Python function that {str(query).strip().rstrip('.') }.",
            response=str(code), source="codesearchnet", quality_score=0.9,
            metadata=metadata, family_id=str(raw.get("family_id", "")),
        )
