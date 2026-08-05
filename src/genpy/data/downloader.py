"""Resumable, bounded source downloading and safe archive extraction."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from genpy.data.source_registry import SourceEntry

LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Metadata for a completed or reused source download."""

    source_id: str
    archive_path: str | None
    extracted_path: str
    downloaded_bytes: int
    extracted_bytes: int
    sha256: str | None
    resumed: bool


def free_disk_bytes(path: Path) -> int:
    """Return available bytes for the volume containing path."""
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    maximum_bytes: int,
    expected_sha256: str | None = None,
) -> tuple[int, str, bool]:
    """Download URL with a resumable partial file and an atomic final rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "GenPy-Phase2/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        status = getattr(response, "status", 200)
        resumed = offset > 0 and status == 206
        if offset and not resumed:
            offset = 0
        mode = "ab" if resumed else "wb"
        total = offset
        with partial.open(mode) as handle:
            while chunk := response.read(CHUNK_SIZE):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError(
                        f"download exceeds configured maximum of {maximum_bytes} bytes"
                    )
                handle.write(chunk)
    digest = _file_sha256(partial)
    if expected_sha256 and expected_sha256 != "unknown" and digest != expected_sha256:
        raise ValueError("download checksum mismatch")
    partial.replace(destination)
    return total, digest, resumed


def _safe_extract_zip(archive: Path, destination: Path) -> int:
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("archive contains an unsafe path")
            if member.is_dir():
                continue
            target = temporary.joinpath(*member_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, CHUNK_SIZE)
            extracted_bytes += member.file_size
    temporary.replace(destination)
    return extracted_bytes


def prepare_source(
    source: SourceEntry,
    raw_root: Path,
    *,
    maximum_download_bytes: int,
    minimum_free_disk_bytes: int,
) -> DownloadResult:
    """Prepare a local directory or a pinned ZIP archive for streaming ingestion."""
    if free_disk_bytes(raw_root) < minimum_free_disk_bytes:
        raise OSError("insufficient free disk space for configured ingestion job")
    if source.access_method == "local_directory":
        local_path = Path(source.local_path or "")
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path
        if not local_path.is_dir():
            raise FileNotFoundError(f"local source directory does not exist: {local_path}")
        extracted_size = sum(
            path.stat().st_size for path in local_path.rglob("*") if path.is_file()
        )
        return DownloadResult(source.id, None, str(local_path), 0, extracted_size, None, False)
    if source.access_method != "github_archive" or not source.archive_url:
        raise ValueError(f"source {source.id} requires unsupported or gated access")

    source_root = raw_root / source.id
    source_root.mkdir(parents=True, exist_ok=True)
    archive = source_root / f"{source.version}.zip"
    extracted = source_root / source.version
    downloaded_bytes = archive.stat().st_size if archive.exists() else 0
    digest = _file_sha256(archive) if archive.exists() else None
    resumed = False
    if not archive.exists():
        downloaded_bytes, digest, resumed = download_file(
            source.archive_url,
            archive,
            maximum_bytes=maximum_download_bytes,
            expected_sha256=source.checksum_sha256,
        )
    if source.checksum_sha256 not in {None, "unknown"} and digest != source.checksum_sha256:
        raise ValueError("cached archive checksum mismatch")
    if not extracted.exists():
        extracted_bytes = _safe_extract_zip(archive, extracted)
    else:
        extracted_bytes = sum(
            path.stat().st_size for path in extracted.rglob("*") if path.is_file()
        )
    result = DownloadResult(
        source.id,
        str(archive),
        str(extracted),
        downloaded_bytes,
        extracted_bytes,
        digest,
        resumed,
    )
    state_path = source_root / "download_state.json"
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(state_path)
    LOGGER.info("source prepared", extra={"source_id": source.id})
    return result


def iter_source_files(
    root: Path,
    include_globs: tuple[str, ...],
    excluded_parts: set[str],
    limit: int | None,
    exclude_globs: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield deterministic source files without loading file contents."""
    count = 0
    seen: set[Path] = set()
    for pattern in include_globs:
        for path in sorted(root.rglob(pattern), key=lambda item: item.as_posix()):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if any(part.lower() in excluded_parts for part in path.parts):
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(relative, glob) for glob in exclude_globs):
                continue
            yield path
            count += 1
            if limit is not None and count >= limit:
                return
