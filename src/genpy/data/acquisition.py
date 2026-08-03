"""Reproducible, resource-bounded acquisition of registered dataset sources.

Supports exactly three source types -- ``local_directory``, ``git_repository``,
and ``http_archive`` -- and produces raw, traceable source material plus a
provenance manifest. This module performs no governance or license
decisions itself: callers (the CLI scripts) must decide whether a source is
eligible to be acquired *before* calling :func:`acquire_source`.

Output layout:

    data/raw/sources/<source-id>/<revision>/   -- acquired file snapshot
    data/manifests/<source-id>-<revision>.json -- provenance manifest

Acquisition is idempotent: re-running against an already-acquired,
manifest-verified source is a safe no-op unless ``force=True``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from genpy.data.checksums import DEFAULT_CHUNK_SIZE, fingerprint_directory, verify_sha256
from genpy.data.exceptions import (
    AcquisitionError,
    StorageLimitError,
    UnsafePathError,
    UnsupportedSourceTypeError,
)
from genpy.data.manifests import (
    AcquisitionRecord,
    FileManifestRecord,
    SourceManifest,
    build_manifest,
    manifest_path_for,
    read_manifest,
    redact_url_credentials,
    sanitize_revision_for_filesystem,
    verify_manifest_against_directory,
    write_manifest,
)
from genpy.data.paths import iter_safe_files, resolve_safe_extraction_path
from genpy.data.schemas import DatasetSource

_SUPPORTED_ARCHIVE_KINDS: Final[tuple[str, ...]] = ("zip", "tar", "tar.gz")
_LOCAL_TEST_HOST_PREFIXES: Final[tuple[str, ...]] = ("http://localhost", "http://127.0.0.1")


@dataclasses.dataclass(frozen=True, slots=True)
class AcquisitionOutcome:
    """The result of one call to :func:`acquire_source`."""

    source_id: str
    status: str
    resolved_revision: str | None
    destination: Path | None
    manifest_path: Path | None
    message: str


def compute_acquisition_identity(source: DatasetSource) -> str:
    """Return a stable identity string for idempotency comparisons.

    Combines source id, source type, revision, and expected checksum (when
    configured), so two registry edits that don't change any of these
    still count as "the same acquisition."
    """
    parts = (source.id, source.source_type, source.revision, source.acquisition.expected_sha256 or "")
    return "|".join(parts)


def _is_local_test_host(url: str) -> bool:
    return any(url.startswith(prefix) for prefix in _LOCAL_TEST_HOST_PREFIXES)


def _stream_copy_tree(source_root: Path, destination_root: Path, *, max_bytes: int) -> int:
    """Stream-copy every safe file under ``source_root`` into ``destination_root``.

    Returns the total number of bytes copied. Raises :class:`StorageLimitError`
    as soon as the running total would exceed ``max_bytes``, before copying
    the file that would push it over.
    """
    total_bytes = 0
    for file_path in iter_safe_files(source_root):
        relative = file_path.relative_to(source_root)
        size = file_path.stat().st_size
        total_bytes += size
        if total_bytes > max_bytes:
            raise StorageLimitError(
                f"Copying {source_root} would exceed the configured maximum of {max_bytes} bytes."
            )
        destination_file = destination_root / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_path, destination_file)
    return total_bytes


def _acquire_local_directory(source: DatasetSource, work_destination: Path) -> None:
    source_root = Path(source.location)
    if not source_root.is_dir():
        raise AcquisitionError(f"Source {source.id!r}: local directory does not exist: {source_root}")

    work_destination.mkdir(parents=True, exist_ok=True)
    _stream_copy_tree(source_root, work_destination, max_bytes=source.acquisition.maximum_extracted_bytes)


def _git_env() -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(args: list[str], *, cwd: Path, timeout: int) -> str:
    redacted_args = [redact_url_credentials(arg) for arg in args]
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise AcquisitionError(f"git command timed out after {timeout}s: {' '.join(redacted_args)}") from exc
    except OSError as exc:
        raise AcquisitionError(f"Could not run git ({' '.join(redacted_args)}): {exc}") from exc

    if result.returncode != 0:
        stderr = redact_url_credentials(result.stderr.strip())
        raise AcquisitionError(f"git command failed ({result.returncode}): {' '.join(redacted_args)}\n{stderr}")

    return result.stdout


def _acquire_git_repository(source: DatasetSource, work_destination: Path, *, timeout_seconds: int) -> str:
    if shutil.which("git") is None:
        raise AcquisitionError("git is required to acquire git_repository sources but was not found on PATH.")

    if source.acquisition.include_submodules:
        raise AcquisitionError(
            f"Source {source.id!r}: acquisition.include_submodules is not supported in Phase 2."
        )

    with tempfile.TemporaryDirectory(prefix="genpy-git-", ignore_cleanup_errors=True) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        clone_dir = tmp_dir / "clone"
        no_hooks_dir = tmp_dir / "no-hooks"  # deliberately never created: disables any hook execution

        init_args = ["git", "init", "--quiet", str(clone_dir)]
        _run_git(init_args, cwd=tmp_dir, timeout=timeout_seconds)

        common_config = ["-c", f"core.hooksPath={no_hooks_dir}"]
        _run_git(
            ["git", *common_config, "remote", "add", "origin", source.location],
            cwd=clone_dir,
            timeout=timeout_seconds,
        )

        fetch_args = ["git", *common_config, "fetch", "--quiet"]
        if source.acquisition.shallow_clone:
            fetch_args += ["--depth", "1"]
        fetch_args += ["origin", source.revision]

        try:
            _run_git(fetch_args, cwd=clone_dir, timeout=timeout_seconds)
        except AcquisitionError as exc:
            raise AcquisitionError(
                f"Source {source.id!r}: could not fetch revision {source.revision!r} "
                f"from {redact_url_credentials(source.location)}: {exc}"
            ) from exc

        _run_git(
            ["git", *common_config, "checkout", "--quiet", "FETCH_HEAD"],
            cwd=clone_dir,
            timeout=timeout_seconds,
        )

        resolved_revision = _run_git(
            ["git", *common_config, "rev-parse", "FETCH_HEAD"], cwd=clone_dir, timeout=timeout_seconds
        ).strip()
        if not resolved_revision:
            raise AcquisitionError(f"Source {source.id!r}: could not resolve revision {source.revision!r}.")

        work_destination.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        max_bytes = source.acquisition.maximum_extracted_bytes
        for file_path in iter_safe_files(clone_dir):
            relative = file_path.relative_to(clone_dir)
            if relative.parts and relative.parts[0] == ".git":
                continue
            size = file_path.stat().st_size
            total_bytes += size
            if total_bytes > max_bytes:
                raise StorageLimitError(
                    f"Source {source.id!r}: git snapshot exceeds maximum_extracted_bytes ({max_bytes})."
                )
            destination_file = work_destination / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_path, destination_file)

        return resolved_revision


def _detect_archive_kind(location: str) -> str:
    lowered = location.lower().split("?")[0].split("#")[0]
    if lowered.endswith(".zip"):
        return "zip"
    if lowered.endswith(".tar.gz") or lowered.endswith(".tgz"):
        return "tar.gz"
    if lowered.endswith(".tar"):
        return "tar"
    raise UnsupportedSourceTypeError(
        f"Unsupported archive type for {redact_url_credentials(location)!r}; "
        f"supported extensions are .zip, .tar, .tar.gz, .tgz."
    )


def _download_with_limit(url: str, destination: Path, *, max_bytes: int, timeout: int, retries: int) -> None:
    """Stream ``url`` to ``destination``, aborting as soon as ``max_bytes`` would be exceeded."""
    last_error: Exception | None = None

    for _attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "genpy-mini-acquisition"})
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_bytes = int(content_length)
                    except ValueError:
                        declared_bytes = None
                    if declared_bytes is not None and declared_bytes > max_bytes:
                        raise StorageLimitError(
                            f"Declared Content-Length ({declared_bytes}) exceeds "
                            f"maximum_download_bytes ({max_bytes})."
                        )

                downloaded = 0
                with destination.open("wb") as out_file:
                    while True:
                        chunk = response.read(DEFAULT_CHUNK_SIZE)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise StorageLimitError(
                                f"Download exceeded maximum_download_bytes ({max_bytes}) while streaming."
                            )
                        out_file.write(chunk)
            return
        except StorageLimitError:
            destination.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            continue

    raise AcquisitionError(
        f"Failed to download {redact_url_credentials(url)} after {retries} attempt(s): {last_error}"
    )


def _safe_extract_zip(archive_path: Path, extract_dir: Path, max_extracted_bytes: int) -> None:
    total_bytes = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise UnsafePathError(f"Zip archive contains a symlink entry, rejected: {info.filename}")

                destination = resolve_safe_extraction_path(extract_dir, info.filename)
                total_bytes += info.file_size
                if total_bytes > max_extracted_bytes:
                    raise StorageLimitError(
                        f"Extraction would exceed maximum_extracted_bytes ({max_extracted_bytes})."
                    )

                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as member, destination.open("wb") as out_file:
                    shutil.copyfileobj(member, out_file, length=DEFAULT_CHUNK_SIZE)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"Archive is not a valid zip file: {exc}") from exc


def _safe_extract_tar(archive_path: Path, extract_dir: Path, max_extracted_bytes: int) -> None:
    total_bytes = 0
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    raise UnsafePathError(
                        f"Tar archive contains a symlink/hardlink entry, rejected: {member.name}"
                    )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise UnsafePathError(
                        f"Tar archive contains an unsupported entry type, rejected: {member.name}"
                    )

                destination = resolve_safe_extraction_path(extract_dir, member.name)
                total_bytes += member.size
                if total_bytes > max_extracted_bytes:
                    raise StorageLimitError(
                        f"Extraction would exceed maximum_extracted_bytes ({max_extracted_bytes})."
                    )

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise UnsafePathError(f"Tar member could not be read: {member.name}")

                destination.parent.mkdir(parents=True, exist_ok=True)
                with extracted, destination.open("wb") as out_file:
                    shutil.copyfileobj(extracted, out_file, length=DEFAULT_CHUNK_SIZE)
    except tarfile.TarError as exc:
        raise AcquisitionError(f"Archive is not a valid tar file: {exc}") from exc


def _acquire_http_archive(
    source: DatasetSource,
    work_destination: Path,
    *,
    downloads_dir: Path,
    timeout_seconds: int,
    retry_count: int,
) -> None:
    if not source.location.startswith("https://") and not _is_local_test_host(source.location):
        raise AcquisitionError(
            f"Source {source.id!r}: http_archive location must use https:// "
            "(or http://localhost / http://127.0.0.1 for local testing)."
        )

    archive_kind = _detect_archive_kind(source.location)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    partial_path = downloads_dir / f"{source.id}-{uuid.uuid4().hex}.partial"

    try:
        _download_with_limit(
            source.location,
            partial_path,
            max_bytes=source.acquisition.maximum_download_bytes,
            timeout=timeout_seconds,
            retries=retry_count,
        )

        if source.acquisition.expected_sha256:
            verify_sha256(partial_path, source.acquisition.expected_sha256)

        with tempfile.TemporaryDirectory(prefix="genpy-archive-", ignore_cleanup_errors=True) as tmp_dir_str:
            extract_dir = Path(tmp_dir_str) / "extracted"
            extract_dir.mkdir()

            if archive_kind == "zip":
                _safe_extract_zip(partial_path, extract_dir, source.acquisition.maximum_extracted_bytes)
            else:
                _safe_extract_tar(partial_path, extract_dir, source.acquisition.maximum_extracted_bytes)

            work_destination.mkdir(parents=True, exist_ok=True)
            _stream_copy_tree(
                extract_dir, work_destination, max_bytes=source.acquisition.maximum_extracted_bytes
            )
    finally:
        partial_path.unlink(missing_ok=True)


def acquire_source(
    source: DatasetSource,
    *,
    sources_root: Path,
    downloads_root: Path,
    manifests_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    governance_override: bool = False,
    override_reason: str | None = None,
    tool_version: str = "0.1.0",
    timeout_seconds: int = 60,
    retry_count: int = 3,
    logger: logging.Logger | None = None,
) -> AcquisitionOutcome:
    """Acquire ``source`` into ``sources_root`` and write its provenance manifest.

    This function performs no governance or license decision -- the caller
    must have already decided the source is eligible to be acquired (see
    ``src/genpy/data/source_registry.py::evaluate_source``). ``force`` and
    ``governance_override``/``override_reason`` are recorded in the manifest
    but are not themselves authorization: the CLI enforces the override
    rules before calling this function.

    Idempotent: if a matching, manifest-verified acquisition already exists
    and ``force`` is ``False``, this is a safe no-op that returns a
    ``"skipped"`` outcome.
    """
    log = logger or logging.getLogger(__name__)
    destination = sources_root / source.id / sanitize_revision_for_filesystem(source.revision)
    manifest_path = manifest_path_for(manifests_dir, source.id, source.revision)

    manifest_exists = manifest_path.is_file()
    destination_exists = destination.is_dir()

    if not force and manifest_exists and destination_exists:
        manifest_data = read_manifest(manifest_path)
        ok, problems = verify_manifest_against_directory(manifest_data, destination)
        if ok:
            log.info(
                "Source %s already acquired at revision %s; skipping (use --force to reacquire).",
                source.id,
                source.revision,
            )
            return AcquisitionOutcome(
                source.id, "skipped", source.revision, destination, manifest_path,
                "already acquired and verified against its manifest",
            )
        raise AcquisitionError(
            f"Source {source.id!r}: existing acquisition at {destination} does not match its manifest "
            f"({'; '.join(problems)}). Use --force to reacquire."
        )

    if not force and destination_exists != manifest_exists:
        raise AcquisitionError(
            f"Source {source.id!r}: destination and manifest are in an inconsistent state "
            f"(destination exists: {destination_exists}, manifest exists: {manifest_exists}). "
            "Use --force to reacquire cleanly."
        )

    if dry_run:
        log.info(
            "[dry-run] would acquire source %s (%s) at revision %s",
            source.id,
            source.source_type,
            source.revision,
        )
        return AcquisitionOutcome(
            source.id, "dry_run", source.revision, destination, None, "dry run: no files were written"
        )

    started_at = datetime.now(timezone.utc)
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_destination = destination.parent / f".{source.id}.tmp-{uuid.uuid4().hex}"

    try:
        if source.source_type == "local_directory":
            resolved_revision = source.revision
            _acquire_local_directory(source, work_destination)
        elif source.source_type == "git_repository":
            resolved_revision = _acquire_git_repository(
                source, work_destination, timeout_seconds=timeout_seconds
            )
        elif source.source_type == "http_archive":
            resolved_revision = source.revision
            _acquire_http_archive(
                source,
                work_destination,
                downloads_dir=downloads_root,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
            )
        else:
            raise UnsupportedSourceTypeError(f"Unsupported source_type: {source.source_type!r}")

        fingerprints = fingerprint_directory(work_destination)
        file_records = tuple(FileManifestRecord.from_fingerprint(fp) for fp in fingerprints)
        completed_at = datetime.now(timezone.utc)
        record = AcquisitionRecord(
            started_at=started_at,
            completed_at=completed_at,
            tool_version=tool_version,
            forced=force,
            governance_override=governance_override,
            override_reason=override_reason,
        )
        manifest: SourceManifest = build_manifest(source, resolved_revision, file_records, record)

        if destination.exists():
            shutil.rmtree(destination)
        work_destination.replace(destination)

        written_manifest_path = write_manifest(manifest, manifests_dir)

        log.info(
            "Acquired source %s: %d file(s), %d byte(s).",
            source.id,
            manifest.file_count,
            manifest.total_bytes,
        )
        return AcquisitionOutcome(
            source.id, "completed", resolved_revision, destination, written_manifest_path,
            "acquisition completed",
        )
    except Exception:
        if destination.exists() and not manifest_path.is_file():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work_destination, ignore_errors=True)
