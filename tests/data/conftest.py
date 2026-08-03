"""Shared fixtures and dict factories for Phase 2 dataset governance tests."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_SOURCE_DIR = FIXTURES_DIR / "sample_source"


def base_license_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "declared_spdx": "MIT",
        "license_file": "LICENSE",
        "attribution_required": True,
        "redistribution_allowed": True,
        "commercial_use_allowed": True,
        "modifications_allowed": True,
        "notes": "Test fixture license.",
    }
    data.update(overrides)
    return data


def base_governance_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "reviewed_by": "Test Suite",
        "reviewed_on": "2026-08-01",
        "approval_status": "approved",
        "approval_notes": "Approved for tests.",
    }
    data.update(overrides)
    return data


def base_acquisition_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "expected_sha256": None,
        "maximum_download_bytes": 1_048_576,
        "maximum_extracted_bytes": 1_048_576,
        "include_submodules": False,
        "shallow_clone": True,
    }
    data.update(overrides)
    return data


def base_source_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": "sample-source",
        "name": "Sample Source",
        "enabled": True,
        "source_type": "local_directory",
        "location": str(SAMPLE_SOURCE_DIR),
        "description": "A sample source used for tests.",
        "revision": "fixture-v1",
        "license": base_license_dict(),
        "governance": base_governance_dict(),
        "acquisition": base_acquisition_dict(),
        "tags": ["python", "fixture"],
    }
    data.update(overrides)
    return data


def base_defaults_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "enabled": False,
        "timeout_seconds": 60,
        "retry_count": 3,
        "maximum_download_bytes": 5_368_709_120,
        "maximum_extracted_bytes": 10_737_418_240,
        "require_pinned_revision": True,
        "require_license_metadata": True,
        "require_checksum_for_http_archives": True,
    }
    data.update(overrides)
    return data


def base_registry_dict(
    sources: list[dict[str, Any]] | None = None, **defaults_overrides: Any
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "defaults": base_defaults_dict(**defaults_overrides),
        "sources": sources if sources is not None else [],
    }


def base_policy_dict(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "schema_version": 1,
        "disclaimer": "Not legal advice; a human must verify every source before approval.",
        "default_status": "review_required",
        "allowed": ["MIT", "Apache-2.0", "BSD-3-Clause"],
        "review_required": ["MPL-2.0"],
        "blocked": ["GPL-3.0-only", "AGPL-3.0-only"],
    }
    data.update(overrides)
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


RegistryPathFactory = Callable[..., Path]
PolicyPathFactory = Callable[..., Path]


@pytest.fixture
def registry_path_factory(tmp_path: Path) -> RegistryPathFactory:
    def _make(sources: list[dict[str, Any]] | None = None, **defaults_overrides: Any) -> Path:
        return write_yaml(tmp_path / "dataset_sources.yaml", base_registry_dict(sources, **defaults_overrides))

    return _make


@pytest.fixture
def policy_path_factory(tmp_path: Path) -> PolicyPathFactory:
    def _make(**overrides: Any) -> Path:
        return write_yaml(tmp_path / "license_policy.yaml", base_policy_dict(**overrides))

    return _make


@dataclasses.dataclass(frozen=True)
class LocalGitRepo:
    """A tiny local git repository with two commits, for git-acquisition tests."""

    path: Path
    first_commit: str
    second_commit: str


@pytest.fixture
def local_git_repo(tmp_path: Path) -> LocalGitRepo:
    if shutil.which("git") is None:
        pytest.skip("git is not available on PATH")

    repo_dir = tmp_path / "git-origin"
    repo_dir.mkdir()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
            env=env,
        )
        return result.stdout.strip()

    run("init", "--quiet", "--initial-branch=main")
    run("config", "user.email", "test@example.invalid")
    run("config", "user.name", "Test Suite")

    (repo_dir / "first.py").write_text("value = 1\n", encoding="utf-8")
    run("add", "first.py")
    run("commit", "--quiet", "-m", "first commit")
    first_commit = run("rev-parse", "HEAD")

    (repo_dir / "second.py").write_text("value = 2\n", encoding="utf-8")
    run("add", "second.py")
    run("commit", "--quiet", "-m", "second commit")
    second_commit = run("rev-parse", "HEAD")

    return LocalGitRepo(path=repo_dir, first_commit=first_commit, second_commit=second_commit)
