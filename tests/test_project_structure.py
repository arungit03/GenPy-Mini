"""Tests that verify the Phase 1 repository layout is present.

These tests only check that required files and directories exist. They do
not inspect the contents of ignored dataset directories beyond confirming
the directories (and their .gitkeep placeholders) exist.
"""

from __future__ import annotations

from config.settings import PROJECT_ROOT

_REQUIRED_DIRECTORIES = (
    "config",
    "data/raw",
    "data/raw/sources",
    "data/raw/downloads",
    "data/interim",
    "data/cleaned",
    "data/tokenized",
    "data/instructions",
    "data/evaluation",
    "data/reports",
    "data/manifests",
    "docs",
    "docs/decisions",
    "notebooks",
    "scripts",
    "src/genpy",
    "src/genpy/data",
    "tests",
    "tests/data",
    "tests/fixtures",
    "tests/fixtures/sample_source",
    ".github/workflows",
)

_REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements-dev.txt",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".env.example",
    ".pre-commit-config.yaml",
    "Makefile",
    "CONTRIBUTING.md",
    "LICENSE-NOTICE.md",
    "config/model_config.yaml",
    "config/settings.py",
    "config/__init__.py",
    "src/genpy/__init__.py",
    "src/genpy/config.py",
    "src/genpy/constants.py",
    "src/genpy/py.typed",
    "scripts/validate_environment.py",
    "docs/architecture.md",
    "docs/development.md",
    "docs/roadmap.md",
    "docs/data-governance.md",
    "docs/decisions/README.md",
    "docs/decisions/ADR-001-project-scope.md",
    "docs/decisions/ADR-002-dataset-acquisition-and-licensing.md",
    "docs/dataset-acquisition.md",
    "docs/dataset-source-template.md",
    "notebooks/README.md",
    ".github/workflows/quality.yml",
    "config/dataset_sources.yaml",
    "config/license_policy.yaml",
    "src/genpy/data/__init__.py",
    "src/genpy/data/acquisition.py",
    "src/genpy/data/checksums.py",
    "src/genpy/data/exceptions.py",
    "src/genpy/data/licenses.py",
    "src/genpy/data/manifests.py",
    "src/genpy/data/paths.py",
    "src/genpy/data/reporting.py",
    "src/genpy/data/schemas.py",
    "src/genpy/data/source_registry.py",
    "scripts/acquire_sources.py",
    "scripts/validate_sources.py",
    "scripts/generate_acquisition_report.py",
    "tests/fixtures/approved_source.yaml",
    "tests/fixtures/rejected_source.yaml",
    "tests/fixtures/sample_source/LICENSE",
    "tests/fixtures/sample_source/example.py",
    "tests/fixtures/sample_source/README.md",
)

_DATA_DIRECTORIES_WITH_GITKEEP = (
    "data/raw",
    "data/raw/sources",
    "data/raw/downloads",
    "data/interim",
    "data/cleaned",
    "data/tokenized",
    "data/instructions",
    "data/evaluation",
    "data/reports",
    "data/manifests",
)


def test_required_directories_exist() -> None:
    missing = [d for d in _REQUIRED_DIRECTORIES if not (PROJECT_ROOT / d).is_dir()]
    assert not missing, f"Missing required directories: {missing}"


def test_required_files_exist() -> None:
    missing = [f for f in _REQUIRED_FILES if not (PROJECT_ROOT / f).is_file()]
    assert not missing, f"Missing required files: {missing}"


def test_data_directories_have_gitkeep_placeholders() -> None:
    missing = [
        d for d in _DATA_DIRECTORIES_WITH_GITKEEP if not (PROJECT_ROOT / d / ".gitkeep").is_file()
    ]
    assert not missing, f"Missing .gitkeep in: {missing}"
