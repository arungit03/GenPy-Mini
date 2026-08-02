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
    "data/interim",
    "data/cleaned",
    "data/tokenized",
    "data/instructions",
    "data/evaluation",
    "data/reports",
    "docs",
    "docs/decisions",
    "notebooks",
    "scripts",
    "src/genpy",
    "tests",
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
    "notebooks/README.md",
    ".github/workflows/quality.yml",
)

_DATA_DIRECTORIES_WITH_GITKEEP = (
    "data/raw",
    "data/interim",
    "data/cleaned",
    "data/tokenized",
    "data/instructions",
    "data/evaluation",
    "data/reports",
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
