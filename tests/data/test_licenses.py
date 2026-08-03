"""Tests for license policy loading and evaluation."""

from __future__ import annotations

from pathlib import Path

import pytest

from genpy.data.exceptions import LicensePolicyError
from genpy.data.licenses import load_license_policy
from tests.data.conftest import base_policy_dict, write_yaml


def test_allowed_spdx_identifier_evaluates_to_allowed(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict())
    policy = load_license_policy(path)
    assert policy.evaluate("MIT") == "allowed"


def test_review_required_spdx_identifier(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict())
    policy = load_license_policy(path)
    assert policy.evaluate("MPL-2.0") == "review_required"


def test_blocked_spdx_identifier(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict())
    policy = load_license_policy(path)
    assert policy.evaluate("GPL-3.0-only") == "blocked"


def test_unknown_spdx_identifier_falls_back_to_default_status(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict(default_status="review_required"))
    policy = load_license_policy(path)
    assert policy.evaluate("Some-Made-Up-License-9000") == "review_required"


def test_missing_license_defaults_conservatively(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict(default_status="blocked"))
    policy = load_license_policy(path)
    assert policy.evaluate("") == "blocked"
    assert policy.evaluate("totally-unlisted") == "blocked"


def test_policy_rejects_identifier_in_multiple_lists(tmp_path: Path) -> None:
    data = base_policy_dict(allowed=["MIT"], review_required=["MIT"])
    path = write_yaml(tmp_path / "policy.yaml", data)
    with pytest.raises(LicensePolicyError):
        load_license_policy(path)


def test_policy_rejects_invalid_default_status(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict(default_status="maybe"))
    with pytest.raises(LicensePolicyError):
        load_license_policy(path)


def test_policy_rejects_missing_required_field(tmp_path: Path) -> None:
    data = base_policy_dict()
    del data["blocked"]
    path = write_yaml(tmp_path / "policy.yaml", data)
    with pytest.raises(LicensePolicyError, match="blocked"):
        load_license_policy(path)


def test_policy_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    data = base_policy_dict()
    data["extra_unexpected_field"] = True
    path = write_yaml(tmp_path / "policy.yaml", data)
    with pytest.raises(LicensePolicyError, match="extra_unexpected_field"):
        load_license_policy(path)


def test_policy_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LicensePolicyError, match="not found"):
        load_license_policy(tmp_path / "does-not-exist.yaml")


def test_policy_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("allowed: [unclosed", encoding="utf-8")
    with pytest.raises(LicensePolicyError, match="YAML"):
        load_license_policy(path)


def test_policy_rejects_blank_entry_in_a_list(tmp_path: Path) -> None:
    data = base_policy_dict(allowed=["MIT", "   "])
    path = write_yaml(tmp_path / "policy.yaml", data)
    with pytest.raises(LicensePolicyError):
        load_license_policy(path)


def test_policy_requires_non_empty_disclaimer(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "policy.yaml", base_policy_dict(disclaimer=""))
    with pytest.raises(LicensePolicyError):
        load_license_policy(path)


def test_real_project_license_policy_loads() -> None:
    """The actual config/license_policy.yaml shipped with the repo must be valid."""
    from config.settings import CONFIG_DIR

    policy = load_license_policy(CONFIG_DIR / "license_policy.yaml")
    assert policy.evaluate("MIT") == "allowed"
    assert policy.evaluate("GPL-3.0-only") == "blocked"
