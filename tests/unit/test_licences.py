from __future__ import annotations

from pathlib import Path

from genpy.data.licences import LicencePolicy


def test_licence_allowlist_and_deny_behavior() -> None:
    policy = LicencePolicy.from_yaml(Path("configs/data/licenses.yaml"))
    assert policy.decision("MIT", provenance_available=True) == (True, None)
    assert policy.decision("GPL-3.0", provenance_available=True)[0] is False
    assert policy.decision("unknown", provenance_available=True)[0] is False
    assert policy.decision("MIT", provenance_available=False) == (False, "missing_provenance")
