from __future__ import annotations

from genpy.data.pii_scan import scan_pii
from genpy.data.pipeline import scanner_false_positive_control
from genpy.data.secret_scan import scan_secrets


def test_fake_secret_detection_returns_category_only() -> None:
    text = 'api_key = "FAKE_TEST_KEY_1234567890_NOT_REAL"'
    result = scan_secrets(text)
    assert "generic_api_key" in result
    assert all("FAKE_TEST" not in category for category in result)


def test_fictional_pii_detection() -> None:
    result = scan_pii('contact = "casey.fixture@example.invalid"')
    assert result == ("email_address",)


def test_fictional_safe_control_false_positive_rate() -> None:
    result = scanner_false_positive_control()
    assert result["safe_control_records"] == 5
    assert result["false_positive_records"] == 0
