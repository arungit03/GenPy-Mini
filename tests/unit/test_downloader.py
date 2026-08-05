from __future__ import annotations

import hashlib

from genpy.data.downloader import download_file, iter_source_files


class _PartialResponse:
    status = 206

    def __init__(self) -> None:
        self._chunks = iter((b"def", b""))

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *args):  # type: ignore[no-untyped-def]
        return None

    def read(self, _size: int) -> bytes:
        return next(self._chunks)


def test_resumable_download_appends_partial_content(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    destination = tmp_path / "archive.zip"
    partial = destination.with_suffix(".zip.part")
    partial.write_bytes(b"abc")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _PartialResponse())
    size, digest, resumed = download_file(
        "https://example.invalid/archive.zip", destination, maximum_bytes=100
    )
    assert destination.read_bytes() == b"abcdef"
    assert size == 6
    assert digest == hashlib.sha256(b"abcdef").hexdigest()
    assert resumed


def test_iter_source_files_applies_source_specific_exclude_globs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text("value = 1\n", encoding="utf-8")
    vendored = tmp_path / "pkg" / "vendored_thirdparty"
    vendored.mkdir()
    (vendored / "helper.py").write_text("value = 2\n", encoding="utf-8")

    without_exclusion = sorted(
        path.name for path in iter_source_files(tmp_path, ("*.py",), set(), None)
    )
    assert without_exclusion == ["core.py", "helper.py"]

    with_exclusion = sorted(
        path.name
        for path in iter_source_files(
            tmp_path, ("*.py",), set(), None, exclude_globs=("pkg/vendored_thirdparty/*",)
        )
    )
    assert with_exclusion == ["core.py"]
