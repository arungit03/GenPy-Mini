import json
import gzip
from dataclasses import replace
from pathlib import Path

from genpy.config import load_data_config
from genpy.data.pipeline import run_pipeline
from genpy.data.source import load_jsonl_rows
from genpy.data.validation import validate_dataset


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    base = load_data_config(ROOT / "configs" / "data.yaml")
    processing = replace(base.processing, min_chars=30, max_chars=5000)
    split = replace(base.split, validation_fraction=0.5, seed=42)
    output = replace(base.output, shard_max_documents=2)
    return replace(base, processing=processing, split=split, output=output)


def test_offline_pipeline_end_to_end(tmp_path):
    config = _config(tmp_path)
    source = list(load_jsonl_rows(ROOT / "tests" / "fixtures" / "sample_documents.jsonl"))
    result = run_pipeline(config, source=source, output_dir=tmp_path / "processed")
    assert result.completed
    assert result.stats.source_documents_seen == len(source)
    assert result.stats.accepted_documents >= 2
    assert result.stats.duplicate_documents == 1
    assert result.stats.rejected_documents >= 3
    assert result.manifest_path.is_file()
    report = validate_dataset(result.output_dir, result.manifest_path)
    assert report["valid"], report["errors"]
    assert report["train_documents"] + report["validation_documents"] == result.stats.accepted_documents


def test_pipeline_resume_skips_completed_source_rows(tmp_path):
    config = _config(tmp_path)
    source = list(load_jsonl_rows(ROOT / "tests" / "fixtures" / "sample_documents.jsonl"))

    def interrupted_source():
        for row in source[:3]:
            yield row
        raise RuntimeError("simulated interruption")

    output = tmp_path / "processed"
    try:
        run_pipeline(config, source=interrupted_source(), output_dir=output)
    except RuntimeError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("interrupted source did not fail")
    resumed = run_pipeline(config, source=source, output_dir=output, resume=True)
    assert resumed.completed
    assert resumed.stats.source_documents_seen == len(source)
    report = validate_dataset(output, resumed.manifest_path)
    assert report["valid"], report["errors"]


def test_incompatible_resume_is_rejected(tmp_path):
    config = _config(tmp_path)
    source = [{"text": "A sufficiently long fixture document for resume compatibility testing." * 2}]
    output = tmp_path / "processed"
    run_pipeline(config, source=source, output_dir=output)
    incompatible = replace(config, processing=replace(config.processing, min_chars=31))
    try:
        run_pipeline(incompatible, source=source, output_dir=output, resume=True)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("incompatible resume was accepted")


def _range_source(count):
    return [
        {
            "id": str(index),
            "text": f"Source document {index} contains enough deterministic text for testing." * 2,
        }
        for index in range(count)
    ]


def _written_ids(output):
    ids = []
    for path in output.glob("*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            ids.extend(json.loads(line)["doc_id"] for line in handle)
    return sorted(ids, key=int)


def test_skip_zero_preserves_existing_range_and_records_provenance(tmp_path):
    config = _config(tmp_path)
    source = _range_source(4)
    result = run_pipeline(config, source=source, max_documents=2, output_dir=tmp_path / "processed")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.stats.source_documents_seen == 2
    assert _written_ids(result.output_dir) == ["0", "1"]
    assert manifest["skip_documents"] == 0
    assert manifest["source_start_index"] == 0
    assert manifest["requested_source_documents"] == 2
    assert manifest["source_end_index_exclusive"] == 2


def test_skip_documents_ignores_rows_before_selected_range(tmp_path):
    config = _config(tmp_path)
    result = run_pipeline(config, source=_range_source(5), skip_documents=3, output_dir=tmp_path / "processed")
    assert result.stats.source_documents_seen == 2
    assert _written_ids(result.output_dir) == ["3", "4"]


def test_max_documents_applies_after_skip(tmp_path):
    config = _config(tmp_path)
    result = run_pipeline(
        config,
        source=_range_source(10),
        skip_documents=3,
        max_documents=2,
        output_dir=tmp_path / "processed",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.stats.source_documents_seen == 2
    assert _written_ids(result.output_dir) == ["3", "4"]
    assert manifest["source_start_index"] == 3
    assert manifest["requested_source_documents"] == 2
    assert manifest["source_end_index_exclusive"] == 5


def test_resume_continues_within_selected_source_range(tmp_path):
    config = _config(tmp_path)
    source = _range_source(8)
    output = tmp_path / "processed"

    def interrupted_source():
        yield from source[:5]
        raise RuntimeError("simulated interruption")

    try:
        run_pipeline(config, source=interrupted_source(), skip_documents=3, output_dir=output)
    except RuntimeError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("interrupted source did not fail")

    resumed = run_pipeline(config, source=source, skip_documents=3, output_dir=output, resume=True)
    assert resumed.completed
    assert resumed.stats.source_documents_seen == 5
    assert _written_ids(output) == ["3", "4", "5", "6", "7"]


def test_resume_rejects_changed_skip_documents(tmp_path):
    config = _config(tmp_path)
    source = _range_source(5)
    output = tmp_path / "processed"
    run_pipeline(config, source=source, skip_documents=3, output_dir=output)
    try:
        run_pipeline(config, source=source, skip_documents=0, output_dir=output, resume=True)
    except ValueError as exc:
        assert "skip_documents" in str(exc)
    else:
        raise AssertionError("incompatible skip_documents was accepted")
