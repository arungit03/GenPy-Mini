import json
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
