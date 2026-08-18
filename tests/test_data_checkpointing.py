import gzip
import json
from dataclasses import replace

import genpy.data.pipeline as pipeline_module
from genpy.data.pipeline import run_pipeline
from tests.test_data_pipeline import _config, _range_source


def _documents(output):
    documents = []
    for path in sorted(output.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            documents.extend(json.loads(line) for line in handle)
    return sorted(
        documents,
        key=lambda document: (document["split"], document["content_hash"], document["doc_id"]),
    )


def test_periodic_state_checkpointing_is_bounded_and_final_state_is_complete(tmp_path, monkeypatch):
    base = _config(tmp_path)
    output_config = replace(base.output, shard_max_documents=100)
    config = replace(base, output=output_config)
    state_writes = []
    original_write_json = pipeline_module._write_json

    def record_write(path, value):
        if path.name == "prepare-state.json":
            state_writes.append(value["source_documents_seen"])
        return original_write_json(path, value)

    monkeypatch.setattr(pipeline_module, "_write_json", record_write)
    result = run_pipeline(
        config,
        source=_range_source(7),
        output_dir=tmp_path / "processed",
        state_checkpoint_interval=3,
    )
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert state_writes == [3, 6, 7, 7]
    assert state["state_checkpoint_interval"] == 3
    assert state["completion_status"] == "complete"
    assert state["statistics"] == result.stats.to_dict()
    assert manifest["completion_status"] == "complete"
    assert manifest["statistics"] == result.stats.to_dict()


def test_interrupted_resume_matches_uninterrupted_corpus_and_statistics(tmp_path):
    base = _config(tmp_path)
    source = _range_source(12)
    # Include a duplicate and a rejected row so both accounting paths are
    # compared, not only accepted documents.
    source[5] = dict(source[2])
    source[8] = {"id": "rejected", "text": "too short"}
    config = replace(base, output=replace(base.output, shard_max_documents=2))
    full = run_pipeline(
        config,
        source=source,
        skip_documents=2,
        max_documents=8,
        output_dir=tmp_path / "full",
        state_checkpoint_interval=3,
    )

    def interrupted_source():
        yield from source[:6]
        raise RuntimeError("simulated interruption")

    interrupted_output = tmp_path / "resumed"
    try:
        run_pipeline(
            config,
            source=interrupted_source(),
            skip_documents=2,
            max_documents=8,
            output_dir=interrupted_output,
            state_checkpoint_interval=3,
        )
    except RuntimeError as error:
        assert "simulated" in str(error)
    else:
        raise AssertionError("interrupted source did not fail")

    resumed = run_pipeline(
        config,
        source=source,
        skip_documents=2,
        max_documents=8,
        output_dir=interrupted_output,
        resume=True,
        state_checkpoint_interval=3,
    )
    assert resumed.completed
    assert resumed.stats.to_dict() == full.stats.to_dict()
    assert _documents(resumed.output_dir) == _documents(full.output_dir)
    state = json.loads(resumed.state_path.read_text(encoding="utf-8"))
    assert state["skip_documents"] == 2
    assert state["source_documents_seen"] == 8
    assert state["completion_status"] == "complete"
