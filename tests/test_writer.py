import gzip
import json

from genpy.data.schema import GenPyDocument
from genpy.data.writer import DocumentShardWriter


def _document(index, split):
    return GenPyDocument(
        doc_id=str(index), text=f"text-{index}", content_hash=f"hash-{index}",
        source_dataset="fixture", source_config="local", source_url=None,
        source_dump=None, language=None, quality_score=None, char_count=len(f"text-{index}"),
        byte_count=len(f"text-{index}"), split=split,
    )


def test_writer_rollover_and_gzip_jsonl(tmp_path):
    writer = DocumentShardWriter(tmp_path, shard_max_documents=2)
    for index in range(3):
        writer.write(_document(index, "train"))
    writer.write(_document(3, "validation"))
    files = writer.close()
    assert files == ["train-00000.jsonl.gz", "train-00001.jsonl.gz", "validation-00000.jsonl.gz"]
    assert not list(tmp_path.glob("*.tmp"))
    with gzip.open(tmp_path / "train-00000.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 2


def test_writer_rejects_write_after_close(tmp_path):
    writer = DocumentShardWriter(tmp_path, shard_max_documents=2)
    writer.close()
    try:
        writer.write(_document(1, "train"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed writer accepted a document")
