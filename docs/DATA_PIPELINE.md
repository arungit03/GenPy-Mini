# GenPy dataset pipeline

Step 2 prepares a text-only corpus for later tokenizer training. The configured source is `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`, split `train`, loaded with `streaming=True`. The source license and terms must be reviewed and respected before any large preparation run.

## Pipeline

```text
stream → clean → filter → exact dedup → deterministic split → shard → manifest/state
```

The source loader is lazy and never collects the upstream stream into a Python list. `--max-documents` is available for smoke tests. Local JSONL fixtures are used for offline tests.

## Cleaning and filtering

Text is normalized to Unicode NFC, CRLF/CR line endings become LF, unwanted control characters are removed while newline and tab are preserved, trailing spaces are removed, repeated ordinary spaces are collapsed, and excessive blank lines are reduced. Paragraphs, punctuation, symbols, Unicode, and code-like text are preserved. Text is stripped at the boundaries.

Documents are rejected when the configured text field is missing, is not a string, is empty after normalization, is shorter than `min_chars`, or is longer than `max_chars`. Each rejection has a reason. No classifier, pretrained model, stemming, lowercasing, or undocumented quality heuristic is used.

## Deduplication and splitting

Exact deduplication hashes normalized UTF-8 text with SHA-256 and retains the first accepted document. Later identical hashes are counted as duplicates. No fuzzy, semantic, MinHash, SimHash, or embedding deduplication is performed.

Train/validation assignment hashes the content hash together with the split seed. The default validation fraction is `0.005` and the default seed is `42`, making the decision deterministic for the same content and configuration.

## Output schema and shards

Accepted documents are written as UTF-8 compressed JSON Lines with deterministic names such as `train-00000.jsonl.gz` and `validation-00000.jsonl.gz`. Shards default to 25,000 documents and are finalized atomically through a `.tmp` file. Each row contains:

`doc_id`, `text`, `content_hash`, `source_dataset`, `source_config`, `source_url`, `source_dump`, `language`, `quality_score`, `char_count`, `byte_count`, and `split`.

Manifests record the pipeline version, timestamp, complete configuration, tokenizer-independent statistics, shard names, completion status, source range, and state-checkpoint policy. A machine-readable `prepare-state.json` stores counters, hashes, completed shards, and a configuration fingerprint.

## Resume behavior

Run preparation with `--resume` to continue a compatible incomplete run. The pipeline rejects changes to the dataset, cleaning thresholds, split settings, output configuration, or other fingerprinted configuration. It skips already-seen source rows while reconstructing a streaming source, so resume is not O(1) when the upstream source cannot seek. Valid existing shards are never silently overwritten; interrupted temporary files are discarded and completed partial shards remain valid.

Before the large-run checkpointing change, `run_pipeline()` serialized
`prepare-state.json` once after every source document, including rejected and
duplicate rows, followed by finalization writes. A 250,000-document run could
therefore perform roughly 250,000 large JSON serializations containing the
growing deduplication hash set. The default policy now checkpoints every 500
selected source documents, using `--state-checkpoint-interval N` to choose
another positive interval. State is also persisted whenever a shard is
finalized, in the `finally` path, and after the final manifest is written.
This bounds normal checkpoint lag to at most `N - 1` selected source
documents while preserving exact rejected, duplicate, accepted, and split
statistics at each persisted checkpoint. Clean completion always writes both
the final state and manifest.

## Tokenization boundary

No tokenization occurs in Step 2. There are no token IDs and no GenPy token counts. GenPy tokenizer training and exact GenPy token statistics belong to Step 3.

## Data limitations

Web-derived datasets can contain factual inaccuracies, bias, low-quality examples, harmful material, and duplicated or malformed content despite upstream filtering. This pipeline is transparent and conservative; it does not claim that the source data is perfect or fully safe.

## Useful commands

```bash
python scripts/inspect_dataset.py --config configs/data.yaml --limit 3
python scripts/prepare_data.py --config configs/data.yaml --max-documents 1000
python scripts/validate_data.py --processed-dir data/processed
```

For offline smoke tests, pass `--source-jsonl tests/fixtures/sample_documents.jsonl` to the inspection or preparation script.
