"""Train the GenPy Byte-Level BPE tokenizer in smoke or production mode."""

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

try:
    from ._bootstrap import ensure_project_root
except ImportError:
    from _bootstrap import ensure_project_root
ensure_project_root()

from genpy.config import load_model_config, load_tokenizer_config, validate_tokenizer_vocab_contract
from genpy.tokenizer.corpus import find_step2_manifest, inspect_corpus_gate
from genpy.tokenizer.trainer import train_from_corpus
from genpy.tokenizer.validation import validate_tokenizer_artifact


def _smoke(config):
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tokenizer_corpus.jsonl.gz"
    smoke_tokenizer = replace(config.tokenizer, vocab_size=512)
    smoke_data = replace(config.training_data, input_dir=str(fixture.parent), train_pattern=fixture.name)
    smoke_config = replace(config, tokenizer=smoke_tokenizer, training_data=smoke_data)
    with tempfile.TemporaryDirectory(prefix="genpy-tokenizer-smoke-") as temporary:
        result = train_from_corpus(smoke_config, fixture.parent, Path(temporary), mode="smoke", vocab_size=512)
        validate_tokenizer_artifact(result["tokenizer_path"], expected_vocab_size=512)
        reloaded = result["tokenizer"].from_file(result["tokenizer_path"])
        assert reloaded.encode("GenPy smoke café") == result["tokenizer"].encode("GenPy smoke café")
        print("Smoke tokenizer: trained")
        print(f"Vocabulary: {reloaded.vocab_size}")
        print(f"Special IDs: {reloaded.pad_token_id}, {reloaded.bos_token_id}, {reloaded.eos_token_id}, {reloaded.unk_token_id}")
        print("Save/reload: PASS")
        print("Round-trip: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tokenizer.yaml")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.production == args.smoke:
        parser.error("choose exactly one of --production or --smoke")
    config = load_tokenizer_config(args.config)
    if args.smoke:
        return _smoke(config)
    validate_tokenizer_vocab_contract(Path(args.config).parent / "model_200m.yaml", config)
    input_dir = args.input_dir or Path(config.training_data.input_dir)
    output_dir = args.output_dir or Path(config.output.output_dir)
    tokenizer_path = output_dir / config.output.tokenizer_file
    if tokenizer_path.exists() and not args.force:
        raise FileExistsError(f"Production tokenizer exists; pass --force to overwrite: {tokenizer_path}")
    print(f"Tokenizer: {config.tokenizer.name}")
    print(f"Vocabulary: {config.tokenizer.vocab_size}")
    print(f"Input directory: {input_dir}")
    print("Mode: production")
    source_manifest = args.source_manifest or find_step2_manifest(input_dir.parent / "manifests")
    if source_manifest is None:
        raise RuntimeError("Production training requires a completed Step 2 manifest; no real corpus provenance was found")
    gate = inspect_corpus_gate(input_dir, config.training_data.train_pattern, config.training_data.text_field, config.training.production_minimum_bytes, config.training.production_target_bytes)
    print(f"Train shards found: {len(gate['train_shards'])}")
    print(f"UTF-8 bytes available: {gate['utf8_bytes']}")
    if gate["classification"] == "development":
        raise RuntimeError(f"Production corpus gate failed: {gate['utf8_bytes']} bytes available, minimum is {config.training.production_minimum_bytes}")
    result = train_from_corpus(config, input_dir, output_dir, mode="production", max_documents=args.max_documents, max_bytes=args.max_bytes, source_manifest=args.source_manifest)
    validate_tokenizer_artifact(result["tokenizer_path"], result["manifest_path"], expected_vocab_size=32000, model_config_path=Path(args.config).parent / "model_200m.yaml")
    print("Production tokenizer: trained and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
