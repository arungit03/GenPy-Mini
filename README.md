# GenPy

GenPy is a from-scratch decoder-only Transformer project intended to become a roughly 200M parameter Python-specialized code-generation language model. It will be trained from randomly initialized weights; no pretrained model weights are part of the project.

## Project status

Checkpoint 4 is complete. The repository contains a reproducible 100,000-example Python instruction dataset, a train-only 32,000-entry custom Byte-Level BPE tokenizer, and the native randomly initialized GenPy-200M Transformer. Checkpoint 5 local verification is complete; CUDA production verification remains pending. Production model training remains future work.

## Architecture target

GenPy-200M is planned as a 24-layer decoder-only Transformer with 768 hidden dimensions, 12 attention heads, 64 dimensions per head, 2176 SwiGLU intermediate dimensions, a 32,000-token vocabulary, 1024-token context, RoPE, RMSNorm, bias-free linear layers, and tied token embeddings/output weights. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Roadmap

1. Checkpoint 1 - Project Setup - COMPLETE
2. Checkpoint 2 - Python Dataset Pipeline - COMPLETE
3. Checkpoint 2.5 - Production Python Dataset - COMPLETE
4. Checkpoint 3 - Custom Byte-Level BPE Tokenizer - COMPLETE
5. Checkpoint 4 - GenPy-200M Transformer - COMPLETE
6. Checkpoint 5 - Deep Model Verification - LOCAL_COMPLETE_GPU_PENDING
7. Checkpoint 6 - Production Training Engine - NOT STARTED
8. Checkpoint 7 - Python Pretraining - NOT STARTED
9. Checkpoint 8 - Instruction Tuning and Evaluation - NOT STARTED

## Repository structure

```text
configs/       Canonical YAML configuration
genpy/         Python package, tokenizer, utilities, and data pipeline
scripts/       Developer, dataset, tokenizer, and environment scripts
tests/         Pytest suite
data/          Runtime datasets and tokenizer corpus
artifacts/     Locally generated tokenizer artifacts
reports/       Verification reports
docs/          Design and checkpoint documentation
```

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -e .
pytest -q
```

## Tokenizer

See [docs/TOKENIZER.md](docs/TOKENIZER.md). The production artifact is `artifacts/tokenizer/genpy-32k/`; it uses PAD=0, BOS=1, EOS=2, and UNK=3 and is trained from the approved train split only.
