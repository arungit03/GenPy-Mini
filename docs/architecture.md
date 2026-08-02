# GenPy-Mini Architecture (Planned)

**Status: planned. Nothing below this line beyond the "Phase 1 (implemented)"
section is implemented yet.**

This document describes the intended end-to-end system. It exists so the
Phase 1 configuration and repository structure can be designed against a
coherent target, not because any of these components exist.

## High-level flow (planned)

```text
Licensed Python source code
        |
        v
Cleaning and governance
        |
        v
Python-focused tokenizer
        |
        v
Tokenized training sequences
        |
        v
Decoder-only Transformer
        |
        v
Base pretraining
        |
        v
Instruction tuning
        |
        v
Evaluation
        |
        v
Inference service
        |
        v
Secure code-execution sandbox
        |
        v
Web application
```

Every stage after "Licensed Python source code" is **planned, not
implemented**. See [docs/roadmap.md](roadmap.md) for the phase that will
introduce each one and its acceptance gate.

## Target model architecture (data only, Phase 1)

The values below are stored in [`config/model_config.yaml`](../config/model_config.yaml)
and loaded through [`src/genpy/config.py`](../src/genpy/config.py). They are
consumed by no other code yet — Phase 1 only guarantees the values are
present, internally consistent, and importable.

| Parameter | Value |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Task | Causal language modeling |
| Language scope | Python, English programming instructions |
| Vocabulary size | 16,000 |
| Context length | 512 tokens |
| Model dimension (`d_model`) | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 2,048 |
| Dropout | 0.1 |
| Tied input/output embeddings | Yes |
| Target parameter count | 33,000,000 – 35,000,000 |

## Phase 1 (implemented)

The only executable pieces that exist today are:

- `src/genpy/config.py` — typed, immutable configuration objects, a YAML
  loader, and architecture consistency validation.
- `config/settings.py` — project path constants and safe environment
  variable helpers.
- `scripts/validate_environment.py` — a local machine / repository health
  check.

No tokenizer, no model layer, no training loop, no inference path, no
sandbox, and no web application exist in this repository yet.

## Environment split

| Responsibility | Local machine | Kaggle |
| --- | --- | --- |
| Coding, docs, config | Yes | No |
| Unit tests / CPU smoke tests | Yes | No |
| Dataset preprocessing (future) | Yes | No |
| GPU pretraining / fine-tuning (future) | No | Yes |
| Checkpoint storage between sessions (future) | Portable target | Portable target |

Local development never requires CUDA. GPU training work happens on Kaggle
notebooks in a later phase.
