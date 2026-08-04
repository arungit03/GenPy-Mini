# Environment Setup

Python 3.11 is the primary supported Python version for GenPy.

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/check_environment.py
pytest
```

Phase 3 uses Hugging Face `tokenizers` only as the implementation library for GenPy's own
empty-state byte-level BPE training. Local verification used `tokenizers` 0.23.1 on Windows in
CPU-only mode. No pretrained tokenizer package or network access is required.

The local Windows computer may not have a CUDA GPU. This is expected. The environment
check should report CPU-only mode instead of failing.

## Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/check_environment.py
pytest
```

## Kaggle

```bash
python -m pip install --upgrade pip
pip install -r requirements-kaggle.txt
python scripts/check_environment.py
pytest
```

Kaggle GPU availability must be checked through PyTorch:

```python
import torch

print(torch.cuda.is_available())
```

Training scripts in later phases must handle missing CUDA clearly and must not assume that
a GPU is available until PyTorch confirms it.

Tokenizer preparation, training, validation, evaluation, counting, and packaging are CPU-only:

```bash
python scripts/tokenizer/check_readiness.py --config configs/tokenizer/genpy_bpe_16k.yaml
python scripts/tokenizer/train_tokenizer.py --config configs/tokenizer/smoke_tokenizer.yaml --mode smoke
python scripts/tokenizer/validate_tokenizer.py --artifact artifacts/tokenizer/smoke
```

Phase 4 also runs without CUDA and adds no dependency beyond the existing PyTorch and NumPy
installations:

```bash
python scripts/model/count_parameters.py --config configs/model/genpy_100m.yaml
python scripts/data/estimate_packing.py --config configs/data/packing.yaml
python scripts/model/smoke_forward.py --config configs/model/smoke_model.yaml
```

Packed arrays use NumPy memory mapping. Project-created smoke model states use PyTorch tensor
state dictionaries with `weights_only=True` on load; arbitrary downloaded pickle files must never
be loaded.
