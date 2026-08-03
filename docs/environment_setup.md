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
