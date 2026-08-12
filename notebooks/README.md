# Kaggle preparation

GenPy will later be imported into Kaggle by cloning this repository or uploading the project as a dataset. From a Kaggle notebook, the planned workflow is:

```bash
git clone <repository-url>
cd GenPy
pip install -r requirements.txt
python scripts/check_environment.py
```

In the Kaggle notebook settings, select the GPU accelerator before starting a serious run. The environment checker reports whether PyTorch sees CUDA, the CUDA/cuDNN versions, GPU count, GPU memory, and the selected GenPy device. A successful GPU check should show `CUDA available: True` and a CUDA selected device.

This document is preparation guidance only. Actual training notebooks will be created in later project steps; no training notebook or dataset is part of Step 1.
