# Disclaimer 
**Work in Progress — Not yet fully validated**
This repository is currently being extracted from a larger research project. The process is not yet complete, so the code has not been fully tested and validated. A stable, reviewed version will be available shortly.

# CadToSeq

**Predicts manufacturing process sequences directly from CAD geometry — using a Transformer decoder.**

Given a set of geometry feature vectors `[1024, 32]` (VecSet embeddings), CadToSeq generates an ordered sequence of manufacturing steps such as:

> *fräsen → bohren → schweißen → schleifen → STOP*

> **Note on geometry embeddings:** VecSet embeddings are produced from surface point clouds using the [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) encoder. That preprocessing step is **not part of this repository** — CadToSeq takes pre-computed `vecset.npy` files as input. See [Preparing your data](#preparing-your-data) below.

---

## How it works

1. A CAD part's geometry is encoded externally into a compact set of vectors (`vecset.npy`).
2. The Transformer decoder reads these vectors and autoregressively predicts manufacturing steps.
3. Generation stops when the model emits a `STOP` token.

---

## Getting started

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- A CUDA-capable GPU is strongly recommended for training

### Installation

```bash
uv sync
source .venv/bin/activate
```

---

## Preparing your data

The current configuration expects each sample its own directory:

```
<data_dir>/
    <part_id>/
        features/
            vecset.npy   # geometry embedding [1024, 32], float32
        plan.csv         # process plan (Schritt;Dauer[min];Kosten[($)])
```

**FabriCAD data:** Download the [FabriCAD](https://github.com/CIMTT-Kiel/cad-api-client) dataset and generate VecSet embeddings with the [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) encoder. Then set `paths.data_dir` in `config.yaml` to point at the dataset root.

**Synthetic test data:** To try out the pipeline without real data, generate trivial random samples:

```bash
python scripts/create_test_data.py            # creates 200 samples in ./test_data
python scripts/create_test_data.py --data_dir /tmp/mydata --n_samples 500
```

Then update `config.yaml`:

```yaml
paths:
  data_dir: ./test_data
```

---

## Training

All hyperparameters are controlled via `config.yaml`. Start training with:

```bash
# Train with the best-known hyperparameters from config.yaml:
python scripts/train.py

# Run Optuna hyperparameter search first, then final training:
python scripts/train.py --tuning=True

# Override individual hyperparameters on the fly:
python scripts/train.py --lr=0.001 --embed_dim=96 --num_layers=3
```

---

## Inference

**From the command line:**

```bash
python scripts/infer.py --vecset path/to/features/vecset.npy
```

**From Python:**

```python
import torch
import numpy as np
from mpp.ml.models.sequence.cadtoseq_module import ARMSTM
from mpp.ml.datasets.fabricad import Fabricad 

model = ARMSTM.load_from_checkpoint("checkpoints/cadtoseq.ckpt", map_location="cpu")
model.eval()

vecset = torch.from_numpy(np.load("path/to/features/vecset.npy")).unsqueeze(0)  # [1, 1024, 32]
with torch.no_grad():
    token_ids = model.generate(vecset, device="cpu")

print(Fabricad.decode_sequence(token_ids[0].tolist()))
# e.g. ['fräsen', 'bohren', 'schweißen', 'STOP']
```

---

## Dataset

The model was trained on [FabriCAD](https://github.com/CIMTT-Kiel/cad-api-client). Each sample consists multiple files. For this project the following are essential:

| File | Description |
|------|-------------|
| `features/vecset.npy` | Geometry embedding of shape `[1024, 32]` |
| `plan.csv` | Process plan with columns `Schritt;Dauer[min];Kosten[($)]` |

Dataset paths can be set via environment variables or directly in `config.yaml`:

```bash
export MPP_FEATURE_DATA=/path/to/fabricad
```

---

## Citation

If you use the VecSet encoder in your workflow, please cite the original work:

```bibtex
@article{10.1145/3592442,
  author    = {Zhang, Biao and Tang, Jiapeng and Nie{\ss}ner, Matthias and Wonka, Peter},
  title     = {3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models},
  journal   = {ACM Trans. Graph.},
  year      = {2023},
  volume    = {42},
  number    = {4},
  articleno = {92},
  doi       = {10.1145/3592442}
}
```

---

## Acknowledgements

- VecSet embeddings are computed with the encoder from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) by Biao Zhang (MIT License).
- This repository was extracted and documented from a larger research project with the assistance of [Claude](https://claude.ai) (Anthropic).
