# Disclaimer
**Work in Progress — Not yet fully validated**
This repository already works but is extracted from a larger research project. We are currently working on optimising the code for the separated CadToSeq approach. This is not fully tested yet. A final version will be available shortly.

# CadToSeq

CadToSeq is an autoregressive transformer-decoder model for manufacturing process sequence prediction from 3D part geometry. Given a CAD model in STEP format, it encodes the part geometry using Vectorset embeddings and predicts an ordered sequence of manufacturing operations — such as milling, drilling, or welding — token by token. Trained on the FabriCAD dataset of 100,000 synthetic CAD models, it achieves an Exact Match Rate of 74.35% and an Elementwise Accuracy of 89.06% on the test set (defined by config/paper_split.json), and demonstrates initial transfer to real industrial data. This repository contains the model architecture, training code, and evaluation scripts.

This repository refers to a paper-submission and reflects the codebase at the time of submission. If accepted, the paper will be referenced here. The project is actively being developed; further work will be referenced here as it becomes available.

---

## Demo data

This repository includes **100 randomly selected samples**  from the FabriCAD dataset in `data/fabricad_examples/`.

> **These are incomplete demonstration samples.** Each sample contains only the files required to run the CadToSeq pipeline (`vecset.npy` and `plan.csv`). The full CAD geometry (STEP files), detailed feature-level information, are available from the fabricad dataset. 

The default `config/config.yaml` points to this demo subset so the pipeline can be run immediately. For training a production model, replace `paths.data_dir` with the path to the full dataset [https://cimtt-kiel.github.io/FabriCAD/](https://cimtt-kiel.github.io/FabriCAD/).

---

## How it works

1. A CAD part's geometry is encoded externally into a compact set of vectors (`vecset.npy`).
2. The Transformer decoder reads these vectors and autoregressively predicts manufacturing steps.
3. Generation stops when the model emits a `STOP` token.


Given a set of geometry feature vectors `[1024, 32]` (VecSet embeddings), CadToSeq generates an ordered sequence of manufacturing steps such as:

> *milling → drilling → welding → grinding → STOP*

> **Note on geometry embeddings:** VecSet embeddings are produced from surface point clouds using the [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) encoder. The preprocessing step is **not part of this repository** — CadToSeq takes pre-computed `vecset.npy` files as input. See [Preparing your data](#preparing-your-data) below.


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

Each sample must have its own directory with the following structure:

```
<data_dir>/
    <part_id>/
        features/
            vecset.npy   # geometry embedding [1024, 32], float32
        plan.csv         # process plan (Schritt;Dauer[min];Kosten[($)])
```

To use the full FabriCAD dataset, see [https://cimtt-kiel.github.io/FabriCAD/](https://cimtt-kiel.github.io/FabriCAD/).

---

## Training

All hyperparameters are controlled via `config/config.yaml`. Start training with:

```bash
# Train with the best-known hyperparameters from config/config.yaml:
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
python scripts/infer.py --vecset path/to/features/vecset.npy --ckpt path/to/cadtoseq_checkpoint.ckpt
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
