# CadToSeq

Predicts manufacturing process sequences from CAD geometry using a Transformer decoder.

Given a set of geometry feature vectors `[1024, 32]` (produced by a VecSet encoder (adapted from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet))), the model generates an ordered sequence of manufacturing steps such as *milling → drilling → welding → grinding*.

---

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1
- CUDA-capable GPU strongly recommended
- Pretrained checkpoints (see below)

---

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Preprocessing
---
Geometry embeddings (`vecset.npy`) are produced from surface point clouds using the bundled VecSet encoder (adapted from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet)). This step is not included in this repo.

## Training

All parameters are controlled via `config.yaml`. Key settings:

```bash
# Train with best-known hyperparameters (from config.yaml):
python scripts/train.py

# Run Optuna HP-tuning first, then final training:
python scripts/train.py --tuning=True

# Override individual hyperparameters:
python scripts/train.py --lr=0.001 --embed_dim=96 --num_layers=3
```

---

## Inference

```bash
python scripts/infer.py --vecset path/to/features/vecset.npy
```

Or via Python:

```python
import torch, numpy as np
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

## Data

Trained on [FabriCAD](https://github.com/CIMTT-Kiel/cad-api-client). Each sample contains:
- `features/vecset.npy` – geometry embedding `[1024, 32]`
- `plan.csv` – process plan with columns `Schritt;Dauer[min];Kosten[($)]`

Set dataset paths via environment variables or `config.yaml`:
---

## Citation

@article{10.1145/3592442,
author = {Zhang, Biao and Tang, Jiapeng and Nie\ss{}ner, Matthias and Wonka, Peter},
title = {3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592442},
doi = {10.1145/3592442},
abstract = {We introduce 3DShape2VecSet, a novel shape representation for neural fields designed for generative diffusion models. Our shape representation can encode 3D shapes given as surface models or point clouds, and represents them as neural fields. The concept of neural fields has previously been combined with a global latent vector, a regular grid of latent vectors, or an irregular grid of latent vectors. Our new representation encodes neural fields on top of a set of vectors. We draw from multiple concepts, such as the radial basis function representation, and the cross attention and self-attention function, to design a learnable representation that is especially suitable for processing with transformers. Our results show improved performance in 3D shape encoding and 3D shape generative modeling tasks. We demonstrate a wide variety of generative applications: unconditioned generation, category-conditioned generation, text-conditioned generation, point-cloud completion, and image-conditioned generation. Code: https://1zb.github.io/3DShape2VecSet/.},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {92},
numpages = {16},
keywords = {3D shape generation, generative models, shape reconstruction, 3D shape representation, diffusion models}
}
```

---

## Acknowledgements

The VecSet encoder (`src/mpp/ml/datasets/preprocessing/`) is adapted from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) by Biao Zhang, used under the MIT License.

This repository was extracted and commented from a larger research project with the assistance of [Claude](https://claude.ai) (Anthropic).
