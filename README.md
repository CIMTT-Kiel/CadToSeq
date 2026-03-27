# CadToSeq

Predicts manufacturing process sequences from CAD geometry using a Transformer decoder.

Given a set of geometry feature vectors `[1024, 32]` (produced by a VecSet encoder from a surface point cloud), the model generates an ordered sequence of manufacturing steps such as *milling → drilling → welding → grinding*.

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

---

## Checkpoints

| Checkpoint | Purpose |
|---|---|
| VecSet ae | Point cloud → geometry embedding |
| CadToSeq model | Geometry embedding → process sequence |

Download links: `<VECSET_CHECKPOINT_URL>` · `<CADTOSEQ_CHECKPOINT_URL>`

Set the paths in `config.yaml`:

```yaml
paths:
  vecset_encoder_ckpt: checkpoints/vecset_encoder.ckpt
  cadtoseq_ckpt:       checkpoints/cadtoseq.ckpt
  data_dir:            /path/to/fabricad
```

---

## Preprocessing

Geometry embeddings (`vecset.npy`) are produced from surface point clouds using the bundled VecSet encoder (adapted from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet)).

**Step 1 – STEP → point cloud:** Sample 8192 surface points from each STEP file and save as `pointcloud.npy` (shape `[8192, 3]`, float32, normalised to `[-1, 1]`). Any CAD tool works (FreeCAD, Open3D, PythonOCC). This step is not included in this repo.

**Step 2 – point cloud → vecset.npy:**

```bash
python scripts/create_vecsets.py --data_dir /path/to/fabricad
```

---

## Training

All parameters are controlled via `config.yaml`. Key settings:

| Key | Default | Description |
|---|---|---|
| `training.final_epochs` | 1000 | Max training epochs |
| `training.final_patience` | 50 | Early stopping patience |
| `training.n_trials` | 50 | Optuna HP-search trials |
| `scheduled_sampling.epsilon_max` | 0.0 | Max scheduled sampling rate (0 = off) |

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

Or from Python:

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

```bibtex
@article{cadtoseq2025,
  title   = {CadToSeq: Predicting Manufacturing Process Sequences from CAD Geometry using Transformer Decoders},
  author  = {Kruse, Michel and ...},
  year    = {2025},
}

@article{10.1145/3592442,
  author  = {Zhang, Biao and Tang, Jiapeng and Nie{\ss}ner, Matthias and Wonka, Peter},
  title   = {3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models},
  journal = {ACM Trans. Graph.},
  year    = {2023},
  doi     = {10.1145/3592442},
}
```

---

## Acknowledgements

The VecSet encoder (`src/mpp/ml/datasets/preprocessing/`) is adapted from [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) by Biao Zhang, used under the MIT License.

This repository was extracted and commented from a larger research project with the assistance of [Claude](https://claude.ai) (Anthropic).
