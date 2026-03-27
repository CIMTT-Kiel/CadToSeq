"""
CadToSeq – Inference Script
============================
Predicts a manufacturing process sequence from a geometry embedding (vecset.npy).

Usage
-----
# Predict from a pre-computed vecset.npy:
    python scripts/infer.py --vecset path/to/features/vecset.npy

# Use a specific checkpoint (overrides config.yaml):
    python scripts/infer.py --vecset path/to/features/vecset.npy --ckpt checkpoints/cadtoseq.ckpt

# Run on GPU:
    python scripts/infer.py --vecset path/to/features/vecset.npy --device cuda
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from mpp.ml.datasets.fabricad import Fabricad
from mpp.ml.models.sequence.cadtoseq_module import ARMSTM


def parse_args():
    parser = argparse.ArgumentParser(description="CadToSeq Inference")
    parser.add_argument(
        "--vecset",
        type=Path,
        required=True,
        help="Path to a vecset.npy file of shape [1024, 32]",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Path to a CadToSeq checkpoint (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on: 'cpu' or 'cuda' (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = args.ckpt or cfg["paths"]["cadtoseq_ckpt"]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set 'paths.cadtoseq_ckpt' in config.yaml or pass --ckpt."
        )

    model = ARMSTM.load_from_checkpoint(ckpt_path, map_location=args.device)
    model.eval()

    vecset = torch.from_numpy(np.load(args.vecset).astype("float32")).unsqueeze(0)

    with torch.no_grad():
        token_ids = model.generate(vecset, device=args.device)

    ids = token_ids[0].tolist()
    # Truncate at STOP token (inclusive) and drop trailing PAD tokens
    from mpp.constants import VOCAB
    stop_id = VOCAB["STOP"]
    if stop_id in ids:
        ids = ids[:ids.index(stop_id) + 1]
    sequence = Fabricad.decode_sequence(ids)
    print(" → ".join(sequence))


if __name__ == "__main__":
    main()
