"""
Create trivial test samples for CadToSeq
=========================================
Generates synthetic data (random VecSet embeddings + process plans) so the
training and inference pipeline can be exercised without real FabriCAD data.

Each sample directory contains:
    <data_dir>/<part_id>/features/vecset.npy   – random float32 [1024, 32]
    <data_dir>/<part_id>/plan.csv              – random manufacturing sequence

Usage
-----
    # Create 200 samples in ./test_data (default):
    python scripts/create_test_data.py

    # Custom location and count:
    python scripts/create_test_data.py --data_dir /tmp/test_data --n_samples 500

Then point config.yaml at the output directory:
    paths:
      data_dir: ./test_data
"""

import argparse
import random
from pathlib import Path

import numpy as np

# Process steps that the model can predict (must match VOCAB in constants.py)
PROCESS_STEPS = ["fräsen", "schleifen", "bohren", "schweißen", "drehen", "prüfen", "kontrollieren"]


def make_plan_csv(steps: list[str]) -> str:
    """Return a plan.csv string for a given list of process steps."""
    lines = ["Schritt;Dauer[min];Kosten[($)]", "planen;10;0"]
    for step in steps:
        duration = random.randint(10, 120)
        cost = round(random.uniform(10, 500), 2)
        lines.append(f"{step};{duration};{cost}")
    lines.append("liefern;5;10")
    return "\n".join(lines) + "\n"


def create_samples(data_dir: Path, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        sample_dir = data_dir / f"{i:05d}"
        features_dir = sample_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # Random VecSet embedding
        vecset = rng.standard_normal((1024, 32)).astype(np.float32)
        np.save(features_dir / "vecset.npy", vecset)

        # Random process sequence (1–4 unique steps, fixed order)
        k = random.randint(1, 4)
        steps = random.sample(PROCESS_STEPS, k)
        (sample_dir / "plan.csv").write_text(make_plan_csv(steps), encoding="utf-8")

    print(f"Created {n_samples} samples in {data_dir.resolve()}")
    print(f"Set  paths.data_dir: {data_dir}  in config.yaml to use them.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CadToSeq test data.")
    parser.add_argument("--data_dir", type=Path, default=Path("test_data"),
                        help="Output directory (default: ./test_data)")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to generate (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_samples(args.data_dir, args.n_samples, args.seed)
