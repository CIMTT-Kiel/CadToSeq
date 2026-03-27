"""
Feature Extraction: point cloud → vecset.npy
=============================================
Encodes CAD parts represented as point clouds into VecSet geometry embeddings
[1024, 32] using the pretrained VecSet encoder published alongside this paper.

Download the checkpoint from: <CHECKPOINT_URL>

Point cloud requirement
-----------------------
The encoder expects a point cloud of exactly 8192 points per part (float32,
shape [8192, 3], normalized to [-1, 1]). If you are starting from STEP files
you need to sample a surface point cloud first using a tool of your choice,
e.g. FreeCAD, Open3D, or PythonOCC. The resulting point cloud should be saved
as a NumPy array (.npy) of shape [8192, 3] in the sample directory:

    <data_dir>/<part_id>/pointcloud.npy

Usage
-----
    python scripts/create_vecsets.py --checkpoint path/to/vecset_encoder.ckpt

    # Custom data directory and GPU:
    python scripts/create_vecsets.py \\
        --checkpoint path/to/vecset_encoder.ckpt \\
        --data_dir   path/to/fabricad \\
        --device     cuda

Expected dataset layout
-----------------------
    <data_dir>/
        <part_id>/
            pointcloud.npy       # [8192, 3], float32, required
            plan.csv             # process plan, required for training
        ...

Output
------
For each part the script writes:
    <data_dir>/<part_id>/features/vecset.npy   # [1024, 32], float32

Already-existing vecset.npy files are skipped.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# VecSet encoder code – bundled with this release
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mpp.ml.datasets.preprocessing.autoencoder import point_vec1024x32_dim1024_depth24_nb

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PC_SIZE = 8192  # point cloud size the encoder was trained with


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_vecset_encoder(ckpt_path: str | Path, device: str = "cpu") -> torch.nn.Module:
    """Instantiate the VecSet encoder and load pretrained weights.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to the ``vecset_encoder.ckpt`` checkpoint file.
    device : str
        Target device, e.g. ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    torch.nn.Module
        Encoder in eval mode on ``device``.
    """
    model = point_vec1024x32_dim1024_depth24_nb(pc_size=PC_SIZE)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    logger.info(f"VecSet encoder loaded from {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, device={device})")
    return model


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def pointcloud_to_vecset(pc_npy: np.ndarray, model: torch.nn.Module, device: str = "cpu") -> np.ndarray:
    """Encode a single point cloud to a VecSet embedding.

    Parameters
    ----------
    pc_npy : np.ndarray
        Float32 array of shape ``[8192, 3]``, coordinates in ``[-1, 1]``.
    model : torch.nn.Module
        Pretrained VecSet encoder (output of :func:`load_vecset_encoder`).
    device : str
        Inference device.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[1024, 32]``.
    """
    pc = torch.from_numpy(pc_npy).float().unsqueeze(0).to(device)  # [1, 8192, 3]
    result = model.encode_to_vecset(pc)
    vecset = result["x"]                                            # [1, 1024, 32]
    return vecset.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_dataset(data_dir: Path, model: torch.nn.Module, device: str = "cpu"):
    """Process all samples in a FabriCAD-style directory.

    Parameters
    ----------
    data_dir : Path
        Root directory containing one subdirectory per part.
    model : torch.nn.Module
        Pretrained VecSet encoder.
    device : str
        Inference device.
    """
    samples = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != ".DS_Store")
    logger.info(f"{len(samples)} samples found in {data_dir}")

    ok, skipped, fail = 0, 0, 0

    for sample_dir in tqdm(samples, desc="Creating vecsets", unit="sample"):
        idx = sample_dir.name
        pc_path     = sample_dir / "pointcloud.npy"
        vecset_path = sample_dir / "features" / "vecset.npy"

        if vecset_path.exists():
            skipped += 1
            continue

        if not pc_path.exists():
            tqdm.write(f"[SKIP] {idx} – pointcloud.npy not found")
            skipped += 1
            continue

        try:
            pc = np.load(pc_path)
            if pc.shape != (PC_SIZE, 3):
                raise ValueError(f"Expected shape ({PC_SIZE}, 3), got {pc.shape}")

            vecset_path.parent.mkdir(parents=True, exist_ok=True)
            vecset = pointcloud_to_vecset(pc, model, device)
            np.save(vecset_path, vecset)
            ok += 1
        except Exception as e:
            tqdm.write(f"[FAIL] {idx} – {e}")
            fail += 1

    logger.info(f"Done.  Created: {ok}  Skipped: {skipped}  Failed: {fail}  Total: {len(samples)}")
    if fail > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PROJECT_ROOT     = Path(__file__).resolve().parents[1]
_PROJECT_CFG_PATH = _PROJECT_ROOT / "config.yaml"


def _read_project_config(config_path: Path) -> dict:
    """Return the parsed project config, or an empty dict if the file is absent."""
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_args():
    cfg = _read_project_config(_PROJECT_CFG_PATH)
    paths_cfg = cfg.get("paths", {})

    default_data = Path(
        os.environ.get(
            "MPP_FEATURE_DATA",
            paths_cfg.get("data_dir", "/path/to/fabricad"),
        )
    )
    default_ckpt = Path(paths_cfg.get("vecset_encoder_ckpt", "checkpoints/vecset_encoder.ckpt"))

    parser = argparse.ArgumentParser(
        description="Encode point clouds to VecSet embeddings using the pretrained encoder."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=default_ckpt,
        help=f"Path to pretrained vecset_encoder.ckpt (default from config.yaml: {default_ckpt})",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=default_data,
        help=f"Dataset root directory (default from config.yaml / $MPP_FEATURE_DATA: {default_data})",
    )
    parser.add_argument(
        "--config", type=Path, default=_PROJECT_CFG_PATH,
        help=f"Path to project config YAML (default: {_PROJECT_CFG_PATH})",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help='Inference device: "cpu" or "cuda" (default: cpu)',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Re-read config from --config path in case user passed a custom file
    if args.config != _PROJECT_CFG_PATH:
        cfg = _read_project_config(args.config)
        paths_cfg = cfg.get("paths", {})
        if args.checkpoint == Path("checkpoints/vecset_encoder.ckpt"):
            args.checkpoint = Path(paths_cfg.get("vecset_encoder_ckpt", args.checkpoint))
        if str(args.data_dir) == "/path/to/fabricad":
            args.data_dir = Path(paths_cfg.get("data_dir", args.data_dir))

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.error("Download from: https://drive.google.com/drive/folders/1tX4pFulWqtICYgchRXmzscHDRJ5q2iSz")
        logger.error("Then set paths.vecset_encoder_ckpt in config.yaml or pass --checkpoint.")
        sys.exit(1)

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Set paths.data_dir in config.yaml, pass --data_dir, or export MPP_FEATURE_DATA=...")
        sys.exit(1)

    model = load_vecset_encoder(args.checkpoint, device=args.device)
    process_dataset(args.data_dir, model, device=args.device)
