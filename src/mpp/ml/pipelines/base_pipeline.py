"""
Shared pipeline utilities for CadToSeq training.

Provides functions to create dataloaders, MLflow loggers, callbacks, and
trainers. Configuration is provided via YAML files in the config/ directory.

Usage
-----
    from mpp.ml.pipelines.base_pipeline import load_config, get_dataloaders, ...

    cfg = load_config(Path("config/cadtoseq.yaml"))
    train_loader, val_loader = get_dataloaders(cfg)

Note
----
Extracted and optimised from a larger research codebase with the assistance
of Claude (Anthropic) – https://claude.ai
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import mlflow
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mpp.constants import PATHS
from mpp.ml.callbacks.artifact_callbacks import MLflowCheckpointCallback
from mpp.ml.datasets.fabricad_datamodule import Fabricad_datamodule

_BASE_CONFIG_PATH = PATHS.CONFIG / "base.yaml"


# ---------------------------------------------------------------------------
# Config-Loading
# ---------------------------------------------------------------------------

def load_config(task_config_path: str | Path) -> dict[str, Any]:
    """Load and merge base.yaml with a task-specific config file.

    The task config overrides values from the base config (deep merge).

    Parameters
    ----------
    task_config_path:
        Path to the task-specific YAML file.

    Returns
    -------
    dict
        Merged configuration.
    """
    with open(_BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(task_config_path) as f:
        task_cfg = yaml.safe_load(f)
    return _deep_merge(cfg, task_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive deep merge: override values replace base values."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict):
    """Create and return (train_loader, val_loader) from the config.

    Parameters
    ----------
    cfg:
        Merged configuration (result of load_config()).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation DataLoaders.
    """
    dm = Fabricad_datamodule(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        input_type=cfg["data"]["input_type"],
        target_type=cfg["data"]["target_type"],
    )
    dm.setup(stage="fit")
    return dm.train_dataloader(), dm.val_dataloader()


def get_test_dataloader(cfg: dict):
    """Create and return the test DataLoader from the config.

    Must only be called after the entire training (including HP tuning) is complete
    to avoid data leakage.

    Parameters
    ----------
    cfg:
        Merged configuration (result of load_config()).

    Returns
    -------
    DataLoader
        Test DataLoader.
    """
    dm = Fabricad_datamodule(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        input_type=cfg["data"]["input_type"],
        target_type=cfg["data"]["target_type"],
    )
    dm.setup(stage="test")
    return dm.test_dataloader()


# ---------------------------------------------------------------------------
# MLflow-Logger
# ---------------------------------------------------------------------------

def build_mlflow_logger(
    cfg: dict,
    experiment_name: str,
    run_name: str | None = None,
    run_id: str | None = None,
) -> MLFlowLogger:
    """Create an MLFlowLogger with the configured tracking URI.

    Parameters
    ----------
    cfg:
        Merged configuration.
    experiment_name:
        Name of the MLflow experiment.
    run_name:
        Optional name of the MLflow run.
    run_id:
        Optional run ID of an already started MLflow run (e.g. for
        nested runs).

    Returns
    -------
    MLFlowLogger
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=cfg["mlflow"]["tracking_uri"],
        run_name=run_name,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(
    cfg: dict,
    checkpoint_subdir: str,
    filename: str,
    patience: int,
) -> list:
    """Create [EarlyStopping, ModelCheckpoint] callbacks.

    Parameters
    ----------
    cfg:
        Merged configuration.
    checkpoint_subdir:
        Subdirectory relative to PATHS.CKPT_DIR for checkpoints.
    filename:
        Filename pattern for ModelCheckpoint.
    patience:
        Patience for EarlyStopping.

    Returns
    -------
    list
        [EarlyStopping, ModelCheckpoint]
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        dirpath=(PATHS.CKPT_DIR / checkpoint_subdir).as_posix(),
        filename=filename,
        save_weights_only=False,
        verbose=True,
    )
    return [early_stop, checkpoint, MLflowCheckpointCallback()]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def build_trainer(
    cfg: dict,
    max_epochs: int,
    logger: MLFlowLogger,
    callbacks: list,
) -> Trainer:
    """Create a PyTorch Lightning Trainer.

    Parameters
    ----------
    cfg:
        Merged configuration.
    max_epochs:
        Maximum number of epochs.
    logger:
        MLFlowLogger instance.
    callbacks:
        List of Lightning callbacks.

    Returns
    -------
    Trainer
    """
    return Trainer(
        max_epochs=max_epochs,
        precision="bf16-mixed",
        logger=logger,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        callbacks=callbacks,
        devices=[cfg["training"]["gpu_id"]],
    )


# ---------------------------------------------------------------------------
# Optuna Tuning
# ---------------------------------------------------------------------------

def run_tuning(cfg: dict, objective) -> optuna.Study:
    """Run Optuna hyperparameter search and return the study.

    Parameters
    ----------
    cfg:
        Merged configuration.
    objective:
        Optuna objective function (trial) -> float.

    Returns
    -------
    optuna.Study
        Completed Optuna study with all trials.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg["training"]["n_trials"])
    return study


# ---------------------------------------------------------------------------
# Helper function: sample hyperparameters from config
# ---------------------------------------------------------------------------

def suggest_hyperparams(trial: optuna.Trial, hp: dict) -> dict:
    """Sample hyperparameters from the configured search space.

    Supported types per parameter:
    - float: {low, high, log?}
    - int:   {low, high, step?}
    - categorical: {choices}

    Parameters
    ----------
    trial:
        Optuna trial.
    hp:
        Hyperparameter search space from cfg["hyperparameter_search"].

    Returns
    -------
    dict
        Sampled hyperparameters {name: value}.
    """
    params: dict[str, Any] = {}
    for name, spec in hp.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif isinstance(spec.get("low"), float) or spec.get("log", False):
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        else:
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )
    return params
