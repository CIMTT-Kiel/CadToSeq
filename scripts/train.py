"""
CadToSeq – Training Script
==========================
Runs hyperparameter tuning and/or final training for the CadToSeq model.

Usage
-----
# Final training only (parameters from config.yaml):
    python scripts/train.py

# Specify a custom config:
    python scripts/train.py --config=my_run.yaml

# Override individual parameters:
    python scripts/train.py --lr=0.001 --num_layers=4 ...

# With hyperparameter tuning (final training is performed with best params from tuning phase):
    python scripts/train.py --tuning=True

Hyperparameter priority (highest first)
----------------------------------------
1. CLI arguments          (--lr, --embed_dim, ...)
2. config.yaml model block / --config
3. Optuna best trial      (only if --tuning=True)
"""

import argparse
from pathlib import Path

import mlflow
import torch
from mpp.ml.callbacks.artifact_callbacks import (
    BestModelPlotCallback,
    SequencePredictionPlotCallback,
    SequenceTestPlotCallback,
)
from mpp.ml.models.sequence.cadtoseq_module import ARMSTM
from mpp.constants import PATHS
from mpp.ml.pipelines.base_pipeline import (
    build_callbacks,
    build_mlflow_logger,
    build_trainer,
    get_dataloaders,
    get_test_dataloader,
    load_config,
    run_tuning,
    suggest_hyperparams,
)

_CONFIG_PATH = PATHS.CONFIG / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="CadToSeq Training Script")

    parser.add_argument(
        "--tuning",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Run Optuna hyperparameter tuning before final training (default: False)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help=f"Path to the config YAML (default: {_CONFIG_PATH})",
    )

    # Model hyperparameters — all optional, override config values
    parser.add_argument("--lr",          type=float, default=None, help="Learning rate")
    parser.add_argument("--embed_dim",   type=int,   default=None, help="Embedding dimension")
    parser.add_argument("--nhead",       type=int,   default=None, help="Number of attention heads")
    parser.add_argument("--num_layers",  type=int,   default=None, help="Number of decoder layers")
    parser.add_argument("--dropout",     type=float, default=None, help="Dropout rate")

    return parser.parse_args()


def resolve_hyperparams(args, cfg: dict, train_cfg: dict, best_trial_params: dict | None = None) -> dict:
    """Determines final hyperparameters by priority:"""
    
    hp_cfg = cfg["hyperparameter_search"]

    def cfg_default(name):
        spec = hp_cfg[name]
        if "choices" in spec:
            return spec["choices"][0]
        return (spec["low"] + spec["high"]) / 2

    # Base: config.yaml hyperparameter_search defaults
    base = {
        "lr":         cfg_default("lr"),
        "embed_dim":  cfg_default("embed_dim"),
        "nhead":      cfg_default("nhead"),
        "num_layers": cfg_default("num_layers"),
        "dropout":    cfg_default("dropout"),
    }

    # Optuna result overrides search space defaults
    if best_trial_params:
        base.update(best_trial_params)

    # config.yaml model block overrides Optuna
    if train_cfg:
        for key in ("lr", "embed_dim", "nhead", "num_layers", "dropout"):
            if key in train_cfg:
                base[key] = train_cfg[key]

    # CLI arguments have highest priority
    if args.lr          is not None: base["lr"]          = args.lr
    if args.embed_dim   is not None: base["embed_dim"]   = args.embed_dim
    if args.nhead       is not None: base["nhead"]       = args.nhead
    if args.num_layers  is not None: base["num_layers"]  = args.num_layers
    if args.dropout     is not None: base["dropout"]     = args.dropout

    return base


def run_hyperparameter_tuning(cfg: dict, train_loader, val_loader) -> dict:
    """Runs Optuna tuning and returns the best trial parameters."""

    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]
        torch.set_float32_matmul_precision("medium")

        model = ARMSTM(
            lr=hp["lr"],
            embed_dim=hp["embed_dim"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            weight_decay=cfg["training"]["weight_decay"],
            max_epochs=max_epochs,
            use_scheduler=False,
            ss_epsilon_max=0.0,
        )

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as child_run:
            mlf_logger = build_mlflow_logger(
                cfg,
                cfg["mlflow"]["tuning_experiment_name"],
                run_id=child_run.info.run_id,
            )
            mlf_logger.log_hyperparams(hp)
            callbacks = build_callbacks(
                cfg,
                cfg["checkpoint"]["tuning_subdir"],
                cfg["checkpoint"]["filename"],
                patience=cfg["training"]["tuning_patience"],
            )
            callbacks.append(SequencePredictionPlotCallback(
                plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
            ))
            callbacks.append(BestModelPlotCallback())
            trainer = build_trainer(cfg, max_epochs, mlf_logger, callbacks)
            trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics["val_loss"].item()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["tuning_experiment_name"])
    with mlflow.start_run(run_name="optuna-study"):
        study = run_tuning(cfg, objective)

    print("\nBest hyperparameters from tuning:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    return study.best_trial.params


def run_final_training(cfg: dict, hp: dict, train_loader, val_loader):
    """Runs the final training with the given hyperparameters."""
    torch.set_float32_matmul_precision("high")

    print("\nFinal training with the following hyperparameters:")
    for k, v in hp.items():
        print(f"  {k}: {v}")

    model = ARMSTM(
        lr=hp["lr"],
        embed_dim=hp["embed_dim"],
        nhead=hp["nhead"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["final_epochs"],
        ss_epsilon_max=cfg["scheduled_sampling"]["epsilon_max"],
        ss_warmup_epochs=cfg["scheduled_sampling"]["warmup_epochs"],
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="best-model") as run:
        mlf_logger = build_mlflow_logger(
            cfg, cfg["mlflow"]["experiment_name"], run_id=run.info.run_id
        )
        callbacks = build_callbacks(
            cfg,
            cfg["checkpoint"]["best_subdir"],
            cfg["checkpoint"]["filename"],
            patience=cfg["training"]["final_patience"],
        )
        callbacks.append(SequencePredictionPlotCallback(
            plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
        ))
        callbacks.append(BestModelPlotCallback())
        callbacks.append(SequenceTestPlotCallback())

        trainer = build_trainer(cfg, cfg["training"]["final_epochs"], mlf_logger, callbacks)
        trainer.fit(model, train_loader, val_loader)

        test_loader = get_test_dataloader(cfg)
        trainer.test(model, test_loader, ckpt_path=None)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_loader, val_loader = get_dataloaders(cfg)

    train_cfg = cfg.get("model", {})
    print(f"Config loaded: {args.config}")

    best_trial_params = None
    if args.tuning:
        best_trial_params = run_hyperparameter_tuning(cfg, train_loader, val_loader)

    hp = resolve_hyperparams(args, cfg, train_cfg, best_trial_params)
    run_final_training(cfg, hp, train_loader, val_loader)


if __name__ == "__main__":
    main()
