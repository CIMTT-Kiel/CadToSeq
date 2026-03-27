"""
Pipeline: Sequence prediction (cadtoseq)

Workflow
--------
1. Hyperparameter tuning with Optuna (number of trials in config/cadtoseq.yaml)
2. Final training with the best hyperparameters

Configuration
-------------
All parameters are read from config/cadtoseq.yaml (+ config/base.yaml).
To adjust, simply edit the YAML files – no code changes required.

Execution
---------
    python -m mpp.ml.pipelines.cadtoseq.model_input_to_tuned_model

Note
----
Extracted and optimised from a larger research codebase with the assistance
of Claude (Anthropic) – https://claude.ai
"""

from pathlib import Path

import mlflow
import torch

from mpp.ml.callbacks.artifact_callbacks import BestModelPlotCallback, SequencePredictionPlotCallback, SequenceTestPlotCallback
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

_CFG_PATH = PATHS.CONFIG / "cadtoseq.yaml"
cfg = load_config(_CFG_PATH)


def main():
    train_loader, val_loader = get_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]
        torch.set_float32_matmul_precision('medium')

        model = ARMSTM(
            lr=hp["lr"],
            embed_dim=hp["embed_dim"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            weight_decay=cfg["training"]["weight_decay"],
            max_epochs=max_epochs,
            use_scheduler=False,
            ss_epsilon_max=0.0,  # Scheduled sampling disabled during tuning (unstable trials)
        )

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as child_run:
            mlf_logger = build_mlflow_logger(
                cfg,
                cfg["mlflow"]["tuning_experiment_name"],
                run_id=child_run.info.run_id,
            )
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

            val_loss = trainer.callback_metrics["val_loss"].item()
            mlf_logger.log_hyperparams(trial.params)
            mlf_logger.log_metrics({"val_loss": val_loss})

        return val_loss

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["tuning_experiment_name"])
    with mlflow.start_run(run_name="optuna-study"):
        study = run_tuning(cfg, objective)

    # ------------------------------------------------------------------
    # Final training with best hyperparameters
    # ------------------------------------------------------------------
    best = study.best_trial.params
    torch.set_float32_matmul_precision('high')

    model = ARMSTM(
        lr=best["lr"],
        embed_dim=best["embed_dim"],
        nhead=best["nhead"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["final_epochs"],
        ss_epsilon_max=cfg["scheduled_sampling"]["epsilon_max"],
        ss_warmup_epochs=cfg["scheduled_sampling"]["warmup_epochs"],
    )

    mlf_logger = build_mlflow_logger(
        cfg, cfg["mlflow"]["experiment_name"], run_name="best-model"
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

    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint (once, after training)
    # ------------------------------------------------------------------
    test_loader = get_test_dataloader(cfg)
    trainer.test(model, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
