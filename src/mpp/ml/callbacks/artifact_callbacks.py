"""
MLflow Artifact Callbacks
--------------------------
MLflowCheckpointCallback          – Logs the best checkpoint as an MLflow artifact.
SequencePredictionPlotCallback     – Logs diagnostic plots every N validation epochs
                                     (with epoch number in the filename).
BestModelPlotCallback              – Logs plots for the currently best model;
                                     files are always overwritten when a
                                     better result is achieved.

Artifact structure (cadtoseq):
  checkpoints/        – Best model checkpoints
  plots/examples/     – Prediction table (per epoch)
  plots/confusion/    – Token confusion matrix (per epoch)
  plots/levenshtein/  – Levenshtein distance distribution (per epoch)
  plots/token_acc/    – Token-wise accuracy (per epoch)
  plots/best/         – All four plots for the currently best model (overwritten)
"""

import logging
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mpp.constants import INV_VOCAB, VOCAB
from mpp.ml.metrics.sequences import Sequence_comparator

logger = logging.getLogger(__name__)

# Process step labels (excluding START, STOP, PAD), in order of token IDs
_SPECIAL_TOKENS = {"START", "STOP", "PAD"}
STEP_LABELS = [INV_VOCAB[i] for i in range(len(VOCAB)) if INV_VOCAB[i] not in _SPECIAL_TOKENS]


# ---------------------------------------------------------------------------
# Shared base class with plot and prediction logic
# ---------------------------------------------------------------------------

class _SequencePlotMixin:
    """
    Mixin with shared prediction collection and plot logic.
    Used by SequencePredictionPlotCallback and BestModelPlotCallback.
    """

    n_examples: int
    _comparator: Sequence_comparator

    # ------------------------------------------------------------------
    # Collect predictions over the entire val loader
    # ------------------------------------------------------------------

    def _collect_predictions(self, val_dl, pl_module):
        """Iterates over the complete val loader and returns aggregated tensors."""
        all_tf_preds, all_gen_preds, all_targets = [], [], []

        pl_module.eval()
        with torch.no_grad():
            for vector_set, padded_targets in val_dl:
                vector_set = vector_set.to(pl_module.device)
                padded_targets = padded_targets.to(pl_module.device)

                logits = pl_module(vector_set, padded_targets[:, :-1])
                all_tf_preds.append(logits.argmax(dim=-1).cpu())
                all_targets.append(padded_targets[:, 1:].cpu())
                all_gen_preds.append(
                    pl_module.generate(vector_set, device=str(pl_module.device)).cpu()
                )

        return (
            torch.cat(all_tf_preds, dim=0),
            torch.cat(all_gen_preds, dim=0),
            torch.cat(all_targets, dim=0),
        )

    # ------------------------------------------------------------------
    # Generate and log all four plots
    # ------------------------------------------------------------------

    def _generate_plots(
        self, tf_preds, gen_preds, targets, title_prefix, run_id, artifact_dir, filename_prefix
    ):
        with tempfile.TemporaryDirectory() as tmp:
            self._plot_examples(tf_preds, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_confusion_matrix(tf_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Teacher-Forced", file_suffix="tf")
            self._plot_confusion_matrix(gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Autoregressive", file_suffix="ar")
            self._plot_token_accuracy(tf_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Teacher-Forced", file_suffix="tf")
            self._plot_token_accuracy(gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Autoregressive", file_suffix="ar")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _decode(self, token_ids):
        return [INV_VOCAB.get(int(t), "?") for t in token_ids if int(t) != VOCAB["PAD"]]

    def _log(self, fig, path, run_id, artifact_dir):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        mlflow.MlflowClient().log_artifact(run_id, path, artifact_path=artifact_dir)

    # ------------------------------------------------------------------
    # Plot methods
    # ------------------------------------------------------------------

    def _plot_examples(self, tf_preds, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        n = min(self.n_examples, targets.size(0))
        fig, ax = plt.subplots(figsize=(15, n * 0.6 + 1.8))
        ax.axis("off")

        rows = []
        for i in range(n):
            valid_mask = targets[i] != VOCAB["PAD"]
            gt = " → ".join(self._decode(targets[i]))
            tf = " → ".join(self._decode(tf_preds[i][valid_mask]))
            ar = " → ".join(self._decode(gen_preds[i]))
            rows.append([gt, tf, ar])

        table = ax.table(
            cellText=rows,
            colLabels=["Ground Truth", "Teacher-Forced", "Autoregressive"],
            cellLoc="left",
            loc="center",
            colWidths=[0.35, 0.35, 0.30],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i, row in enumerate(rows):
            color = "#c8e6c9" if row[0] == row[2] else "#ffffff"
            for j in range(3):
                table[i + 1, j].set_facecolor(color)

        ax.set_title(
            f"{title_prefix} – Prediction Examples  (green = exact AR match)",
            pad=12, fontsize=11,
        )
        path = os.path.join(tmp, f"{filename_prefix}examples.png")
        self._log(fig, path, run_id, f"{artifact_dir}/examples")

    def _plot_confusion_matrix(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Teacher-Forced", file_suffix="tf"):
        targets = targets[:, :preds.size(1)]  # align lengths (gen_preds shorter than targets)
        mask = targets != VOCAB["PAD"]
        p = preds[mask].cpu().numpy()
        t = targets[mask].cpu().numpy()

        n = len(VOCAB)
        cm = np.zeros((n, n), dtype=int)
        for pi, ti in zip(p, t):
            cm[ti, pi] += 1

        labels = [INV_VOCAB[i] for i in range(n)]

        # Absolute confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{title_prefix} – Token Confusion Matrix absolute ({label})")
        path = os.path.join(tmp, f"{filename_prefix}confusion_{file_suffix}.png")
        self._log(fig, path, run_id, f"{artifact_dir}/confusion")

        # Relative confusion matrix (row-normalized, i.e. recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_rel = np.where(row_sums > 0, cm / row_sums, 0.0)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rel, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
                    xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Ground Truth")
        ax2.set_title(f"{title_prefix} – Token Confusion Matrix relative ({label})")
        path2 = os.path.join(tmp, f"{filename_prefix}confusion_rel_{file_suffix}.png")
        self._log(fig2, path2, run_id, f"{artifact_dir}/confusion")

    def _plot_levenshtein(self, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        mask_t = self._comparator._create_mask(targets)
        mask_p = self._comparator._create_mask(gen_preds)
        dists = self._comparator.levenshtein_distance(
            gen_preds, targets, mask_p, mask_t
        ).cpu().numpy().astype(float)

        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        sns.kdeplot(data=dists, ax=ax, fill=True, color="#4c72b0",
                    alpha=0.15, linewidth=2, clip=(0.0, None), bw_adjust=0.4)
        ax.axvline(dists.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {dists.mean():.2f}")
        ax.set_xlabel("Levenshtein Distance")
        ax.set_ylabel("Density")
        ax.set_title(f"{title_prefix} – Levenshtein-Distanz (Autoregressive)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}levenshtein.png")
        self._log(fig, path, run_id, f"{artifact_dir}/levenshtein")

    def _plot_token_accuracy(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix, label="Teacher-Forced", file_suffix="tf"):
        targets = targets[:, :preds.size(1)]  # align lengths (gen_preds shorter than targets)
        token_acc = self._comparator.stepwise_accuracy(preds, targets)
        sorted_keys = sorted(token_acc.keys())
        labels = [INV_VOCAB[k] for k in sorted_keys]
        values = [token_acc[k] if not np.isnan(token_acc[k]) else 0.0
                  for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, values, edgecolor="black", color="#55a868")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_prefix} – Token-wise Accuracy ({label})")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        path = os.path.join(tmp, f"{filename_prefix}token_acc_{file_suffix}.png")
        self._log(fig, path, run_id, f"{artifact_dir}/token_acc")


# ---------------------------------------------------------------------------
# Checkpoint-Callback
# ---------------------------------------------------------------------------

class MLflowCheckpointCallback(Callback):
    """
    Logs the best checkpoints (save_top_k) as MLflow artifacts under
    'checkpoints/' at the end of training, then deletes the local copies.
    """

    def on_train_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                dirs_to_clean = set()
                for path in cb.best_k_models:
                    if not path or not os.path.exists(path):
                        continue
                    mlflow.MlflowClient().log_artifact(
                        trainer.logger.run_id,
                        path,
                        artifact_path="checkpoints",
                    )
                    logger.info(f"Checkpoint logged to MLflow: {path}")
                    dirs_to_clean.add(os.path.dirname(path))
                    os.remove(path)
                for d in dirs_to_clean:
                    if os.path.isdir(d) and not os.listdir(d):
                        os.removedirs(d)


# ---------------------------------------------------------------------------
# Per-epoch plots
# ---------------------------------------------------------------------------

class SequencePredictionPlotCallback(_SequencePlotMixin, Callback):
    """
    Logs four diagnostic plots every `plot_every_n_epochs` epochs as MLflow artifacts.
    Filenames contain the epoch number → full history is preserved.

    Parameters
    ----------
    plot_every_n_epochs : int
        Frequency of plot generation.
    n_examples : int
        Number of examples in the prediction table.
    """

    def __init__(self, plot_every_n_epochs: int = 10, n_examples: int = 8):
        self.plot_every_n_epochs = plot_every_n_epochs
        self.n_examples = n_examples
        self._comparator = Sequence_comparator(VOCAB)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        tf_preds, gen_preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        self._generate_plots(
            tf_preds, gen_preds, targets,
            title_prefix=f"Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots",
            filename_prefix=f"ep{epoch:04d}_",
        )


# ---------------------------------------------------------------------------
# Best-model plots (overwritten on improvement)
# ---------------------------------------------------------------------------

class BestModelPlotCallback(_SequencePlotMixin, Callback):
    """
    Logs four diagnostic plots for the currently best model under 'plots/best/'.
    Files have fixed names and are overwritten as soon as a
    better result is achieved.

    Parameters
    ----------
    n_examples : int
        Number of examples in the prediction table.
    """

    def __init__(self, n_examples: int = 8):
        self.n_examples = n_examples
        self._comparator = Sequence_comparator(VOCAB)
        self._last_best_path: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        # Only trigger when ModelCheckpoint has saved a new best
        best_path = ""
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_path = cb.best_model_path
                break

        if not best_path or best_path == self._last_best_path:
            return

        self._last_best_path = best_path

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        tf_preds, gen_preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        logger.info(f"New best model (Epoch {epoch}) – best plots will be overwritten.")

        self._generate_plots(
            tf_preds, gen_preds, targets,
            title_prefix=f"Best Model – Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/best",
            filename_prefix="",
        )


# ---------------------------------------------------------------------------
# Test evaluation (once after trainer.test())
# ---------------------------------------------------------------------------

class SequenceTestPlotCallback(_SequencePlotMixin, Callback):
    """
    After trainer.test(), generates the four standard diagnostic plots on the test dataset
    and logs them under 'plots/test/' as MLflow artifacts.

    Executed only once at the end of trainer.test() – not during training.

    Parameters
    ----------
    n_examples : int
        Number of examples in the prediction table (default: 16).
    """

    def __init__(self, n_examples: int = 16):
        self.n_examples = n_examples
        self._comparator = Sequence_comparator(VOCAB)

    def on_test_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        test_dl = trainer.test_dataloaders
        if isinstance(test_dl, list):
            test_dl = test_dl[0]

        tf_preds, gen_preds, targets = self._collect_predictions(test_dl, pl_module)

        self._generate_plots(
            tf_preds, gen_preds, targets,
            title_prefix="Test",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/test",
            filename_prefix="test_",
        )
