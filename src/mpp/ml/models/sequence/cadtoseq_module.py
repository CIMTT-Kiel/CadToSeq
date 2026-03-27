#standard library imports
import logging

# third party imports
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import mlflow
from torch.nn.attention import sdpa_kernel, SDPBackend

#custom imports
from mpp.constants import VOCAB
from mpp.ml.models.sequence.vecset_transformer import ARMSTD
from mpp.ml.metrics.sequences import Sequence_comparator

class ARMSTM(pl.LightningModule):
    """
    PyTorch Lightning Module for training a Transformer-based model for 
    autoregressive manufacturing step prediction.

    This module wraps the ARMSTD decoder model and provides:
    - Autoregressive sequence modeling using teacher forcing during training
    - Training, validation, and inference logic
    - Cross-entropy loss (with PAD-token masking)
    - Sequence-level evaluation metrics (exact match, Levenshtein distance, etc.)
    - Learning rate scheduling and optimizer configuration

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer (default: 3e-5).
    embed_dim : int, optional
        Embedding dimension used in the transformer (default: 128).
    nhead : int, optional
        Number of attention heads in the transformer (default: 4).
    num_layers : int, optional
        Number of transformer decoder layers (default: 3).
    dropout : float, optional
        Dropout probability (default: 0.3).
    weight_decay : float, optional
        Weight decay for regularization (default: 0.01).
    max_epochs : int, optional
        Maximum number of epochs for training (default: 100).
    ss_epsilon_max : float, optional
        Maximum scheduled sampling rate (0.0 = disabled, default: 0.3).
        After ss_warmup_epochs, epsilon grows linearly from 0 to this value.
    ss_warmup_epochs : int, optional
        Epochs without scheduled sampling at the start of training (default: 50).
    """

    def __init__(self, lr=0.00003, embed_dim=128, nhead=4, num_layers=3, dropout=0.3, weight_decay=0.01, max_epochs=100, use_scheduler=True, ss_epsilon_max=0.3, ss_warmup_epochs=50):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.use_scheduler = use_scheduler

        #model spezific
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.model = ARMSTD(embed_dim=self.embed_dim, num_layers=self.num_layers, nhead=self.nhead, dropout=self.dropout)

        # Scheduled Sampling
        self.ss_epsilon = 0.0
        self.ss_epsilon_max = ss_epsilon_max
        self.ss_warmup_epochs = ss_warmup_epochs

        self.save_hyperparameters()
        self.save_hyperparameters("lr", "embed_dim", "nhead", "num_layers", "dropout", "weight_decay", "max_epochs")

        self.criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["PAD"])  # ignore PAD tokens in the loss calculation

        # Initialize the sequence comparator to evaluate additional sequence metrics
        self.comparator = Sequence_comparator(VOCAB)


    def forward(self, vector_set, tgt_seq):
        """
        Forward pass of the model.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input features of shape (batch_size, set_size, input_dim).
        tgt_seq : torch.Tensor
            Target sequence input to the decoder (e.g., shifted ground truth tokens).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, seq_len, num_classes).
        """

        
        return self.model(vector_set, tgt_seq)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step with scheduled sampling.

        With ss_epsilon > 0, each decoder input position is replaced with probability
        epsilon by the model's own prediction (scheduled sampling).
        Position 0 (START token) is never replaced.

        Parameters
        ----------
        batch : tuple
            A batch containing input vector sets and padded target sequences.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed training loss.
        """
        vector_set, padded_targets = batch
        decoder_input = padded_targets[:, :-1]  # [START, step1, ..., STOP] – remove last token

        if self.ss_epsilon > 0.0:
            with torch.no_grad():
                predicted = self(vector_set, decoder_input).argmax(dim=-1)
            mask = torch.bernoulli(
                torch.full(decoder_input.shape, self.ss_epsilon, dtype=torch.float, device=decoder_input.device)
            ).bool()
            mask[:, 0] = False                          # START-Token nie ersetzen
            mask &= (decoder_input != VOCAB["PAD"])     # PAD-Positionen nie ersetzen
            decoder_input = torch.where(mask, predicted, decoder_input)

        logits = self(vector_set, decoder_input)
        loss = self.criterion(logits.view(-1, VOCAB.__len__()), padded_targets[:, 1:].reshape(-1))

        self.log("train_loss", loss)
        self.log("ss_epsilon", self.ss_epsilon)

        return loss

    def on_train_epoch_end(self):
        """Linearly updates ss_epsilon from 0 to ss_epsilon_max after the warmup phase."""
        if self.ss_epsilon_max > 0.0 and self.current_epoch >= self.ss_warmup_epochs:
            progress = (self.current_epoch - self.ss_warmup_epochs) / max(
                self.trainer.max_epochs - self.ss_warmup_epochs, 1
            )
            self.ss_epsilon = min(self.ss_epsilon_max, progress * self.ss_epsilon_max)
    
    def generate(self, vector_set, return_probs=False, device="cpu"):
        """
        Generate sequences in autoregressive fashion using the decoder model.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input vector sets.
        return_probs : bool, optional
            Whether to return token probabilities (default: False).
        device : str, optional
            Device to run generation on (default: "cpu").

        Returns
        -------
        torch.Tensor
            Generated sequences (without START token).
        """
        return self.model.generate(vector_set, return_probs=False, device=device)

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step with optional metric logging.

        Parameters
        ----------
        batch : tuple
            A batch containing input vector sets and padded target sequences.
        batch_idx : int
            Index of the current validation batch.
        """
        vector_set, padded_targets = batch

        # Teacher forcing: val_loss + val_acc as optimization signal
        decoder_input = padded_targets[:, :-1]  # [START, step1, ..., STOP] – remove last token
        logits = self(vector_set, decoder_input)
        targets_shifted = padded_targets[:, 1:]  # [step1, ..., STOP, PAD, ...] – remove START

        val_loss = self.criterion(logits.view(-1, VOCAB.__len__()), targets_shifted.reshape(-1))
        self.log("val_loss", val_loss, prog_bar=True)

        preds_tf = logits.argmax(dim=-1)
        mask = targets_shifted != VOCAB["PAD"]
        acc = ((preds_tf == targets_shifted) & mask).sum().float() / mask.sum()
        self.log("val_acc", acc, prog_bar=True)

        # Autoregressive generation: sequence metrics under real inference conditions
        if batch_idx % 2 == 0:
            with torch.no_grad():
                gen_seqs = self.model.generate(vector_set, device=vector_set.device)

            targets_for_eval = padded_targets[:, 1:gen_seqs.size(1) + 1]

            s_metrics_ar = self.comparator.compare(gen_seqs, targets_for_eval)
            self.log("val_exact_match", s_metrics_ar["exact_match"].to(torch.float).mean(), prog_bar=True)
            self.log("val_elementwise_accuracy", s_metrics_ar["elementwise_accuracy"].mean(), prog_bar=True)
            self.log("val_levenshtein_distance", s_metrics_ar["levenshtein_distance"].mean(), prog_bar=True)

            # Teacher-forcing counterparts for directly reading the exposure-bias gap
            # preds_tf has length max_len-1=9, targets_for_eval has length max_seq_len=6 → align lengths
            s_metrics_tf = self.comparator.compare(preds_tf[:, :targets_for_eval.size(1)], targets_for_eval)
            self.log("val_exact_match_tf", s_metrics_tf["exact_match"].to(torch.float).mean())
            self.log("val_elementwise_accuracy_tf", s_metrics_tf["elementwise_accuracy"].mean())

        # Log the learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

    def test_step(self, batch, batch_idx):
        """
        Computes sequence metrics on the test dataset – exclusively autoregressive generation,
        no teacher forcing. Called once after training via trainer.test().
        """
        vector_set, padded_targets = batch

        with torch.no_grad():
            gen_seqs = self.model.generate(vector_set, device=vector_set.device)

        targets_for_eval = padded_targets[:, 1:gen_seqs.size(1) + 1]
        s_metrics = self.comparator.compare(gen_seqs, targets_for_eval)

        self.log("test_exact_match", s_metrics["exact_match"].to(torch.float).mean())
        self.log("test_elementwise_accuracy", s_metrics["elementwise_accuracy"].mean())
        self.log("test_levenshtein_distance", s_metrics["levenshtein_distance"].mean())

        
    def on_train_start(self):
        """Logs the actual Flash Attention status as an MLflow tag."""
        device = next(self.parameters()).device
        precision = self.trainer.precision  # e.g. "bf16-mixed", "32"

        on_cuda = device.type == "cuda"
        is_low_precision = any(p in str(precision) for p in ("16", "bf16"))

        # Check whether Flash Attention is actually running: force a small dummy forward pass
        flash_active = False
        if on_cuda and is_low_precision:
            try:
                # SDPA expects (batch, heads, seq_len, head_dim)
                nhead = self.hparams.nhead
                head_dim = self.hparams.embed_dim // nhead
                dummy = torch.randn(1, nhead, 4, head_dim, device=device, dtype=torch.bfloat16)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.nn.functional.scaled_dot_product_attention(dummy, dummy, dummy)
                flash_active = True
            except Exception:
                flash_active = False

        mlflow.log_params({
            "flash_attention_active": flash_active,
            "training_device": str(device),
            "training_precision": str(precision),
            "batch_size": self.trainer.train_dataloader.batch_size,
        })

        status = "ACTIVE" if flash_active else "INACTIVE"
        logging.getLogger(__name__).info(
            f"Flash Attention: {status}  (device={device}, precision={precision})"
        )

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary with optimizer and scheduler configuration.
        """
        # AdamW-Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),          
            lr=self.hparams.lr,        
            weight_decay=self.hparams.weight_decay  
        )

        if not self.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }