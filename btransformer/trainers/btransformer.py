"""
Contains the trainer implementation for the Behavior Transformer (BTransformer) model.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from btransformer.loggers.compose import ComposeLogger
from btransformer.metrics.compose import ComposeMetric
from btransformer.models.btransformer import BehaviorTransformer
from btransformer.trainers.base import Trainer

BatchType = tuple[Tensor, Tensor]               # (obs, action) for BehaviorTransformer
BatchTypeMO = tuple[Tensor, Tensor, Tensor]     # (img_obs, prop_obs, action) for BT with mixed obs


class BTransformerTrainer(Trainer):
    """Trainer for the Behavior Transformer (BTransformer) model using the AdamW optimizer.
    
    Args:
        model: BTransformer model to be trained.
        ckpt_path: Optional path to save/load model checkpoints.
        gamma: Exponent for the focal loss applied to the discrete action bin predictions.
        max_grad_norm: Optional max norm for gradient clipping.
    """
    
    def __init__(
        self,
        model: BehaviorTransformer,
        ckpt_path: Path | None = None,
        gamma: float = 2,
        max_grad_norm: float | None = None,
    ) -> None:
        self.model = model
        self.ckpt_path = ckpt_path
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        # Balancing factor for focal and multi-task losses (initialized in first call to `_loss_fn`)
        self.alpha = None

    def train(
        self,
        n_epochs: int,
        learning_rate: float,
        loss_key: str,
        metrics: ComposeMetric,
        loggers: ComposeLogger,
        dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
        weight_decay: float = 1e-2,
    ) -> None:
        """Train and evaluate model for a number of epochs.
        
        Args:
            n_epochs: Number of epochs to train for.
            learning_rate: Learning rate used by optimizer.
            loss_key: Key for the loss metric to use in determining model quality.
            metrics: Composed metrics to track during training and evaluation.
            loggers: Composed loggers for logging tracked metrics.
            dataloader: Dataloader for training dataset.
            test_dataloader: Optional dataloader for testing dataset.
            weight_decay: Weight decay factor for AdamW optimizer. Defaults to 1e-2.
        """
        self.alpha = None
        loss_min = torch.tensor(torch.inf)
        # Setup AdamW optimizer with weight decay parameter groups
        decay, no_decay = self.model.split_parameters()
        optimizer = AdamW(
            [
                {'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=learning_rate,
        )

        for epoch in range(n_epochs):
            # Training loop
            self.model.train()
            for batch in dataloader:
                self._train_step(optimizer, metrics, batch)
            loggers.log(epoch, metrics, reset=False)

            # Evaluation loop
            if test_dataloader is not None:
                self.model.eval()
                for batch in test_dataloader:
                    self._eval_step(metrics, batch)
                loggers.log(epoch, metrics, reset=False)
            
            # Checkpoint best model so far
            loss = metrics.compute()[loss_key]
            if loss < loss_min:
                loss_min = loss
                self.model.save(self.ckpt_path)
            metrics.reset()

    def _train_step(
        self,
        optimizer: AdamW,
        metrics: ComposeMetric,
        batch: BatchType | BatchTypeMO,
    ) -> None:
        """Perform a single training step using given batch and update metrics.
        
        Args:
            optimizer: PyTorch optimizer to use for updating model.
            metrics: Composed metrics to update with training stats.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
        # Update model
        optimizer.zero_grad()
        *obs, acts = batch
        pred_acts, logits, offsets = self.model(*obs)
        target_bins, target_offsets = self.model.clusterer.encode(acts)
        loss, focal_loss, mt_loss = self._loss_fn(logits, offsets, target_bins, target_offsets)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        optimizer.step()

        # Update training metrics
        metrics.update(
            train_loss=loss.detach(),
            train_focal_loss=focal_loss,
            train_multi_task_loss=mt_loss,
            train_pred_acts=pred_acts.detach(),
            train_targ_acts=acts,
        )
    
    @torch.no_grad
    def _eval_step(
        self,
        metrics: ComposeMetric,
        batch: BatchType | BatchTypeMO,
    ) -> None:
        """Perform a single evaluation step using given batch and update metrics.
        
        Args:
            metrics: Composed metrics to update with evaluation stats.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
        # Get model predictions and losses
        *obs, acts = batch
        pred_acts, logits, offsets = self.model(*obs)
        target_bins, target_offsets = self.model.clusterer.encode(acts)
        loss, focal_loss, mt_loss = self._loss_fn(logits, offsets, target_bins, target_offsets)

        # Update evaluation metrics
        metrics.update(
            eval_loss=loss,
            eval_focal_loss=focal_loss,
            eval_multi_task_loss=mt_loss,
            eval_pred_acts=pred_acts,
            eval_targ_acts=acts,
        )

    def _loss_fn(
        self,
        logits: Tensor,
        offsets: Tensor,
        target_bins: Tensor,
        target_offsets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the combined focal + mult-task loss for the BTransformer model.
        
        Args:
            logits: (B, T, n_bins) Tensor of predicted logits for discrete action bins.
            offsets: (B, T, n_bins, A) Tensor of predicted continuous offsets for each action bin.
            target_bins: (B, T) Tensor of ground truth discrete action bin indices.
            target_offsets: (B, T, A) Tensor of ground truth continuous action offsets.
        
        Returns:
            tuple:
            - loss: Scalar combined loss tensor.
            - focal_loss: Detached focal loss for logging.
            - mt_loss: Detached multi-task loss for logging.
        """
        focal_loss = self._focal_loss(logits, target_bins)
        mt_loss = self._multi_task_loss(offsets, target_bins, target_offsets)
        if self.alpha is None:
            # Make initial losses equal in magnitude
            self.alpha = focal_loss.detach().item() / mt_loss.detach().item()
        loss = focal_loss + self.alpha * mt_loss
        return loss, focal_loss.detach(), mt_loss.detach()
    
    def _focal_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute a multi-class focal loss for discrete action bin predictions.

        The focal loss is a modification to the standard CE loss for handling imbalanced classes.
        Proposed in `Focal Loss for Dense Object Detectio` <https://arxiv.org/abs/1708.02002>.
        
        Args:
            logits: (B, T, n_bins) Tensor of predicted logits for discrete action bins.
            targets: (B, T) Tensor of ground truth discrete action bin indices.
        
        Returns:
            Scalar focal loss tensor.
        """
        ce_loss = F.cross_entropy(
            logits.flatten(0, 1),  # (B * T, n_bins)
            targets.flatten(),     # (B * T,)
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return loss
    
    def _multi_task_loss(
        self,
        offsets: Tensor,
        target_bins: Tensor,
        target_offsets: Tensor,
    ) -> Tensor:
        """Compute the multi-task loss for continuous action offset predictions.
        
        The multi-task loss is a MSE loss computed only between the predicted offsets corresponding
        to the ground truth labels.
        
        Args:
            offsets: (B, T, n_bins, A) Tensor of predicted continuous offsets for each action bin.
            target_bins: (B, T) Tensor of ground truth discrete action bin indices.
            target_offsets: (B, T, A) Tensor of ground truth continuous action offsets.
        
        Returns:
            Scalar multi-task loss tensor.
        """
        B, T, _, A = offsets.shape
        target_bins = target_bins[:, :, None, None].expand(-1, -1, 1, A)
        offsets = offsets.gather(dim=2, index=target_bins).squeeze(2)  # (B, T, A)
        loss = F.mse_loss(offsets, target_offsets, reduction="mean")
        return loss
