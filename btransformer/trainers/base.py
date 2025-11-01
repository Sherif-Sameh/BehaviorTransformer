"""
Contains base abstract class for PyTorch model trainers in the btransformer library.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from btransformer.loggers.compose import ComposeLogger
from btransformer.metrics.compose import ComposeMetric
from btransformer.models.base import Model


class Trainer(ABC):
    """Base abstract class for model trainers in the btransformer library.
    
    Args:
        model: Model to be trained.
        ckpt_path: Optional path to save/load model checkpoints.
    """

    def __init__(self, model: Model, ckpt_path: Path | None = None) -> None:
        self.model = model
        self.ckpt_path = ckpt_path

    @abstractmethod
    def train(
        self,
        n_epochs: int,
        learning_rate: float,
        metrics: ComposeMetric,
        loggers: ComposeLogger,
        dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
    ) -> None:
        """Train and evaluate model for a number of epochs.
        
        Args:
            n_epochs: Number of epochs to train for.
            learning_rate: Learning rate used by optimizer.
            metrics: Composed metrics to track during training and evaluation.
            loggers: Composed loggers for logging tracked metrics.
            dataloader: Dataloader for training dataset.
            test_dataloader: Optional dataloader for testing dataset.
        """
        pass
    
    @abstractmethod
    def _train_step(
        self,
        optimizer: Optimizer,
        metrics: ComposeMetric,
        batch: tuple[Tensor, ...],
    ) -> None:
        """Perform a single training step using given batch and update metrics.
        
        Args:
            optimizer: PyTorch optimizer to use for updating model.
            metrics: Composed metrics to update with training stats.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
        pass
    
    @abstractmethod
    def _eval_step(
        self,
        metrics: ComposeMetric,
        batch: tuple[Tensor, ...],
    ) -> None:
        """Perform a single evaluation step using given batch and update metrics.
        
        Args:
            metrics: Composed metrics to update with evaluation stats.
            batch: A tuple of arbitrary length of model input/s and target/s.
        """
        pass
