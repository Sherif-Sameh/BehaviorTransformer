"""
Contains base abstract class for trainable models in the btransformer library.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn


class Model(ABC, nn.Module):
    """
    Abstract base class for trainable models in the btransformer library.

    This class enforces the implementation of a method to split model parameters into two groups:
    those that should be regularized with weight decay and those that should not. In addition to
    this, models are required to provide methods for saving and loading model checkpoints.
    """

    @abstractmethod
    def split_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Splits the model's parameters into two groups:
        1. Parameters to apply weight decay (e.g., nn.Linear.weight).
        2. Parameters to exclude from weight decay (e.g., nn.Linear.bias, nn.LayerNorm.weight).

        Returns:
            tuple
            - decay: List of parameters to apply weight decay.
            - no_decay: List of parameters to exclude from weight decay.
        """
        pass

    @abstractmethod
    def save(self, path: Path | None = None) -> None:
        """Save the model's state dict to the specified path.
        
        Args:
            path: Optional path to save the model checkpoint. Should default to a 'checkpoints'
                directory within the 'models' directory if not provided.
        """
        pass

    @abstractmethod
    def load(self, path: Path | None = None) -> None:
        """Load the model's state dict from the specified path.
        
        Args:
            path: Optional path to load the model checkpoint from. If not provided, method should
                default to looking inside a 'checkpoints' directory within the 'models' directory.
        """
        pass