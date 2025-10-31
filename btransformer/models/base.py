"""
Contains base abstract class for trainable models in the btransformer library.

Extends torch.nn.Module class to enforce the definition of a split parameters method for weight regularization.
"""

from abc import ABC, abstractmethod

import torch.nn as nn


class Model(ABC, nn.Module):
    """
    Abstract base class for trainable models in the btransformer library.

    This class enforces the implementation of a method to split model parameters into two groups:
    those that should be regularized with weight decay and those that should not.
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