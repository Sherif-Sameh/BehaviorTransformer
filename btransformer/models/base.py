"""
Contains base abstract class for trainable models in the btransformer library.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class Model(ABC, nn.Module):
    """
    Abstract base class for trainable models in the btransformer library.

    This class enforces the implementation of a method to split model parameters into two groups:
    those that should be regularized with weight decay and those that should not. In addition to
    this, models are required to provide methods for saving and loading model checkpoints.
    """
    DEFAULT_PATH = Path(__file__).parent / "checkpoints/model.pt"

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

    def save(self, path: Path | None = None) -> None:
        """Save the model's state dict to the specified path.
        
        Args:
            path: Optional path to save the model checkpoint. Defaults to `DEFAULT_PATH` if not
                provided.
        """
        path = self.DEFAULT_PATH if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: Path | None = None) -> None:
        """Load the model's state dict from the specified path.
        
        Args:
            path: Optional path to load the model checkpoint from. If not provided, `DEFAULT_PATH`
                is checked for existing compatible checkpoints.
        """
        load_path = self.DEFAULT_PATH if path is None else path
        if not load_path.exists():
            raise FileNotFoundError(
                f"No checkpoints at {str(path)} or default path {str(self.DEFAULT_PATH)}."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        try:
            self.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as e:
            print(f"Error while loading checkpoint from {str(load_path)}: {e}")