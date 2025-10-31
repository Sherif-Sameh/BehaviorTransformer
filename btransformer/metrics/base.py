"""
Contains base abstract class for metrics in the btransformer library.

Heavily inspired by Flax NNX's metrics implementation.
"""

from abc import ABC, abstractmethod

from torch import Tensor


class Metric(ABC):
    """Base abstract class for tracking metrics in the btransformer library.

    Heavily based on the flax.nnx.Metric class.

    Args:
        name: Optional name for the metric. If None, defaults to "Metric".
    """
    
    def __init__(self, name: str | None = None):
        self._name = "Metric" if name is None else name
    
    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Sets the name of the metric."""
        self._name = value

    @abstractmethod
    def compute(self) -> Tensor:
        """Computes and returns the metric value based on the accumulated state.
        
        Returns:
            Tensor containing the computed metric value.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the metric's internal accumulated state."""
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Updates the metric's internal accumulated state based on input data.
        
        Args:
            **kwargs: Keyword arguments containing input tensor data to update metric with.
        """
        pass
