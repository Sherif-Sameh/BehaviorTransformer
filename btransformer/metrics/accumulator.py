"""
Contains the accumulator metric implementation for the btransformer library.
"""

from typing import Literal

import torch
from torch import Tensor

from btransformer.metrics.base import Metric


class AccumulatorMetric(Metric):
    """Accumulator metric.

    The metric accumulates values provided during updates. Then, the metric value is computed
    by reducing those accumulated values according to the chosen reduction method.

    Args:
        argname: Name of the argument to accumulate from during metric updates.
        red: Reduction method for computing metric. Must be one of "sum", "mean", "cnt".
        name: Optional name for the metric. If None, name is determined from argname and red.
    """

    def __init__(
        self,
        argname: str,
        red: Literal["sum", "mean", "cnt"],
        name: str | None = None,
    ):
        assert red in ["sum", "mean", "cnt"], \
            f"Reduction method {red} not supported."
        name = argname.replace("_", " ").title() + f"({red.title()})" if name is None else name
        super().__init__(name=name)
        self.argname = argname
        self.red = red
        self.reset()

    def compute(self) -> Tensor:
        """Computes and returns the metric value by reducing the accumulated state.
        
        Returns:
            Tensor containing the reduced state according to the set reduction method.
        """
        if self.state is None:
            return torch.tensor(float('nan'))
        match self.red:
            case "sum":
                return self.state
            case "mean":
                return self.state / self.count
            case "cnt":
                return torch.tensor(self.count, device=self.state.device)
            
    def reset(self) -> None:
        """Resets the internal state and count to their default values."""
        self.state = None
        self.count = 0

    def update(self, **kwargs) -> None:
        """Updates the internal state and count with the provided value.

        **Warning**: The provided value is reduced with `sum()` before being added to the existing
        state.
        
        Args:
            **kwargs: Keyword arguments containing input tensor data to update metric with.
        """
        value: Tensor | None = kwargs.get(self.argname)
        if value is None:
            return  # There's nothing to update.
        if self.state is None:
            self.state = torch.zeros(1, device=value.device)
        self.state += value.sum()
        self.count += value.numel()
