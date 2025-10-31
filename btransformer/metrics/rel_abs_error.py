"""
Contains the relative error metric implementation for the btransformer library.
"""

from typing import Literal

import torch
from torch import Tensor

from btransformer.metrics.accumulator import AccumulatorMetric


class RelAbsErrorMetric(AccumulatorMetric):
    """Relative absolute error metric.

    Accumulates relative absolute errors between predictions and targets. Then, the metric value is
    computed by reducing those accumulated values according to the chosen reduction method.

    Args:
        pred_argname: Name of the argument containing predictions during metric updates.
        targ_argname: Name of the argument containing targets during metric updates.
        red: Reduction method for computing metric.
            Must be one of "sum", "mean", "std", "max", "min".
        name: Optional name for the metric. If None, name is determined from both argnames and red.
    """

    def __init__(
        self,
        pred_argname: str,
        targ_argname: str,
        red: Literal["sum", "mean", "std", "max", "min"],
        name: str | None = None,
    ):
        pred_name = pred_argname.replace('_', ' ').title()
        targ_name = targ_argname.replace('_', ' ').title()
        name = f"Rel. Error of {pred_name} vs {targ_name}, ({red.title()})" \
            if name is None else name
        super().__init__("rel_error_temp", red, name=name)
        self.pred_argname = pred_argname
        self.targ_argname = targ_argname
    
    def update(self, **kwargs) -> None:
        """Updates the internal state and count with the provided value.

        **Warning**: The provided tensors' shapes **must be broadcastable** to each other and the
        existing state's shape.
        
        Args:
            **kwargs: Keyword arguments containing input tensor data to update metric with.
        """
        pred: Tensor | None = kwargs.get(self.pred_argname)
        targ: Tensor | None = kwargs.get(self.targ_argname)
        if pred is None or targ is None:
            return  # There's nothing to update.
        value = torch.abs(pred - targ) / (torch.abs(targ) + 1e-8)
        if self.state is None:
            self.state = torch.zeros_like(value)
        self.state += value
        self.count += 1
