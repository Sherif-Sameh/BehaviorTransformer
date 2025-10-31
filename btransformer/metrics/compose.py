"""
Contains the metric composition implementation for the btransformer library.
"""

from torch import Tensor

from btransformer.metrics.base import Metric


class ComposeMetric(Metric):
    """Composes multiple metrics into a single metric.

    Args:
        metrics: List of metrics to compose.
        name: Optional name for the composed metric. If None, defaults to "ComposedMetric".
    """

    def __init__(self, metrics: list[Metric], name: str | None = None):
        name = "Composed Metric" if name is None else name
        super().__init__(name=name)
        self.metrics = metrics

    def compute(self) -> dict[str, Tensor]:
        """Computes and returns the values of all composed metrics.

        Returns:
            Dictionary mapping metric names to their computed Tensor values.
        """
        return {metric.name: metric.compute() for metric in self.metrics}

    def reset(self) -> None:
        """Resets all composed metrics' internal accumulated states."""
        for metric in self.metrics:
            metric.reset()

    def update(self, **kwargs) -> None:
        """Updates all composed metrics' internal accumulated states based on input data.

        Args:
            **kwargs: Keyword arguments containing input tensor data to update metrics with.
        """
        for metric in self.metrics:
            metric.update(**kwargs)
