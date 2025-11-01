"""
Contains base abstract class for loggers in the btransformer library.
"""

from abc import ABC, abstractmethod

from torch import Tensor

from btransformer.metrics.compose import ComposeMetric


class Logger(ABC):
    """Base abstract class for loggers in the btransformer library.
    
    Args:
        interval: Interval for flushing logger in steps. Defaults to 1.
        filter: Optional filter string to filter metrics for logging.
    """

    def __init__(self, interval: int = 1, filter: str | None = None,
    ):
        self._interval = interval
        self._filter = filter
        self._log = dict()
        self._count = 0
    
    @abstractmethod
    def flush(self) -> None:
        """Flushes stored logs to the logging output destination."""
        pass

    def log(self, step: int, metrics: ComposeMetric, reset: bool = False) -> None:
        """Logs all tracked metrics at the given step.
        
        Logger can optionally trigger the metrics' reset method after logging. Logs are flushed
        automatically if the set logging interval is reached.
        
        Args:
            step: Current step for logging metrics.
            metrics: Composed metrics to track and log.
            reset: Reset metrics after logging.
        """
        metrics_dict = metrics.compute()
        metrics_dict = self.filter(metrics_dict)
        self._log[step] = metrics_dict
        self._count += 1
        if self._count % self._interval == 0:
            self.flush()
            self._count = 0
        if reset:
            metrics.reset()

    def filter(self, metrics: dict[str, Tensor]) -> dict[str, Tensor]:
        """Filters the provided metrics based on the logger's filter string.
        
        Args:
            metrics: Dictionary mapping metric names to their Tensor values.
        
        Returns:
            Filtered dictionary of metrics.
        """
        if self._filter is None:
            return metrics
        return {k: v for k, v in metrics.items() if self._filter in k}
