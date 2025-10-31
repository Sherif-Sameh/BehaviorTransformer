"""
Contains the logger composition implementation for the btransformer library.
"""

from btransformer.loggers.base import Logger
from btransformer.metrics.compose import ComposeMetric


class ComposeLogger(Logger):
    """Composes mutliple loggers into a single logger.
    
    Args:
        loggers: List of loggers to compose.
    """

    def __init__(self, loggers: list[Logger]):
        super().__init__()
        self.loggers = loggers
    
    def flush(self) -> None:
        """Flushes all loggers' stored logs to their logging output destinations."""
        for logger in self.loggers:
            logger.flush()

    def log(self, step: int, metrics: ComposeMetric, reset: bool = False) -> None:
        """Logs all tracked metrics for all loggers at the given step.
        
        Logger can optionally trigger the metrics' reset method after logging. Individual Logs are
        flushed automatically if their set logging interval is reached.
        
        Args:
            step: Current step for logging metrics.
            metrics: Composed metrics to track and log.
            reset: Reset metrics after logging.
        """
        # TODO: Optimize to avoid redundant calls to `metrics.compute()` across all loggers.
        for logger in self.loggers:
            logger.log(step, metrics, reset=reset)
