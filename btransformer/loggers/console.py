"""
Contains console logger implementation for the btransformer library.
"""

from btransformer.loggers.base import Logger


class ConsoleLogger(Logger):
    """Console-based logger that prints metrics to the console.
    
    Args:
        interval: Interval for flushing logger in steps. Defaults to 1.
        filter: Optional filter string to filter metrics for logging.
    """

    def flush(self) -> None:
        """Prints stored logs to the console."""
        for step, metrics in self._log.items():
            print(f"\nStep {step}:")
            for name, value in metrics.items():
                print(f"\t{name}: {value.item():.2f}")
        self._log.clear()
