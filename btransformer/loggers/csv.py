"""
Contains CSV logger implementation using Pandas for the btransformer library.
"""

from pathlib import Path

import pandas as pd

from btransformer.loggers.base import Logger


class CSVLogger(Logger):
    """CSV-based logger that saves metrics to a CSV file using Pandas.

    Args:
        path: Path to the CSV file where logs will be saved.
        interval: Interval for flushing logger in steps. Defaults to 1.
        filter: Optional filter string to filter metrics for logging.
    """

    def __init__(self, path: str | Path, interval: int = 1, filter: str | None = None):
        super().__init__(interval=interval, filter=filter)
        self.path = Path(path) if isinstance(path, str) else path
        assert self.path.suffix == '.csv', \
            f"CSVLogger requires a .csv file extension. Got {self.path.suffix} instead."
        self.path = self.path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()  # Remove old logs

    def flush(self) -> None:
        """Saves stored logs to a CSV file."""
        if not self._log:
            return
        # Convert from tensor.Tensor to float for saving
        for step in self._log.keys():
            for name in self._log[step].keys():
                self._log[step][name] = self._log[step][name].item()
        df = pd.DataFrame.from_dict(self._log, orient='index')
        df.to_csv(self.path, mode='a', header=not self.path.exists())
        self._log.clear()
