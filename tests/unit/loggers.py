from pathlib import Path

import pandas as pd
import pytest
import torch

from btransformer.loggers import (
    ConsoleLogger,
    CSVLogger,
)
from btransformer.metrics import (
    AccumulatorMetric,
    ComposeMetric,
)

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_console_logger(device: torch.device, capsys: pytest.CaptureFixture):
    with capsys.disabled():
        print(f"Console logs from device: {str(device).upper()}")
        # Initialize some metrics to log
        metric1 = AccumulatorMetric("values1", red="mean", name="MeanMetric")
        metric2 = AccumulatorMetric("values2", red="sum", name="SumMetric")
        metrics = ComposeMetric([metric1, metric2])
        
        # Update metrics with synthetic data
        values1, values2 = torch.randn(20, device=device).split(10)
        metrics.update(values1=values1, values2=values2)
        
        # Initialize and test standard console logger
        logger = ConsoleLogger(interval=1, filter=None)
        logger.log(step=1, metrics=metrics, reset=True)
        metrics_dict = metrics.compute()
        assert logger._log == {}
        assert torch.isnan(metrics_dict["MeanMetric"])
        assert torch.isnan(metrics_dict["SumMetric"])

        # Initialize and test non-standard console logger with filter
        logger = ConsoleLogger(interval=2, filter="Mean")
        metrics.update(values1=values1, values2=values2)
        logger.log(step=1, metrics=metrics, reset=False)
        assert logger._log != {}
        metrics.update(values1=values1, values2=values2)
        logger.log(step=2, metrics=metrics, reset=True)
        metrics_dict = metrics.compute()
        assert torch.isnan(metrics_dict["MeanMetric"])
        assert torch.isnan(metrics_dict["SumMetric"])
        print("\n")


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_csv_logger(device: torch.device):
    # Initialize some metrics to log
    metric1 = AccumulatorMetric("values1", red="mean", name="MeanMetric")
    metric2 = AccumulatorMetric("values2", red="sum", name="SumMetric")
    metrics = ComposeMetric([metric1, metric2])
    
    # Update metrics with synthetic data
    values1, values2 = torch.randn(20, device=device).split(10)
    metrics.update(values1=values1, values2=values2)
    
    # Initialize and test standard CSV logger
    csv_path = Path(__file__).parent / "test.csv"
    if csv_path.exists():
        csv_path.unlink()
    logger = CSVLogger(path=csv_path, interval=1, filter=None)
    logger.log(step=1, metrics=metrics, reset=True)
    metrics_dict = metrics.compute()
    assert logger._log == {}
    assert torch.isnan(metrics_dict["MeanMetric"])
    assert torch.isnan(metrics_dict["SumMetric"])
    df = pd.read_csv(csv_path, index_col=0)
    assert "MeanMetric" in df.columns and "SumMetric" in df.columns
    assert df.shape[0] == 1

    # Attempt to append more logs to the same CSV file
    metrics.update(values1=values1, values2=values2)
    logger.log(step=2, metrics=metrics, reset=False)
    metrics.update(values1=values1, values2=values2)
    logger.log(step=3, metrics=metrics, reset=True)
    df = pd.read_csv(csv_path, index_col=0)
    assert "MeanMetric" in df.columns and "SumMetric" in df.columns
    assert df.shape[0] == 3
    csv_path.unlink()  # Clean up test CSV file
