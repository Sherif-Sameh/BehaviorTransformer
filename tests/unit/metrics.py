import pytest
import torch

from btransformer.metrics.abs_error import AbsErrorMetric
from btransformer.metrics.accumulator import AccumulatorMetric
from btransformer.metrics.compose import ComposeMetric

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_accumulator_metric(device: torch.device):
    # Initialize accumulator metric with mean reduction
    argname = "values"
    metric = AccumulatorMetric(argname, red="mean")
    
    # Test updates(), compute() and reset()
    true_values = []
    for _ in range(5):
        data = torch.randn(10, device=device)
        true_values.append(data.clone())
        metric.update(values=data)
    true_values = torch.cat(true_values).mean()
    assert torch.allclose(metric.compute(), true_values)
    metric.reset()
    assert metric.state is None and torch.isnan(metric.compute())

    # Test that updating with other argname does not change state
    metric.update(other_values=torch.randn(10, device=device))
    assert metric.state is None and torch.isnan(metric.compute())


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_abs_error_metric(device: torch.device):
    # Initialize absolute error metric with sum reduction
    pred_argname, targ_argname = "pred", "target"
    metric = AbsErrorMetric(pred_argname, targ_argname, red="sum")
    
    # Test updates(), compute() and reset()
    true_values = []
    for _ in range(5):
        pred, target = torch.randn(20, device=device).split(10)
        abs_error = torch.abs(pred - target)
        true_values.append(abs_error.clone())
        metric.update(pred=pred, target=target)
    true_values = torch.cat(true_values).sum()
    assert torch.allclose(metric.compute(), true_values)
    metric.reset()
    assert metric.state is None and torch.isnan(metric.compute())

    # Test that updating with missing argnames does not change state
    metric.update(value=torch.randn(10, device=device))
    metric.update(pred=torch.randn(10, device=device))
    metric.update(target=torch.randn(10, device=device))
    assert metric.state is None and torch.isnan(metric.compute())


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_compose_metric(device: torch.device):
    # Initialize two simple accumulator and abs error metrics
    metric1 = AccumulatorMetric("values", red="sum", name="SumMetric")
    metric2 = AbsErrorMetric("pred", "target", red="mean", name="MeanMetric")
    
    # Compose the two metrics
    composed_metric = ComposeMetric([metric1, metric2])
    
    # Test updates(), compute() and reset()
    true_values = []
    true_abs_errors = []
    for _ in range(5):
        values, pred, target = torch.randn(30, device=device).split(10)
        abs_error = torch.abs(pred - target)
        true_values.append(values.clone())
        true_abs_errors.append(abs_error.clone())
        composed_metric.update(values=values, pred=pred, target=target)
    true_values = torch.cat(true_values).sum()
    true_abs_errors = torch.cat(true_abs_errors).mean()
    computed_metrics = composed_metric.compute()
    assert torch.allclose(computed_metrics["SumMetric"], true_values)
    assert torch.allclose(computed_metrics["MeanMetric"], true_abs_errors)    
    composed_metric.reset()
    computed_metrics = composed_metric.compute()
    assert torch.isnan(computed_metrics["SumMetric"])
    assert torch.isnan(computed_metrics["MeanMetric"])

    # Test that partial updates work correctly
    composed_metric.update(values=values)
    computed_metrics = composed_metric.compute()
    assert torch.allclose(computed_metrics["SumMetric"], values.sum())
    assert torch.isnan(computed_metrics["MeanMetric"])
    composed_metric.reset()

    # Test that updating with missing argnames does not change state
    composed_metric.update(other_values=torch.randn(10, device=device))
    computed_metrics = composed_metric.compute()
    assert torch.isnan(computed_metrics["SumMetric"])
    assert torch.isnan(computed_metrics["MeanMetric"])
