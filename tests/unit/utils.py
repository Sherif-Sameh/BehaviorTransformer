import pytest
import numpy as np
import torch

from btransformer.utils.normalization import (
    unnormalize_imgs,
    rescale_actions,
    unscale_actions,
)

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_unnormalize_imgs(device: torch.device):
    # Generate synthetic normalized images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    imgs = torch.normal(
        torch.tensor(mean, device=device).repeat((16, 64, 64, 1)),
        torch.tensor(std, device=device).repeat((16, 64, 64, 1)),
    ).permute(0, 3, 1, 2)  # (B, C, H, W)

    # Unnormalize images
    unnormalized_imgs = unnormalize_imgs(imgs, mean, std)
    assert type(unnormalized_imgs) is np.ndarray
    assert unnormalized_imgs.dtype == np.uint8
    assert unnormalized_imgs.shape == (16, 64, 64, 3)
    out_of_bounds = np.logical_or(
        unnormalized_imgs < 0,
        unnormalized_imgs > 255,
    ).sum()
    assert out_of_bounds == 0


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_rescale_actions(device: torch.device):
    # Generate synthetic unscaled actions
    acts_low = [-2.0, 0.0, -1.0]
    acts_high = [2.0, 4.0, 1.0]
    acts_low_t = torch.tensor(acts_low, device=device)
    acts_high_t = torch.tensor(acts_high, device=device)
    acts = torch.rand(
        (1000, 3), device=device
    ) * (acts_high_t - acts_low_t) + acts_low_t

    # Rescale actions
    rescaled_acts = rescale_actions(acts, acts_low, acts_high)
    assert rescaled_acts.shape == acts.shape
    assert torch.all(rescaled_acts >= -1.0) and torch.all(rescaled_acts <= 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_unscale_actions(device: torch.device):
    # Generate synthetic unscaled actions
    acts_low = [-2.0, 0.0, -1.0]
    acts_high = [2.0, 4.0, 1.0]
    acts_low_t = torch.tensor(acts_low, device=device)
    acts_high_t = torch.tensor(acts_high, device=device)
    acts = torch.rand(
        (1000, 3), device=device
    ) * (acts_high_t - acts_low_t) + acts_low_t

    # Rescale and then unscale actions
    rescaled_acts = rescale_actions(acts, acts_low, acts_high)
    unscaled_acts = unscale_actions(rescaled_acts, acts_low, acts_high)
    assert unscaled_acts.shape == acts.shape
    assert torch.allclose(unscaled_acts, acts, atol=1e-6)
