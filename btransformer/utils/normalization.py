"""
Contains all utility functions related to normalization used throughout the btransformer library. 
"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def unnormalize_imgs(
    imgs: Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> NDArray[np.uint8]:
    """Unnormalizes a batch of images given the mean and standard deviation.

    Images are converted back to a NumPy uint8 array format of shape (B, H, W, C). 
    
    Args:
        imgs: (B, C, H, W) Tensor of normalized images.
        mean: Sequence of means used for normalization for each channel.
        std: Sequence of standard deviations used for normalization for each channel.
    
    Returns:
        (B, H, W, C) Array of unnormalized images of type np.uint8.
    """
    C = imgs.shape[1]
    assert len(mean) == C, f"Length of means {len(mean)} should match number of channels {C}."
    assert len(std) == C, f"Length of stds {len(std)} should match number of channels {C}."
    device = imgs.device
    mean = torch.tensor(mean, device=device).view(1, C, 1, 1)
    std = torch.tensor(std, device=device).view(1, C, 1, 1)
    unnormalized_imgs = (imgs * std + mean).clamp(min=0, max=1)
    unnormalized_imgs = (unnormalized_imgs * 255).byte().permute(0, 2, 3, 1)
    return unnormalized_imgs.cpu().numpy()


def rescale_actions(
    acts: Tensor,
    acts_low: Sequence[float],
    acts_high: Sequence[float],
) -> Tensor:
    """Rescales a batch of actions to the range [-1, 1] given the action space bounds.
    
    Args:
        acts: (B, A) Tensor of unnormalized actions.
        acts_low: Sequence of lower bounds for action space.
        acts_high: Sequence of upper bounds for action space.
    
    Returns:
        (B, A) Tensor of rescaled actions in the range [-1, 1].
    """
    A = acts.shape[1]
    assert len(acts_low) == A, \
        f"Length of action lows {len(acts_low)} should match action dimension {A}."
    assert len(acts_high) == A, \
        f"Length of action highs {len(acts_high)} should match action dimension {A}."
    device = acts.device
    acts_low = torch.tensor(acts_low, device=device).view(1, A)
    acts_high = torch.tensor(acts_high, device=device).view(1, A)
    rescaled_acts = (acts - acts_low) / (acts_high - acts_low)
    rescaled_acts = rescaled_acts * 2 - 1
    return rescaled_acts


def unscale_actions(
    rescaled_acts: Tensor,
    acts_low: Sequence[float],
    acts_high: Sequence[float],
) -> Tensor:
    """Unscales a batch of actions from the range [-1, 1] back to the original action space bounds.

    Reverse the rescaling performed by the `rescale_actions` function.
    
    Args:
        rescaled_acts: (B, A) Tensor of rescaled actions in the range [-1, 1].
        acts_low: Sequence of lower bounds for action space.
        acts_high: Sequence of upper bounds for action space.
    
    Returns:
        (B, A) Tensor of unscaled actions in the original action space bounds.
    """
    A = rescaled_acts.shape[1]
    assert len(acts_low) == A, \
        f"Length of action lows {len(acts_low)} should match action dimension {A}."
    assert len(acts_high) == A, \
        f"Length of action highs {len(acts_high)} should match action dimension {A}."
    device = rescaled_acts.device
    acts_low = torch.tensor(acts_low, device=device).view(1, A)
    acts_high = torch.tensor(acts_high, device=device).view(1, A)
    acts = (rescaled_acts + 1) / 2
    acts = acts * (acts_high - acts_low) + acts_low
    return acts

