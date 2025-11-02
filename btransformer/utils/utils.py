"""
Contains all utility functions used throughout the btransformer library. 
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms.v2 as tf

def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility across NumPy and PyTorch.
    
    Args:
        seed: Seed value to set. If None, no action is taken.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sqr_l2_norm(x: Tensor, y: Tensor) -> Tensor:
    """Compute the pairwise squared L2 norm between each datapoint in x and y.
    
    Args:
        x: (..., N, D) tensor of datapoints.
        y: (..., M, D) tensor of datapoints.
    
    Returns:
        (..., N, M) tensor of pairwise squared L2 norms between datapoints.
    """
    return torch.sum((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2, dim=-1)


def convert_transforms(transforms: list[dict[str, Any]]) -> tf.Compose:
    """Convert transforms from list configuration to a Compose tranform."""
    transforms_list = []
    for transform in transforms:
        assert "type" in transform, \
            "Each transform dict must have a 'type' entry specifying its class."
        assert hasattr(tf, transform["type"]), \
            f"Found no valid transform with type {transform['type']} in transforms module."
        type = transform["type"]
        kwargs = transform.get("kwargs", {})
        cls = getattr(tf, type)
        transforms_list.append(cls(**kwargs))
    return tf.Compose(transforms_list)