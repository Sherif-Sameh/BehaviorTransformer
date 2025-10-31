"""
Contains all utility functions used throughout the btransformer library. 
"""

import numpy as np
import torch
from torch import Tensor


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
