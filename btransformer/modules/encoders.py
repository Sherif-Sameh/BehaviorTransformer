"""
Contains all submodules needed for the image encoders in the btransformer library.

Largely adapted from Andrej Karpathy's implementation of minGPT available at: 

<https://github.com/karpathy/minGPT/tree/master>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialSoftMax(nn.Module):
    """Implementation of the spatial soft-max layer from:
    `Deep Spatial Autoencoders for Visuomotor Learning` <https://arxiv.org/abs/1509.06113>.
    
    Args:
        alpha: Initial value for temperature parameter alpha.
        learn_alpha: Learn the temperature parameter alpha.
        normalize_coords: Use normalized coordinates [-1, 1] instead of [0, h] and [0, w].
    """

    def __init__(
        self,
        alpha: float = 1.0,
        learn_alpha: bool = False,
        normalize_coords: bool = True,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor([alpha], dtype=torch.float32)), requires_grad=learn_alpha
        )
        self.normalize_coords = normalize_coords
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the spatial soft-max layer.
        
        Args:
            x: (B, C, H, W) Tensor of input feature maps.
        
        Returns:
            (B, C * 2) Tensor of output expected 2D positions of each channel weighted according to
                the stength of their activations.
        """
        assert x.ndim == 4, f"Input must be a 4-dim tensor, got {x.ndim} dims."
        B, C, H, W = x.shape

        # Apply softmax with temperature parameter to each channel
        x = x.view(B * C, H * W)
        logits = x / (torch.exp(self.log_alpha) + 1e-8)
        probs = F.softmax(logits, dim=-1)

        # Get grid coordinates and computed their expected values according to probs
        x_coords, y_coords = self._get_grid_coords(x.view(B, C, H, W))
        x_coords = torch.sum(x_coords * probs, dim=-1).view(B, C)
        y_coords = torch.sum(y_coords * probs, dim=-1).view(B, C)

        # Combine x and y-coordinates along channel dimension
        out = torch.cat([x_coords, y_coords], dim=-1)  # (B, C * 2)
        return out
    
    def _get_grid_coords(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Get the X and Y coordinates of each spatial position in the feature maps.
        
        Args:
            x: (B, C, H, W) Tensor of input feature maps.
        
        Returns:
            tuple
            - x_coords: (H * W,) Tensor of flattened grid X coordinates.
            - y_coords: (H * W,) Tensor of flattened grid Y coordinates.
            """
        dtype, device = x.dtype, x.device
        H, W = x.shape[2:]
        if self.normalize_coords:
            x_coords, y_coords = torch.meshgrid(
                torch.linspace(-1, 1, W, dtype=dtype, device=device),
                torch.linspace(-1, 1, H, dtype=dtype, device=device),
                indexing="xy",
            )
        else:
            x_coords, y_coords = torch.meshgrid(
                torch.arange(W, dtype=dtype, device=device),
                torch.arange(H, dtype=dtype, device=device),
                indexing="xy",
            )
        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        return x_coords, y_coords
