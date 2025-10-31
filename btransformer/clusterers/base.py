"""
Contains base abstract class for clusterers in the btransformer library.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from btransformer.utils.utils import sqr_l2_norm


class Clusterer(ABC, nn.Module):
    """
    Abstract base class for clusterers in the btransformer library.

    Clusterers have three main responsibilities in the Behavior Transformer framework:
    1. Determine the locations of action bins to discretize continuous action spaces into discrete
        ones and store those locations.
    2. Encode continuous actions into their discrete bins + continuous offsets.
    3. Decode discrete actions + continuous offsets back into continuous actions.

    Note: Clusterers extend `nn.Module` to ease pickling/unpickling when saving/loading models.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("centers", torch.empty(0))
    
    @property
    def act_dim(self) -> int:
        """Returns the action dimension of the clusterer.

        Returns:
            Action dimension (A).
        """
        return self.centers.shape[-1]
    
    @property
    def centers(self) -> FloatTensor:
        """Returns a copy of the stored cluster centers.

        Returns:
            (n_clusters, A) Tensor of cluster center locations.
        """
        return self.centers.clone()

    @abstractmethod
    def forward(self, acts: FloatTensor) -> FloatTensor:
        """Forward pass of the clusterer.

        Computes and returns the clusters' "center" locations based on the input actions.

        Args:
            acts: (B, A) or (B, T, A) Input actions to cluster.
        
        Returns:
            (n_clusters, A) Tensor of computed cluster center locations.
        """
        pass

    @torch.no_grad
    def encode(self, acts: FloatTensor) -> tuple[LongTensor, FloatTensor]:
        """Encodes continuous actions into their discrete bins + continuous offsets.

        Args:
            acts: (B, T, A) Input continuous actions to encode.

        Returns:
            tuple
            - bins: (B, T) Discrete action bin indices.
            - offsets: (B, T, A) Continuous offsets from the cluster centers.
        """
        assert acts.ndim == 3, f"Actions must be a 3-dim tensor, got {acts.ndim} dims."
        assert acts.shape[-1] == self.act_dim, \
            f"Action dimension does not match. Expected {self.act_dim}, got {acts.shape[-1]}."
        centers = self.centers.unsqueeze(0)     # (1, n_clusters, A)
        dists = sqr_l2_norm(acts, centers)      # (B, T, n_clusters)
        bins = torch.argmin(dists, dim=-1)      # (B, T)
        offsets = acts - self.centers[bins]     # (B, T, A)
        return bins, offsets

    @torch.no_grad
    def decode(self, bins: LongTensor, offsets: FloatTensor) -> FloatTensor:
        """Decodes discrete actions bins + continuous offsets into continuous actions.

        Args:
            bins: (B, T) Discrete action bin indices.
            offsets: (B, T, A) Continuous offsets from the cluster centers.

        Returns:
            (B, T, A) Decoded continuous actions.
        """
        assert bins.ndim == 2, f"Bins must be a 2-dim tensor, got {bins.ndim} dims."
        assert offsets.ndim == 3, f"Offsets must be a 3-dim tensor, got {offsets.ndim} dims."
        assert offsets.shape[-1] == self.act_dim, \
            f"Offset dimension does not match. Expected {self.act_dim}, got {offsets.shape[-1]}."
        acts = self.centers[bins] + offsets  # (B, T, A)
        return acts