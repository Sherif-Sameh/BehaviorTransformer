"""
Contains K-Means clusterer implementation for the btransformer library.
"""

import torch
from torch import FloatTensor

from btransformer.clusterers.base import Clusterer
from btransformer.utils.utils import sqr_l2_norm


class KMeansClusterer(Clusterer):
    """
    Clusterer based on K-Means clustering.
    
    Implements the k-means algorithm to determine cluster centers from input actions.
    """

    def __init__(self, n_clusters: int, max_iters: int, tol: float = 1e-3):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
    
    @torch.no_grad
    def forward(self, acts: FloatTensor) -> FloatTensor:
        """Forward pass of the k-means clusterer.

        Computes and returns the clusters' "center" locations based on the input actions using the
        k-means clustering algorithm.

        Args:
            acts: (B, A) or (B, T, A) Input actions to cluster.
        
        Returns:
            (n_clusters, A) Tensor of computed cluster center locations.
        """
        assert acts.ndim == 2 or acts.ndim == 3, "Actions must be of shape (B, A) or (B, T, A)."
        acts = acts.view(-1, acts.shape[-1])
        N, A = acts.shape
        assert N >= self.n_clusters, \
            f"Number of actions must be >= number of clusters. Got {N} and {self.n_clusters}."
        device = acts.device
        # Randomly initialize cluster centers
        rand_indices = torch.randperm(N, device=device)[:self.n_clusters]
        centers = acts[rand_indices]  # (n_clusters, A)

        # K-Means iterations
        for i in range(self.max_iters):
            centers_prev = centers.clone()
            # Update cluster center locations
            dists = sqr_l2_norm(acts, centers)  # (N, n_clusters)
            bins = torch.argmin(dists, dim=-1)
            centers = torch.zeros((self.n_clusters, A), device=device)
            centers.index_add_(0, bins, acts)
            counts = torch.bincount(bins, minlength=self.n_clusters)
            centers /= counts.clamp_min(1).unsqueeze(1)
            empty = counts == 0
            if empty.any():
                centers[empty] = acts[torch.randperm(N, device=device)[:empty.sum()]]

            # Compare to old centers for convergence
            delta = torch.norm(centers - centers_prev, dim=-1).max()
            if delta < self.tol:
                break
        
        self._centers = centers
        return centers