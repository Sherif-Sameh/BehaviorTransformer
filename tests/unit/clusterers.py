import pytest
import torch
from torch import Tensor

from btransformer.clusterers.kmeans import KMeansClusterer

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


def generate_data(n_samples: int, device: torch.device) -> tuple[Tensor, list[Tensor]]:
    torch.manual_seed(42)
    # Generate 3 batches of normally distributed tensors
    batch_1 = torch.normal(0.0, 1.0, size=(n_samples, 2), device=device)
    batch_2 = torch.normal(10.0, 1.0, size=(n_samples, 2), device=device)
    batch_3 = torch.normal(-10.0, 1.0, size=(n_samples, 2), device=device)
    
    # Get empirical sample means
    mean_1 = batch_1.mean(dim=0)
    mean_2 = batch_2.mean(dim=0)
    mean_3 = batch_3.mean(dim=0)

    # Concatenate the batches
    data = torch.cat([batch_1, batch_2, batch_3], dim=0)
    return data, [mean_1, mean_2, mean_3]


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_kmeans_clustering(device: torch.device):
    # Initialize KMeansClusterer
    clusterer = KMeansClusterer(n_clusters=3, max_iters=100, tol=1e-3).to(device)

    # Generate synthetic data and run k-means clustering
    acts, true_centers = generate_data(n_samples=1000, device=device)
    cluster_centers = clusterer(acts)
    
    # Check if the cluster centers are close to the true means
    for tc in true_centers:
        allclose = [torch.allclose(cc, tc, atol=0.1) for cc in cluster_centers]
        assert any(allclose), f"No cluster centers found near target center {tc}."


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_kmeans_encode_decode(device: torch.device):
    # Initialize KMeansClusterer
    clusterer = KMeansClusterer(n_clusters=3, max_iters=100, tol=1e-3).to(device)

    # Generate synthetic data and run k-means clustering
    acts, true_centers = generate_data(n_samples=1000, device=device)
    cluster_centers = clusterer(acts)

    # Attempt to encode and decode nearby tensors to true centers
    for tc in true_centers:
        # Sample nearby actions and get their true bins and offsets
        acts = torch.normal(tc.repeat((100, 1)), 0.1)
        tc_bin = torch.argmin(torch.norm(tc - cluster_centers, dim=-1))
        true_offsets = acts - cluster_centers[tc_bin]

        # Test encoding
        bins, offsets = clusterer.encode(acts.unsqueeze(1))
        bins, offsets = bins.squeeze(1), offsets.squeeze(1)
        assert torch.equal(bins, tc_bin.repeat(100)), \
            f"Encoded bins do not match expected cluster bin {tc_bin}."
        assert torch.allclose(offsets, true_offsets, atol=0.01), \
            "Encoded offsets do not match expected offsets."
        
        # Test decoding
        decoded_acts = clusterer.decode(bins.unsqueeze(1), offsets.unsqueeze(1)).squeeze(1)
        assert torch.allclose(decoded_acts, acts, atol=0.01), \
            "Decoded actions do not match original actions."
