import pytest
import torch
import torchvision.models as models

from btransformer.clusterers.kmeans import KMeansClusterer
from btransformer.models.btransformer import (
    BehaviorTransformer,
    BehaviorTransformerMixedObs,
)
from btransformer.models.gpt import PolicyGPT

Devices = [torch.device("cpu")]
Devices = Devices + [torch.device("cuda")] if torch.cuda.is_available() else Devices


def get_policy_and_clusterer(
    obs_dim: int,
    act_dim: int,
    n_bins: int,
) -> tuple[PolicyGPT, KMeansClusterer]:
    # Initialize a small GPT policy model
    policy = PolicyGPT(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_bins=n_bins,
        n_blocks=2,
        embd_dim=20,
        n_heads=2,
        seq_len=3,
        embd_pdrop=0.5,
        attn_pdrop=0.5,
        proj_pdrop=0.5,
    )

    # Initialize a KMeans clusterer with random data
    data = torch.normal(0.0, 1.0, size=(100, act_dim))
    clusterer = KMeansClusterer(n_clusters=n_bins, max_iters=100, tol=1e-4)
    clusterer(data)
    return policy, clusterer


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_btransformer(device: torch.device):
    # Initialize Behavior Transformer model
    obs_dim, act_dim, n_bins = 2, 2, 3
    policy, clusterer = get_policy_and_clusterer(obs_dim, act_dim, n_bins)
    bt = BehaviorTransformer(policy, clusterer).to(device)

    # Forward pass synthetic observation data and verify shapes
    obs = torch.randn(size=(8, 3, obs_dim), device=device)
    pred_acts, logits, offsets = bt(obs)
    assert pred_acts.shape == (8, 3, act_dim), \
        f"Predicted actions shape mismatch. Expected (8, 3, {act_dim}), got {pred_acts.shape}."
    assert logits.shape == (8, 3, n_bins), \
        f"Logits shape mismatch. Expected (8, 3, {n_bins}), got {logits.shape}."
    assert offsets.shape == (8, 3, n_bins, act_dim), \
        f"Offsets shape mismatch. Expected (8, 3, {n_bins}, {act_dim}), got {offsets.shape}."
    
    # Verify that both logits and offsets require gradients
    assert logits.requires_grad, "Logits tensor does not require gradients."
    assert offsets.requires_grad, "Offsets tensor does not require gradients."

    # Verify that splitting pred_acts yields the same bins and offsets
    bins = torch.argmax(logits, dim=-1)
    bins_exp = bins[:, :, None, None].expand(-1, -1, 1, act_dim)
    bins_offsets = offsets.gather(2, bins_exp).squeeze(2)
    bins_encoded, offsets_encoded = bt.clusterer.encode(pred_acts)
    assert torch.equal(bins, bins_encoded), "Encoded bins do not match predicted bins."
    assert torch.allclose(offsets_encoded, bins_offsets, atol=1e-4), \
        "Encoded offsets do not match predicted offsets."


@pytest.mark.unit
@pytest.mark.parametrize("device", Devices)
def test_btransformer_mixed_obs(device: torch.device):
    # Initialize Behavior Transformer model
    obs_dim, act_dim, n_bins = 514, 2, 3
    policy, clusterer = get_policy_and_clusterer(obs_dim, act_dim, n_bins)
    resnet18 = models.resnet18(weights=None)
    img_encoder = torch.nn.Sequential(*list(resnet18.children())[:-1])
    bt = BehaviorTransformerMixedObs(policy, clusterer, img_encoder).to(device)

    # Forward pass synthetic observation data and verify shapes
    img_obs = torch.randn((2, 3, 3, 96, 96), device=device)  # (B, T, C, H, W)
    prop_obs = torch.randn((2, 3, obs_dim - 512), device=device)
    pred_acts, logits, offsets = bt(img_obs, prop_obs)
    assert pred_acts.shape == (2, 3, act_dim), \
        f"Predicted actions shape mismatch. Expected (2, 3, {act_dim}), got {pred_acts.shape}."
    assert logits.shape == (2, 3, n_bins), \
        f"Logits shape mismatch. Expected (2, 3, {n_bins}), got {logits.shape}."
    assert offsets.shape == (2, 3, n_bins, act_dim), \
        f"Offsets shape mismatch. Expected (2, 3, {n_bins}, {act_dim}), got {offsets.shape}."
    
    # Verify that both logits and offsets require gradients
    assert logits.requires_grad, "Logits tensor does not require gradients."
    assert offsets.requires_grad, "Offsets tensor does not require gradients."

    # Verify that splitting pred_acts yields the same bins and offsets
    bins = torch.argmax(logits, dim=-1)
    bins_exp = bins[:, :, None, None].expand(-1, -1, 1, act_dim)
    bins_offsets = offsets.gather(2, bins_exp).squeeze(2)
    bins_encoded, offsets_encoded = bt.clusterer.encode(pred_acts)
    assert torch.equal(bins, bins_encoded), "Encoded bins do not match predicted bins."
    assert torch.allclose(offsets_encoded, bins_offsets, atol=1e-4), \
        "Encoded offsets do not match predicted offsets."
