"""
Contains the Behavior Transformer (BTransformer) model implementation.

The Behavior Transformer model was proposed in the paper:

"Behavior Transformers: Cloning k modes with one stone" available at <https://arxiv.org/abs/2206.11251>.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from btransformer.clusterers.base import Clusterer
from btransformer.models.base import Model
from btransformer.models.gpt import PolicyGPT


class BehaviorTransformer(Model):
    """Behavior Transformer (BTransformer) model for multi-modal behavior cloning.

    First proposed in `Behavior Transformers: Cloning k modes with one stone` available at
    <https://arxiv.org/abs/2206.11251>.

    The BTransformer model integrates a GPT-based policy network with an action clustering
    mechanism to enable continuous action multi-modal behavior cloning. The model takes sequences
    of observations as input and outputs corresponding sequences of actions conditioned on those
    observations. Actions are predicted as a combination of discrete tokens and continuous offsets.
    The locations of the discrete tokens are determined by the employed clustering module. 

    Args:
        policy: GPT-based policy model for encoding observations and predicting actions.
        clusterer: Clustering module for clustering actions, then encoding/decoding them from
            continuous action -> discrete token + continuous offset and vice versa.
    """
    DEFAULT_PATH = Path(__file__).parent / "checkpoints/BTransformer.pt"

    def __init__(self, policy: PolicyGPT, clusterer: Clusterer):
        super().__init__()
        assert clusterer.centers.numel() > 0, \
            "Clusterer must be initialized with cluster centers."
        assert policy.act_dim == clusterer.act_dim, \
            "Policy action dimension must match clusterer action dimension."
        self.policy = policy
        self.clusterer = clusterer

    @property
    def act_dim(self) -> int:
        """Returns the action dimension of the BTransformer model.

        Returns:
            Action dimension (A).
        """
        return self.policy.act_dim
    
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for the BTransformer model.

        Args:
            obs: (B, T, obs_dim) Tensor of observation sequences.

        Returns:
            tuple
            - pred_acts: (B, T, A) Tensor of predicted action sequences.
            - logits: (B, T, n_bins) Tensor of predicted logits for discrete action bins.
            - offsets: (B, T, n_bins, A) Tensor of predicted continuous offsets for each action
                bin.
        """
        # Get policy predicitions
        logits, offsets = self.policy(obs)
        
        # Combine discrete bins and offsets to get final action predictions
        bins = torch.argmax(logits, dim=-1)
        bins_exp = bins[:, :, None, None].expand(-1, -1, 1, self.act_dim)
        bins_offsets = offsets.gather(2, bins_exp).squeeze(2)
        pred_acts = self.clusterer.decode(bins, bins_offsets)
        return pred_acts, logits, offsets
    
    @torch.no_grad
    def inference(self, obs: Tensor, deterministic: bool = True) -> Tensor:
        """Run inference using the BTransformer model.
        
        Args:
            obs: (B, T, obs_dim) Tensor of observation sequences.
            deterministic: Choose discrete action bins with highest probability deterministically.
                Otherwise, sample from discrete action bins according to their probabilities.
        
        Returns:
            (B, T, A) Tensor of predicted action sequences.
        """
        # Get logits and determine action bins
        B, T = obs.shape[:2]
        logits, offsets = self.policy(obs)
        if deterministic:
            bins = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            bins = torch.multinomial(probs.view(B * T, -1), 1).view(B, T)
        
        # Combine discrete bins and corresponding offsets to get final predicitions
        bins_exp = bins[:, :, None, None].expand(-1, -1, 1, self.act_dim)
        bins_offsets = offsets.gather(2, bins_exp).squeeze(2)
        pred_acts = self.clusterer.decode(bins, bins_offsets)
        return pred_acts
        
    def split_parameters(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        """
        Splits the model's parameters into two groups:
        1. Parameters to apply weight decay (e.g., nn.Linear.weight).
        2. Parameters to exclude from weight decay (e.g., nn.Linear.bias, nn.LayerNorm.weight).

        Returns:
            tuple
            - decay: List of parameters to apply weight decay.
            - no_decay: List of parameters to exclude from weight decay.
        """
        # Only the policy has trainable parameters
        decay, no_decay = self.policy.split_parameters()
        return decay, no_decay
    
    def save(self, path: Path | None = None) -> None:
        """Save the model's state dict to the specified path.
        
        Args:
            path: Optional path to save the model checkpoint. Default to `DEFAULT_PATH` if not
                provided.
        """
        path = self.DEFAULT_PATH if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def load(self, path: Path | None = None) -> None:
        """Load the model's state dict from the specified path.
        
        Args:
            path: Optional path to load the model checkpoint from. If not provided, `DEFAULT_PATH`
                is checked for existing compatible checkpoints.
        """
        load_path = self.DEFAULT_PATH if path is None else path
        if not load_path.exists():
            raise FileNotFoundError(
                f"No checkpoints at {str(path)} or default path {str(self.DEFAULT_PATH)}."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        try:
            self.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as e:
            print(f"Error while loading checkpoint from {str(load_path)}: {e}")


class BehaviorTransformerMixedObs(BehaviorTransformer):
    """Behavior Transformer (BTransformer) model for multi-modal behavior cloning with mixed
    observation types (i.e. images + proprioceptive data).

    This class extends the standard `BehaviorTransformer` to handle mixed observation inputs.
    Image observations are encoded separately using an image encoder then concatenated with
    the proprioceptive observations before being passed to the GPT-based policy network.

    Note: The image encoder is assumed to be a pre-trained model with frozen weights.

    Args:
        policy: GPT-based policy model for encoding mixed observations and predicting actions.
        clusterer: Clustering module for clustering actions, then encoding/decoding them from
            continuous action -> discrete token + continuous offset and vice versa.
        img_encoder: Image encoder module for encoding image observations.
    """
    DEFAULT_PATH = Path(__file__).parent / "checkpoints/BTransformerMixedObs.pt"

    def __init__(self, policy: PolicyGPT, clusterer: Clusterer, img_encoder: nn.Module):
        super().__init__(policy, clusterer)
        self.img_encoder = img_encoder
        for param in self.img_encoder.parameters():
            param.requires_grad = False  # Ensure image encoder weights are frozen
    
    def forward(self, img_obs: Tensor, prop_obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for the BTransformer model with mixed observations.

        Args:
            img_obs: (B, T, C, H, W) Tensor of image observation sequences.
            prop_obs: (B, T, P) Tensor of proprioceptive observation sequences.

        Returns:
            tuple
            - pred_acts: (B, T, A) Tensor of predicted action sequences.
            - logits: (B, T, n_bins) Tensor of predicted logits for discrete action bins.
            - offsets: (B, T, n_bins, A) Tensor of predicted continuous offsets for each action
                bin.
        """
        B, T, C, H, W = img_obs.shape
        assert prop_obs.shape[0] == B and prop_obs.shape[1] == T, \
            "Batch size and sequence length of image and proprioceptive observations must match."
        img_obs = img_obs.view(B * T, C, H, W)
        with torch.no_grad():
            img_features = self.img_encoder(img_obs)
        img_features = img_features.view(B, T, -1)
        obs = torch.cat([img_features, prop_obs], dim=-1)
        return super().forward(obs)

    @torch.no_grad
    def inference(self, img_obs: Tensor, prop_obs: Tensor, deterministic: bool = True) -> Tensor:
        """Run inference using the BTransformer model with mixed observations.
        
        Args:
            img_obs: (B, T, C, H, W) Tensor of image observation sequences.
            prop_obs: (B, T, P) Tensor of proprioceptive observation sequences.
            deterministic: Choose discrete action bins with highest probability deterministically.
                Otherwise, sample from discrete action bins according to their probabilities.
        
        Returns:
            (B, T, A) Tensor of predicted action sequences.
        """
        B, T, C, H, W = img_obs.shape
        assert prop_obs.shape[0] == B and prop_obs.shape[1] == T, \
            "Batch size and sequence length of image and proprioceptive observations must match."
        img_obs = img_obs.view(B * T, C, H, W)
        with torch.no_grad():
            img_features = self.img_encoder(img_obs)
        img_features = img_features.view(B, T, -1)
        obs = torch.cat([img_features, prop_obs], dim=-1)
        return super().inference(obs, deterministic=deterministic)
