"""
Contains all GPT-based models in the btransformer library.

Largely adapted from Andrej Karpathy's implementation of minGPT available at: 

<https://github.com/karpathy/minGPT/tree/master>
"""

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from btransformer.models.base import Model
from btransformer.modules.transformer import TransformerBlock


class PolicyGPT(Model):
    """GPT-based policy model for the behavior transformer.
    
    The policy is setup for continuous observation and action spaces. The model recieves a sequence
    of observations as input and outputs a corresponding sequence of actions. Actions are predicted
    as discrete tokens over a set number of bins as well as continuous offsets to be added to each
    discrete action.

    Args:
        obs_dim: Observation dimension.
        act_dim: Action dimension.
        n_bins: Number of discrete action bins.
        n_blocks: Number of transformer blocks.
        embd_dim: Embedding dimension for input sequences.
        n_heads: Number of parllel attention heads.
        seq_len: Expected sequence length for input sequences.
        embd_pdrop: Dropout probability for input embeddings.
        attn_pdrop: Dropout probability for attention weights.
        proj_pdrop: Dropout probability for transformer output projection.
    """
    DEFAULT_PATH = Path(__file__).parent / "checkpoints/policyGPT.pt"

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_bins: int,
        n_blocks: int,
        embd_dim: int,
        n_heads: int,
        seq_len: int,
        embd_pdrop: float,
        attn_pdrop: float,
        proj_pdrop: float,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_bins = n_bins
        # Embedding layers for observations and positional encodings
        self.obs_embd = nn.Linear(obs_dim, embd_dim)
        self.pos_embd = nn.Embedding(seq_len, embd_dim)
        self.embd_dropout = nn.Dropout(embd_pdrop)
        
        # Main transformer blocks for encoding observation sequences
        self.transformer = nn.Sequential(*[
            TransformerBlock(embd_dim, n_heads, seq_len, attn_pdrop, proj_pdrop) \
                for _ in range(n_blocks)
        ])
        self.ln_final = nn.LayerNorm(embd_dim)

        # Output heads for logits and continuous offsets
        self.logits_head = nn.Linear(embd_dim, n_bins)
        self.offset_head = nn.Linear(embd_dim, n_bins * act_dim)

        # Initialize weights and apply a special scaled init to the residual output projections
        self.apply(partial(self._init_weights, mean=0.0, std=0.02))
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))

        # Report number of parameters across transformer (without output heads)
        n_params_transformer = sum((
            sum(param.numel() for param in self.obs_embd.parameters()),
            sum(param.numel() for param in self.pos_embd.parameters()),
            sum(param.numel() for param in self.transformer.parameters()),
            sum(param.numel() for param in self.ln_final.parameters()),
        ))
        print(f"Number of parameters in transformer: {n_params_transformer/1e6:.2f}M")
    
    @staticmethod
    def _init_weights(module: nn.Module, mean: float, std: float) -> None:
        """Initialize weights of the transformer's layers according to their types.
        
        Args:
            module: Module to initialize.
            mean: Mean value for normal distribution.
            std: Standard deviation for normal distribution.
        """
        match module:
            case nn.Linear():
                nn.init.normal_(module.weight, mean=mean, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            case nn.Embedding():
                nn.init.normal_(module.weight, mean=mean, std=std)
            case nn.LayerNorm():
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
            case _:
                pass
            
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the GPT-based policy model.
        
        Args:
            obs: (B, T, obs_dim) Tensor of input observations.
        
        Returns:
            tuple
            - logits: (B, T, n_bins) Tensor of action logits.
            - offsets: (B, T, n_bins, act_dim) Tensor of continuous offsets for each action bin.
        """
        assert obs.ndim == 3, f"Input must be a 3-dim tensor, got {obs.ndim} dims."
        assert obs.shape[2] == self.obs_dim, \
            f"Observation dimension must be {self.obs_dim}, got {obs.shape[2]}."
        device = obs.device
        B, T, _ = obs.shape
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Embed observations and add positional encodings
        obs_embd = self.obs_embd(obs)
        pos_embd = self.pos_embd(pos)
        x = self.embd_dropout(obs_embd + pos_embd)

        # Pass through transformer blocks to get output embeddings
        x = self.transformer(x)
        x = self.ln_final(x)

        # Compute action logits and continuous offsets
        logits = self.logits_head(x)
        offsets = self.offset_head(x).view(B, T, self.n_bins, self.act_dim)
        return logits, offsets
    
    def split_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """
        Splits the model's parameters into two groups:
        1. Parameters to apply weight decay (e.g., nn.Linear.weight).
        2. Parameters to exclude from weight decay (e.g., nn.Linear.bias, nn.LayerNorm.weight).

        Returns:
            tuple
            - decay: List of parameters to apply weight decay.
            - no_decay: List of parameters to exclude from weight decay.
        """
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        # Iterate over all modules and store parameter names in the appropriate set
        for name, module in self.named_modules():
            for pname, _ in module.named_parameters(recurse=False):
                full_pname = f"{name}.{pname}" if name else pname
                if pname.endswith("weight"):
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_pname)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_pname)
                elif pname.endswith("bias"):
                    no_decay.add(full_pname)
        
        # Validate that every parameter has been considered
        param_dict = {name: param for (name, param) in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            f"Parameters {str(inter_params)} made it into both sets!"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters {str(param_dict.keys() - union_params)} were missed from both sets!"
        
        # Create final parameter lists from decay/no_decay sets
        decay = [param_dict[name] for name in sorted(list(decay))]
        no_decay = [param_dict[name] for name in sorted(list(no_decay))]
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
