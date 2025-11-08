"""
Contains modified torchvision ResNet models used in the btransformer library.

The models are adapted according to the description given in:

`Diffusion Policy: Visuomotor Policy Learning via Action Diffusion`
<https://arxiv.org/abs/2303.04137>
"""

from pathlib import Path
from typing import Callable, Literal

import torch.nn as nn
import torchvision.models.resnet as resnet
from torch import Tensor
from torchvision.models.resnet import ResNet, WeightsEnum

from btransformer.models.base import Model
from btransformer.modules.encoders import SpatialSoftMax


class ResNetDP(Model):
    """Modified ResNet image encoders from
    `Diffusion Policy: Visuomotor Policy Learning via Action Diffusion`
    <https://arxiv.org/abs/2303.04137>.
    
    The ResNet models are modified in two ways:
        1) Global average pooling is replaced with spatial soft-max.
        2) Each BatchNorm layer is replaced with a GroupNorm of fixed group size.
    
    Args:
        model: ResNet model to use from torchvision. One of ""
        pretrained: Load pre-trained model weights from ImageNet.
        use_spatial_softmax: Replace global average pooling with spatial soft-max.
        use_group_norm: Replace BatchNorm layers with GroupNorm of fixed group size.
        group_size: Group size to use for GroupNorm layers if applicable.
    """
    DEFAULT_PATH = Path(__file__).parent / "checkpoints/ResNet.pt"

    def __init__(
        self,
        model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        pretrained: bool = False,
        use_spatial_softmax: bool = True,
        use_group_norm: bool = True,
        group_size: int = 16,
    ):
        super().__init__()
        # Initialize torchvision ResNet base model and remove FC layers
        resnet_fn, weights = self._get_resnet_fn_and_weights(model)
        weights = weights if pretrained else None
        resnet_model = resnet_fn(weights=weights)
        self.encoder = nn.Sequential(*list(resnet_model.children())[:-1])

        # Replace global average pooling with spatial soft-max
        if use_spatial_softmax:
            assert isinstance(self.encoder[-1], nn.AdaptiveAvgPool2d)
            self.encoder[-1] = SpatialSoftMax()
        
        # Replace all BatchNorm layers with GroupNorm layers
        if use_group_norm:
            for i, mod in enumerate(self.encoder):
                if isinstance(mod, nn.BatchNorm2d):
                    self.encoder[i] = nn.GroupNorm(
                        mod.num_features // group_size, mod.num_features
                    )
                
    @staticmethod
    def _get_resnet_fn_and_weights(
        model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    ) -> tuple[Callable[..., ResNet], WeightsEnum]:
        """Get ResNet function and weights according to the chosen model."""
        match model:
            case "resnet18":
                return resnet.resnet18, resnet.ResNet18_Weights
            case "resnet34":
                return resnet.resnet34, resnet.ResNet34_Weights
            case "resnet50":
                return resnet.resnet50, resnet.ResNet50_Weights
            case "resnet101":
                return resnet.resnet101, resnet.ResNet101_Weights
            case "resnet152":
                return resnet.resnet152, resnet.ResNet152_Weights
            case _:
                raise ValueError(f"Unrecognized ResNet model {model}.")
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the modified ResNet network.
        
        Args:
            x: (B, C, H, W) Tensor of input images.
        
        Returns:
            (B, out_features) Tensor of output features from ResNet image encoder.
        """
        features = self.encoder(x)
        return features

    def split_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Splits the model's parameters into two groups:
        1. Parameters to apply weight decay (e.g., nn.Conv2d.weight).
        2. Parameters to exclude from weight decay (e.g., nn.Conv2d.bias, nn.BatchNorm.weight).

        Returns:
            tuple
            - decay: List of parameters to apply weight decay.
            - no_decay: List of parameters to exclude from weight decay.
        """
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Conv2d, )
        blacklist_weight_modules = (nn.BatchNorm2d, nn.GroupNorm)
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
