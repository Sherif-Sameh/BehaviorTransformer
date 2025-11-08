"""
Contains modified torchvision ResNet models used in the btransformer library.

The models are adapted according to the description given in:

`Diffusion Policy: Visuomotor Policy Learning via Action Diffusion`
<https://arxiv.org/abs/2303.04137>
"""

from typing import Callable, Literal

import torch.nn as nn
import torchvision.models.resnet as resnet
from torch import Tensor
from torchvision.models.resnet import ResNet, WeightsEnum

from btransformer.modules.encoders import SpatialSoftMax


class ResNetDP(nn.Module):
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
