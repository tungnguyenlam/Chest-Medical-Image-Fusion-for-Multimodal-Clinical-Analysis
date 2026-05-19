from typing import Optional

import timm
import torch
import torch.nn as nn


class TimmImageEncoder(nn.Module):
    """A timm backbone with the classification head stripped.

    Outputs feature maps in [N, C, H', W'] for ConvNeXt-style backbones used
    by CaMCheX. ``name`` is a human-readable label used by RunLoggerCallback
    to group grad norms / metrics per component.
    """

    def __init__(
        self,
        timm_init_args: dict,
        pretrained_path: Optional[str] = None,
        name: str = "image_encoder",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.component_name = name
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        if gradient_checkpointing:
            self.model.set_grad_checkpointing(enable=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
