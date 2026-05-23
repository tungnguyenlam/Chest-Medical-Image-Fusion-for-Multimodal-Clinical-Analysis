import timm
import torch
import torch.nn as nn


class TimmImageEncoder(nn.Module):
    def __init__(self, timm_init_args, pretrained_path=None):
        super().__init__()
        self.model = timm.create_model(**timm_init_args)
        self.model.head = nn.Identity()

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)
