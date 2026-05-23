import torch
import torch.nn as nn

from src.encoder.TimmImageEncoder import TimmImageEncoder


class CaMCheXImageEncoder(nn.Module):
    def __init__(self, timm_init_args, frontal_pretrained_path=None, lateral_pretrained_path=None):
        super().__init__()
        self.frontal_encoder = TimmImageEncoder(
            timm_init_args=timm_init_args,
            pretrained_path=frontal_pretrained_path,
        )
        self.lateral_encoder = TimmImageEncoder(
            timm_init_args=timm_init_args,
            pretrained_path=lateral_pretrained_path,
        )

    def forward(self, x, view_positions):
        b, s, _, h, w = x.shape

        x = x.reshape(b * s, *x.shape[2:])
        view_positions = view_positions.reshape(b * s)

        nonzero_mask = x.sum(dim=(1, 2, 3)) != 0
        x_nonzero = x[nonzero_mask]
        view_positions_nonzero = view_positions[nonzero_mask]

        frontal_mask = view_positions_nonzero == 1
        lateral_mask = view_positions_nonzero == 2

        feats = torch.zeros(
            (x_nonzero.shape[0], 768, h // 32, w // 32),
            device=x.device,
            dtype=x.dtype,
        )

        if frontal_mask.any():
            feats[frontal_mask] = self.frontal_encoder(x_nonzero[frontal_mask])
        if lateral_mask.any():
            feats[lateral_mask] = self.lateral_encoder(x_nonzero[lateral_mask])

        return feats, nonzero_mask
