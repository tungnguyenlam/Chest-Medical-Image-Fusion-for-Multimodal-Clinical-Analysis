from typing import Optional

import einops
import torch
import torch.nn as nn


class CaMCheXModel(nn.Module):
    """Assembly that wires per-view image encoders + text encoder + fusion + head.

    No timm / transformers calls live here; submodules are injected. Swapping
    a backbone or decoder is a one-line change in the caller (Lightning module).

    Forward expects ``data`` in the order produced by CaMCheXDataset:
        (study_id, images, view_positions,
         clinical_input_ids, clinical_attention_mask,
         clinical_obs_input_ids, clinical_obs_attention_mask)

    ``images`` is [B, S, C, H, W]; ``view_positions`` is [B, S] with
    1 = frontal, 2 = lateral, 0 = padding. S must match ``fusion.num_views``.
    """

    FRONTAL_VIEW = 1
    LATERAL_VIEW = 2

    def __init__(
        self,
        frontal_encoder: nn.Module,
        lateral_encoder: nn.Module,
        text_encoder: nn.Module,
        fusion: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.frontal_encoder = frontal_encoder
        self.lateral_encoder = lateral_encoder
        self.text_encoder = text_encoder
        self.fusion = fusion
        self.head = head

    def forward(self, data):
        (_, x, view_positions,
         clin_ids, clin_mask,
         obs_ids, obs_mask) = data

        b, s, _, _, _ = x.shape
        x = einops.rearrange(x, "b s c h w -> (b s) c h w")
        view_positions = einops.rearrange(view_positions, "b s -> (b s)")

        nonzero_mask = (x.sum(dim=(1, 2, 3)) != 0)
        x_nonzero = x[nonzero_mask]
        vp_nonzero = view_positions[nonzero_mask]

        frontal_sel = (vp_nonzero == self.FRONTAL_VIEW)
        lateral_sel = (vp_nonzero == self.LATERAL_VIEW)

        out_frontal = self.frontal_encoder(x_nonzero[frontal_sel]) if frontal_sel.any() else None
        out_lateral = self.lateral_encoder(x_nonzero[lateral_sel]) if lateral_sel.any() else None

        ref = out_frontal if out_frontal is not None else out_lateral
        if ref is None:
            feats = x_nonzero.new_zeros((0, self.fusion.feature_dim, 1, 1))
        else:
            feats = ref.new_zeros((x_nonzero.shape[0], *ref.shape[1:]))
            if out_frontal is not None:
                feats[frontal_sel] = out_frontal
            if out_lateral is not None:
                feats[lateral_sel] = out_lateral

        clin_emb = self.text_encoder(clin_ids, clin_mask)
        obs_emb = self.text_encoder(obs_ids, obs_mask)

        seq, mask = self.fusion(feats, nonzero_mask, [clin_emb, obs_emb])
        return self.head(seq, mask)


class SingleViewModel(nn.Module):
    """Single-image baseline: one image encoder + 2D pos-enc + head.

    Useful as an architectural sanity check. Mirrors the original
    SingleViewModel but takes injected submodules.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        head: nn.Module,
        pos_encoding: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.pos_encoding = pos_encoding if pos_encoding is not None else nn.Identity()
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_encoder(x)
        x = self.pos_encoding(x)
        return self.head(x)
