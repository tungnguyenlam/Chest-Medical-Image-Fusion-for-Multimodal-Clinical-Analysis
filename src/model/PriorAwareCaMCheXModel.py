"""Prior-aware CaMCheX model.

Same backbone as CaMCheXModel, plus a parallel "prior" branch sharing weights
with the current branch. Adds:

  - 13 segment embedding slots (4 cur views + clin/obs + 4 prior views + clin/obs + prior labels)
  - Time-delta bucket embedding broadcast onto every prior token
  - Linear(26 -> 768) projection of the prior CheXpert/NegBio label vector
  - Per-sample masking when has_prior is False (matches dataset's label_dropout)

Forward consumes a dict (see PriorAwareDataset) rather than a positional tuple.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.decoder.MLDecoder import MLDecoder
from src.encoder import CaMCheXImageEncoder, CaMCheXTextEncoder
from src.dataloader.PriorAwareDataset import N_DELTA_BUCKETS, N_CLASSES, bucket_days


# Segment slot indices
SEG_CUR_VIEWS = (0, 1, 2, 3)
SEG_CUR_CLIN = 4
SEG_CUR_OBS = 5
SEG_PRV_VIEWS = (6, 7, 8, 9)
SEG_PRV_CLIN = 10
SEG_PRV_OBS = 11
SEG_PRV_LABELS = 12
N_SEGMENTS = 13


class PriorAwareCaMCheXModel(nn.Module):
    def __init__(
        self,
        timm_init_args: dict,
        frontal_pretrained_path: str | None = None,
        lateral_pretrained_path: str | None = None,
        text_model: str = "dmis-lab/biobert-v1.1",
        d_model: int = 768,
        n_classes: int = N_CLASSES,
        transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # --- shared encoders ---------------------------------------------
        self.image_encoder = CaMCheXImageEncoder(
            timm_init_args=timm_init_args,
            frontal_pretrained_path=frontal_pretrained_path,
            lateral_pretrained_path=lateral_pretrained_path,
        )
        self.text_encoder = CaMCheXTextEncoder(text_model=text_model)

        self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))

        # --- segment + delta + label projections -------------------------
        self.padding_token = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(N_SEGMENTS, d_model, 1, 1))
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.delta_embedding = nn.Embedding(N_DELTA_BUCKETS, d_model)
        nn.init.normal_(self.delta_embedding.weight, std=0.02)

        self.prior_label_proj = nn.Linear(n_classes, d_model)

        # --- fusion + decoder --------------------------------------------
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=transformer_layers,
        )
        self.head = MLDecoder(num_classes=n_classes, initial_num_features=d_model)

    # ---------------------------------------------------------------------
    # Image branch
    # ---------------------------------------------------------------------
    def _encode_image_block(self, x: torch.Tensor, view_positions: torch.Tensor,
                            view_seg_indices: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 4, 3, H, W)
            view_positions: (B, 4) int64
            view_seg_indices: 4 segment indices for the 4 image slots.

        Returns:
            tokens: (B, 4, C, h2, w2) with segment embeddings already added and padding slots zeroed.
            slot_valid: (B, 4) bool — True where the slot held a real (nonzero) image.
        """
        b, s = x.shape[:2]
        feats, nonzero_mask = self.image_encoder(x, view_positions)
        feats = self.conv2d(feats)
        feats = self.pos_encoding(feats)

        h2, w2 = feats.shape[2], feats.shape[3]
        pad_tokens = einops.repeat(
            self.padding_token, "1 c 1 1 -> (b s) c h w", b=b, s=s, h=h2, w=w2
        ).type_as(feats).clone()

        seg_for_views = torch.stack([self.segment_embedding[i] for i in view_seg_indices], dim=0)
        seg = einops.repeat(seg_for_views, "s c 1 1 -> (b s) c h w", b=b, h=h2, w=w2).type_as(feats)

        pad_tokens[nonzero_mask] = feats + seg[nonzero_mask]
        block = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s)
        slot_valid = nonzero_mask.view(b, s)
        return block, slot_valid

    def _expand_block_tokens(self, block: torch.Tensor) -> torch.Tensor:
        """(B, S, C, h2, w2) -> (B, S*h2*w2, C)."""
        return einops.rearrange(block, "b s c h w -> b (s h w) c")

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, data: dict) -> torch.Tensor:
        # ---- current branch -----------------------------------------------
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )
        b, s_img, cdim, h2, w2 = cur_block.shape

        cur_clin_cls, cur_obs_cls = self.text_encoder(
            clinical_input_ids=data["clin_input_ids"],
            clinical_attention_mask=data["clin_attn_mask"],
            clinical_obs_input_ids=data["obs_input_ids"],
            clinical_obs_attention_mask=data["obs_attn_mask"],
        )
        cur_clin_tok = cur_clin_cls + self.segment_embedding[SEG_CUR_CLIN].view(cdim)  # (B, C)
        cur_obs_tok = cur_obs_cls + self.segment_embedding[SEG_CUR_OBS].view(cdim)

        cur_img_tokens = self._expand_block_tokens(cur_block)                          # (B, s*h*w, C)
        cur_text_tokens = torch.stack([cur_clin_tok, cur_obs_tok], dim=1)             # (B, 2, C)

        cur_img_valid = cur_slot_valid.unsqueeze(-1).expand(-1, -1, h2 * w2).reshape(b, s_img * h2 * w2)
        cur_text_valid = torch.ones(b, 2, dtype=torch.bool, device=cur_img_tokens.device)

        # ---- prior branch -------------------------------------------------
        has_prior = data["has_prior"].to(cur_img_tokens.device).bool()                # (B,)
        days_since = data["days_since_prior"].to(cur_img_tokens.device).float()       # (B,)
        delta_idx = bucket_days(days_since, has_prior)                                # (B,)
        delta_emb = self.delta_embedding(delta_idx)                                   # (B, C)

        prv_block, prv_slot_valid = self._encode_image_block(
            data["prior_img"], data["prior_view_positions"], SEG_PRV_VIEWS
        )
        prv_img_tokens = self._expand_block_tokens(prv_block)                         # (B, s*h*w, C)

        prv_clin_cls, prv_obs_cls = self.text_encoder(
            clinical_input_ids=data["prior_clin_input_ids"],
            clinical_attention_mask=data["prior_clin_attn_mask"],
            clinical_obs_input_ids=data["prior_obs_input_ids"],
            clinical_obs_attention_mask=data["prior_obs_attn_mask"],
        )
        prv_clin_tok = prv_clin_cls + self.segment_embedding[SEG_PRV_CLIN].view(cdim)
        prv_obs_tok = prv_obs_cls + self.segment_embedding[SEG_PRV_OBS].view(cdim)

        prior_label = data["prior_label"].to(cur_img_tokens.device).float()           # (B, 26)
        prv_label_tok = self.prior_label_proj(prior_label) + self.segment_embedding[SEG_PRV_LABELS].view(cdim)

        # Add the time-delta embedding to every prior token.
        delta_for_img = delta_emb.unsqueeze(1).expand(-1, prv_img_tokens.size(1), -1)
        prv_img_tokens = prv_img_tokens + delta_for_img
        prv_clin_tok = prv_clin_tok + delta_emb
        prv_obs_tok = prv_obs_tok + delta_emb
        prv_label_tok = prv_label_tok + delta_emb

        prv_text_tokens = torch.stack([prv_clin_tok, prv_obs_tok, prv_label_tok], dim=1)  # (B, 3, C)

        prv_img_slot_valid = prv_slot_valid & has_prior.unsqueeze(-1)
        prv_img_valid = prv_img_slot_valid.unsqueeze(-1).expand(-1, -1, h2 * w2).reshape(b, s_img * h2 * w2)
        prv_text_valid = has_prior.unsqueeze(-1).expand(-1, 3)                         # (B, 3)

        # ---- fuse + decode ------------------------------------------------
        tokens = torch.cat([cur_img_tokens, cur_text_tokens, prv_img_tokens, prv_text_tokens], dim=1)
        valid = torch.cat([cur_img_valid, cur_text_valid, prv_img_valid, prv_text_valid], dim=1)
        key_padding_mask = ~valid

        x = self.transformer_encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.head(x, key_padding_mask)
