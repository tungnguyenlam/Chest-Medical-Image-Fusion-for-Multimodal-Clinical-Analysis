from typing import Sequence, Tuple

import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class TransformerFusion(nn.Module):
    """CaMCheX fusion block: combines per-view image feature maps with
    per-stream text [CLS] embeddings into a single token sequence and
    runs a TransformerEncoder over it.

    Layout of the segment-embedding bank (size = num_views + num_text_streams):
      [0 .. num_views-1]                       -> one per image-view slot
      [num_views .. num_views+num_text_streams-1] -> one per text stream

    Args:
        feature_dim: channel dim of image feature maps and text CLS vectors.
        num_views: max image views per study (CaMCheX = 4).
        num_text_streams: number of text [CLS] streams (CaMCheX = 2: indication, observations).
        num_layers, nhead, dropout: TransformerEncoder hyperparams.
        stem_kernel, stem_stride: shape of the image feature reducer applied
            before fusion. Stride=2 halves spatial; set stride=1 to disable.
        name: human-readable component label for run-logger grouping.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_views: int = 4,
        num_text_streams: int = 2,
        num_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        stem_kernel: int = 3,
        stem_stride: int = 2,
        name: str = "fusion",
    ):
        super().__init__()
        self.component_name = name
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.num_text_streams = num_text_streams

        self.conv2d = nn.Conv2d(
            feature_dim, feature_dim,
            kernel_size=stem_kernel, stride=stem_stride, padding=stem_kernel // 2,
        )
        self.pos_encoding = Summer(PositionalEncoding2D(feature_dim))

        self.padding_token = nn.Parameter(torch.randn(1, feature_dim, 1, 1))

        n_segments = num_views + num_text_streams
        self.segment_embedding = nn.Parameter(torch.randn(n_segments, feature_dim, 1, 1))
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim, nhead=nhead, dropout=dropout, batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(
        self,
        image_feats: torch.Tensor,          # [nonzero_count, feature_dim, H, W]
        nonzero_mask: torch.Tensor,         # [b*num_views] bool, True where a view exists
        text_embeds: Sequence[torch.Tensor],  # each [b, feature_dim], len == num_text_streams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(text_embeds) != self.num_text_streams:
            raise ValueError(
                f"Expected {self.num_text_streams} text streams, got {len(text_embeds)}"
            )

        b = text_embeds[0].shape[0]
        s_img = self.num_views

        feats_nonzero = self.conv2d(image_feats)
        feats_nonzero = self.pos_encoding(feats_nonzero)
        _, c, h2, w2 = feats_nonzero.shape

        pad_tokens = einops.repeat(
            self.padding_token, "1 c 1 1 -> (b s) c h w", b=b, s=s_img, h=h2, w=w2,
        ).type_as(feats_nonzero).clone()

        view_segments = einops.repeat(
            self.segment_embedding[:s_img], "s c 1 1 -> (b s) c h w", b=b, h=h2, w=w2,
        ).type_as(feats_nonzero)

        pad_tokens[nonzero_mask] = feats_nonzero + view_segments[nonzero_mask]
        feats_img = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s_img)

        text_token_blocks = []
        for i, emb in enumerate(text_embeds):
            seg = self.segment_embedding[s_img + i].view(c).to(emb.device)
            tok = (emb + seg).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h2, w2).unsqueeze(1)
            text_token_blocks.append(tok)

        feats_cat = torch.cat([feats_img, *text_token_blocks], dim=1)
        x = einops.rearrange(feats_cat, "b s c h w -> b (s h w) c")

        img_slots_valid = nonzero_mask.view(b, s_img)
        img_valid_tokens = (
            img_slots_valid.unsqueeze(-1).unsqueeze(-1)
            .expand(b, s_img, h2, w2)
            .reshape(b, s_img * h2 * w2)
        )
        text_valid_tokens = torch.ones(
            b, self.num_text_streams * h2 * w2, dtype=torch.bool, device=x.device,
        )
        valid_tokens = torch.cat([img_valid_tokens, text_valid_tokens], dim=1)
        key_padding_mask = ~valid_tokens

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        return x, key_padding_mask
