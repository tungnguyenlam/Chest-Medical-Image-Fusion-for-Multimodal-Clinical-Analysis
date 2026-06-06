from __future__ import annotations

import einops
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.dataloader.PriorAwareDataset import N_CLASSES, N_DELTA_BUCKETS, bucket_days
from src.decoder.MLDecoder import MLDecoder
from src.encoder import CaMCheXTextEncoder
from src.model.CaMCheXV2NanoVitalsModel import ConvNeXtV2NanoImageEncoder, VitalsTokenProjector


SEG_CUR_VIEWS = (0, 1, 2, 3)
SEG_CUR_CLIN = 4
SEG_CUR_VITALS = 5
SEG_PRV_VIEWS = (6, 7, 8, 9)
SEG_PRV_CLIN = 10
SEG_PRV_VITALS = 11
SEG_PRV_LABELS = 12
N_SEGMENTS = 13


class PriorAwareV2NanoModel(nn.Module):
    """Prior-aware CaMCheX variant using v2nano image routing and numeric vitals."""

    def __init__(
        self,
        timm_init_args: dict,
        frontal_pretrained_path: str | None = None,
        lateral_pretrained_path: str | None = None,
        text_model: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        d_model: int = 768,
        n_classes: int = N_CLASSES,
        transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        use_precomputed_text_embeddings: bool = False,
        vital_dropout_p: float = 0.0,
        vitals_dropout: float = 0.1,
        vitals_hidden_dim: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_text_encoder = freeze_text_encoder
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings

        self.image_encoder = ConvNeXtV2NanoImageEncoder(
            timm_init_args=timm_init_args,
            frontal_pretrained_path=frontal_pretrained_path,
            lateral_pretrained_path=lateral_pretrained_path,
        )
        self.text_encoder = None
        if not use_precomputed_text_embeddings:
            self.text_encoder = CaMCheXTextEncoder(text_model=text_model)
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                self.text_encoder.eval()

        self.image_proj = nn.Conv2d(640, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))
        self.vitals_projector = VitalsTokenProjector(
            num_vitals=7,
            d_model=d_model,
            grid_size=8,
            hidden_dim=vitals_hidden_dim,
            dropout=vitals_dropout,
            vital_dropout_p=vital_dropout_p,
        )

        self.padding_token = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(N_SEGMENTS, d_model, 1, 1))
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.delta_embedding = nn.Embedding(N_DELTA_BUCKETS, d_model)
        nn.init.normal_(self.delta_embedding.weight, std=0.02)
        self.prior_label_proj = nn.Linear(n_classes, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers=transformer_layers,
        )
        self.head = MLDecoder(num_classes=n_classes, initial_num_features=d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def _encode_image_block(
        self,
        x: torch.Tensor,
        view_positions: torch.Tensor,
        view_seg_indices: tuple[int, int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s = x.shape[:2]
        feats, nonzero_mask = self.image_encoder(x, view_positions)
        feats = self.image_proj(feats)
        feats = self.pos_encoding(feats)
        h2, w2 = feats.shape[2], feats.shape[3]

        pad_tokens = einops.repeat(
            self.padding_token,
            "1 c 1 1 -> (b s) c h w",
            b=b,
            s=s,
            h=h2,
            w=w2,
        ).type_as(feats).clone()
        seg_for_views = torch.stack([self.segment_embedding[i] for i in view_seg_indices], dim=0)
        seg = einops.repeat(seg_for_views, "s c 1 1 -> (b s) c h w", b=b, h=h2, w=w2).type_as(feats)
        pad_tokens[nonzero_mask] = feats + seg[nonzero_mask]
        block = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s)
        return block, nonzero_mask.view(b, s)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.is_floating_point() and input_ids.ndim == 2:
            return input_ids.float()
        if self.use_precomputed_text_embeddings:
            raise TypeError("use_precomputed_text_embeddings=True requires float clinical embedding tensors")
        if self.text_encoder is None:
            raise RuntimeError("text_encoder is not initialized; disable use_precomputed_text_embeddings for token batches")
        if self.freeze_text_encoder:
            with torch.no_grad():
                return self.text_encoder.biobert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_encoder.biobert_encoder(input_ids=input_ids, attention_mask=attention_mask)

    def _expand_text_block(self, cls: torch.Tensor, segment_index: int, token_count: int, cdim: int) -> torch.Tensor:
        seg = self.segment_embedding[segment_index].view(cdim).to(cls.device)
        return (cls + seg).unsqueeze(1).expand(-1, token_count, -1)

    def _valid_image_tokens(self, slot_valid: torch.Tensor, h2: int, w2: int) -> torch.Tensor:
        b, s = slot_valid.shape
        return slot_valid.unsqueeze(-1).expand(b, s, h2 * w2).reshape(b, s * h2 * w2)

    def forward(self, data: dict) -> torch.Tensor:
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )
        b, s_img, cdim, h2, w2 = cur_block.shape
        block_tokens = h2 * w2

        cur_img_tokens = einops.rearrange(cur_block, "b s c h w -> b (s h w) c")
        cur_clin_cls = self._encode_text(data["clin_input_ids"], data["clin_attn_mask"])
        cur_clin_tokens = self._expand_text_block(cur_clin_cls, SEG_CUR_CLIN, block_tokens, cdim)
        cur_vital_tokens = self.vitals_projector(
            data["vital_values"].to(cur_img_tokens.device),
            data["vital_missing_mask"].to(cur_img_tokens.device),
        )
        cur_vital_tokens = cur_vital_tokens + self.segment_embedding[SEG_CUR_VITALS].view(cdim).to(cur_vital_tokens.device)

        has_prior = data["has_prior"].to(cur_img_tokens.device).bool()
        days_since = data["days_since_prior"].to(cur_img_tokens.device).float()
        delta_emb = self.delta_embedding(bucket_days(days_since, has_prior))

        prv_block, prv_slot_valid = self._encode_image_block(
            data["prior_img"], data["prior_view_positions"], SEG_PRV_VIEWS
        )
        prv_img_tokens = einops.rearrange(prv_block, "b s c h w -> b (s h w) c")
        prv_clin_cls = self._encode_text(data["prior_clin_input_ids"], data["prior_clin_attn_mask"])
        prv_clin_tokens = self._expand_text_block(prv_clin_cls, SEG_PRV_CLIN, block_tokens, cdim)
        prv_vital_tokens = self.vitals_projector(
            data["prior_vital_values"].to(cur_img_tokens.device),
            data["prior_vital_missing_mask"].to(cur_img_tokens.device),
        )
        prv_vital_tokens = prv_vital_tokens + self.segment_embedding[SEG_PRV_VITALS].view(cdim).to(prv_vital_tokens.device)
        prior_label = data["prior_label"].to(cur_img_tokens.device).float()
        prv_label_token = self.prior_label_proj(prior_label) + self.segment_embedding[SEG_PRV_LABELS].view(cdim).to(cur_img_tokens.device)

        prv_img_tokens = prv_img_tokens + delta_emb.unsqueeze(1)
        prv_clin_tokens = prv_clin_tokens + delta_emb.unsqueeze(1)
        prv_vital_tokens = prv_vital_tokens + delta_emb.unsqueeze(1)
        prv_label_token = prv_label_token + delta_emb

        tokens = torch.cat(
            [
                cur_img_tokens,
                cur_clin_tokens,
                cur_vital_tokens,
                prv_img_tokens,
                prv_clin_tokens,
                prv_vital_tokens,
                prv_label_token.unsqueeze(1),
            ],
            dim=1,
        )

        cur_img_valid = self._valid_image_tokens(cur_slot_valid, h2, w2)
        cur_clin_valid = torch.ones(b, block_tokens, dtype=torch.bool, device=tokens.device)
        cur_vital_valid = torch.ones(b, block_tokens, dtype=torch.bool, device=tokens.device)
        prv_img_valid = self._valid_image_tokens(prv_slot_valid & has_prior.unsqueeze(-1), h2, w2)
        prv_clin_valid = has_prior.unsqueeze(1).expand(-1, block_tokens)
        prv_vital_valid = has_prior.unsqueeze(1).expand(-1, block_tokens)
        prv_label_valid = has_prior.unsqueeze(1)
        valid = torch.cat(
            [
                cur_img_valid,
                cur_clin_valid,
                cur_vital_valid,
                prv_img_valid,
                prv_clin_valid,
                prv_vital_valid,
                prv_label_valid,
            ],
            dim=1,
        )
        mask = ~valid
        x = self.transformer_encoder(tokens, src_key_padding_mask=mask)
        return self.head(x, mask)
