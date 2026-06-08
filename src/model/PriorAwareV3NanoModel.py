"""Prior-aware CaMCheX v3 Nano: single-token clinical/report/vitals fusion.

Same prior-aware design as PriorAwareV2NanoModel (current + prior branches sharing
the ConvNeXtV2 Nano image router, CXR-BERT text encoder, numeric VitalsTokenProjector,
time-delta embedding, and Linear(26->768) prior-label token) with ONE change: every
non-image signal becomes a single token instead of a 64-wide (8x8) block.

v2nano repeated each text CLS vector across the 8x8 image grid (64 identical tokens)
and emitted 64 vital tokens. v3nano collapses each to one token -- the prior-aware
analogue of how CaMCheXV3NanoModel slims CaMCheXV2NanoVitalsModel. This is an ablation
of whether the 64x text duplication carries any signal; it also shortens the sequence
from ~833 to ~518 tokens at 512px.

Per-token layout (default size=512, image grid 8x8 -> 64 tokens/view, 4 views):
  current:  256 image  + 1 clinical + 1 vitals
  prior:    256 image  + 1 clinical + 1 report + 1 vitals + 1 label
  total:    518 tokens

The prior radiology report (findings + impression) is fed only for the PRIOR study:
it states the labels for its own exam (leakage if used for the current study) but was
authored before the current exam, so it is legitimate prior context here. The prior
label is the prior study's 26-dim multi-hot vector projected to one token by
Linear(26->768) -- multiple positive labels are handled by that projection, no
per-label tokens needed. The time gap between studies is bucketed (bucket_days, 8
buckets) and added to every prior token via a learned delta embedding.
"""

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
SEG_PRV_REPORT = 13  # prior study's findings + impression (legitimate prior info)
N_SEGMENTS = 14


class PriorAwareV3NanoModel(nn.Module):
    """Prior-aware CaMCheX variant: v2nano backbone/vitals, single-token fusion."""

    gradcam_runner_module = "src.interpret.run_prior_gradcam"

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
        # Opt-in GPU-resident text embedding table (attach_text_embedding_table).
        # None by default -> per-sample float-embedding contract is unchanged.
        # persistent=False: derived from the cache, kept out of checkpoints.
        self.register_buffer("text_embedding_table", None, persistent=False)

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
        # grid_size=1: numeric vitals -> a SINGLE token (v2nano uses grid_size=8 -> 64).
        self.vitals_projector = VitalsTokenProjector(
            num_vitals=7,
            d_model=d_model,
            grid_size=1,
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

    # ---- encoders ---------------------------------------------------------
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
            self.padding_token, "1 c 1 1 -> (b s) c h w", b=b, s=s, h=h2, w=w2
        ).type_as(feats).clone()
        seg_for_views = torch.stack([self.segment_embedding[i] for i in view_seg_indices], dim=0)
        seg = einops.repeat(seg_for_views, "s c 1 1 -> (b s) c h w", b=b, h=h2, w=w2).type_as(feats)
        pad_tokens[nonzero_mask] = feats + seg[nonzero_mask]
        block = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s)
        return block, nonzero_mask.view(b, s)

    def attach_text_embedding_table(self, table: torch.Tensor) -> None:
        """Register a frozen ``[N, d_model]`` precomputed embedding table as a
        (non-persistent) buffer. It then rides ``.to(device)`` onto the GPU once,
        and integer index batches from the dataset are gathered against it in
        ``_encode_text``. Opt-in; default runs leave this None."""
        self.register_buffer("text_embedding_table", table.float(), persistent=False)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.is_floating_point() and input_ids.ndim == 2:
            return input_ids.float()
        if self.text_embedding_table is not None and not input_ids.is_floating_point() and input_ids.ndim == 1:
            # GPU-resident table path: (B,) row indices -> (B, d_model) CLS vectors.
            return self.text_embedding_table[input_ids.long()]
        if self.use_precomputed_text_embeddings:
            raise TypeError("use_precomputed_text_embeddings=True requires float clinical embedding tensors")
        if self.text_encoder is None:
            raise RuntimeError("text_encoder is not initialized; disable use_precomputed_text_embeddings for token batches")
        if self.freeze_text_encoder:
            with torch.no_grad():
                return self.text_encoder.biobert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_encoder.biobert_encoder(input_ids=input_ids, attention_mask=attention_mask)

    def _text_token(self, cls: torch.Tensor, segment_index: int, cdim: int) -> torch.Tensor:
        """One CLS vector -> a single (B, 1, C) fusion token (v3: no 64-wide expand)."""
        seg = self.segment_embedding[segment_index].view(cdim).to(cls.device)
        return (cls + seg).unsqueeze(1)

    def _valid_image_tokens(self, slot_valid: torch.Tensor, h2: int, w2: int) -> torch.Tensor:
        b, s = slot_valid.shape
        return slot_valid.unsqueeze(-1).expand(b, s, h2 * w2).reshape(b, s * h2 * w2)

    # ---- forward ----------------------------------------------------------
    def forward(self, data: dict) -> torch.Tensor:
        # ---- current branch ----------------------------------------------
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )
        b, s_img, cdim, h2, w2 = cur_block.shape
        cur_img_tokens = einops.rearrange(cur_block, "b s c h w -> b (s h w) c")

        cur_clin_cls = self._encode_text(data["clin_input_ids"], data["clin_attn_mask"])
        cur_clin_tok = self._text_token(cur_clin_cls, SEG_CUR_CLIN, cdim)                    # (B, 1, C)
        cur_vital_tok = self.vitals_projector(
            data["vital_values"].to(cur_img_tokens.device),
            data["vital_missing_mask"].to(cur_img_tokens.device),
        )                                                                                    # (B, 1, C)
        cur_vital_tok = cur_vital_tok + self.segment_embedding[SEG_CUR_VITALS].view(cdim).to(cur_vital_tok.device)

        # ---- prior branch ------------------------------------------------
        has_prior = data["has_prior"].to(cur_img_tokens.device).bool()                       # (B,)
        days_since = data["days_since_prior"].to(cur_img_tokens.device).float()              # (B,)
        delta_emb = self.delta_embedding(bucket_days(days_since, has_prior))                  # (B, C)
        delta_tok = delta_emb.unsqueeze(1)                                                    # (B, 1, C)

        prv_block, prv_slot_valid = self._encode_image_block(
            data["prior_img"], data["prior_view_positions"], SEG_PRV_VIEWS
        )
        prv_img_tokens = einops.rearrange(prv_block, "b s c h w -> b (s h w) c")

        prv_clin_cls = self._encode_text(data["prior_clin_input_ids"], data["prior_clin_attn_mask"])
        prv_clin_tok = self._text_token(prv_clin_cls, SEG_PRV_CLIN, cdim)
        prv_report_cls = self._encode_text(data["prior_report_input_ids"], data["prior_report_attn_mask"])
        prv_report_tok = self._text_token(prv_report_cls, SEG_PRV_REPORT, cdim)
        prv_vital_tok = self.vitals_projector(
            data["prior_vital_values"].to(cur_img_tokens.device),
            data["prior_vital_missing_mask"].to(cur_img_tokens.device),
        )
        prv_vital_tok = prv_vital_tok + self.segment_embedding[SEG_PRV_VITALS].view(cdim).to(prv_vital_tok.device)
        prior_label = data["prior_label"].to(cur_img_tokens.device).float()                  # (B, 26)
        prv_label_tok = (self.prior_label_proj(prior_label) + self.segment_embedding[SEG_PRV_LABELS].view(cdim)).unsqueeze(1)

        # Time-delta embedding broadcast onto every prior token.
        prv_img_tokens = prv_img_tokens + delta_tok
        prv_clin_tok = prv_clin_tok + delta_tok
        prv_report_tok = prv_report_tok + delta_tok
        prv_vital_tok = prv_vital_tok + delta_tok
        prv_label_tok = prv_label_tok + delta_tok

        tokens = torch.cat(
            [
                cur_img_tokens,
                cur_clin_tok,
                cur_vital_tok,
                prv_img_tokens,
                prv_clin_tok,
                prv_report_tok,
                prv_vital_tok,
                prv_label_tok,
            ],
            dim=1,
        )

        # ---- valid-token mask --------------------------------------------
        ones1 = torch.ones(b, 1, dtype=torch.bool, device=tokens.device)
        prior1 = has_prior.unsqueeze(1)                                                       # (B, 1)
        valid = torch.cat(
            [
                self._valid_image_tokens(cur_slot_valid, h2, w2),                             # current image
                ones1,                                                                        # current clinical
                ones1,                                                                        # current vitals
                self._valid_image_tokens(prv_slot_valid & has_prior.unsqueeze(-1), h2, w2),   # prior image
                prior1,                                                                       # prior clinical
                prior1,                                                                       # prior report
                prior1,                                                                       # prior vitals
                prior1,                                                                       # prior label
            ],
            dim=1,
        )
        mask = ~valid
        x = self.transformer_encoder(tokens, src_key_padding_mask=mask)
        return self.head(x, mask)
