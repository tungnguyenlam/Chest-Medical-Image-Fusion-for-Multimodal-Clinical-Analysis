"""Prior-Aware v5 Nano: bottlenecked asymmetric fusion that protects small findings.

Successor to ``PriorAwareV4NanoModel``. v4 made attention *asymmetric* -- the 258
current tokens are the residual stream (``tgt``) and the 261 prior tokens are
read-only ``memory`` -- which structurally weakens the prior-label copy shortcut.
v5 keeps that asymmetry and adds an explicit **information bottleneck on the prior
context**, while deliberately *not* bottlenecking the current image, so that focal
disease evidence (nodule, mass, pneumothorax line, subtle effusion) survives.

The design separates two concerns that v4 conflated:

* **Where the small detail lives:** the *current* image, at spatial resolution.
* **Where the memorization / copy risk lives:** the *prior* context (especially the
  ``Linear(26->768)`` prior-label token, which v4 still let reach fusion as a clean,
  near-copyable channel).

So the compression is applied where it helps and withheld where it hurts:

1. **Learned selective pooling of the prior memory** (``n_prior_latents`` > 0). A small
   bank of learned query tokens runs one Perceiver/Q-Former-style block
   (``self-attn(latents) + cross-attn(latents -> 261 prior tokens) + FFN``) and emits
   ``K`` prior *latents* (default 16). Because it is attention-based *selection* (not
   average/strided pooling), a single change-relevant prior patch can win a query --
   focal prior evidence is preserved, but the prior label can no longer arrive
   un-mixed. Cross-attention cost in fusion drops from ``258*261`` to ``258*K``.
   ``n_prior_latents=0`` disables pooling and falls back to v4's full 261-token memory
   (the "pool only" vs "baseline" ablation cell lives inside this one class).

2. **Current branch kept at (optionally higher) full spatial resolution.**
   ``current_image_stride`` controls the ``image_proj`` stride: ``2`` -> 8x8 = 64
   tokens/view (v4 default), ``1`` -> 16x16 = 256 tokens/view (higher resolution, the
   small-finding ablation). The current image is never spatially pooled.

3. **High-resolution skip to the head** (``highres_skip``). A max-pool over the valid,
   *un-fused* current image tokens is appended as one extra token to the head input.
   Max (not mean) keeps focal bright/dark lesions that mean-pooling would average away,
   and the skip guarantees the classifier sees full-resolution current evidence even if
   fusion dilutes it.

4. **Context information bottleneck** (``context_bottleneck_dim``, optional). A
   down-up projection (``Linear(d->b) -> GELU -> Linear(b->d)``) squeezes each *non-image*
   context token (clinical, prior report, prior vitals, prior label) through ``b`` dims
   before fusion, forcing it to carry only what is class-relevant. ``None`` disables it.

5. **Prior-latent dropout** (``prior_latent_dropout``). During training whole prior
   latents are randomly masked out of fusion (at least one always kept), forcing
   redundancy across latents so no single latent becomes "the label channel."

Everything else matches v4: the shared ConvNeXtV2-Nano image router, CXR-BERT text
encoder, single-token-per-signal layout, per-modality pre-fusion LayerNorms (the only
normalization the cross-attention K/V ever get), the always-valid ``no_prior_token``
memory sentinel, ``norm_first=True`` decoder with an explicit final norm, and the
``MLDecoder`` head.

Per-token layout (default size=512, stride=2 -> 8x8 grid, 4 views):
  current (tgt):   256 image + 1 clinical + 1 vitals             = 258  (+1 skip token)
  prior  (memory): 1 sentinel + 256 image + 1 clin + 1 report + 1 vitals + 1 label = 261
  prior latents:   K (default 16) after selective pooling

Not checkpoint-compatible with v4: the prior pooler, context bottlenecks, and high-res
skip token are new modules and the head reads a different token count.
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


def _context_bottleneck(d_model: int, bottleneck_dim: int | None) -> nn.Module:
    """Down-up projection that squeezes a context token through ``bottleneck_dim`` dims
    (an information bottleneck) while preserving width ``d_model``. Identity if disabled."""
    if not bottleneck_dim:
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(d_model, bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, d_model),
    )


class PriorAwareV5NanoModel(nn.Module):
    """Prior-aware v5: bottlenecked prior memory + detail-preserving current branch."""

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
        # ---- v5 knobs ----
        current_image_stride: int = 2,
        n_prior_latents: int = 16,
        pooler_nhead: int = 8,
        prior_latent_dropout: float = 0.1,
        context_bottleneck_dim: int | None = None,
        highres_skip: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_text_encoder = freeze_text_encoder
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        self.n_prior_latents = n_prior_latents
        self.prior_latent_dropout = prior_latent_dropout
        self.highres_skip = highres_skip
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

        # stride=2 -> 8x8 grid/view (v4 default); stride=1 -> 16x16 (higher current resolution).
        self.image_proj = nn.Conv2d(640, d_model, kernel_size=3, stride=current_image_stride, padding=1)
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))
        # grid_size=1: numeric vitals -> a SINGLE token (matches v4).
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

        # Learned, always-valid memory sentinel for no-prior samples (avoids an
        # all-masked cross-attention row -> NaN; doubles as an "I have no prior" code).
        self.no_prior_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Information bottlenecks for the non-image context tokens (shared current/prior
        # per modality, like the norms). Squeeze each through context_bottleneck_dim dims.
        self.bottleneck_clin = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_vitals = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_report = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_label = _context_bottleneck(d_model, context_bottleneck_dim)

        # Per-modality pre-fusion LayerNorms. Current/prior share a norm per modality
        # (same encoder/projector source); the segment/delta codes carry the branch id.
        # These are the ONLY normalization the prior K/V memory tokens get (the pre-LN
        # decoder normalizes queries, not cross-attention memory).
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_clin = nn.LayerNorm(d_model)
        self.norm_vitals = nn.LayerNorm(d_model)
        self.norm_report = nn.LayerNorm(d_model)
        self.norm_label = nn.LayerNorm(d_model)
        self.norm_sentinel = nn.LayerNorm(d_model)

        # Learned selective pooling of the prior memory: K query tokens run one Perceiver
        # block (self-attn + cross-attn to the 261 prior tokens + FFN) -> K prior latents.
        if n_prior_latents > 0:
            self.prior_latent_queries = nn.Parameter(torch.randn(n_prior_latents, d_model))
            nn.init.normal_(self.prior_latent_queries, std=0.02)
            pooler_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=pooler_nhead,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.prior_pooler = nn.TransformerDecoder(
                pooler_layer, num_layers=1, norm=nn.LayerNorm(d_model)
            )
        else:
            self.prior_latent_queries = None
            self.prior_pooler = None

        # Asymmetric fusion: current = tgt, prior (latents or full memory) = memory.
        fusion_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerDecoder(
            fusion_layer, num_layers=transformer_layers, norm=nn.LayerNorm(d_model)
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

    def enable_input_normalization(self, mean, std) -> None:
        """Normalize raw uint8 image batches on-device (see ConvNeXtV2NanoImageEncoder).
        Both branches route through the shared image encoder, so this covers both."""
        self.image_encoder.enable_input_normalization(mean, std)

    def attach_text_embedding_table(self, table: torch.Tensor) -> None:
        """Register a frozen ``[N, d_model]`` precomputed embedding table as a
        (non-persistent) buffer for the GPU-resident text path. Opt-in."""
        self.register_buffer("text_embedding_table", table.float(), persistent=False)

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.is_floating_point() and input_ids.ndim == 2:
            return input_ids.float()
        if self.text_embedding_table is not None and not input_ids.is_floating_point() and input_ids.ndim == 1:
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
        """One CLS vector -> a single (B, 1, C) fusion token."""
        seg = self.segment_embedding[segment_index].view(cdim).to(cls.device)
        return (cls + seg).unsqueeze(1)

    def _valid_image_tokens(self, slot_valid: torch.Tensor, h2: int, w2: int) -> torch.Tensor:
        b, s = slot_valid.shape
        return slot_valid.unsqueeze(-1).expand(b, s, h2 * w2).reshape(b, s * h2 * w2)

    def _pool_prior(self, memory: torch.Tensor, mem_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Selectively pool the 261-token prior memory to ``K`` latents, then optionally
        drop whole latents (training only, always keeping >=1). Returns (latents, pad)."""
        b = memory.shape[0]
        queries = self.prior_latent_queries.unsqueeze(0).expand(b, -1, -1)                 # (B, K, C)
        latents = self.prior_pooler(queries, memory, memory_key_padding_mask=mem_pad)      # (B, K, C)

        if self.training and self.prior_latent_dropout > 0.0:
            keep = torch.rand(latents.shape[:2], device=latents.device) >= self.prior_latent_dropout
            keep[:, 0] = True  # guarantee at least one valid latent (no all-masked row -> NaN)
            return latents, ~keep
        return latents, None

    # ---- forward ----------------------------------------------------------
    def forward(self, data: dict) -> torch.Tensor:
        # ---- current branch (tgt) ----------------------------------------
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

        # Context information bottleneck (identity if disabled), then per-modality LayerNorm.
        cur_clin_tok = self.bottleneck_clin(cur_clin_tok)
        cur_vital_tok = self.bottleneck_vitals(cur_vital_tok)
        cur_img_tokens = self.norm_img(cur_img_tokens)
        cur_clin_tok = self.norm_clin(cur_clin_tok)
        cur_vital_tok = self.norm_vitals(cur_vital_tok)
        tgt = torch.cat([cur_img_tokens, cur_clin_tok, cur_vital_tok], dim=1)                 # (B, 258, C)

        # ---- prior branch (memory) ---------------------------------------
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

        # Time-delta embedding broadcast onto every prior token (before bottleneck/LN).
        prv_img_tokens = prv_img_tokens + delta_tok
        prv_clin_tok = prv_clin_tok + delta_tok
        prv_report_tok = prv_report_tok + delta_tok
        prv_vital_tok = prv_vital_tok + delta_tok
        prv_label_tok = prv_label_tok + delta_tok

        # Context information bottleneck (identity if disabled).
        prv_clin_tok = self.bottleneck_clin(prv_clin_tok)
        prv_report_tok = self.bottleneck_report(prv_report_tok)
        prv_vital_tok = self.bottleneck_vitals(prv_vital_tok)
        prv_label_tok = self.bottleneck_label(prv_label_tok)

        # Per-modality LayerNorm -> the ONLY normalization the cross-attention K/V get.
        prv_img_tokens = self.norm_img(prv_img_tokens)
        prv_clin_tok = self.norm_clin(prv_clin_tok)
        prv_report_tok = self.norm_report(prv_report_tok)
        prv_vital_tok = self.norm_vitals(prv_vital_tok)
        prv_label_tok = self.norm_label(prv_label_tok)

        sentinel = self.norm_sentinel(self.no_prior_token).expand(b, 1, cdim)                 # (B, 1, C)
        memory = torch.cat(
            [sentinel, prv_img_tokens, prv_clin_tok, prv_report_tok, prv_vital_tok, prv_label_tok],
            dim=1,
        )                                                                                     # (B, 261, C)

        # ---- padding masks ------------------------------------------------
        ones1 = torch.ones(b, 1, dtype=torch.bool, device=tgt.device)
        prior1 = has_prior.unsqueeze(1)                                                       # (B, 1)
        cur_img_valid = self._valid_image_tokens(cur_slot_valid, h2, w2)
        cur_valid = torch.cat([cur_img_valid, ones1, ones1], dim=1)                            # img, clin, vitals
        mem_valid = torch.cat(
            [
                ones1,                                                                        # sentinel (always valid)
                self._valid_image_tokens(prv_slot_valid & has_prior.unsqueeze(-1), h2, w2),    # prior image
                prior1,                                                                       # prior clinical
                prior1,                                                                       # prior report
                prior1,                                                                       # prior vitals
                prior1,                                                                       # prior label
            ],
            dim=1,
        )
        tgt_pad = ~cur_valid
        mem_pad = ~mem_valid

        # ---- selective prior pooling (or v4-style full memory if disabled) ----
        if self.prior_pooler is not None:
            fusion_memory, fusion_mem_pad = self._pool_prior(memory, mem_pad)
        else:
            fusion_memory, fusion_mem_pad = memory, mem_pad

        x = self.fusion(
            tgt, fusion_memory, tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=fusion_mem_pad
        )

        # ---- high-resolution skip: max-pool over valid un-fused current image tokens --
        if self.highres_skip:
            masked = cur_img_tokens.masked_fill(~cur_img_valid.unsqueeze(-1), float("-inf"))
            skip_tok = masked.max(dim=1).values.unsqueeze(1)                                   # (B, 1, C)
            x = torch.cat([x, skip_tok], dim=1)
            tgt_pad = torch.cat([tgt_pad, torch.zeros(b, 1, dtype=torch.bool, device=tgt.device)], dim=1)

        return self.head(x, tgt_pad)
