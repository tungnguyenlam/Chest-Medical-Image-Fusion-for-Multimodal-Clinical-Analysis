"""Prior-Aware v6 Nano: native-width (640) fusion with a lower-capacity geometry.

Successor to ``PriorAwareV5NanoModel``. v5 ran the whole fusion bus at ``d_model=768``
to match CXR-BERT's hidden size, which forced a learned ``Conv2d(640->768, stride 2)``
on the image path (the backbone's native channel width is 640). v6 keeps every v5
idea -- asymmetric current/prior cross-attention, selective prior pooling into
``n_prior_latents`` latents, the high-res skip, per-modality pre-fusion LayerNorms,
the always-valid memory sentinel -- but changes the *geometry* to cut capacity, which
is the v6 thesis (this family overfit at 768).

Three geometry changes vs v5:

1. **Native bus width ``d_model=640``.** The fusion bus is the ConvNeXtV2-Nano output
   width, so the image needs **no channel projection**. Every fusion/pooler tensor
   shrinks by ``640/768`` (attention projections by ~0.69x, FFN linearly), a free
   capacity cut across the whole stack. ``640 / 8 heads = 80`` stays integer.

2. **Image: pool instead of strided conv.** The backbone's ``640x16x16`` feature grid
   is spatially pooled to ``640x8x8`` (``image_pool_type`` in {max, avg, depthwise};
   ``image_pool_stride=1`` keeps 16x16, the expensive high-resolution cell). This
   deletes v5's ~4.4M-param ``image_proj`` conv. **Max** pool is the default -- it keeps
   focal bright/dark lesions that average-pooling would wash out, consistent with the
   small-finding thesis and the ``highres_skip`` max-pool. A 1x1 projection is inserted
   ONLY if ``d_model != 640``; at the default 640 it is an identity.

3. **Text: one frozen CLS -> ``n_text_tokens`` fusion tokens.** CXR-BERT emits a single
   768-d CLS per signal. A learned ``Linear(text_embed_dim -> n_text_tokens * d_model)``
   expands it to ``n_text_tokens`` (default 2) tokens of width ``d_model``, giving the
   fusion cross-attention more than one slot per text signal. The two tokens are derived
   from the same CLS (no new information beyond what CLS carries), but they let attention
   read the text through more than a single bottleneck. The frozen CLS cache is reused
   unchanged: it stores raw 768-d CLS vectors and the projection happens inside the model,
   so ``use_precomputed_text_embeddings`` / the GPU-resident table still work.

Regularization (these were off in v5; v6 turns them on via config -- the family overfit):
  - ``drop_path_rate`` on the shared backbone (set in ``timm_init_args``),
  - ``context_bottleneck_dim`` to squeeze the non-image context tokens,
  - ``fusion_ffn_dim`` exposed (v5 silently used the 2048 ``TransformerDecoderLayer``
    default) so the fusion + pooler FFN width is a knob.

Per-token layout (default d_model=640, image_pool_stride=2 -> 8x8 grid, 4 views, n_text_tokens=2):
  current (tgt):   256 image + 2 clinical + 1 vitals                       = 259  (+1 skip token)
  prior  (memory): 1 sentinel + 256 image + 2 clin + 2 report + 1 vitals + 1 label = 263
  prior latents:   K (default 16) after selective pooling

Not checkpoint-compatible with v5: the bus width changed (640 vs 768), the image
projection conv is gone, and the text path now emits a different token count.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Native ConvNeXtV2-Nano final feature width. The fusion bus runs at this when
# d_model == IMG_FEAT_CHANNELS, so the image path needs no channel projection.
IMG_FEAT_CHANNELS = 640


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


def _make_image_pool(pool_type: str, channels: int, stride: int) -> nn.Module:
    """Spatial downsampler for the backbone feature grid (640x16x16 -> 640x8x8 at
    stride 2). ``stride=1`` is identity (keep full resolution). ``max`` keeps focal
    peaks, ``avg`` blurs, ``depthwise`` is a learned per-channel 3x3 strided conv."""
    if stride == 1:
        return nn.Identity()
    if pool_type == "max":
        return nn.MaxPool2d(kernel_size=stride, stride=stride)
    if pool_type == "avg":
        return nn.AvgPool2d(kernel_size=stride, stride=stride)
    if pool_type == "depthwise":
        return nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels)
    raise ValueError(f"image_pool_type must be one of max|avg|depthwise, got {pool_type!r}")


class PriorAwareV6NanoModel(nn.Module):
    """Prior-aware v6: native-640-width fusion, pooled image path, multi-token text."""

    gradcam_runner_module = "src.interpret.run_prior_gradcam"

    def __init__(
        self,
        timm_init_args: dict,
        frontal_pretrained_path: str | None = None,
        lateral_pretrained_path: str | None = None,
        text_model: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        d_model: int = 640,
        n_classes: int = N_CLASSES,
        transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        use_precomputed_text_embeddings: bool = False,
        vital_dropout_p: float = 0.0,
        vitals_dropout: float = 0.1,
        vitals_hidden_dim: int = 256,
        # ---- v5 knobs (carried over) ----
        n_prior_latents: int = 16,
        pooler_nhead: int = 8,
        prior_latent_dropout: float = 0.1,
        context_bottleneck_dim: int | None = None,
        highres_skip: bool = True,
        background_penalty_lambda: float = 0.0,
        # ---- v6 geometry knobs ----
        image_pool_type: str = "max",
        image_pool_stride: int = 2,
        text_embed_dim: int = 768,
        n_text_tokens: int = 2,
        fusion_ffn_dim: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_text_encoder = freeze_text_encoder
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        self.n_prior_latents = n_prior_latents
        self.prior_latent_dropout = prior_latent_dropout
        self.highres_skip = highres_skip
        self.n_text_tokens = n_text_tokens
        # >0 turns on the confident-background attention penalty: discourage current
        # image features that land on outside-patient cells (see docs/background_attention_penalty.md).
        # 0.0 = exactly the base v6 behaviour (forward returns only logits).
        self.background_penalty_lambda = float(background_penalty_lambda)
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

        # Image path: spatial pool (640x16x16 -> 640x8x8), then project to d_model ONLY if
        # d_model != 640 (identity at the native width -- the whole point of v6's geometry).
        self.image_pool = _make_image_pool(image_pool_type, IMG_FEAT_CHANNELS, image_pool_stride)
        self.image_proj = (
            nn.Identity() if d_model == IMG_FEAT_CHANNELS
            else nn.Conv2d(IMG_FEAT_CHANNELS, d_model, kernel_size=1)
        )
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))
        # grid_size=1: numeric vitals -> a SINGLE token; built at d_model so no extra projection.
        self.vitals_projector = VitalsTokenProjector(
            num_vitals=7,
            d_model=d_model,
            grid_size=1,
            hidden_dim=vitals_hidden_dim,
            dropout=vitals_dropout,
            vital_dropout_p=vital_dropout_p,
        )

        # Text: one frozen CLS (text_embed_dim, e.g. 768) -> n_text_tokens tokens of width d_model.
        self.text_proj = nn.Linear(text_embed_dim, n_text_tokens * d_model)

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
        # These are the ONLY normalization the cross-attention K/V memory tokens get (the
        # pre-LN decoder normalizes queries, not cross-attention memory).
        self.norm_img = nn.LayerNorm(d_model)
        self.norm_clin = nn.LayerNorm(d_model)
        self.norm_vitals = nn.LayerNorm(d_model)
        self.norm_report = nn.LayerNorm(d_model)
        self.norm_label = nn.LayerNorm(d_model)
        self.norm_sentinel = nn.LayerNorm(d_model)

        # Learned selective pooling of the prior memory: K query tokens run one Perceiver
        # block (self-attn + cross-attn to the prior tokens + FFN) -> K prior latents.
        if n_prior_latents > 0:
            self.prior_latent_queries = nn.Parameter(torch.randn(n_prior_latents, d_model))
            nn.init.normal_(self.prior_latent_queries, std=0.02)
            pooler_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=pooler_nhead,
                dim_feedforward=fusion_ffn_dim,
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
            dim_feedforward=fusion_ffn_dim,
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
        feats = self.image_pool(feats)   # 640x16x16 -> 640x8x8 (identity if stride 1)
        feats = self.image_proj(feats)   # 640 -> d_model (identity at native width)
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
        """Register a frozen ``[N, text_embed_dim]`` precomputed embedding table as a
        (non-persistent) buffer for the GPU-resident text path. The stored CLS width is
        the encoder's (768); ``text_proj`` maps it to ``n_text_tokens * d_model``. Opt-in."""
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

    def _text_tokens(self, cls: torch.Tensor, segment_index: int) -> torch.Tensor:
        """One frozen CLS vector (B, text_embed_dim) -> (B, n_text_tokens, d_model).

        A learned linear expands the single CLS into ``n_text_tokens`` fusion tokens;
        all tokens of a signal share the same segment embedding (same modality/branch)
        and differ only through the projection weights."""
        proj = self.text_proj(cls)                                            # (B, n_text_tokens * C)
        toks = proj.view(cls.shape[0], self.n_text_tokens, self.d_model)      # (B, n_text_tokens, C)
        seg = self.segment_embedding[segment_index].view(self.d_model).to(cls.device)
        return toks + seg

    def _valid_image_tokens(self, slot_valid: torch.Tensor, h2: int, w2: int) -> torch.Tensor:
        b, s = slot_valid.shape
        return slot_valid.unsqueeze(-1).expand(b, s, h2 * w2).reshape(b, s * h2 * w2)

    def _pool_prior(self, memory: torch.Tensor, mem_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Selectively pool the prior memory to ``K`` latents, then optionally drop whole
        latents (training only, always keeping >=1). Returns (latents, pad)."""
        b = memory.shape[0]
        queries = self.prior_latent_queries.unsqueeze(0).expand(b, -1, -1)                 # (B, K, C)
        latents = self.prior_pooler(queries, memory, memory_key_padding_mask=mem_pad)      # (B, K, C)

        if self.training and self.prior_latent_dropout > 0.0:
            keep = torch.rand(latents.shape[:2], device=latents.device) >= self.prior_latent_dropout
            keep[:, 0] = True  # guarantee at least one valid latent (no all-masked row -> NaN)
            return latents, ~keep
        return latents, None

    def _background_penalty(
        self, cur_block: torch.Tensor, cur_slot_valid: torch.Tensor, bg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean current-image feature energy in confident-background cells.

        ``cur_block`` (B,S,C,H,W) is the PRE-LayerNorm image grid (energy is only
        meaningful pre-norm; LayerNorm would flatten every token to ~unit norm).
        ``bg_mask`` (B,S,G,G) is the per-view confident-background weight; we
        average-pool it to the (H,W) grid so each cell's weight is the fraction of
        it that is background. Invalid (padded) views are zeroed via ``cur_slot_valid``.
        See docs/background_attention_penalty.md for the full derivation.
        """
        b, s, c, h, w = cur_block.shape
        energy = cur_block.pow(2).sum(dim=2)                                  # (B,S,H,W)
        bg = bg_mask.to(cur_block.dtype).reshape(b * s, 1, bg_mask.shape[-2], bg_mask.shape[-1])
        bg = F.adaptive_avg_pool2d(bg, (h, w)).reshape(b, s, h, w)            # (B,S,H,W) fractional weight
        weight = bg * cur_slot_valid.to(cur_block.dtype).view(b, s, 1, 1)
        return (weight * energy).sum() / (weight.sum() + 1e-6)

    # ---- forward ----------------------------------------------------------
    def forward(self, data: dict):
        # ---- current branch (tgt) ----------------------------------------
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )
        b, s_img, cdim, h2, w2 = cur_block.shape
        cur_img_tokens = einops.rearrange(cur_block, "b s c h w -> b (s h w) c")

        # Confident-background attention penalty (opt-in; pre-norm energy).
        bg_penalty = None
        if self.background_penalty_lambda > 0.0 and "bg_mask" in data:
            bg_penalty = self.background_penalty_lambda * self._background_penalty(
                cur_block, cur_slot_valid, data["bg_mask"]
            )

        cur_clin_cls = self._encode_text(data["clin_input_ids"], data["clin_attn_mask"])
        cur_clin_tok = self._text_tokens(cur_clin_cls, SEG_CUR_CLIN)                          # (B, n_text, C)
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
        tgt = torch.cat([cur_img_tokens, cur_clin_tok, cur_vital_tok], dim=1)                 # (B, 256+n_text+1, C)

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
        prv_clin_tok = self._text_tokens(prv_clin_cls, SEG_PRV_CLIN)                          # (B, n_text, C)
        prv_report_cls = self._encode_text(data["prior_report_input_ids"], data["prior_report_attn_mask"])
        prv_report_tok = self._text_tokens(prv_report_cls, SEG_PRV_REPORT)                    # (B, n_text, C)
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
        )                                                                                     # (B, 1+256+2n_text+2, C)

        # ---- padding masks ------------------------------------------------
        ones1 = torch.ones(b, 1, dtype=torch.bool, device=tgt.device)
        ones_text = torch.ones(b, self.n_text_tokens, dtype=torch.bool, device=tgt.device)
        prior1 = has_prior.unsqueeze(1)                                                       # (B, 1)
        prior_text = prior1.expand(b, self.n_text_tokens)                                     # (B, n_text)
        cur_img_valid = self._valid_image_tokens(cur_slot_valid, h2, w2)
        cur_valid = torch.cat([cur_img_valid, ones_text, ones1], dim=1)                        # img, clin(n_text), vitals
        mem_valid = torch.cat(
            [
                ones1,                                                                        # sentinel (always valid)
                self._valid_image_tokens(prv_slot_valid & has_prior.unsqueeze(-1), h2, w2),    # prior image
                prior_text,                                                                   # prior clinical (n_text)
                prior_text,                                                                   # prior report (n_text)
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

        logits = self.head(x, tgt_pad)
        # When the background penalty is active, hand it back as an auxiliary loss
        # (train_step adds it to the criterion). Otherwise behave exactly like base v6.
        if bg_penalty is not None:
            return logits, bg_penalty
        return logits
