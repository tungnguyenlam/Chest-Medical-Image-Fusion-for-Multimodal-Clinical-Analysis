"""Prior-Aware v7 Nano: resolution-gated learned-query image pooler.

Successor to ``PriorAwareV6NanoModel``. v7 keeps the v6 encoder, the asymmetric
current/prior cross-attention fusion, every v6 regularization knob, **the v6 2x2
max-pool**, and the prior Perceiver pooler; it adds a per-view learned-query
pooler that runs *on top of* the max-pool and only when it buys something:

1. **v6's 2x2 max-pool is kept (``image_pool_stride=2``).** The backbone grid is
   max-pooled exactly as in v6 (640x16x16 -> 640x8x8 at 512px input). v7 does NOT
   drop it; the learned query consumes the post-max-pool grid.

2. **Current image: per-view learned-query pooler, gated on input resolution.**
   A per-view Perceiver with ``n_cur_image_latents`` (default 64) learned query
   tokens cross-attends to the post-max-pool grid of each view through one
   ``TransformerDecoder`` block -> K latents/view. It is **skipped at 512x512
   input**, where the post-max-pool grid is already 8x8 = 64 tokens/view: pooling
   that further would only lose information for no resolution gain (the design-doc
   risk #1), so at 512 the current path is byte-for-byte v6 (max-pooled flat
   tokens, 4*64 = 256). Above 512 (e.g. 1024 -> 32x32 -> max-pool -> 16x16) the
   pooler runs: K=64/view -> 4*64 = 256 latents, so the fusion sequence length is
   256 current image tokens at *every* resolution. The query is what decouples
   fusion cost from input resolution.

3. **Prior image: per-view learned-query pooler, K=32/view (always on).** A second
   per-view Perceiver (``n_prior_image_latents``, default 32) reduces the prior
   image to 4*32 = 128 latents. It is NOT resolution-gated (the user's
   "current-only" choice); at 1024px the un-pooled prior would be 4*16*16 = 1024
   tokens, which this avoids.

4. **Prior memory pooler: K=32 (was 16).** Same Perceiver primitive v6 already
   uses on the *full* prior memory; doubled budget. ``n_prior_latents=32`` passed
   up to ``PriorAwareV6NanoModel``. (This is distinct from the per-view prior
   *image* pooler above: this one selects across all prior modalities.)

5. **High-res skip disabled by default.** v6's ``highres_skip`` is a max-pool
   over the un-fused current image tokens; the per-view pooler supersedes it.
   Re-enable via config to recreate the v6 behaviour.

Per-token layout (defaults: d_model=640, n_cur_image_latents=64, n_prior_image_latents=32, 4 views, n_text_tokens=2):
    current (tgt) @512:  256 image (4*8*8, pooler skipped) + 2 clinical + 1 vitals       = 259
    current (tgt) @1024: 256 image (4*64 latents)          + 2 clinical + 1 vitals       = 259
    prior  (memory): 1 sentinel + 128 image (4*32) + 2 clin + 2 report + 1 vitals + 1 label = 135
    prior latents:   K=32 (was 16) after selective pooling of the full prior memory

Not checkpoint-compatible with v6: the per-view poolers are new and the prior
memory K doubled. The image path geometry (max-pool, stride 2) matches v6.

Risks acknowledged in ``docs/learned_query_image_pooling.md``:
- Per-view pooler fights the small-finding thesis if queries collapse; the primary
  readout is the small-finding-subset mAP, not overall mAP.
- Spatial grounding: Grad-CAM will need to attribute through the pooler's
  cross-attention weights (a follow-up; the post-max-pool grid is still
  recoverable as ``cur_block`` until after pooling).
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn

from src.dataloader.PriorAwareDataset import N_CLASSES, bucket_days
from src.model.PriorAwareV6NanoModel import (
    IMG_FEAT_CHANNELS,
    SEG_CUR_VIEWS,
    SEG_CUR_CLIN,
    SEG_CUR_VITALS,
    SEG_PRV_VIEWS,
    SEG_PRV_CLIN,
    SEG_PRV_VITALS,
    SEG_PRV_LABELS,
    SEG_PRV_REPORT,
    PriorAwareV6NanoModel,
)

# Input edge length (px) at which the current-image learned-query pooler is
# skipped. At 512 the backbone /32 grid is 16x16 and v6's 2x2 max-pool takes it
# to 8x8 = 64 tokens/view; pooling that further loses information for no
# resolution gain, so the current path falls back to v6 exactly. Hardcoded (the
# user's explicit choice over a config knob); the equivalent condition is a
# post-max-pool grid of 8x8.
SKIP_CUR_POOLER_INPUT_SIZE = 512


class PriorAwareV7NanoModel(PriorAwareV6NanoModel):
    """Prior-aware v7: per-view learned-query image pooler (current) + K=32 prior latents.

    All v6 init args remain valid; the only changes vs v6 are:

    - ``n_cur_image_latents`` (default 64) — K per view for the current-image Perceiver.
    - ``cur_pooler_nhead`` (default 8) — heads of the current-image pooler decoder.
    - ``cur_pooler_dropout`` (default ``dropout``) — pooler dropout.
    - ``cur_pooler_ffn_dim`` (default ``fusion_ffn_dim``) — pooler FFN width.
    - ``n_prior_image_latents`` (default 32) — K per view for the prior-image Perceiver.
    - ``prv_pooler_nhead`` / ``prv_pooler_dropout`` / ``prv_pooler_ffn_dim`` — prior-image
      pooler decoder knobs (default to 8 / ``dropout`` / ``fusion_ffn_dim``).
    - ``n_prior_latents`` default raised to 32 (was 16 in v6) — the full-memory pooler.
    - ``image_pool_stride`` keeps v6's default of 2 (the 2x2 max-pool is retained;
      the learned query runs on the post-max-pool grid).
    - ``highres_skip`` default forced to ``False`` (pooler supersedes it).

    The current-image pooler is skipped when the input is
    ``SKIP_CUR_POOLER_INPUT_SIZE`` x ``SKIP_CUR_POOLER_INPUT_SIZE`` (512x512),
    where the post-max-pool grid is already 8x8 and the current path equals v6.
    """

    gradcam_runner_module = "src.interpret.run_prior_gradcam"

    def __init__(
        self,
        timm_init_args: dict,
        frontal_pretrained_path: str | None = None,
        lateral_pretrained_path: str | None = None,
        text_model: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        d_model: int = IMG_FEAT_CHANNELS,
        n_classes: int = N_CLASSES,
        transformer_layers: int = 2,
        nhead: int = 8,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
        use_precomputed_text_embeddings: bool = False,
        vital_dropout_p: float = 0.0,
        vitals_dropout: float = 0.1,
        vitals_hidden_dim: int = 256,
        n_prior_latents: int = 32,
        pooler_nhead: int = 8,
        prior_latent_dropout: float = 0.1,
        context_bottleneck_dim: int | None = 64,
        highres_skip: bool = False,
        background_penalty_lambda: float = 0.0,
        image_pool_type: str = "max",
        image_pool_stride: int = 2,
        text_embed_dim: int = 768,
        n_text_tokens: int = 2,
        fusion_ffn_dim: int = 1024,
        n_cur_image_latents: int = 64,
        cur_pooler_nhead: int = 8,
        cur_pooler_dropout: float | None = None,
        cur_pooler_ffn_dim: int | None = None,
        n_prior_image_latents: int = 32,
        prv_pooler_nhead: int = 8,
        prv_pooler_dropout: float | None = None,
        prv_pooler_ffn_dim: int | None = None,
    ):
        if cur_pooler_dropout is None:
            cur_pooler_dropout = dropout
        if cur_pooler_ffn_dim is None:
            cur_pooler_ffn_dim = fusion_ffn_dim
        if prv_pooler_dropout is None:
            prv_pooler_dropout = dropout
        if prv_pooler_ffn_dim is None:
            prv_pooler_ffn_dim = fusion_ffn_dim

        super().__init__(
            timm_init_args=timm_init_args,
            frontal_pretrained_path=frontal_pretrained_path,
            lateral_pretrained_path=lateral_pretrained_path,
            text_model=text_model,
            d_model=d_model,
            n_classes=n_classes,
            transformer_layers=transformer_layers,
            nhead=nhead,
            dropout=dropout,
            freeze_text_encoder=freeze_text_encoder,
            use_precomputed_text_embeddings=use_precomputed_text_embeddings,
            vital_dropout_p=vital_dropout_p,
            vitals_dropout=vitals_dropout,
            vitals_hidden_dim=vitals_hidden_dim,
            n_prior_latents=n_prior_latents,
            pooler_nhead=pooler_nhead,
            prior_latent_dropout=prior_latent_dropout,
            context_bottleneck_dim=context_bottleneck_dim,
            highres_skip=highres_skip,
            background_penalty_lambda=background_penalty_lambda,
            image_pool_type=image_pool_type,
            image_pool_stride=image_pool_stride,
            text_embed_dim=text_embed_dim,
            n_text_tokens=n_text_tokens,
            fusion_ffn_dim=fusion_ffn_dim,
        )

        self.n_cur_image_latents = n_cur_image_latents

        # Per-view Perceiver pooler for the current image. K=64 by default, applied
        # independently to each of the 4 views, and only above 512px input (see
        # SKIP_CUR_POOLER_INPUT_SIZE). K=64 was chosen so the post-pool token count
        # (4*64=256) matches v6's max-pooled 256 -> the fusion sequence length is
        # 256 current image tokens at every resolution. At 1024px the pooler turns
        # the post-max-pool 16x16 grid into 64 content-adaptive latents/view; at
        # 512px it is skipped and the 8x8 max-pooled grid is used directly (= v6).
        if n_cur_image_latents > 0:
            self.cur_image_queries = nn.Parameter(
                torch.randn(n_cur_image_latents, d_model)
            )
            nn.init.normal_(self.cur_image_queries, std=0.02)
            cur_pooler_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=cur_pooler_nhead,
                dim_feedforward=cur_pooler_ffn_dim,
                dropout=cur_pooler_dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.cur_image_pooler = nn.TransformerDecoder(
                cur_pooler_layer, num_layers=1, norm=nn.LayerNorm(d_model)
            )
        else:
            self.cur_image_queries = None
            self.cur_image_pooler = None

        self.n_prior_image_latents = n_prior_image_latents

        # Per-view Perceiver pooler for the PRIOR image. Same primitive as the
        # current-image pooler but a smaller budget (K=32/view default -> 4*32=128
        # prior image latents) and NOT resolution-gated (always on). At 1024px the
        # post-max-pool prior grid is 16x16 -> 4*16*16=1024 tokens; this learned
        # pooler keeps the prior cheap (128 latents) instead of dumping 1024 keys
        # into the prior memory Perceiver. At 512px (post-max-pool 8x8 = 64/view)
        # it still pools 64 -> 32/view, which is the user's "current-only" choice
        # for the resolution gate.
        if n_prior_image_latents > 0:
            self.prv_image_queries = nn.Parameter(
                torch.randn(n_prior_image_latents, d_model)
            )
            nn.init.normal_(self.prv_image_queries, std=0.02)
            prv_pooler_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=prv_pooler_nhead,
                dim_feedforward=prv_pooler_ffn_dim,
                dropout=prv_pooler_dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.prv_image_pooler = nn.TransformerDecoder(
                prv_pooler_layer, num_layers=1, norm=nn.LayerNorm(d_model)
            )
        else:
            self.prv_image_queries = None
            self.prv_image_pooler = None

    def _perceiver_pool_per_view(
        self,
        block: torch.Tensor,
        slot_valid: torch.Tensor,
        queries: torch.Tensor,
        pooler: nn.Module,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-view learned-query (Perceiver) pooler on a 16x16 image grid.

        ``block`` (B,S,C,H,W) is the post-pos-encoding grid; ``slot_valid`` (B,S)
        marks which views are real (False = padded). Returns:

        - img_tokens (B, S*K, C) — K learned latents per view, concatenated along S*K.
        - img_valid (B, S*K) — False only for padded views (the per-view pooler
          runs unconditionally on a view's tokens; a fully-padded view's queries
          are unused by the rest of the model via this mask).

        The pooler runs *per view* (independent Perceiver) rather than over the
        concatenated 4x16x16 grid to preserve view identity — the segment
        embedding already encodes view, but per-view keeps the query budget
        explicitly per view and makes the bookkeeping symmetric with v6's
        per-view 8x8 max-pool.
        """
        b, s, c, h, w = block.shape

        # (B*S, C, H, W) -> (B*S, H*W, C) so each view is a separate sequence the
        # shared Perceiver can run on with batch_first=True.
        tokens = einops.rearrange(block, "b s c h w -> (b s) (h w) c")
        q = queries.unsqueeze(0).expand(b * s, -1, -1)                              # (B*S, K, C)
        latents = pooler(q, tokens)                                                 # (B*S, K, C)
        img_tokens = einops.rearrange(latents, "(b s) k c -> b (s k) c", b=b, s=s)

        img_valid = slot_valid.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        return img_tokens, img_valid

    def _pool_current_image_per_view(
        self, cur_block: torch.Tensor, cur_slot_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K=``n_cur_image_latents`` learned latents per current-image view."""
        return self._perceiver_pool_per_view(
            cur_block, cur_slot_valid, self.cur_image_queries, self.cur_image_pooler, self.n_cur_image_latents
        )

    def _pool_prior_image_per_view(
        self, prv_block: torch.Tensor, prv_slot_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K=``n_prior_image_latents`` learned latents per prior-image view."""
        return self._perceiver_pool_per_view(
            prv_block, prv_slot_valid, self.prv_image_queries, self.prv_image_pooler, self.n_prior_image_latents
        )

    def forward(self, data: dict):
        # ---- current branch (tgt) ----------------------------------------
        # _encode_image_block runs the backbone + v6's 2x2 max-pool + 2D pos
        # encoding. At 512px input cur_block is (B, S, 640, 8, 8); at 1024px it is
        # (B, S, 640, 16, 16). cur_slot_valid is (B, S).
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )

        # Confident-background attention penalty (opt-in; pre-norm energy on the
        # post-max-pool grid, exactly as in v6).
        bg_penalty = None
        if self.background_penalty_lambda > 0.0 and "bg_mask" in data:
            bg_penalty = self.background_penalty_lambda * self._background_penalty(
                cur_block, cur_slot_valid, data["bg_mask"]
            )

        # Resolution gate: skip the current-image pooler at 512x512 (post-max-pool
        # grid already 8x8). At 512 the current path is byte-for-byte v6 -- the
        # max-pooled grid flattened to (B, S*H*W, C) -- which is also the fallback
        # when the pooler is disabled (n_cur_image_latents=0). Above 512 the
        # per-view learned-query pooler runs -> (B, S*K, C), keeping the current
        # image at 256 fusion tokens regardless of input resolution.
        in_h, in_w = data["img"].shape[-2], data["img"].shape[-1]
        skip_cur_pooler = in_h == SKIP_CUR_POOLER_INPUT_SIZE and in_w == SKIP_CUR_POOLER_INPUT_SIZE
        if self.cur_image_pooler is not None and not skip_cur_pooler:
            cur_img_tokens, cur_img_valid = self._pool_current_image_per_view(
                cur_block, cur_slot_valid
            )
        else:
            ch, cw = cur_block.shape[-2], cur_block.shape[-1]
            cur_img_tokens = einops.rearrange(cur_block, "b s c h w -> b (s h w) c")
            cur_img_valid = self._valid_image_tokens(cur_slot_valid, ch, cw)

        b, _, cdim = cur_img_tokens.shape

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
        tgt = torch.cat([cur_img_tokens, cur_clin_tok, cur_vital_tok], dim=1)                 # (B, S*K + n_text + 1, C)

        # ---- prior branch (memory) ---------------------------------------
        has_prior = data["has_prior"].to(cur_img_tokens.device).bool()                       # (B,)
        days_since = data["days_since_prior"].to(cur_img_tokens.device).float()              # (B,)
        delta_emb = self.delta_embedding(bucket_days(days_since, has_prior))                   # (B, C)
        delta_tok = delta_emb.unsqueeze(1)                                                    # (B, 1, C)

        prv_block, prv_slot_valid = self._encode_image_block(
            data["prior_img"], data["prior_view_positions"], SEG_PRV_VIEWS
        )
        # Per-view learned-query pooler -> (B, S*K_prv, C). Always on (not
        # resolution-gated). At 1024px the post-max-pool prior grid is 16x16, so
        # without this the prior image would arrive as 4*16*16=1024 tokens; the
        # pooler keeps it at 4*K_prv latents instead.
        prv_img_tokens, prv_img_valid = self._pool_prior_image_per_view(
            prv_block, prv_slot_valid
        )

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
        )                                                                                     # (B, 1 + S*K_prv + 4*n_text + 2, C)

        # ---- padding masks ------------------------------------------------
        ones1 = torch.ones(b, 1, dtype=torch.bool, device=tgt.device)
        ones_text = torch.ones(b, self.n_text_tokens, dtype=torch.bool, device=tgt.device)
        prior1 = has_prior.unsqueeze(1)                                                       # (B, 1)
        prior_text = prior1.expand(b, self.n_text_tokens)                                     # (B, n_text)
        # cur_img_valid already comes from the per-view pooler (B, S*K) so
        # padded views are masked out; the rest of v6's bookkeeping is unchanged.
        cur_valid = torch.cat([cur_img_valid, ones_text, ones1], dim=1)                        # img, clin(n_text), vitals
        mem_valid = torch.cat(
            [
                ones1,                                                                        # sentinel (always valid)
                prv_img_valid & has_prior.unsqueeze(-1),                                       # prior image (per-view pooled, K_prv each)
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

        # ---- high-resolution skip (opt-in; default off in v7) ----
        if self.highres_skip:
            # When highres_skip is enabled alongside the pooler, max-pool the
            # un-fused 16x16 cur_block — preserves v6's "focal evidence reaches
            # the classifier" path. v7 disables this by default because the
            # per-view pooler already does content-adaptive selection.
            h, w = cur_block.shape[-2], cur_block.shape[-1]
            masked = einops.rearrange(cur_block, "b s c h w -> b (s h w) c").masked_fill(
                ~self._valid_image_tokens(cur_slot_valid, h, w).unsqueeze(-1), float("-inf")
            )
            skip_tok = masked.max(dim=1).values.unsqueeze(1)                                   # (B, 1, C)
            x = torch.cat([x, skip_tok], dim=1)
            tgt_pad = torch.cat([tgt_pad, torch.zeros(b, 1, dtype=torch.bool, device=tgt.device)], dim=1)

        logits = self.head(x, tgt_pad)
        if bg_penalty is not None:
            return logits, bg_penalty
        return logits
