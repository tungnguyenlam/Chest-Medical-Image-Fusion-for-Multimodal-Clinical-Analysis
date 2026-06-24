"""Prior-Aware v7 Nano: learned-query image pooler (current and prior).

Successor to ``PriorAwareV6NanoModel``. v7 keeps the v6 encoder, the asymmetric
current/prior cross-attention fusion, every v6 regularization knob, and the prior
Perceiver pooler; it changes the **image path**:

1. **Current image: per-view learned-query pooler, no 16x16->8x8 max-pool.**
   The backbone's native 640x16x16 feature grid is kept (no max-pool — pass
   ``image_pool_stride=1`` so the existing v6 pooler is an identity), then a
   per-view Perceiver pooler with ``n_cur_image_latents`` (default 64) learned
   query tokens cross-attends to the 16x16=256 spatial tokens of that view
   through one ``TransformerDecoder`` block. With 4 views that yields
   4 * 64 = 256 current image latents — same total count as v6's max-pooled
   256, but learned and content-adaptive rather than fixed 2x2 windows.

2. **Prior image: per-view learned-query pooler, K=32/view.** ``image_pool_stride=1``
   removed v6's fixed 8x8 max-pool from the *shared* image path, which would
   otherwise leave the prior image at the full 4*16*16=1024 tokens. A second
   per-view Perceiver pooler (``n_prior_image_latents``, default 32) restores a
   cheap, content-adaptive reduction symmetric with the current path:
   4 * 32 = 128 prior image latents instead of 1024 raw tokens.

3. **Prior memory pooler: K=32 (was 16).** Same Perceiver primitive v6 already
   uses on the *full* prior memory; doubled budget. ``n_prior_latents=32`` passed
   up to ``PriorAwareV6NanoModel``. (This is distinct from the per-view prior
   *image* pooler above: this one selects across all prior modalities.)

4. **High-res skip disabled by default.** v6's ``highres_skip`` is a max-pool
   over the un-fused current image tokens; the per-view pooler supersedes it
   (the pooler is the new "what to attend to" mechanism). Re-enable via config
   to recreate the v6 behaviour.

Per-token layout (defaults: d_model=640, n_cur_image_latents=64, n_prior_image_latents=32, 4 views, n_text_tokens=2):
    current (tgt):   256 image (4*64) + 2 clinical + 1 vitals            = 259
    prior  (memory): 1 sentinel + 128 image (4*32) + 2 clin + 2 report + 1 vitals + 1 label = 135
    prior latents:   K=32 (was 16) after selective pooling of the full prior memory

Not checkpoint-compatible with v6: the per-view poolers are new, the prior memory K
doubled, and ``image_pool_stride=1`` keeps the un-fused 16x16 grid (v6 used stride 2).

Risks acknowledged in ``docs/learned_query_image_pooling.md``:
- Per-view pooler fights the small-finding thesis if queries collapse; the primary
  readout is the small-finding-subset mAP, not overall mAP.
- Spatial grounding: Grad-CAM will need to attribute through the pooler's
  cross-attention weights (a follow-up; the 16x16 grid is still recoverable as
  ``cur_block`` until after pooling).
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
    - ``image_pool_stride`` default forced to 1 (no 16x16->8x8 max-pool) when not provided.
    - ``highres_skip`` default forced to ``False`` (pooler supersedes it).
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
        image_pool_stride: int = 1,
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
        # independently to each of the 4 views. K=64 was chosen so the post-pool
        # token count (4*64=256) matches v6's max-pooled 256 and the fusion
        # sequence length stays the same — the difference is content-adaptive
        # selection vs. fixed 2x2 max-pool windows.
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
        # prior image latents). v7's `image_pool_stride=1` removed the fixed 8x8
        # max-pool from the shared image path, which would otherwise leave the prior
        # image at the full 4*16*16=1024 tokens; this learned pooler restores a
        # cheap, content-adaptive reduction symmetric with the current path instead
        # of dumping 1024 keys into the prior Perceiver.
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
        # _encode_image_block runs the backbone + 2D pos encoding. With
        # image_pool_stride=1 the spatial pooler is identity, so cur_block is
        # (B, S, 640, 16, 16) and cur_slot_valid is (B, S).
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )

        # Confident-background attention penalty (opt-in; pre-norm energy on the
        # un-fused 16x16 grid, exactly as in v6).
        bg_penalty = None
        if self.background_penalty_lambda > 0.0 and "bg_mask" in data:
            bg_penalty = self.background_penalty_lambda * self._background_penalty(
                cur_block, cur_slot_valid, data["bg_mask"]
            )

        # Per-view learned-query pooler -> (B, S*K, C) image latents. Replaces
        # v6's `einops.rearrange(cur_block, "b s c h w -> b (s h w) c")` flat
        # reshape (which yielded 256 spatial tokens; now we yield 256 learned
        # latents with the same count).
        cur_img_tokens, cur_img_valid = self._pool_current_image_per_view(
            cur_block, cur_slot_valid
        )

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
        # Per-view learned-query pooler -> (B, S*K_prv, C). Mirrors the current
        # image path; without it the un-pooled prior grid would be 4*16*16=1024
        # tokens (image_pool_stride=1 disabled v6's 8x8 max-pool for both branches).
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
