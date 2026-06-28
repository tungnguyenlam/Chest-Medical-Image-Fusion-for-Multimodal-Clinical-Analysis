"""Prior-Aware v8 Nano: v6 geometry + a noise-aware label-correlation graph head.

Successor to ``PriorAwareV6NanoModel``. v8 keeps the **entire v6 encoder + asymmetric
fusion stack unchanged** (native-640 bus, pooled image path, multi-token text, selective
prior pooling, high-res skip, context bottlenecks, per-modality LayerNorms, sentinel) and
adds exactly one thing at the head: the 26 classifier query vectors are no longer frozen
random ``nn.Embedding`` rows -- they are produced by a directed **label-correlation graph**
(``src/model/graph_head.py``) so correlated classes share representation. This is the
ML-GCN / CheXGCN injection slot, dimension-matched to ``MLDecoder``'s ``decoder_embedding``.

This is a *head-only* contribution (clean attribution vs the v6 baseline). It does **not**
touch the image/text/prior modality balance and does **not** add multi-prior / temporal
modeling -- those are the v9 line. The graph is **noise-aware**: built from train-split
co-occurrence with Bayesian shrinkage, BH-significance, deterministic-edge pruning and a
curated clinical hierarchy (see ``docs/prior_aware_v8_label_graph.md``); the operative
sparsifier is ``top_k`` / ``lift_threshold`` (the 2026-06-25 gate found significance alone
passes 42% of pairs).

Two optional, config-gated extras on top of the baseline (A) structural prior:
  * **graph-aware prior label** (doc secondary injection): the prior study's label vector
    enters fusion through the graph embeddings ``Z`` instead of a bare ``Linear``.
  * **co-occurrence consistency loss** (doc §4B): a small detached-coupled aux penalty
    (same plumbing as the v6 background penalty) that punishes predictions violating the
    graph. Off by default.

Knobs (see ``training/prior_aware_v8nano/config.yaml``): ``graph_path``, ``head_mode``
(graph|independent), ``gnn`` (gcn|gat), ``graph_layers``, ``graph_dir`` (directed|
symmetrized), ``lift_threshold``, ``graph_top_k``, ``use_significance``,
``use_hierarchy_edges``, ``reweight_p``, ``graph_mode`` (joint|pretrain_freeze),
``graph_pretrain_steps``, ``graph_prior_label``, ``graph_consistency_lambda``.

``head_mode: independent`` makes the class behave exactly like v6 (frozen-random queries,
graph module never built) -- the in-class baseline arm of the ablation grid.

Not checkpoint-compatible with v6 when ``head_mode: graph`` (the head gains the graph
module and the MLDecoder queries change source).
"""

from __future__ import annotations

from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

from src.dataloader.PriorAwareDataset import N_CLASSES, N_DELTA_BUCKETS, bucket_days
from src.decoder.MLDecoder import MLDecoder
from src.encoder import CaMCheXTextEncoder
from src.model.CaMCheXV2NanoVitalsModel import ConvNeXtV2NanoImageEncoder, VitalsTokenProjector
from src.model.graph_head import LabelGraphHead

ROOT = Path(__file__).resolve().parents[2]

SEG_CUR_VIEWS = (0, 1, 2, 3)
SEG_CUR_CLIN = 4
SEG_CUR_VITALS = 5
SEG_PRV_VIEWS = (6, 7, 8, 9)
SEG_PRV_CLIN = 10
SEG_PRV_VITALS = 11
SEG_PRV_LABELS = 12
SEG_PRV_REPORT = 13
N_SEGMENTS = 14

IMG_FEAT_CHANNELS = 640


def _context_bottleneck(d_model: int, bottleneck_dim: int | None) -> nn.Module:
    if not bottleneck_dim:
        return nn.Identity()
    return nn.Sequential(
        nn.Linear(d_model, bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, d_model),
    )


def _make_image_pool(pool_type: str, channels: int, stride: int) -> nn.Module:
    if stride == 1:
        return nn.Identity()
    if pool_type == "max":
        return nn.MaxPool2d(kernel_size=stride, stride=stride)
    if pool_type == "avg":
        return nn.AvgPool2d(kernel_size=stride, stride=stride)
    if pool_type == "depthwise":
        return nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=channels)
    raise ValueError(f"image_pool_type must be one of max|avg|depthwise, got {pool_type!r}")


class PriorAwareV8NanoModel(nn.Module):
    """Prior-aware v8: v6 stack + noise-aware label-graph classifier head."""

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
        # ---- v5/v6 knobs (carried over unchanged) ----
        n_prior_latents: int = 16,
        pooler_nhead: int = 8,
        prior_latent_dropout: float = 0.1,
        context_bottleneck_dim: int | None = None,
        highres_skip: bool = True,
        background_penalty_lambda: float = 0.0,
        image_pool_type: str = "max",
        image_pool_stride: int = 2,
        text_embed_dim: int = 768,
        n_text_tokens: int = 2,
        fusion_ffn_dim: int = 2048,
        # ---- v8 label-graph knobs ----
        graph_path: str = "data/data-camchex/label_graph.pt",
        head_mode: str = "graph",            # graph | independent (independent == v6 baseline)
        gnn: str = "gcn",                    # gcn | gat
        graph_layers: int = 2,
        graph_hidden_dim: int | None = None,
        graph_dropout: float = 0.1,
        lift_threshold: float = 1.5,
        graph_top_k: int = 6,
        use_significance: bool = True,
        use_hierarchy_edges: bool = True,
        graph_dir: str = "directed",         # directed | symmetrized
        reweight_p: float = 0.25,
        gat_heads: int = 4,
        graph_mode: str = "joint",           # joint | pretrain_freeze
        graph_pretrain_steps: int = 400,
        graph_prior_label: bool = False,
        graph_consistency_lambda: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.freeze_text_encoder = freeze_text_encoder
        self.use_precomputed_text_embeddings = use_precomputed_text_embeddings
        self.n_prior_latents = n_prior_latents
        self.prior_latent_dropout = prior_latent_dropout
        self.highres_skip = highres_skip
        self.n_text_tokens = n_text_tokens
        self.background_penalty_lambda = float(background_penalty_lambda)
        self.head_mode = head_mode
        self.graph_prior_label = bool(graph_prior_label)
        self.graph_consistency_lambda = float(graph_consistency_lambda)
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

        self.image_pool = _make_image_pool(image_pool_type, IMG_FEAT_CHANNELS, image_pool_stride)
        self.image_proj = (
            nn.Identity() if d_model == IMG_FEAT_CHANNELS
            else nn.Conv2d(IMG_FEAT_CHANNELS, d_model, kernel_size=1)
        )
        self.pos_encoding = Summer(PositionalEncoding2D(d_model))
        self.vitals_projector = VitalsTokenProjector(
            num_vitals=7,
            d_model=d_model,
            grid_size=1,
            hidden_dim=vitals_hidden_dim,
            dropout=vitals_dropout,
            vital_dropout_p=vital_dropout_p,
        )

        self.text_proj = nn.Linear(text_embed_dim, n_text_tokens * d_model)

        self.padding_token = nn.Parameter(torch.randn(1, d_model, 1, 1))
        self.segment_embedding = nn.Parameter(torch.randn(N_SEGMENTS, d_model, 1, 1))
        with torch.no_grad():
            self.segment_embedding.data = self.segment_embedding.data.clamp(-1.0, 1.0)

        self.delta_embedding = nn.Embedding(N_DELTA_BUCKETS, d_model)
        nn.init.normal_(self.delta_embedding.weight, std=0.02)
        self.prior_label_proj = nn.Linear(n_classes, d_model)

        self.no_prior_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.bottleneck_clin = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_vitals = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_report = _context_bottleneck(d_model, context_bottleneck_dim)
        self.bottleneck_label = _context_bottleneck(d_model, context_bottleneck_dim)

        self.norm_img = nn.LayerNorm(d_model)
        self.norm_clin = nn.LayerNorm(d_model)
        self.norm_vitals = nn.LayerNorm(d_model)
        self.norm_report = nn.LayerNorm(d_model)
        self.norm_label = nn.LayerNorm(d_model)
        self.norm_sentinel = nn.LayerNorm(d_model)

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

        # ---- the v8 addition: the label-correlation graph head -------------------
        # decoder_embedding (== query width) and group count come straight from MLDecoder.
        qw = self.head.decoder.query_embed.weight
        embed_len_decoder, decoder_embedding = qw.shape
        self.graph_head = None
        if head_mode == "graph":
            if embed_len_decoder != n_classes:
                raise ValueError(
                    f"graph head needs one query per class (MLDecoder embed_len_decoder="
                    f"{embed_len_decoder}, n_classes={n_classes}); set num_of_groups=n_classes"
                )
            artifact = self._load_graph_artifact(graph_path, n_classes)
            self.graph_head = LabelGraphHead(
                artifact,
                out_dim=decoder_embedding,
                gnn=gnn,
                layers=graph_layers,
                hidden_dim=graph_hidden_dim,
                dropout=graph_dropout,
                lift_threshold=lift_threshold,
                top_k=graph_top_k,
                use_significance=use_significance,
                use_hierarchy_edges=use_hierarchy_edges,
                symmetrize=(graph_dir == "symmetrized"),
                reweight_p=reweight_p,
                gat_heads=gat_heads,
            )
            if graph_mode == "pretrain_freeze":
                self.graph_head.pretrain_and_freeze(steps=graph_pretrain_steps)
            elif graph_mode != "joint":
                raise ValueError(f"graph_mode must be joint|pretrain_freeze, got {graph_mode!r}")
            # Secondary injection: prior-label vector -> graph embeddings -> d_model.
            if self.graph_prior_label:
                self.graph_label_proj = nn.Linear(decoder_embedding, d_model)
        elif head_mode != "independent":
            raise ValueError(f"head_mode must be graph|independent, got {head_mode!r}")

    @staticmethod
    def _load_graph_artifact(graph_path: str, n_classes: int) -> dict:
        path = Path(graph_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"label graph artifact not found: {path}. Build it with "
                f"`python src/prepare/05_build_label_graph.py` (head_mode: graph requires it)."
            )
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        nf = artifact["node_features"]
        if nf.shape[0] != n_classes:
            raise ValueError(
                f"graph artifact has {nf.shape[0]} classes, model expects {n_classes}; "
                f"rebuild label_graph.pt with the matching class list."
            )
        return artifact

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder and self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    # ---- encoders ---------------------------------------------------------
    def _encode_image_block(self, x, view_positions, view_seg_indices):
        b, s = x.shape[:2]
        feats, nonzero_mask = self.image_encoder(x, view_positions)
        # When no slot in the block holds a real view (e.g. a sample with no prior
        # study), the encoder returns a 0-row feats tensor. Feeding a zero-batch
        # tensor through pool/proj/pos_encoding is fine on CUDA but MPS mishandles
        # empty-batch conv/pool kernels and can hand back a collapsed (non-4d)
        # tensor, crashing pos_encoding. Substitute a single dummy row so the
        # shape math stays reliable; its result is discarded since the scatter
        # below selects zero slots.
        any_valid = bool(nonzero_mask.any())
        if not any_valid:
            feats = feats.new_zeros((1, *feats.shape[1:]))
        feats = self.image_pool(feats)
        feats = self.image_proj(feats)
        feats = self.pos_encoding(feats)
        h2, w2 = feats.shape[2], feats.shape[3]

        pad_tokens = einops.repeat(
            self.padding_token, "1 c 1 1 -> (b s) c h w", b=b, s=s, h=h2, w=w2
        ).type_as(feats).clone()
        seg_for_views = torch.stack([self.segment_embedding[i] for i in view_seg_indices], dim=0)
        seg = einops.repeat(seg_for_views, "s c 1 1 -> (b s) c h w", b=b, h=h2, w=w2).type_as(feats)
        if any_valid:
            pad_tokens[nonzero_mask] = feats + seg[nonzero_mask]
        block = einops.rearrange(pad_tokens, "(b s) c h w -> b s c h w", b=b, s=s)
        return block, nonzero_mask.view(b, s)

    def enable_input_normalization(self, mean, std) -> None:
        self.image_encoder.enable_input_normalization(mean, std)

    def attach_text_embedding_table(self, table: torch.Tensor) -> None:
        self.register_buffer("text_embedding_table", table.float(), persistent=False)

    def _encode_text(self, input_ids, attention_mask):
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
        proj = self.text_proj(cls)
        toks = proj.view(cls.shape[0], self.n_text_tokens, self.d_model)
        seg = self.segment_embedding[segment_index].view(self.d_model).to(cls.device)
        return toks + seg

    def _valid_image_tokens(self, slot_valid, h2, w2):
        b, s = slot_valid.shape
        return slot_valid.unsqueeze(-1).expand(b, s, h2 * w2).reshape(b, s * h2 * w2)

    def _pool_prior(self, memory, mem_pad):
        b = memory.shape[0]
        queries = self.prior_latent_queries.unsqueeze(0).expand(b, -1, -1)
        latents = self.prior_pooler(queries, memory, memory_key_padding_mask=mem_pad)
        if self.training and self.prior_latent_dropout > 0.0:
            keep = torch.rand(latents.shape[:2], device=latents.device) >= self.prior_latent_dropout
            keep[:, 0] = True
            return latents, ~keep
        return latents, None

    def _background_penalty(self, cur_block, cur_slot_valid, bg_mask):
        b, s, c, h, w = cur_block.shape
        energy = cur_block.pow(2).sum(dim=2)
        bg = bg_mask.to(cur_block.dtype).reshape(b * s, 1, bg_mask.shape[-2], bg_mask.shape[-1])
        bg = F.adaptive_avg_pool2d(bg, (h, w)).reshape(b, s, h, w)
        weight = bg * cur_slot_valid.to(cur_block.dtype).view(b, s, 1, 1)
        return (weight * energy).sum() / (weight.sum() + 1e-6)

    # ---- forward ----------------------------------------------------------
    def forward(self, data: dict):
        # Graph queries Z (recomputed each forward from fixed Z0 + fixed A + learned GNN
        # weights; a frozen buffer under graph_mode=pretrain_freeze). None == v6 head.
        z_graph = self.graph_head() if self.graph_head is not None else None

        # ---- current branch (tgt) ----------------------------------------
        cur_block, cur_slot_valid = self._encode_image_block(
            data["img"], data["view_positions"], SEG_CUR_VIEWS
        )
        b, s_img, cdim, h2, w2 = cur_block.shape
        cur_img_tokens = einops.rearrange(cur_block, "b s c h w -> b (s h w) c")

        bg_penalty = None
        if self.background_penalty_lambda > 0.0 and "bg_mask" in data:
            bg_penalty = self.background_penalty_lambda * self._background_penalty(
                cur_block, cur_slot_valid, data["bg_mask"]
            )

        cur_clin_cls = self._encode_text(data["clin_input_ids"], data["clin_attn_mask"])
        cur_clin_tok = self._text_tokens(cur_clin_cls, SEG_CUR_CLIN)
        cur_vital_tok = self.vitals_projector(
            data["vital_values"].to(cur_img_tokens.device),
            data["vital_missing_mask"].to(cur_img_tokens.device),
        )
        cur_vital_tok = cur_vital_tok + self.segment_embedding[SEG_CUR_VITALS].view(cdim).to(cur_vital_tok.device)

        cur_clin_tok = self.bottleneck_clin(cur_clin_tok)
        cur_vital_tok = self.bottleneck_vitals(cur_vital_tok)
        cur_img_tokens = self.norm_img(cur_img_tokens)
        cur_clin_tok = self.norm_clin(cur_clin_tok)
        cur_vital_tok = self.norm_vitals(cur_vital_tok)
        tgt = torch.cat([cur_img_tokens, cur_clin_tok, cur_vital_tok], dim=1)

        # ---- prior branch (memory) ---------------------------------------
        has_prior = data["has_prior"].to(cur_img_tokens.device).bool()
        days_since = data["days_since_prior"].to(cur_img_tokens.device).float()
        delta_emb = self.delta_embedding(bucket_days(days_since, has_prior))
        delta_tok = delta_emb.unsqueeze(1)

        prv_block, prv_slot_valid = self._encode_image_block(
            data["prior_img"], data["prior_view_positions"], SEG_PRV_VIEWS
        )
        prv_img_tokens = einops.rearrange(prv_block, "b s c h w -> b (s h w) c")

        prv_clin_cls = self._encode_text(data["prior_clin_input_ids"], data["prior_clin_attn_mask"])
        prv_clin_tok = self._text_tokens(prv_clin_cls, SEG_PRV_CLIN)
        prv_report_cls = self._encode_text(data["prior_report_input_ids"], data["prior_report_attn_mask"])
        prv_report_tok = self._text_tokens(prv_report_cls, SEG_PRV_REPORT)
        prv_vital_tok = self.vitals_projector(
            data["prior_vital_values"].to(cur_img_tokens.device),
            data["prior_vital_missing_mask"].to(cur_img_tokens.device),
        )
        prv_vital_tok = prv_vital_tok + self.segment_embedding[SEG_PRV_VITALS].view(cdim).to(prv_vital_tok.device)
        prior_label = data["prior_label"].to(cur_img_tokens.device).float()
        # Prior-label token: graph-aware (prior_label @ Z -> d_model) if enabled, else the
        # bare independent-class projection (v6 behaviour).
        if self.graph_prior_label and z_graph is not None:
            prv_label_vec = self.graph_label_proj(prior_label @ z_graph)
        else:
            prv_label_vec = self.prior_label_proj(prior_label)
        prv_label_tok = (prv_label_vec + self.segment_embedding[SEG_PRV_LABELS].view(cdim)).unsqueeze(1)

        prv_img_tokens = prv_img_tokens + delta_tok
        prv_clin_tok = prv_clin_tok + delta_tok
        prv_report_tok = prv_report_tok + delta_tok
        prv_vital_tok = prv_vital_tok + delta_tok
        prv_label_tok = prv_label_tok + delta_tok

        prv_clin_tok = self.bottleneck_clin(prv_clin_tok)
        prv_report_tok = self.bottleneck_report(prv_report_tok)
        prv_vital_tok = self.bottleneck_vitals(prv_vital_tok)
        prv_label_tok = self.bottleneck_label(prv_label_tok)

        prv_img_tokens = self.norm_img(prv_img_tokens)
        prv_clin_tok = self.norm_clin(prv_clin_tok)
        prv_report_tok = self.norm_report(prv_report_tok)
        prv_vital_tok = self.norm_vitals(prv_vital_tok)
        prv_label_tok = self.norm_label(prv_label_tok)

        sentinel = self.norm_sentinel(self.no_prior_token).expand(b, 1, cdim)
        memory = torch.cat(
            [sentinel, prv_img_tokens, prv_clin_tok, prv_report_tok, prv_vital_tok, prv_label_tok],
            dim=1,
        )

        # ---- padding masks ------------------------------------------------
        ones1 = torch.ones(b, 1, dtype=torch.bool, device=tgt.device)
        ones_text = torch.ones(b, self.n_text_tokens, dtype=torch.bool, device=tgt.device)
        prior1 = has_prior.unsqueeze(1)
        prior_text = prior1.expand(b, self.n_text_tokens)
        cur_img_valid = self._valid_image_tokens(cur_slot_valid, h2, w2)
        cur_valid = torch.cat([cur_img_valid, ones_text, ones1], dim=1)
        mem_valid = torch.cat(
            [
                ones1,
                self._valid_image_tokens(prv_slot_valid & has_prior.unsqueeze(-1), h2, w2),
                prior_text,
                prior_text,
                prior1,
                prior1,
            ],
            dim=1,
        )
        tgt_pad = ~cur_valid
        mem_pad = ~mem_valid

        if self.prior_pooler is not None:
            fusion_memory, fusion_mem_pad = self._pool_prior(memory, mem_pad)
        else:
            fusion_memory, fusion_mem_pad = memory, mem_pad

        x = self.fusion(
            tgt, fusion_memory, tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=fusion_mem_pad
        )

        if self.highres_skip:
            masked = cur_img_tokens.masked_fill(~cur_img_valid.unsqueeze(-1), float("-inf"))
            skip_tok = masked.max(dim=1).values.unsqueeze(1)
            x = torch.cat([x, skip_tok], dim=1)
            tgt_pad = torch.cat([tgt_pad, torch.zeros(b, 1, dtype=torch.bool, device=tgt.device)], dim=1)

        logits = self.head(x, tgt_pad, query_embed=z_graph)

        # ---- auxiliary losses (detached-coupled in train_step) ------------
        # Combine the optional background penalty and the optional graph consistency loss
        # into the single aux term train_step expects. Both are already scaled by their own
        # lambda; train_step couples the sum to the detached criterion loss.
        aux = bg_penalty
        if (
            self.graph_consistency_lambda > 0.0
            and self.graph_head is not None
            and self.training
        ):
            cons = self.graph_consistency_lambda * self.graph_head.consistency_loss(
                torch.sigmoid(logits)
            )
            aux = cons if aux is None else aux + cons
        if aux is not None:
            return logits, aux
        return logits
