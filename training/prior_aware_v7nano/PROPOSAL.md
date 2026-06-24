# Prior-Aware v7 Nano — proposal

Successor to [`prior_aware_v6nano`](../prior_aware_v6nano/). v7 keeps the entire v6
encoder + asymmetric fusion stack **unchanged** and replaces the 16×16→8×8 max-pool
on the current image with a per-view **learned-query (Perceiver) pooler**. The
prior Perceiver pooler (already in v6) is kept and its budget doubled from
K=16 to K=32.

This is a **modality-path / image-resolution contribution**. It does **not** touch
the classifier head and does **not** add a label graph — those are deliberately
deferred. The label-correlation graph is the **v8 line** (see its
[PROPOSAL.md](../prior_aware_v8nano/PROPOSAL.md) and
[`docs/prior_aware_v8_label_graph.md`](../../docs/prior_aware_v8_label_graph.md)).
v7 and v8 are independent contributions kept separate so each thesis claim has
clean attribution; they can later compose into one model.

## One-line thesis

> v6's 2×2 max-pool on the current image keeps focal lesions but throws away
> information destructively (only the strongest activation in each window
> survives). A per-view Perceiver pooler replaces it with a learned,
> content-adaptive selection that can carry **more** localized signal at the
> same fusion cost — and as a bonus, it decouples fusion token count from
> input resolution, which is the only way to afford 768/1024 input on the
> current GPU.

## Why now

The motivation and the risks are in
[`docs/learned_query_image_pooling.md`](../../docs/learned_query_image_pooling.md).
The v5/v6 line overfit; the v6 response was a geometry-level capacity cut
(native-640 bus, no image channel projection, FFN 1024 instead of 2048, max-pool
instead of strided conv). v7 is the *next* geometry move: the 16×16 backbone
grid has 256 spatial tokens/view, but v6 collapses them to 64 with a 2×2 max
before they reach the fusion cross-attention. A learned-query pooler with K=64
latents can carry strictly more signal at the same count, because each query
learns where to attend.

Two honest caveats this proposal has to respect:

1. **Fights the small-finding thesis** if queries collapse. The v-series exists
   to catch focal lesions, and the current image's spatial detail was
   protected precisely for that reason. The pooler replaces fixed windows
   with learned selection; if it doesn't learn to focus, focal evidence is
   lost. The primary readout is the small-finding-subset mAP, not overall
   mAP.
2. **Loses clean spatial grounding for Grad-CAM** unless we attribute through
   the pooler's cross-attention weights. The pre-pool `cur_block` is still
   available, but the dominant current signal in fusion is now the latents.
   Grad-CAM support is a follow-up; this commit makes the model trainable
   first.

## What changes vs v6 (and what does not)

| component | v6 | v7 |
|---|---|---|
| image / text / prior encoders | — | **unchanged** |
| asymmetric cross-attention fusion (current=tgt, prior=memory) | — | **unchanged** |
| `context_bottleneck_dim`, `n_text_tokens`, `fusion_ffn_dim`, `drop_path_rate` | — | **unchanged** |
| current image: 16×16→8×8 max-pool | `image_pool_stride=2` (`max`) | **removed** (`image_pool_stride=1`) |
| current image: per-view Perceiver pooler | — | **added** — K=64/view (4×64=256 latents) |
| prior image: per-view Perceiver pooler | — (8×8 max-pool → 4×64=256 tokens) | **added** — K=32/view (4×32=128 latents) |
| prior memory Perceiver pooler | K=16 latents | **K=32 latents** (same primitive) |
| `highres_skip` (max-pool un-fused current) | ON by default | **OFF by default** (pooler supersedes) |
| `n_cur_image_latents`, `cur_pooler_*`, `n_prior_image_latents`, `prv_pooler_*` | n/a | **new** |

Per-token layout (defaults: d_model=640, n_cur_image_latents=64, n_prior_image_latents=32, 4 views, n_text_tokens=2):

```
current (tgt):   256 image (4*64) + 2 clinical + 1 vitals            = 259
prior  (memory): 1 sentinel + 128 image (4*32) + 2 clin + 2 report + 1 vitals + 1 label = 135
prior latents:   K=32 (was 16) after selective pooling of the full prior memory
```

Both image branches keep `image_pool_stride=1` (v6's fixed 8×8 max-pool is off);
the learned per-view poolers do the reduction instead. The prior gets a smaller
per-view budget (K=32 vs the current's K=64) because the prior image is context,
not the prediction target — and a 1024-token un-pooled prior is the expensive case
this avoids.

## The per-view Perceiver pooler

- **K=64** learned query tokens per view, total **4×64=256** current image
  latents (same count as v6's max-pooled 256).
- One `TransformerDecoder` block, `d_model=640`, 8 heads, FFN 1024, GELU,
  pre-LN, dropout 0.1 — same recipe v6's prior Perceiver uses.
- Runs **per view** (independent Perceiver on each view's 16×16=256 spatial
  tokens) rather than over the concatenated 4×16×16=1024 grid, to preserve
  view identity and keep the query budget explicitly per view. Symmetric with
  v6's per-view 8×8 max-pool.
- The pooler consumes the **un-fused 16×16 grid** (post-pos-encoding). The
  v6 8×8 max-pool is replaced by `image_pool_stride=1` (an identity), so the
  pooler sees the full spatial tokens the backbone produced.

## What it should help (and what it won't)

- **Should help (small findings):** nodule, mass, pneumothorax, focal
  classes. The pooler is asked to learn where to look; if it succeeds, focal
  evidence is preserved *and* the lesion-vs-background discrimination is
  better than a fixed max-pool window.
- **Shouldn't hurt much (common classes):** Support Devices, Lung Opacity,
  Cardiomegaly — large diffuse findings that a 2×2 max already captures.
- **Won't move much (already-pooled text):** clinical and report text are
  unchanged; the pooler doesn't touch them.
- **Not addressed by v7 at all:** label-graph head, multi-prior / temporal-
  decay. Those are v8/v9. Saying so up front keeps the thesis claims honest.

## Risks (with mitigations)

| risk | mitigation |
|---|---|
| Queries collapse / pooler loses focal detail | primary readout is small-finding-subset mAP, not overall mAP |
| Higher resolution needed to win (v6 already overfits at 512) | track 512 vs 768 (requires memory headroom) |
| Grad-CAM loses spatial-token → pixel mapping | attribute through pooler cross-attention weights (follow-up) |
| Background-attention penalty breaks on permutation-free latents | penalty consumes the un-fused `cur_block`, which v7 keeps available |
| Doubled prior K adds parameters | K=32 prior latents is ~3.4M extra params (negligible vs ~50M backbone) |
| Loses checkpoint compatibility with v6 | train fresh; warm-start only the shared backbone / text encoder by name |

## Ablation grid (config-only where possible)

| variant | isolates |
|---|---|
| `n_cur_image_latents: 16 / 32 / 96` | how much latent budget the current image needs |
| `n_prior_latents: 16` (match v6) | whether the doubled prior budget helps |
| `highres_skip: true` (re-enable v6's path) | whether the per-view pooler alone is enough |
| `image_pool_stride: 2` (re-enable v6's max-pool) | pooler *on top of* max-pool (sanity) |
| `cur_pooler_ffn_dim: 2048` (match v6) | whether the pooler FFN shrink hurts |

The headline comparison for the thesis is **`v6` (max-pool, K_prior=16) vs
`v7` (per-view pooler K=64, K_prior=32)** on the small-finding-subset mAP.
If v7 doesn't move the small-finding needle, the per-view pooler isn't
learning to look — fall back to a smaller K or pair with a resolution bump.

## Files added / edited

- `src/model/PriorAwareV7NanoModel.py` — fork of v6; adds `cur_image_pooler`
  (per-view Perceiver, K=64) and overrides `forward` to use it. Background
  penalty and `highres_skip` paths preserved.
- `training/prior_aware_v7nano/{config.yaml, prior_aware_train.py, prior_aware_eval.py, README.md}`
  — mirror of v6; new knobs above.
- `test/benchmark_pipeline.py` — `prior_aware_v7nano` registered in
  `_build_prior_aware_model` and `resolve_pipeline`.
- `src/interpret/run_prior_gradcam.py` — `prior_aware_v7nano` registered in
  `_MODEL_CLASSES` and `_DEFAULT_TEXT_MODEL`.

## Open question to settle before training

The K=64 default is a hypothesis (matches v6's max-pooled count of 256 across
4 views, so fusion cost is unchanged). The first experiment should be a **K
sweep at 512px** (16 / 32 / 64 / 96) on the small-finding-subset mAP to
confirm the default. If K=32 is best, the v7 model is *cheaper* than v6 in
fusion cost; if K=96 is best, the extra fusion cost needs to be justified
by the small-finding gain.
