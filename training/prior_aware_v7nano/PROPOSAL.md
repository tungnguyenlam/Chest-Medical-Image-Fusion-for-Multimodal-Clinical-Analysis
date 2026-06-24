# Prior-Aware v7 Nano — proposal

Successor to [`prior_aware_v6nano`](../prior_aware_v6nano/). v7 keeps the entire v6
encoder + asymmetric fusion stack **and v6's 2×2 max-pool** unchanged, and adds a
per-view **learned-query (Perceiver) pooler** that runs *on top of* the max-pool —
but only above 512px input. At 512px the current pooler is skipped (the post-max-pool
grid is already 8×8 and the current path equals v6). The prior Perceiver pooler
(already in v6) is kept and its budget doubled from K=16 to K=32.

This is a **modality-path / image-resolution contribution**. It does **not** touch
the classifier head and does **not** add a label graph — those are deliberately
deferred. The label-correlation graph is the **v8 line** (see its
[PROPOSAL.md](../prior_aware_v8nano/PROPOSAL.md) and
[`docs/prior_aware_v8_label_graph.md`](../../docs/prior_aware_v8_label_graph.md)).
v7 and v8 are independent contributions kept separate so each thesis claim has
clean attribution; they can later compose into one model.

## One-line thesis

> v6's 2×2 max-pool is fine at 512px (the grid is only 8×8). The win from a
> learned-query pooler only appears when you raise input resolution: a per-view
> Perceiver over the larger post-max-pool grid carries **more** localized signal
> than a fixed pool, *and* it holds the fusion token count constant (256 current
> image tokens) regardless of resolution — the only way to afford 768/1024 input
> on the current GPU. So v7 keeps v6's max-pool, skips the pooler at 512 (≡ v6),
> and turns it on above 512.

## Why now

The motivation and the risks are in
[`docs/learned_query_image_pooling.md`](../../docs/learned_query_image_pooling.md).
The v5/v6 line overfit; the v6 response was a geometry-level capacity cut
(native-640 bus, no image channel projection, FFN 1024 instead of 2048, max-pool
instead of strided conv). v7 is the *next* geometry move, but a resolution-aware
one: the design note is explicit that compressing the current image with learned
queries only helps when paired with a resolution bump — *"Pooling current at 512px
would just lose information for no gain."* So v7 keeps v6's 2×2 max-pool, skips the
learned query at 512px (≡ v6), and turns the query on above 512, where the larger
post-max-pool grid (16×16 at 1024px) can be selected into K=64 latents/view that
carry strictly more signal than a fixed pool — at the same 256-token fusion cost.

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
| current image: 2×2 max-pool | `image_pool_stride=2` (`max`) | **kept** (`image_pool_stride=2`) |
| current image: per-view Perceiver pooler | — | **added** — K=64/view, **skipped at 512px**, active above |
| prior image: per-view Perceiver pooler | — (8×8 max-pool → 4×64=256 tokens) | **added** — K=32/view (4×32=128 latents), always on |
| prior memory Perceiver pooler | K=16 latents | **K=32 latents** (same primitive) |
| `highres_skip` (max-pool un-fused current) | ON by default | **OFF by default** (pooler supersedes) |
| `n_cur_image_latents`, `cur_pooler_*`, `n_prior_image_latents`, `prv_pooler_*` | n/a | **new** |

Per-token layout (defaults: d_model=640, n_cur_image_latents=64, n_prior_image_latents=32, 4 views, n_text_tokens=2):

```
current (tgt) @512:  256 image (4*8*8, pooler skipped) + 2 clinical + 1 vitals            = 259
current (tgt) @1024: 256 image (4*64 latents)          + 2 clinical + 1 vitals            = 259
prior  (memory): 1 sentinel + 128 image (4*32) + 2 clin + 2 report + 1 vitals + 1 label   = 135
prior latents:   K=32 (was 16) after selective pooling of the full prior memory
```

Both image branches keep `image_pool_stride=2` (v6's 2×2 max-pool). The current
pooler runs on the post-max-pool grid and only above 512px; the prior pooler runs
always (the user's "current-only" gate) and gets a smaller per-view budget (K=32 vs
the current's K=64) because the prior image is context, not the prediction target —
and a 1024-token un-pooled prior (at 1024px) is the expensive case it avoids.

## The per-view Perceiver pooler

- **K=64** learned query tokens per view, total **4×64=256** current image
  latents (same count as v6's max-pooled 256), so fusion cost is constant across
  resolutions.
- One `TransformerDecoder` block, `d_model=640`, 8 heads, FFN 1024, GELU,
  pre-LN, dropout 0.1 — same recipe v6's prior Perceiver uses.
- Runs **per view** (independent Perceiver on each view's post-max-pool grid)
  rather than over the concatenated multi-view grid, to preserve view identity
  and keep the query budget explicitly per view. Symmetric with v6's per-view
  max-pool.
- The pooler consumes the **post-max-pool grid** (post-pos-encoding):
  `image_pool_stride=2` is kept, so the pooler sees the 2×2-max-pooled tokens.
  **Resolution gate:** at 512px the grid is 8×8 (64/view) and the pooler is
  skipped (current path ≡ v6, hardcoded `SKIP_CUR_POOLER_INPUT_SIZE=512`); at
  1024px it is 16×16 (256/view) and the pooler reduces it to K=64 latents/view.

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
| Higher resolution needed to win (v6 already overfits at 512) | pooler is **skipped at 512** (≡ v6); it only activates at >512, so its cost is only paid where it can help |
| Grad-CAM loses spatial-token → pixel mapping | only above 512 (at 512 the v6 spatial mapping is intact); attribute through pooler cross-attention weights (follow-up) |
| Background-attention penalty breaks on permutation-free latents | penalty consumes the post-max-pool `cur_block`, which v7 keeps available |
| Doubled prior K adds parameters | K=32 prior latents is ~3.4M extra params (negligible vs ~50M backbone) |
| Loses checkpoint compatibility with v6 | train fresh; warm-start only the shared backbone / text encoder by name |

## Ablation grid (config-only where possible)

| variant | isolates |
|---|---|
| `size: 1024` (vs `512`) | **the point of v7** — the current pooler is only active above 512 |
| `n_cur_image_latents: 16 / 32 / 96` (at `size: 1024`) | how much latent budget the current image needs |
| `n_prior_latents: 16` (match v6) | whether the doubled prior budget helps |
| `highres_skip: true` (re-enable v6's path) | whether the per-view pooler alone is enough |
| `image_pool_stride: 1` (drop max-pool) | pooler on the un-pooled grid vs on top of max-pool |
| `cur_pooler_ffn_dim: 2048` (match v6) | whether the pooler FFN shrink hurts |

The headline comparison for the thesis is **`v6` @1024 (max-pool flat, K_prior=16)
vs `v7` @1024 (per-view pooler K=64, K_prior=32)** on the small-finding-subset mAP.
At 512 v7's current path is identical to v6, so the comparison only means something
above 512. If v7 doesn't move the small-finding needle at 1024, the per-view pooler
isn't learning to look — fall back to a smaller K.

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

The current pooler is bypassed at the default `size: 512`, so a 512px run is a
v6 sanity check (same current path) that only exercises the new prior pooler and
the doubled prior-memory K. The pooler's actual test is at **`size: 1024`**: the
first real experiment should be a **K sweep at 1024px** (16 / 32 / 64 / 96) on the
small-finding-subset mAP. K=64 is the default because 4×64=256 matches the fusion
token count v6/512 uses; if K=32 wins, v7 is cheaper, if K=96 wins, the extra
fusion cost needs to be justified by the small-finding gain.

(The 512-vs-1024 cutoff is hardcoded as `SKIP_CUR_POOLER_INPUT_SIZE=512` in
`src/model/PriorAwareV7NanoModel.py`. If you train at an intermediate size like
768, the pooler is active there too — only an exactly-512×512 input skips it.)
