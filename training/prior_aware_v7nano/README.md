# Prior-Aware v7 Nano

Successor to [`prior_aware_v6nano`](../prior_aware_v6nano/). v7 keeps the entire v6
encoder + asymmetric fusion stack **unchanged** and replaces the 16×16→8×8 max-pool
on the current image with a per-view **learned-query (Perceiver) pooler**. The
prior Perceiver pooler (already in v6) is kept and its budget doubled from
K=16 to K=32.

The motivation and the risks are in
[`docs/learned_query_image_pooling.md`](../../docs/learned_query_image_pooling.md).
The short version: the v6 2×2 max-pool keeps focal lesions but throws away
information destructively (only the strongest activation in each window
survives), and the per-view Perceiver replaces it with a learned, content-
adaptive selection that can carry **more** localized signal at the same fusion
cost. Same total token count (256 image latents / study) as v6's max-pooled 256.

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

## Knobs (all in `config.yaml` → `model.model_init_args`)

| knob | default | effect |
|---|---|---|
| `n_cur_image_latents` | `64` | K learned latents per view for the current-image Perceiver. K=64 gives 4×64=256 current image latents, matching v6's max-pooled 256. |
| `cur_pooler_nhead` | `8` | heads of the current-image pooler decoder (640/8=80, integer). |
| `cur_pooler_dropout` | `0.1` (= `dropout`) | pooler dropout. |
| `cur_pooler_ffn_dim` | `1024` (= `fusion_ffn_dim`) | pooler FFN width. |
| `n_prior_image_latents` | `32` | K learned latents per view for the **prior-image** Perceiver. 4×32=128 prior image latents. Without it the un-pooled prior would be 4×16×16=1024 tokens. |
| `prv_pooler_nhead` / `prv_pooler_dropout` / `prv_pooler_ffn_dim` | `8` / `0.1` / `1024` | prior-image pooler decoder knobs (default to `dropout` / `fusion_ffn_dim`). |
| `n_prior_latents` | `32` (was 16 in v6) | K latents the prior **memory** Perceiver pools the whole prior memory to (all modalities). |
| `image_pool_stride` | `1` (was 2 in v6) | keep the un-fused 16×16 grid; the per-view poolers consume it. |
| `highres_skip` | `false` (was `true` in v6) | v6's max-pool skip is replaced by the per-view pooler. Set `true` to recreate v6's behaviour on top of the pooler. |

## Not checkpoint-compatible with v6

The per-view pooler is new, `n_prior_latents` doubled, and the un-fused 16×16
grid is the input to the pooler (v6 used the 8×8 max-pooled grid). Train v7
fresh (or warm-start only the shared backbone / text-encoder tensors by name).

## Risks (from `docs/learned_query_image_pooling.md`)

1. **Fights the small-finding thesis** if queries collapse. The pooler is
   asked to learn where to look; if it doesn't, focal lesions are lost.
   **Primary readout: small-finding-subset mAP** (nodule, mass, pneumothorax,
   focal classes), not overall mAP.
2. **Loses clean spatial grounding for Grad-CAM.** The pooler replaces
   spatial tokens with permutation-free latents; `cur_block` is still
   available pre-pool, but the dominant current-image signal in fusion is
   now the latents. Grad-CAM needs to attribute through the pooler's
   cross-attention weights (query→grid-cell attention). **This is a follow-up;
   not addressed in this commit.**
3. **Background-attention penalty still works** because the penalty consumes
   `cur_block` directly (`PriorAwareV6NanoModel._background_penalty`), which
   v7 keeps available.

## Quick start

```bash
# train (default v7: K=64/view current pooler, K=32 prior pooler, no max-pool)
python training/prior_aware_v7nano/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# eval
python training/prior_aware_v7nano/prior_aware_eval.py \
  --checkpoint-path output/prior_aware_v7nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings
```

The same host-RAM / throughput flags apply (`--text-embeddings-gpu-resident`,
`--uint8-image-pipeline`, `gc.freeze()`); see [`FLAGS.md`](../FLAGS.md).

## Grad-CAM / Attribution

`config.yaml` sets `arch: prior_aware_v7nano`, registered in
[`src/interpret/run_prior_gradcam.py`](../../src/interpret/run_prior_gradcam.py).
The attribution machinery hooks `image_encoder`, CXR-BERT embeddings,
`delta_embedding`, and grad×value on `prior_label`/vitals — all preserved by
name from v6. **The current-image Grad-CAM panel will need a follow-up that
attributes through the per-view pooler's cross-attention weights** (the
parked design note flags this; not done here).

```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v7nano/config.yaml \
  --checkpoint-path output/prior_aware_v7nano/runs/<run>/checkpoints/best.pt \
  --split val --scan-limit 800 --device cuda
```

## Ablation grid (config-only)

| variant | isolates |
|---|---|
| `n_cur_image_latents: 16 / 32 / 96` | how much latent budget the current image needs |
| `n_prior_latents: 16` (match v6) | whether the doubled prior budget helps |
| `highres_skip: true` (re-enable v6's path) | whether the per-view pooler alone is enough |
| `image_pool_stride: 2` (re-enable v6's max-pool) | pooler *on top of* max-pool (sanity) |

## File map

- `src/model/PriorAwareV7NanoModel.py` — v7 fork of v6; adds `cur_image_pooler`
  (per-view Perceiver, K=64) and overrides `forward` to use it.
- `training/prior_aware_v7nano/{config.yaml, prior_aware_train.py, prior_aware_eval.py, README.md}`
  — mirror of v6; new knobs above.
- `test/benchmark_pipeline.py` — `prior_aware_v7nano` registered in
  `_build_prior_aware_model` and `resolve_pipeline`.
- `src/interpret/run_prior_gradcam.py` — `prior_aware_v7nano` registered in
  `_MODEL_CLASSES` and `_DEFAULT_TEXT_MODEL`.
