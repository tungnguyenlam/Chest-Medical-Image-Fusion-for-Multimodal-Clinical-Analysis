# Prior-Aware v6 Nano

Prior-aware successor to [`prior_aware_v5nano`](../prior_aware_v5nano/), kept in the
`prior_aware` line. Same shared encoders and asymmetric fusion (current = `tgt` queries,
prior = read-only `memory`), same selective prior pooling — implemented in
[`src/model/PriorAwareV6NanoModel.py`](../../src/model/PriorAwareV6NanoModel.py).

## The idea: same fusion, lower-capacity geometry

The v5 family overfit. v6 keeps every v5 mechanism (prior latents, high-res skip,
context bottleneck, per-modality LayerNorms, the memory sentinel) but changes the
**geometry** to cut capacity, and turns on the regularizers v5 left off.

Three geometry changes:

| change | v5 | v6 | why |
|---|---|---|---|
| fusion bus width | `d_model=768` | `d_model=640` | 640 is ConvNeXtV2-Nano's native width. Every fusion/pooler tensor shrinks by `640/768` (attention ≈0.69×, FFN linearly). `640/8 heads = 80`. |
| image path | `Conv2d(640→768, stride 2)` (~4.4M params) | spatial **pool** 640×16×16 → 640×8×8, no channel projection | deletes the projection conv; pooling keeps the native features. **Max** pool by default (keeps focal lesions). |
| text path | 1 CLS → 1 token | 1 frozen CLS → `n_text_tokens` (=2) tokens via a learned `Linear(768→2·640)` | gives cross-attention >1 slot per text signal. The frozen CLS cache is reused unchanged (it stores 768-d CLS; the projection lives in the model). |

Plus the regularizers v5 had switched off, now on in `config.yaml`:

- `drop_path_rate: 0.15` (backbone stochastic depth — with text frozen the backbone is the largest trainable block),
- `context_bottleneck_dim: 64` (squeeze the non-image context tokens — the copy-shortcut path),
- `fusion_ffn_dim: 1024` (v5 silently used the `nn` default 2048).

See the earlier-discussion rationale in the v5 overfitting analysis; v6 is the
geometry-level answer to "the model still has too much capacity."

## Knobs (all in `config.yaml` → `model.model_init_args`)

Carried over from v5: `n_prior_latents` (16), `prior_latent_dropout` (0.1),
`context_bottleneck_dim` (now `64`), `highres_skip` (true). See
[`docs/prior_latents.md`](../../docs/prior_latents.md) for the prior-latent design.

New in v6:

| knob | default | effect |
|---|---|---|
| `d_model` | `640` | fusion bus width = backbone native width. At 640 the image needs no channel projection. |
| `image_pool_type` | `max` | `max` keeps focal bright/dark lesions; `avg` blurs (baseline-equivalent); `depthwise` is a learned per-channel 3×3 strided conv (~5.8K params). |
| `image_pool_stride` | `2` | `2` → 8×8 = 64 tok/view; `1` → keep 16×16 = 256 tok/view (no pooling, the high-resolution cell). |
| `n_text_tokens` | `2` | fusion tokens per text signal, expanded from the single frozen CLS. `1` = v5 behaviour. |
| `text_embed_dim` | `768` | width of the frozen CXR-BERT CLS the text projection consumes. |
| `fusion_ffn_dim` | `1024` | feed-forward width inside each fusion + pooler decoder layer. |

Per-token layout (defaults: d_model=640, pool stride 2 → 8×8, 4 views, n_text_tokens=2):

```
current (tgt):   256 image + 2 clinical + 1 vitals                         = 259  (+1 skip token)
prior  (memory): 1 sentinel + 256 image + 2 clin + 2 report + 1 vitals + 1 label = 263
prior latents:   K (=16) after selective pooling
```

## Ablation grid (config-only)

| variant | change from default | isolates |
|---|---|---|
| pool: avg | `image_pool_type: avg` | whether max-pool's focal preservation matters |
| pool: depthwise | `image_pool_type: depthwise` | learned vs. parameter-free downsample |
| high-res | `image_pool_stride: 1` | current-image resolution (256 tok/view) |
| single text token | `n_text_tokens: 1` | whether the 2-token text expansion helps |
| no FFN cut | `fusion_ffn_dim: 2048` | whether the FFN shrink hurts |
| no context bottleneck | `context_bottleneck_dim: null` | the context-token squeeze |

As with v5, **track a small-finding subset mAP separately** (nodule, mass, pneumothorax,
focal classes) — the geometry changes (pooling, smaller bus) most plausibly cost there.

## Not checkpoint-compatible with v5

The bus width changed (640 vs 768), the `image_proj` conv is gone, and the text path
emits a different token count. Train v6 fresh (or warm-start only the shared backbone /
text-encoder tensors by name).

## Quick start

```bash
# train (default v6: 640-wide bus, max-pooled image, 2 text tokens, regularizers on)
python training/prior_aware_v6nano/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# eval (full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_v6nano/prior_aware_eval.py \
  --checkpoint-path output/prior_aware_v6nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings
```

The same host-RAM / throughput flags apply (`--text-embeddings-gpu-resident`,
`--uint8-image-pipeline`, `gc.freeze()`); see [`FLAGS.md`](../FLAGS.md).

## Grad-CAM / Attribution

`config.yaml` sets `arch: prior_aware_v6nano`, registered in
[`src/interpret/run_prior_gradcam.py`](../../src/interpret/run_prior_gradcam.py). The
attribution machinery hooks `image_encoder`, CXR-BERT embeddings, `delta_embedding`, and
grad×value on `prior_label`/vitals — all preserved by name — so it works unchanged.

```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v6nano/config.yaml \
  --checkpoint-path output/prior_aware_v6nano/runs/<run>/checkpoints/best.pt \
  --split val --scan-limit 800 --device cuda
```
