# Prior-Aware v5 Nano

Prior-aware successor to [`prior_aware_v4nano`](../prior_aware_v4nano/), kept in the
`prior_aware` line. Same shared encoders and single-token-per-signal layout, same
asymmetric fusion (current = `tgt` queries, prior = read-only `memory`) — implemented in
[`src/model/PriorAwareV5NanoModel.py`](../../src/model/PriorAwareV5NanoModel.py).

## The idea: bottleneck the prior, protect the current image

v4 deliberately kept *every* prior patch at full resolution ("no lossy pooling") and let
the `Linear(26→768)` prior-label token reach fusion as a clean channel. That maximizes
information but leaves a memorization / prior-label copy shortcut, and a uniform bottleneck
would risk the small-finding classes (nodule, mass, pneumothorax line, subtle effusion)
that depend on fine current-image detail.

v5 separates the two concerns:

- **Where the small detail lives** → the *current* image, at spatial resolution. Never
  pooled; optionally given *higher* resolution.
- **Where the memorization risk lives** → the *prior* context. Compressed hard.

Five knobs implement this (all in `config.yaml` → `model.model_init_args`):

| knob | default | effect |
|---|---|---|
| `n_prior_latents` | `16` | learned **selective** pooling: `K` query tokens run one Perceiver block (`self-attn + cross-attn → 261 prior tokens + FFN`) → `K` prior latents. Selection (not averaging) keeps focal prior evidence; the prior label can no longer arrive un-mixed. `0` → v4-style full 261-token memory. |
| `current_image_stride` | `2` | `image_proj` stride. `2` → 8×8 = 64 tok/view (v4 default); `1` → 16×16 = 256 tok/view (higher current resolution). |
| `highres_skip` | `true` | append a **max**-pool over the un-fused current image tokens as one extra head token — focal evidence reaches the classifier even if fusion dilutes it. |
| `context_bottleneck_dim` | `null` | down-up projection squeezing each non-image context token (clin / report / vitals / label) through this many dims before fusion. `null` disables. |
| `prior_latent_dropout` | `0.1` | drop whole prior latents during training (≥1 always kept) so no single latent becomes "the label channel." |

Cross-attention cost in fusion drops from `258·261` to `258·K` (with the pooler adding a
one-off `K·261`), so default v5 is also cheaper than v4 despite the extra pooler.

## Ablation grid

The knobs are designed so the grid is config-only — each cell isolates one mechanism:

| variant | `n_prior_latents` | `current_image_stride` | story |
|---|---|---|---|
| baseline (≈v4) | `0` | `2` | reference inside the v5 class |
| pool only | `16` | `2` | isolates prior-memory compression |
| high-res only | `0` | `1` | isolates current-image resolution |
| v5 full | `16` | `1` | combined |

**Track a small-finding subset mAP separately** (nodule, mass, pneumothorax, focal
classes), not just mean mAP — the failure mode this design guards against shows up only
there, and confirming the per-class pattern is the strongest thesis evidence either way.

## Not checkpoint-compatible with v4

The prior pooler, context bottlenecks, and high-res skip token are new modules and the
head reads a different token count. Train v5 fresh (or warm-start only the shared
backbone / text / vitals / prior-label tensors by name).

## Quick start

```bash
# train (default v5: pooled prior, stride-2 current, high-res skip)
python training/prior_aware_v5nano/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# eval (full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_v5nano/prior_aware_eval.py \
  --checkpoint-path output/prior_aware_v5nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings
```

The same host-RAM / throughput flags as v4 apply (`--text-embeddings-gpu-resident`,
`--uint8-image-pipeline`, `gc.freeze()`); see [`FLAGS.md`](../FLAGS.md) and the
[`prior_aware_v4nano` README](../prior_aware_v4nano/README.md).

## Grad-CAM / Attribution

`config.yaml` sets `arch: prior_aware_v5nano`, registered in
[`src/interpret/run_prior_gradcam.py`](../../src/interpret/run_prior_gradcam.py), so the
attribution machinery (it hooks `image_encoder`, CXR-BERT embeddings, `delta_embedding`,
grad×value on `prior_label`/vitals — all preserved by name) works unchanged. With
`n_prior_latents > 0`, prior-branch contributions now flow through the pooler's
cross-attention, so the prior attention map doubles as a "what the model chose to remember
from the prior" figure.

```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v5nano/config.yaml \
  --checkpoint-path output/prior_aware_v5nano/runs/<run>/checkpoints/epoch_000.pt \
  --split val --scan-limit 800 --device cuda
```
