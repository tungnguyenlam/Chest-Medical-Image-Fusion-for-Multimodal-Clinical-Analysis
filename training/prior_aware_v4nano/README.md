# Prior-Aware v4 Nano

Prior-aware successor to [`prior_aware_v3nano`](../prior_aware_v3nano/), kept in the
`prior_aware` line. Same prior-aware design and same single-token-per-signal layout
as v3nano (current + prior branches sharing the ConvNeXtV2-Nano image router, CXR-BERT
text encoder, numeric vitals projector, time-delta embedding, and `Linear(26→768)`
prior-label token) — implemented in
[`src/model/PriorAwareV4NanoModel.py`](../../src/model/PriorAwareV4NanoModel.py).

## The one change: asymmetric prior cross-attention

v3nano runs **full self-attention** over the concatenated current + prior sequence
(258 + 260 = 518 tokens), so attention costs `518²`. v4 makes the fusion **asymmetric**:

| role | tokens | in attention |
|---|---|---|
| current (`tgt`) | 256 image + 1 clin + 1 vitals = **258** | queries — the residual stream |
| prior (`memory`) | 1 sentinel + 256 image + 1 clin + 1 report + 1 vitals + 1 label = **261** | keys/values only — never updated |

Each fusion layer does `self-attn(current) + cross-attn(current → prior) + FFN`, i.e. an
`nn.TransformerDecoder` with `tgt=current`, `memory=prior`. Attention cost drops from
`518²` (~268k) to `258² + 258·261` (~134k) — **~2× cheaper** — while **every prior patch
is kept at full 8×8 resolution as a key/value (no lossy pooling)**.

The head then reads only the 258 prior-informed current tokens, so the prior label/report
can no longer be a *direct* classifier input — it has to be integrated with current image
evidence first. That structurally weakens the prior-label copy shortcut (the main
overfitting risk in a prior-aware model).

## Three smaller changes that the layout requires

- **A learned `no_prior_token` is prepended to the memory and is always valid.** Without
  it, a no-prior sample (or a `label_dropout` drop) masks *every* memory token → an
  all-`-inf` cross-attention row → NaN. The sentinel guarantees one valid key and doubles
  as a learned "I have no prior" representation.
- **Per-modality LayerNorm before fusion.** A pre-LN decoder normalizes its *queries* but
  leaves the cross-attention **memory (K/V) un-normalized**, so this is the *only*
  normalization the prior tokens ever get. Every token group (image / clin / vitals /
  report / label / sentinel) is LayerNorm'd to unit scale before fusion, harmonizing the
  very different scales of ConvNeXt patches, BERT CLS vectors, the `Linear(26→768)` label
  projection, and the raw vitals-projector output.
- **`norm_first=True` + explicit final LayerNorm.** Pre-LN leaves the residual stream
  un-normalized at the output, so the `nn.TransformerDecoder` is given a final `norm`.

## Not checkpoint-compatible with v3nano

The fusion stack is a decoder (self + cross + FFN) instead of an encoder, the per-modality
norms are new, and the head reads 258 tokens instead of 518. Train v4 fresh (or warm-start
only the shared backbone/text/vitals/prior-label tensors by name).

## Quick start

```bash
# 0. build the shared prior-aware parquet once (shared with every prior-aware variant)
python src/prepare/04_build_prior_aware_dataset.py

# 1. train
python training/prior_aware_v4nano/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# 2. eval (two passes: full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_v4nano/prior_aware_eval.py \
  --checkpoint-path output/prior_aware_v4nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings
```

`config.yaml` mirrors `prior_aware_v3nano` exactly except `model.arch` — so a v4-vs-v3nano
run isolates the fusion change. `model.timm_init_args.drop_path_rate` is left at `0.0` for
that reason; bump it to `0.1–0.15` once the cross-attention effect is measured (it's a
cheap additional regularizer for the shared backbone). The same host-RAM flags as v3nano
apply (`--text-embeddings-gpu-resident`, `--uint8-image-pipeline`, the pyarrow backend,
`gc.freeze()`, arena capping); see [`TRAINING_FLAGS.md`](../TRAINING_FLAGS.md) and the
[`prior_aware_v3nano` README](../prior_aware_v3nano/README.md) for details.

## Grad-CAM / Attribution

`config.yaml` sets `arch: prior_aware_v4nano`, registered in
[`src/interpret/run_prior_gradcam.py`](../../src/interpret/run_prior_gradcam.py), so the
model reuses the prior-aware attribution machinery unchanged (it hooks `image_encoder`,
the CXR-BERT embeddings, `delta_embedding`, and grad×value on `prior_label`/vitals — all
preserved by name in v4). Panels dump automatically after each epoch's validation. Note
that prior-branch attribution now flows through the cross-attention rather than direct head
access, so the prior modality contributions reflect the v4 "prior as evidence" design.
```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v4nano/config.yaml \
  --checkpoint-path output/prior_aware_v4nano/runs/<run>/checkpoints/epoch_000.pt \
  --split val --scan-limit 800 --device cuda
```
