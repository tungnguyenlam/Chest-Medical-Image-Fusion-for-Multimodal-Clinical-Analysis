# Prior-Aware V2 Nano

## Quick start

```bash
# 0. build the shared prior-aware parquet once (no --tokenizer; tokenized at load)
python src/prepare/04_build_prior_aware_dataset.py

# 1. train (CXR-BERT frozen via the text-embedding cache; EMA + single-cosine)
python training/prior_aware_v2nano/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# 2. eval (two passes: full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_v2nano/prior_aware_eval.py \
  --checkpoint-path <ckpt> --use-precomputed-text-embeddings
```

Optional warm-start of the image/text/fusion stack from a trained `camchex_v2nano_vitals`
checkpoint (weights-only, fresh optimizer; mismatched/prior-only params init fresh):
`--checkpoint-path <camchex_v2nano_vitals ckpt>`. Tune `--batch-size` / `--num-workers`;
leave `val_num_workers` at 0; don't `--resume-from` an EMA checkpoint.

This variant keeps the prior-aware current/prior study structure, but pulls in
the v2nano changes from `training/camchex_v2nano_vitals`:

- `convnextv2_nano.fcmae_ft_in22k_in1k_384` frontal/lateral image routing.
- CXR-BERT clinical indication encoding.
- Optional train-time shared text embedding cache.
- Numeric ED vitals projected as token blocks for current and prior studies.
- The shared 3-channel image cache and common training loop.

Rebuild the prior-aware parquet before using this model so it contains the raw
text and numeric vital columns added by `src/prepare/04_build_prior_aware_dataset.py`:

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer microsoft/BiomedVLP-CXR-BERT-specialized
```

Train with live CXR-BERT:

```bash
python training/prior_aware_v2nano/prior_aware_train.py
```

Train with frozen cached text embeddings:

```bash
python training/prior_aware_v2nano/prior_aware_train.py \
  --use-precomputed-text-embeddings \
  --text-embedding-cache-dir data/text_embeddings
```

## Grad-CAM / Attribution

This model is a *superset* of `camchex_v2nano_vitals`, so its attribution reuses the
same machinery (`src/interpret/{attribution,visualize}.py`) through a prior-aware wrapper
(`src/interpret/{prior_attribution,run_prior_gradcam}.py`) and adds the prior branch. One
`logit.backward()` per class yields:

- **current**: image Grad-CAM per CXR view, clinical-indication token grad×embedding,
  numeric vitals grad×value,
- **prior**: prior-image Grad-CAM, prior clinical text, and the **prior radiology report**
  (findings + impression — legitimate prior context, fed only for the prior study),
- **prior label**: per-class grad×value over the projected 26-dim prior CheXpert vector
  (`Linear(26→768)` token) — *which prior findings drove the current prediction*,
- **time delta**: grad×embedding on the time-gap bucket token (signed contribution + bucket),
- **modality**: current-vs-prior contribution breakdown across every token group.

Each class writes one folder of inspect-by-hand PNGs:
`<Class>/{image,prior_image,text,prv_clin,prv_report,vitals,prior_vitals,prior_label,time_delta,modality}.png`.

The model declares `gradcam_runner_module = "src.interpret.run_prior_gradcam"`, so panels
are dumped automatically after each epoch's validation (controlled by `--gradcam-epochs`
/ `--gradcam-device`, like the other models), reusing the validation logits to pick two
representative studies per class (`best/` and `first/`). Studies without a prior render
placeholder prior panels and a masked time-delta. The dump forces the live CXR-BERT path
(cache off, grads on) so per-token attribution works even with cached-embedding training;
predictions are identical.

Standalone:

```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v2nano/config.yaml \
  --checkpoint-path output/prior_aware_v2nano/runs/<run>/checkpoints/best.pt \
  --split val --scan-limit 800
```
