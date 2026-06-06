# Prior-Aware V2 Nano

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

Grad-CAM infrastructure is now model-extensible in `training/common.py`, but
this model does not yet define a custom attribution runner. The existing v2nano
runner is still specific to the non-prior vitals model.
