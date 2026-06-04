# CaMCheX ConvNeXtV2 Nano + Numeric Vitals

This is an additive CaMCheX variant. It does not change the legacy
`CaMCheXModel`, `CaMCheXDataset`, or existing `training/camchex*` entrypoints.

## What changed

- Image backbone: `convnextv2_nano.fcmae_ft_in22k_in1k_384`.
- Text backbone: `microsoft/BiomedVLP-CXR-BERT-specialized`.
- Text encoder freezing is opt-in by config or `--freeze-text-encoder`.
- Structured ED vitals are numeric features, not observation text.
- Numeric vitals are projected to one 8x8 token block.
- Optional vital dropout masks individual vital fields during training.
- Optional clinical embedding cache can skip repeated frozen CXR-BERT calls.
- Optional decoded image cache can skip repeated JPEG decode while preserving
  train-time augmentations.

## Main files

| Concern | File |
|---|---|
| Model assembly | `src/model/CaMCheXV2NanoVitalsModel.py` |
| Numeric vitals dataset | `src/dataloader/CaMCheXVitalsDataset.py` |
| Train script | `training/camchex_v2nano_vitals/camchex_v2nano_vitals_train.py` |
| Eval script | `training/camchex_v2nano_vitals/camchex_v2nano_vitals_eval.py` |
| Config | `training/camchex_v2nano_vitals/config.yaml` |
| Decoded image cache script | `scripts/precompute_image_cache.py` |
| Grad-CAM / attribution | `src/interpret/{attribution,visualize,run_gradcam}.py` |

## Token layout

For a study with up to 4 image views:

```text
image views:       4 * 8 * 8 = 256 tokens
clinical text:     1 * 8 * 8 =  64 tokens
numeric vitals:    1 * 8 * 8 =  64 tokens
total max:                     384 tokens
```

ConvNeXtV2 Nano returns `(B, 640, 16, 16)` features for 512px inputs. The model
uses a `640 -> 768` stride-2 convolution to produce `(B, 768, 8, 8)` before
fusion. The model has a local Nano image router because the older shared
`CaMCheXImageEncoder` preallocates 768-channel features for ConvNeXt Tiny.

## Vitals

The dataset emits seven structured fields:

```text
temperature, heartrate, resprate, o2sat, sbp, dbp, gender
```

Each field is normalized and paired with a missing-mask bit. Missing values are
encoded as value `0.0` plus `missing=True`. During training,
`vital_dropout_p` can randomly mask individual fields using the same mechanism.

Config knobs:

```yaml
model:
  model_init_args:
    vital_dropout_p: 0.1
    vitals_dropout: 0.1
    vitals_hidden_dim: 256
data:
  datamodule_cfg:
    vital_stats:
      temperature: {mean: 98.6, std: 3.0}
```

`vital_stats` is optional. If omitted, the dataset uses conservative defaults
from `CaMCheXVitalsDataset.py`.

## Train

```bash
python training/camchex_v2nano_vitals/camchex_v2nano_vitals_train.py \
  --train-df-path data/subset/labels/train.csv \
  --val-df-path data/subset/labels/val.csv \
  --test-df-path data/subset/labels/test.csv \
  --batch-size 4 \
  --max-epochs 30
```

For an offline or shape-only smoke run, disable timm pretrained image weights:

```bash
python training/camchex_v2nano_vitals/camchex_v2nano_vitals_train.py \
  --train-df-path data/subset/labels/train.csv \
  --val-df-path data/subset/labels/val.csv \
  --test-df-path data/subset/labels/test.csv \
  --no-pretrained \
  --fast-dev-run
```

CXR-BERT still needs to be available from Hugging Face or local cache on the
first run that misses the shared text embedding cache.

## Eval

```bash
python training/camchex_v2nano_vitals/camchex_v2nano_vitals_eval.py \
  --checkpoint-path output/camchex_v2nano_vitals/runs/<run>/checkpoints/best.pt \
  --test-df-path data/subset/labels/test.csv
```

Predictions default to:

```text
output/camchex_v2nano_vitals/predictions.csv
output/camchex_v2nano_vitals/metrics.json
```

## Grad-CAM / Attribution

Per-class attribution panels show *where* a prediction came from across all three
modalities from a single `logit.backward()`:

- image: Grad-CAM heatmap on each CXR view's ConvNeXtV2 feature map,
- text: grad x input-embedding per CXR-BERT token (per-word highlighting),
- vitals: signed grad x value per vital, plus a rough modality-share bar.

Each class writes three PNGs you can inspect by hand:
`<Class>/{image,text,vitals}.png`.

### During training (default: on, every epoch)

After each epoch's validation, the trainer reuses the validation logits (no extra
scan) to pick, per class, two representative studies and dumps panels to:

```text
<run_dir>/gradcam/epoch_<N>/
  best/<Class>/{image,text,vitals}.png    # highest-confidence true positive (varies per epoch)
  first/<Class>/{image,text,vitals}.png   # first true positive in val order (FIXED across epochs)
```

Flip through `epoch_*/first/<Class>/image.png` to watch one fixed study's heatmap
evolve as training progresses. Control with:

```bash
# default is every epoch; restrict, disable, or move off CPU as needed
python training/camchex_v2nano_vitals/camchex_v2nano_vitals_train.py \
  --gradcam-epochs 0,4,9   # or 'all' (default) / 'none'
  --gradcam-device cuda    # default cpu (the dump runs in a subprocess to protect GPU memory)
```

The dump forces the live CXR-BERT path (cache off, grads on) so per-word text
attribution works even if you train with cached embeddings; predictions are identical.

### Standalone (any checkpoint)

```bash
python -m src.interpret.run_gradcam \
  --config training/camchex_v2nano_vitals/config.yaml \
  --checkpoint-path output/camchex_v2nano_vitals/runs/<run>/checkpoints/best.pt \
  --split val --scan-limit 800
```

Here it scans the split for the highest-confidence true positive per class. A study
with multiple findings gets a class-conditional panel under each class it is positive
for; the panel header lists the co-occurring labels.

## Optional Clinical Embedding Cache

When the text encoder is frozen, clinical indication CLS embeddings can be
cached automatically by the train/eval data path. There is no separate
precompute command for this model. Enable it with
`--use-precomputed-text-embeddings` or the config flags below. The loader then
gathers the needed `study_id -> clinical_indication` texts, computes only cache
misses, and feeds float CLS embeddings to the model so CXR-BERT does not need to
stay loaded during training.

```bash
python training/camchex_v2nano_vitals/camchex_v2nano_vitals_train.py \
  --use-precomputed-text-embeddings \
  --text-embedding-cache-dir data/text_embeddings
```

```yaml
model:
  model_init_args:
    freeze_text_encoder: true
    use_precomputed_text_embeddings: true
data:
  datamodule_cfg:
    use_text_embedding_cache: true
    text_embedding_cache_dir: data/text_embeddings
    text_embedding_batch_size: 32
    text_embedding_device: auto
```

The shared cache is grouped by embedding model:

```text
data/text_embeddings/<embedding-model-name>-<model-hash>/
  metadata.json
  embeddings/<key-prefix>/<cache-key>.npy
```

Each entry key also includes the text model, max token length, and raw text, so
the same cache root can be shared across model variants that use the same frozen
text backbone. The train/eval dataset streams one `.npy` vector per sample
instead of loading every cached embedding into memory at startup.

For prior-aware text embedding caches, use:

```bash
python src/prepare/04_build_prior_aware_dataset.py \
  --tokenizer microsoft/BiomedVLP-CXR-BERT-specialized \
  --precompute-text-embeddings
```

That builder writes the embedded current/prior text streams into parquet, but
the underlying frozen CLS embeddings are still read from and written to the same
shared `data/text_embeddings/...` cache.

## Optional Decoded Image Cache

This cache stores decoded RGB uint8 arrays, not final normalized tensors. That
means JPEG decode is skipped, but Albumentations can still apply random
train-time transforms each epoch.

```bash
python scripts/precompute_image_cache.py \
  --input-csv data/data-camchex/03_mimic_train.csv \
  --input-csv data/data-camchex/03_mimic_development.csv \
  --input-csv data/data-camchex/03_mimic_test.csv \
  --cache-dir data/data-camchex/image_cache_rgb
```

Then set:

```yaml
data:
  datamodule_cfg:
    image_cache_dir: data/data-camchex/image_cache_rgb
```

The cache is keyed by resolved image path. Moving the cache to a machine with
different absolute paths may cause misses; the dataset falls back to JPEG decode
on a miss.

## Compatibility Notes

- Existing CaMCheX train/eval scripts are unchanged.
- Existing CaMCheX datasets still return their old observation-text tuple.
- Stage-1 frontal/lateral checkpoint paths must be ConvNeXtV2 Nano weights for
  this model. Old ConvNeXt Tiny stage-1 weights are not shape-compatible.
- The config uses the timm registry name without the `timm/` prefix because this
  repo passes `model_name` directly to `timm.create_model`.
