# CaMCheX ConvNeXtV2 Nano + Numeric Vitals

This is an additive CaMCheX variant. It does not change the legacy
`CaMCheXModel`, `CaMCheXDataset`, or existing `training/camchex*` entrypoints.

## What changed

- Image backbone: `convnextv2_nano.fcmae_ft_in22k_in1k_384`.
- Text backbone: `microsoft/BiomedVLP-CXR-BERT-specialized`.
- Text encoder is frozen by default.
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
| Clinical embedding cache script | `scripts/precompute_clinical_embeddings.py` |
| Decoded image cache script | `scripts/precompute_image_cache.py` |

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

CXR-BERT still needs to be available from Hugging Face or local cache unless a
clinical embedding cache is configured.

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

## Optional Clinical Embedding Cache

Because the text encoder is frozen, clinical indication CLS embeddings can be
precomputed once per `study_id`.

Use `scripts/precompute_clinical_embeddings.py` only for this non-prior model
path. It writes a simple `study_id -> clinical_indication CLS embedding` cache
used through `clinical_embedding_path`. It is not the right tool for
prior-aware training, where the parquet must contain current clinical, current
observation, prior clinical, and prior observation embeddings per row.

```bash
python scripts/precompute_clinical_embeddings.py \
  --input-csv data/data-camchex/03_mimic_train.csv \
  --input-csv data/data-camchex/03_mimic_development.csv \
  --input-csv data/data-camchex/03_mimic_test.csv \
  --output-path data/data-camchex/cxrbert_clinical_embeddings.pt
```

Then set:

```yaml
data:
  datamodule_cfg:
    clinical_embedding_path: data/data-camchex/cxrbert_clinical_embeddings.pt
```

If an embedding is missing for a study, the dataset falls back to tokenization
and live frozen CXR-BERT inference.

For prior-aware text embedding caches, use:

```bash
python src/prepare/04_build_prior_aware_dataset.py --precompute-text-embeddings
```

See `training/prior_aware/README.md`.

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
