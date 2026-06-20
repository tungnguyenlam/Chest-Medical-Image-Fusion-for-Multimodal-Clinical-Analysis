# Prior-Aware v5 Nano — CXR-LT 2024 (task1)

Same model as [`prior_aware_v5nano`](../prior_aware_v5nano/README.md) trained on the
**CXR-LT 2024 task1** label space (40 labels) instead of the 2023 26-label set. Only
the label space, the ASL class frequencies, the backbone `num_classes`, and the
dataframe paths differ — the architecture and all v5 knobs are identical, so this is a
clean dataset-swap ablation.

The 2024 release uses the **same MIMIC-CXR images** as 2023; only the labels and the
train/val/test assignment change. So the expensive image-channel cache
(`../cache/channels`) and frozen text-embedding cache (`../cache/text_embeddings`) are
keyed by image path / text content and are **reused verbatim** — only the prior-aware
parquets are rebuilt in the 40-label space.

## Build the data (one-time)

Stage 04 now builds **every** CXR-LT release/task variant (including this one) by
default, so the simplest path is one command:

```bash
# Builds prior_aware_cxrlt2024_task1_{train,val,test}.parquet (and all other
# variants) from the 03_mimic_* base. Re-attaches labels + re-splits per release.
python src/prepare/04_build_prior_aware_dataset.py
```

To rebuild *only* this 40-label variant (faster), use single-variant mode:

```bash
# 1. Relabel + re-split the prepared CSVs for CXR-LT 2024 task1.
python scripts/relabel_prepared_cxrlt.py --cxr-lt-version cxr-lt-2024 --cxr-lt-label-set task1

# 2. Build the prior-aware parquets in the 40-label space.
#    Prior linkage is re-resolved here (it is resolved within a split, and 2024
#    reassigns splits). Emits a *_label_metadata.json sidecar with per-split counts.
python src/prepare/04_build_prior_aware_dataset.py \
  --in-dir data/data-camchex/cxrlt2024_task1 --in-prefix "" --splits train val test \
  --out-dir data/data-camchex --out-prefix prior_aware_cxrlt2024_task1_ \
  --cxr-lt-version cxr-lt-2024 --cxr-lt-label-set task1
```

## Fill the loss weights

`config.yaml` ships with `class_instance_nums: null` / `total_instance_num: null` — the
ASL loss has no uniform fallback, so these **must** be filled before training. Copy them
straight from the build's sidecar:

```
data/data-camchex/prior_aware_cxrlt2024_task1_label_metadata.json
  -> splits.train.class_instance_nums   (40 values, same order as model.classes)
  -> splits.train.total_instance_num
```

These are per-**study** positive counts for the prior-aware parquet — not the per-image
counts in `camchex_v2nano_vitals_cxrlt2024/config.yaml` — so they must come from this
build, not be copied from the image-level config.

## Train

```bash
python training/prior_aware_v5nano/prior_aware_train.py \
  --config training/prior_aware_v5nano_cxrlt2024/config.yaml
```

(The 2024 variant reuses the `prior_aware_v5nano` train/eval entry points; only the
`--config` changes.)

## Eval (incl. the task2 gold set)

```bash
python training/prior_aware_v5nano/prior_aware_eval.py \
  --config training/prior_aware_v5nano_cxrlt2024/config.yaml \
  --checkpoint-path <ckpt>
```

Because this is a 2024 task1 (40-label) model, eval **automatically also scores it on
the CXR-LT 2024 task2 (gold) test set** — the small expert-labeled set. The model's 40
outputs are sliced to the 26 task2 labels (task2 ⊂ task1) and scored against the gold
labels, written to `metrics_task2_gold.json` / `predictions_task2_gold.csv` alongside the
normal task1 metrics. This needs the task2 gold file next to the eval file (built by
`python src/prepare/04_build_prior_aware_dataset.py`, which produces every variant). Pass
`--skip-task2-gold` to turn it off.
