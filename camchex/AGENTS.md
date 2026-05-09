# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CaMCheX (Clinically-aligned Multi-modal Chest X-ray Classification) — a PyTorch/Lightning model that fuses chest X-ray images with clinical text (reports, indication/history) and ED vital signs for 26-class multi-label disease classification. Accepted at ML4H 2025.

All training code lives under `camchex/`. Data preparation scripts live under `camchex/data/`. Training is always launched from `camchex/` as the working directory.

## Data pipeline (run from project root)

Three steps must be run in order. All scripts enforce project root via a `sys.exit()` guard.

```bash
# Step 1 — merge all sources into 01_merged.csv and produce initial splits
python camchex/data/01_make_dataset.py

# Step 2 — re-split from 01_merged.csv (skip if step 1 already produced 02_*.csv)
python camchex/data/02_split_dataset.py

# Step 3 — filter rows to only images that exist on disk; writes to data/data-camchex/03_*.csv
python camchex/data/03_filter_existing_images.py
```

Intermediate files live in `data/data-camchex/` with step-number prefixes:
- `01_merged.csv` — full merged dataset (all sources joined)
- `01_progress.csv` — mid-run checkpoint saved after report parsing
- `02_train/development/test.csv` — splits by CXR-LT 2023 IDs
- `03_train/development/test.csv` — image-filtered, ready for training

## Training

```bash
cd camchex
sbatch train.sh          # SLURM cluster
bash train.sh            # local (also works)
```

`train.sh` loads `config.yaml` and, if present, `config.local.yaml` (machine-specific overrides). Per-machine setup:

```bash
cp camchex/config.local.yaml.example camchex/config.local.yaml
# edit devices, batch_size, num_workers
```

`config.local.yaml` is gitignored and excluded from all rsync scripts — it never leaves the machine.

## Architecture

The model (`camchex/model/`) has two backbones selectable in `wrapper.py`:
- `SingleViewModel` — single image per sample
- `CaMCheXModel` — active default; fuses up to 4 views per study with clinical text and vitals via BioBERT (`dmis-lab/biobert-v1.1`) token embeddings

`CaMCheXDataset` (`camchex/dataset/dataset.py`) groups rows by `study_id`, loads up to 4 images, encodes clinical indication (max 384 tokens) and vital signs as text (max 128 tokens). Image paths in CSVs are relative to `camchex/` — e.g. `images/p10/p10000032/s50414267/<dicom>.jpg`.

Loss is Asymmetric Loss (ASL) with per-class instance counts from `config.yaml` for long-tail reweighting. Validation tracks mAP split into head / medium / tail frequency groups (indices hardcoded in `wrapper.py`).

LightningCLI wires everything; `config.yaml` is the single source of truth for hyperparameters, paths, and class lists.

## Key data sources and symlinks

| Symlink / path | Points to | Contents |
|---|---|---|
| `data/MIMIC-CXR-JPG/files` | `~/Programming/split-4/files/` | JPG images |
| `data/MIMIC-CXR/files` | `~/Programming/download-mimic-cxr-txt/files/` | `.txt` reports |
| `camchex/images/p1x/` | Kaggle cache (per-folder symlinks) | Subset of JPG images |
| `data/MIMIC-IV-ED-2-2/` | local | ED triage, vitals, stays |
| `data/CXR-LT/` | local | 26-class labels (train/dev/test) |

Symlink setup scripts are in `scripts/create-symlink/`. Rsync scripts are in `scripts/rsync-scripts/` — they exclude `camchex/config.local.yaml` so machine configs are never overwritten.

## Config structure

`camchex/config.yaml` has three top-level sections: `trainer`, `model`, `data`. Notable fields:
- `data.datamodule_cfg.train/devel/pred_df_path` — point to `../data/data-camchex/03_*.csv` (relative to `camchex/` cwd at training time)
- `model.loss_init_args.class_instance_nums` — must match the 26 classes in order if changed
- `trainer.val_check_interval: 0.25` — validates 4× per epoch
- `trainer.accumulate_grad_batches: 16` — effective batch = batch_size × 16
