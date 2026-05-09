# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Reproduce and extend the CaMCheX paper ("Clinically-aligned Multi-modal Chest X-ray Classification", ML4H 2025). The model fuses chest X-ray images with ED clinical text and vital signs for 26-class multi-label disease classification. The `camchex/` subdirectory contains the model code with its own `AGENTS.md`.

## Repository layout

```
camchex/          Training code (model, dataset, config) — see camchex/AGENTS.md
data/             All datasets (mostly gitignored; symlinked to external storage)
mimic-cxr/        Git submodule — MIT-LCP MIMIC-CXR repo (used for section_parser.py)
scripts/
  create-symlink/ Per-machine symlink setup scripts
  rsync-scripts/  Push/pull code between machines
  dataset-download/ Kaggle download helpers
```

## Multi-machine workflow

Four machines in use (kubuntu is primary): `kubuntu`, `ict14`, `macmini`, `richmadam`, plus a potential cloud instance. Code is synced via rsync — **not git push** — because some machines require manual login to pull.

```bash
# Push from kubuntu to a machine
bash scripts/rsync-scripts/kubuntu2ict14.sh
bash scripts/rsync-scripts/kubuntu2macmini.sh
bash scripts/rsync-scripts/kubuntu2richmadam.sh

# Pull from ict14 back to kubuntu
bash scripts/rsync-scripts/ict142kubuntu.sh
```

All rsync scripts exclude `camchex/config.local.yaml` so per-machine training configs are never overwritten. When adding a new rsync script for a cloud instance, copy an existing one and add `--exclude 'camchex/config.local.yaml'`.

On each machine, run the relevant symlink script once after first sync:
```bash
bash scripts/create-symlink/<machine>.sh
```

## Data pipeline

All three scripts must be run from the **project root**. Each enforces this with a `sys.exit()` guard. Outputs are named with step-number prefixes so provenance is clear.

```bash
python camchex/data/01_make_dataset.py      # merge all sources → data/data-camchex/01_merged.csv
python camchex/data/02_split_dataset.py     # split by CXR-LT IDs → data/data-camchex/02_*.csv
python camchex/data/03_filter_existing_images.py  # drop missing images → data/data-camchex/03_*.csv
```

Step 1 is slow (parses ~200k text reports). If it completes but crashes before the end, `data/data-camchex/01_progress.csv` is the checkpoint saved after report parsing — the most expensive part.

Step 2 can be skipped if step 1 already produced `02_*.csv` (it does so at the end). Run it only to resplit without rerunning step 1.

## Data sources and symlinks

| Path | What it is | Where data lives |
|---|---|---|
| `data/MIMIC-CXR-JPG/files` | JPG X-ray images | `~/Programming/split-4/files/` (symlink) |
| `data/MIMIC-CXR/files` | Text reports (`.txt`) | `~/Programming/download-mimic-cxr-txt/files/` (symlink) |
| `camchex/images/p1x/` | Kaggle image subset | Per-folder symlinks into Kaggle cache |
| `data/MIMIC-IV-ED-2-2/` | ED triage + vitals | Local `.csv.gz` files |
| `data/CXR-LT/` | 26-class labels | Local CSVs |

The full image set (~500 GB) is unlikely to be completely downloaded. `03_filter_existing_images.py` handles this — it silently drops any study whose image file is absent.

## Gitignore notes

`data/*` is ignored except `data/data-tcia-download/`. CSV files (`*.csv`, `*.csv.gz`) are also ignored globally — the generated intermediates in `data/data-camchex/` are not committed.
