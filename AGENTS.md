# AGENTS.md

Guidance for Claude Code (and any other agent) working on this repository.

## Goal

Reproduce and extend CaMCheX ("Clinically-aligned Multi-modal Chest X-ray
Classification", ML4H 2025). The model fuses chest X-rays with ED clinical
text and vital signs for 26-class multi-label disease classification.

The current active refactor is a split root `src/` package, not the older
`src/modules/` plan. Recent work moved the pieces into
`src/{encoder,decoder,dataloader,model,loss}` so encoders, decoders, models,
and dataloaders can be combined incrementally. Treat `camchex/` as the
legacy paper code unless the user explicitly asks to work there.

## Repository layout

```text
src/
  encoder/           TimmImageEncoder, BioBertEncoder,
                     CaMCheXImageEncoder, CaMCheXTextEncoder.
  decoder/           MLDecoder and transformer decoder helpers.
  dataloader/        CaMCheXDataset/DataModule, SingleViewDataset, transforms.
  model/             CaMCheXModel and SingleViewModel assemblies.
  loss/              AsymetricLoss.
scripts/
  build_mimic_subset.py        Sample patients, copy JPGs + reports, optional 7z+HF upload.
  download_subset.py           Pull split subset bundles from HF and extract with 7z.
  prepare_subset_labels.py     Joins metadata + reports + ED vitals + CXR-LT labels;
                               writes data/<subset>/labels/{train,val,test}.csv.
  analyze_subset_dataset.py    Summarize prepared subset health and write plots/CSVs.
camchex/             Legacy paper code and old LightningCLI/YAML baseline.
data/                Mostly gitignored. Subsets live at data/<subset_name>/.
output/              Analysis outputs and training/run outputs.
configs/             Older YAML configs; verify current entrypoints before using.
mimic-cxr/           Git submodule (section_parser.py).
```

## Current code conventions

**Active work goes under root `src/`.** The existing component split is
singular package names (`src/encoder`, `src/decoder`, `src/dataloader`,
`src/model`, `src/loss`). Do not assume `src/modules/` exists or recreate it
without asking; worklog entries from 2026-05-24 indicate it was replaced by
the current split layout.

**Keep reusable backbones separate from CaMCheX-specific routing.**
`TimmImageEncoder` wraps one timm model. `BioBertEncoder` wraps one BioBERT
text input. `CaMCheXImageEncoder` is the CaMCheX-specific frontal/lateral
router and returns `(features, nonzero_mask)`. `CaMCheXTextEncoder` is the
CaMCheX-specific two-stream wrapper over one shared `BioBertEncoder`.

**Model assembly still lives in `src/model/CaMCheXModel.py`.** The current
assembly constructs the CaMCheX encoders plus fusion/segment/padding,
transformer encoder, and MLDecoder head. This is not yet the fully
dependency-injected per-model-folder design described in older docs.

**Be careful with device portability.** Avoid new `.cuda()` calls. Existing
code may still contain legacy `.cuda()` usage, especially in loss code; if
you touch that area, prefer registered buffers or tensors moved with the
module/device.

**Do not silently modify legacy `camchex/`.** It is the preserved comparison
baseline. If a change is about the active refactor, make it in root `src/` or
`scripts/`, not under `camchex/`, unless the user explicitly asks for legacy
behavior.

## Data pipeline

Run subset scripts from the project root:

```bash
# 1. Build a subset. Skip archive/upload for local-only experiments.
python scripts/build_mimic_subset.py --fraction 0.1 --seed 42 --skip-archive --skip-upload

# 2. Produce label CSVs against that subset.
python scripts/prepare_subset_labels.py --subset-name subset

# 3. Optional: inspect split health, missingness, views, labels, text, and vitals.
python scripts/analyze_subset_dataset.py --subset-name subset
```

`build_mimic_subset.py` reads full MIMIC (`data/MIMIC-CXR-JPG/files`,
`data/MIMIC-CXR/files`) and produces
`data/<subset>/{MIMIC-CXR-JPG,MIMIC-CXR}/files/` plus small metadata CSVs. It
can optionally bundle into split 7z volumes and upload to a private HF
dataset repo.

`prepare_subset_labels.py` ports the four-stage `src/prepare/0{1,2,3,4}_*.py`
pipeline into one subset-targeted script. It writes CSVs under
`data/<subset>/labels/`.

Subset name auto-resolves: defaults `(0.1, 42)` -> `subset`; otherwise
`subset_seed{S}_{P}pct`. Override with `--subset-name foo` everywhere.

## Current component map

| Concern | File |
|---|---|
| Single timm backbone wrapper | `src/encoder/TimmImageEncoder.py` |
| Generic BioBERT CLS wrapper | `src/encoder/BioBertEncoder.py` |
| CaMCheX frontal/lateral image router | `src/encoder/CaMCheXImageEncoder.py` |
| CaMCheX clinical + observation text wrapper | `src/encoder/CaMCheXTextEncoder.py` |
| MLDecoder | `src/decoder/MLDecoder.py` |
| CaMCheX assembly/fusion/head wiring | `src/model/CaMCheXModel.py` |
| Single-view assembly | `src/model/SingleViewModel.py` |
| ASL loss | `src/loss/AsymetricLoss.py` |
| CaMCheX dataset/datamodule | `src/dataloader/CaMCheXDataset.py`, `src/dataloader/CaMCheXDataLoader.py` |
| Subset builder | `scripts/build_mimic_subset.py` |
| Subset downloader | `scripts/download_subset.py` |
| Label prep | `scripts/prepare_subset_labels.py` |
| Dataset analysis | `scripts/analyze_subset_dataset.py` |

## Combining components

The repo has enough pieces to start incremental combining/refactoring, but
not a finished train-entry abstraction for arbitrary architecture swaps. When
adding a new combination:

1. Reuse `src/encoder/TimmImageEncoder.py` and `src/encoder/BioBertEncoder.py`
   for generic backbones when possible.
2. Keep model-specific routing/fusion in the relevant model assembly rather
   than hiding it in generic encoder wrappers.
3. Promote shared code only after it is reused by more than one model.
4. Preserve the current dataset output contract unless changing both
   `CaMCheXDataset` and the model forward path together.

## Multi-machine workflow

The user works across `kubuntu` (primary), `ict14`, `macmini`, `richmadam`,
and potentially cloud. Current preference is **git push/pull** for code sync;
earlier sessions used rsync. If you see rsync scripts in
`scripts/rsync-scripts/`, check with the user before assuming they are active.

The subset bundle is the heavy artifact to move between machines:

```bash
python scripts/build_mimic_subset.py     # default: bundle + upload to HF
# on another machine:
python scripts/download_subset.py --repo-id <repo> --subset-name subset
```

`DATA_PASSWORD` and `HF_TOKEN` come from `.env` (see `.env.example`).

## Gitignore

`data/*` is gitignored except `data/data-tcia-download/`. CSVs are ignored
globally, so generated label files (`data/<subset>/labels/*.csv`) are not
committed. Build artifacts under `output/` are also gitignored.

## Legacy `camchex/`

The original paper code is preserved for reference. It uses LightningCLI +
`camchex/config.yaml`. The data-prep pipeline previously under
`camchex/data/` now lives at `src/prepare/0{1,2,3,4}_*.py` and is shared by
both the legacy and the refactored training paths. **Do not modify
`camchex/`** unless explicitly asked. New refactor work goes in root `src/`.

## Worklog

After finishing a non-trivial request, **append** a dated entry to the end
of [WORKLOG.md](WORKLOG.md) at the repo root. Entries are chronological
(oldest first, newest at bottom).

**Always append via a bash heredoc - never use the Edit/Write tools on
WORKLOG.md.** This rule exists because past agents have silently rewritten
or "tidied" earlier entries while editing; appending via shell guarantees
prior entries are byte-for-byte untouched.

### What an entry must contain

The worklog is the primary handoff to the next agent and to the user across
sessions. The diff already shows *what* changed; the worklog should capture
what is not recoverable from `git diff` or `git log`: reasoning,
alternatives rejected, constraints discovered, and assumptions baked into the
choice.

Every non-trivial entry should explicitly cover:

1. **Goal** - what the user asked for, restated in your own words.
2. **What changed** - concrete edits, with `path:line` references.
3. **Why this approach** - the reasoning behind the chosen design.
4. **Assumptions and constraints** - anything a future agent might challenge.
5. **Gotchas / non-obvious findings** - surprises, footguns, things noticed
   but not fixed.
6. **Follow-ups** - anything left to verify or do next.

### Format

```bash
cat >> WORKLOG.md <<'EOF'

## YYYY-MM-DD - one-line summary

**Goal.** Restate the request in 1-2 sentences.

**Changes.**
- `path/to/file.py:42` - what you changed and the role it plays.

**Reasoning.** Explain *why* this shape of solution. Mention the alternatives
you considered and why you did not pick them.

**Assumptions.** Bullet anything you took as given that is not obvious from
the code.

**Gotchas.** Anything subtle the next agent should know.

**Follow-ups.** What is left, what to verify, what you would do with more time.
EOF
```

Use the literal `'EOF'` (quoted) so backticks and `$` in the body are not
expanded. Not every section is mandatory; if there genuinely are no
assumptions or no follow-ups, omit that header rather than writing "none".
**Reasoning** and **Gotchas** should almost always appear.

Skip the worklog entirely for trivial edits. For everything else, prefer too
much context over too little.
