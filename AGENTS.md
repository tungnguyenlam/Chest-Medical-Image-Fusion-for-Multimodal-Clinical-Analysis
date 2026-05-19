# AGENTS.md

Guidance for Claude Code (and any other agent) working on this repository.

## Goal

Reproduce and extend CaMCheX ("Clinically-aligned Multi-modal Chest X-ray
Classification", ML4H 2025). The model fuses chest X-rays with ED clinical
text and vital signs for 26-class multi-label disease classification.

The user is migrating from the original paper code (`camchex/`) to a
component-organised codebase under `src/modules/` so different encoders,
decoders, and architectures can be swapped per experiment. Training now
runs on a deterministic patient-level subset of MIMIC (default 10%) so
experiments are tractable on a single machine.

## Repository layout

```text
src/modules/
  encoders/          Shared image / text encoders. Each accepts name=.
  decoders/          Shared decoders (currently MLDecoder).
  dataloaders/       MimicMultiViewDataset + DataModule, transforms.
  callbacks/         EMA, PredictionWriter, RunLogger (component-aware).
  models/<name>/     One folder per architecture. Owns its assembly,
                     fusion (if architecture-specific), LightningModule,
                     loss, and train_<name>.py argparse entry.
scripts/
  build_mimic_subset.py        Sample patients, copy JPGs + reports, optional 7z+HF upload.
  prepare_subset_labels.py     Joins metadata + reports + ED vitals + CXR-LT labels;
                               writes data/<subset>/labels/{train,val,test}.csv.
camchex/             Legacy. Kept verbatim for reference; not imported by src/.
data/                Mostly gitignored. Subsets live at data/<subset_name>/.
output/              Run outputs at output/<model_name>/runs/<run_id>-<run_name>/.
configs/             Legacy YAMLs; the new entry uses pure argparse.
mimic-cxr/           Git submodule (section_parser.py).
```

## Core conventions

**Per-model folders.** Each architecture is a self-contained package under
`src/modules/models/<name>/` with its own `train_<name>.py`. Shared
encoders/decoders/dataloaders/callbacks live one level up under
`src/modules/` and are imported by each model's assembly.

**Dependency injection.** `CaMCheXModel(frontal_encoder, lateral_encoder,
text_encoder, fusion, head)` takes pre-built submodules. No `timm.create_model`
or `AutoModel.from_pretrained` calls live inside the assembly — those happen
in the `LightningModule.__init__` so swapping a backbone is a few-line change.

**`component_name` convention.** Every encoder, decoder, fusion, and head
accepts `name="..."` and stores `self.component_name`. `RunLoggerCallback`
walks `pl_module.named_modules()`, groups grad norms by the nearest ancestor
with a `component_name`, and logs `grad_norm/<component_name>`. Logs stay
stable across architecture swaps. If a component has no `component_name`,
the callback falls back to Python attribute path.

**Argparse, not YAML.** The new training entry is pure argparse:
`python -m src.modules.models.camchex.train_camchex --help`. No LightningCLI,
no jsonargparse, no YAML. Each run dumps `args` to `config.resolved.json`
inside its run dir.

**Device-agnostic code.** No `.cuda()` calls — use `register_buffer` or
let Lightning move things via `.to(device)`. The stack must run on CUDA,
ROCm (presents as CUDA), MPS, and CPU. The ASL loss in
`src/modules/models/camchex/loss.py` was fixed to register buffers
specifically for this.

**Output paths.** Always `output/<model_name>/runs/<run_id>-<run_name>/`
relative to project root. `<model_name>` is a module-level constant inside
each model's `train_<name>.py` (e.g. `MODEL_NAME = "camchex"`).

## Data pipeline

Two scripts, run from project root:

```bash
# 1. Build a subset (skip --skip-upload --skip-archive for local experiments)
python scripts/build_mimic_subset.py --fraction 0.1 --seed 42 --skip-archive --skip-upload

# 2. Produce label CSVs against that subset
python scripts/prepare_subset_labels.py --subset-name subset
```

`build_mimic_subset.py` reads full MIMIC (`data/MIMIC-CXR-JPG/files`,
`data/MIMIC-CXR/files`) and produces `data/<subset>/{MIMIC-CXR-JPG,MIMIC-CXR}/files/`
plus the small metadata CSVs. It also (optionally) bundles into a 7z
archive and uploads to a private HF dataset repo so other machines can
pull just the subset.

`prepare_subset_labels.py` ports the camchex 3-step pipeline
(`camchex/data/01,02,03_*.py`) into one file, targeted at the subset.
It outputs CSVs whose `path` column is **relative to**
`data/<subset>/MIMIC-CXR-JPG/files/`. The DataModule prepends `image_root`
at load time — no per-machine symlinks, no `../` escape paths.

Subset name auto-resolves: defaults `(0.1, 42)` → `subset`; otherwise
`subset_seed{S}_{P}pct`. Override with `--subset-name foo` everywhere.

## Training

```bash
python -m src.modules.models.camchex.train_camchex \
    --subset-name subset \
    --batch-size 4 --max-epochs 30 --lr 1e-4
```

The 26 CXR-LT classes are a module constant in `train_camchex.py`. ASL
class-instance counts are auto-computed from `train.csv` at startup —
no need to maintain a 26-element list in config.

## What lives where (cheat sheet)

| Concern | File |
|---|---|
| Image backbone wrapping (timm) | `src/modules/encoders/timm_image.py` |
| Text encoder (BioBERT) | `src/modules/encoders/biobert_text.py` |
| MLDecoder | `src/modules/decoders/ml_decoder.py` |
| CaMCheX fusion (segment + padding + transformer) | `src/modules/models/camchex/fusion.py` |
| CaMCheX assembly | `src/modules/models/camchex/camchex.py` |
| LightningModule (CaMCheX) | `src/modules/models/camchex/lightning_module.py` |
| ASL loss | `src/modules/models/camchex/loss.py` |
| Train entry | `src/modules/models/camchex/train_camchex.py` |
| MIMIC multi-view dataset | `src/modules/dataloaders/mimic_multiview_dataset.py` |
| DataModule | `src/modules/dataloaders/mimic_multiview_datamodule.py` |
| Subset builder | `scripts/build_mimic_subset.py` |
| Label prep | `scripts/prepare_subset_labels.py` |

## Adding a new model

1. `mkdir src/modules/models/<name>/`
2. Write the assembly that pulls encoders/decoders from `src/modules/`.
3. Write the LightningModule.
4. Copy `src/modules/models/camchex/train_camchex.py` to
   `train_<name>.py`, change `MODEL_NAME`, adjust argparse + imports.
5. Train via `python -m src.modules.models.<name>.train_<name>`.

If the new model needs a fusion block that camchex doesn't have, add it
under `src/modules/models/<name>/fusion.py`. Only promote things to
`src/modules/{encoders,decoders,dataloaders}/` once they're reused by ≥2
models — premature shared abstractions are a known footgun here.

## Multi-machine workflow

The user works across `kubuntu` (primary), `ict14`, `macmini`,
`richmadam`, and potentially cloud. Current preference is **git push/pull**
for code sync; earlier sessions used rsync — if you see rsync scripts in
`scripts/rsync-scripts/`, check with the user before assuming they're still
active.

The subset bundle is the heavy thing to move between machines. Workflow:
build subset on the machine that has full MIMIC, bundle to 7z, upload to
the private HF dataset repo, pull + extract on other machines:

```bash
python scripts/build_mimic_subset.py     # default: bundle + upload to HF
# on other machine:
# huggingface-cli download <repo> --repo-type dataset --local-dir data/_bundles
7z x data/_bundles/bundle-a3f9.7z.001 -odata/
```

`DATA_PASSWORD` and `HF_TOKEN` come from `.env` (see `.env.example`).

## Gitignore

`data/*` is gitignored except `data/data-tcia-download/`. CSVs are
ignored globally — generated label files (`data/<subset>/labels/*.csv`)
are not committed. Build artifacts under `output/` are also gitignored.

## Legacy `camchex/`

The original paper code is preserved untouched. It uses LightningCLI +
`camchex/config.yaml` + the 3-step `camchex/data/` pipeline against the
full MIMIC tree. **Do not modify `camchex/`** unless explicitly asked —
it's the comparison baseline. New work goes in `src/modules/`.

## Worklog

After finishing a non-trivial request, **append** a dated entry to the end
of [WORKLOG.md](WORKLOG.md) at the repo root. Entries are chronological
(oldest first, newest at bottom).

**Always append via a bash heredoc — never use the Edit/Write tools on
WORKLOG.md.** This rule exists because past agents have silently rewritten
or "tidied" earlier entries while editing; appending via shell guarantees
prior entries are byte-for-byte untouched.

### What an entry must contain

The worklog is the primary handoff to the next agent (and to the user
across sessions). The diff already shows *what* changed — your job in the
worklog is to capture everything that **isn't recoverable from `git diff`
or `git log`**: the reasoning, the alternatives you rejected, the
constraints you discovered, and the assumptions baked into the choice. Err
on the side of writing more, not less.

Every non-trivial entry should explicitly cover:

1. **Goal** — what the user actually asked for, restated in your own words.
2. **What changed** — concrete edits, with `path:line` references.
3. **Why this approach** — the reasoning behind the chosen design. What
   were the candidate options? Why did you pick this one over the others?
4. **Assumptions and constraints** — anything you took as given that a
   future agent might want to challenge.
5. **Gotchas / non-obvious findings** — surprises, footguns, things that
   almost broke, things you noticed but didn't fix. Highest-value section.
6. **Follow-ups** — anything you didn't do that probably should be done
   next, or that the user might want to verify.

### Format

```bash
cat >> WORKLOG.md <<'EOF'

## YYYY-MM-DD — one-line summary

**Goal.** Restate the request in 1–2 sentences.

**Changes.**
- `path/to/file.py:42` — what you changed and the role it plays.

**Reasoning.** Explain *why* this shape of solution. Mention the
alternatives you considered and why you didn't pick them.

**Assumptions.** Bullet anything you took as given that isn't obvious from the code.

**Gotchas.** Anything subtle the next agent should know.

**Follow-ups.** What's left, what to verify, what you'd do with more time.
EOF
```

Use the literal `'EOF'` (quoted) so backticks and `$` in the body aren't
expanded. Not every section is mandatory — if there genuinely are no
assumptions or no follow-ups, omit that header rather than writing "none".
**Reasoning** and **Gotchas** should almost always appear; if you find
yourself wanting to skip them, you're probably under-explaining.

Skip the worklog entirely for trivial edits. For everything else, prefer
too much context over too little.
