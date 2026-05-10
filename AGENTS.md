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

## Worklog

After finishing a non-trivial request, **append** a dated entry to the end of [WORKLOG.md](WORKLOG.md) at the repo root. Entries are chronological (oldest first, newest at bottom).

**Always append via a bash heredoc — never use the Edit/Write tools on WORKLOG.md.** This rule exists because past agents have silently rewritten or "tidied" earlier entries while editing; appending via shell guarantees prior entries are byte-for-byte untouched.

### What an entry must contain

The worklog is the primary handoff to the next agent (and to the user across sessions). The diff already shows *what* changed — your job in the worklog is to capture everything that **isn't recoverable from `git diff` or `git log`**: the reasoning, the alternatives you rejected, the constraints you discovered, and the assumptions baked into the choice. Err on the side of writing more, not less. A worklog entry that's just a bullet list of file paths is failing its purpose.

Every non-trivial entry should explicitly cover:

1. **Goal** — what the user actually asked for, restated in your own words (so a future agent can tell whether they're picking up the same task or a related one).
2. **What changed** — concrete edits, with `path:line` references. Group related edits; don't just dump a list.
3. **Why this approach** — the reasoning behind the chosen design. What were the candidate options? Why did you pick this one over the others? What trade-off did you accept?
4. **Assumptions and constraints** — anything you took as given that a future agent might want to challenge (e.g. "assumed cwd is `camchex/` because `train.sh` cd's there", "kept `../` paths because rsync scripts copy `camchex/` as a unit").
5. **Gotchas / non-obvious findings** — surprises, footguns, things that almost broke, things you noticed but didn't fix. This is the highest-value section for future agents.
6. **Follow-ups** — anything you didn't do that probably should be done next, or that the user might want to verify.

### Format

```bash
cat >> WORKLOG.md <<'EOF'

## YYYY-MM-DD — one-line summary

**Goal.** Restate the request in 1–2 sentences.

**Changes.**
- `path/to/file.py:42` — what you changed and the role it plays.
- `other/file.yaml` — …

**Reasoning.** Explain *why* this shape of solution. Mention the alternatives you considered and why you didn't pick them. If a choice is reversible-but-annoying-to-change later, say so. If it's load-bearing for something downstream, say so.

**Assumptions.** Bullet anything you took as given that isn't obvious from the code.

**Gotchas.** Anything subtle the next agent should know — e.g. SLURM `--output` is resolved before the script's `cd`, so the path is relative to submit dir; an earlier checkpoint format expected a flat dir; rsync excludes `config.local.yaml`; etc.

**Follow-ups.** What's left, what to verify, what you'd do with more time.
EOF
```

Use the literal `'EOF'` (quoted) so backticks and `$` in the body aren't expanded. Not every section is mandatory for every entry — if there genuinely are no assumptions or no follow-ups, omit that header rather than writing "none". But **Reasoning** and **Gotchas** should almost always appear; if you find yourself wanting to skip them, you're probably under-explaining.

Skip the worklog entirely for trivial edits (typo fixes, formatting-only changes, single-line config tweaks the user dictated verbatim). For everything else, prefer too much context over too little — future agents skim this to catch up on multi-machine/multi-session state that isn't visible in the diff, and a thin entry forces them to re-derive your reasoning from scratch.
