# Worklog

Chronological log of agent-completed tasks (oldest first, newest at bottom). Each entry: date, one-line summary, then bullets covering what changed and any non-obvious findings. Keep entries short — link to files with `path:line` instead of pasting code.

**Append-only.** New entries must be added with `cat >> WORKLOG.md <<'EOF' ... EOF` from bash — never edited in via Edit/Write. See AGENTS.md "Worklog" for the rationale.

## 2026-05-10 — Two-source image filtering + Kaggle symlink clarification

- Rewrote [camchex/data/03_filter_existing_images.py](camchex/data/03_filter_existing_images.py) to filter against two image sources and emit per-source CSVs:
  - `data/MIMIC-CXR-JPG/files` → `data/data-camchex/03_mimic_{train,development,test}.csv`
  - `data/data-kaggle/official_data_iccv_final/files` → `data/data-camchex/03_kaggle_{train,development,test}.csv`
  - Strips the `images/` prefix from each CSV `path` and joins against each source's base dir. Skips a source if its base dir is missing on the current machine.
  - Training config now picks both the CSV set and the matching image base dir; old single-output `03_{train,development,test}.csv` is no longer produced.
- Confirmed [scripts/dataset-download/kaggle-mimic-cxr-jpg-setup.py](scripts/dataset-download/kaggle-mimic-cxr-jpg-setup.py) symlink direction is correct: `data/data-kaggle -> ~/.cache/kagglehub/.../versions/2`. Reversing it would put the link outside the workspace and break the project's read path.
- VSCode "can't open the symlink" symptom is a workspace-traversal limitation (the target lives under `~/.cache`, outside the workspace), not a script bug. Workarounds: open the cache path directly, or rely on terminal/training tooling which follows the symlink fine.

## 2026-05-10 — Refactor image dataset filtering

- updated camchex/data/03_filter_existing_images.py:17 to support multiple image sources (mimic/kaggle) instead of relying on a single symlinked `camchex/images` directory.
- removed legacy symlinks from camchex/images and updated scripts/create-symlink/ict14.sh:8.
- added documentation to AGENTS.md:68 regarding the WORKLOG.md maintenance protocol.
