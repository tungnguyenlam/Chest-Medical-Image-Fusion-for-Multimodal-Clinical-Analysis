# Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis

Reproduces and extends CaMCheX — a multimodal chest X-ray classifier that
fuses up to 4 image views with ED clinical text and vital signs for 26-class
multi-label disease classification.

Active reusable components live under the root `src/` package, and active
train/eval entrypoints live under `training/`. The original paper code
remains in `camchex/` as legacy / reference.

## Setup

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source miniforge3/bin/activate

conda create -n camchex python=3.13 libjpeg-turbo -y
conda activate camchex
conda install -c conda-forge p7zip -y       # only needed to (un)bundle subsets

pip install uv

git clone https://github.com/tungnguyenlam/Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis.git
cd Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis
uv pip install -r requirements.txt

curl -fsSL https://herdr.dev/install.sh | sh
export PATH="/root/.local/bin:$PATH"
```

Copy the env template if you'll bundle/upload subsets or pull a private HF dataset:

```bash
cp .env.example .env
# fill in HF_TOKEN and DATA_PASSWORD
```

## Repository layout

```text
src/
  encoder/           TimmImageEncoder, BioBertEncoder,
                     CaMCheXImageEncoder, CaMCheXTextEncoder
  decoder/           MLDecoder and transformer decoder helpers
  dataloader/        CaMCheXDataset, SingleViewDataset, transforms
  model/             CaMCheXModel, SingleViewModel
  loss/              AsymetricLoss
training/
  camchex/           train/eval entrypoints for multimodal CaMCheX
  camchex_v2nano_vitals/
                     ConvNeXtV2 Nano + CXR-BERT + numeric vitals variant
  prior_aware/       current study + nearest previous study variant
                     (also: prior_aware_v2nano = Nano backbone + numeric vitals;
                     prior_aware_v3nano = same, single-token fusion)
  camchex_v4nano/    prior-aware v4: single-token fusion + asymmetric prior
                     cross-attention (current = queries, prior = memory)
  singleview/        train/eval entrypoints for single-view image models
  common.py          shared CLI flags (add_common_args); re-exports utils/ helpers
  utils/             plain PyTorch train/eval helpers, split by concern
                     (constants, system, config, model, data, metrics,
                      train, evaluation)
scripts/
  build_mimic_subset.py        Patient-level subset bundler (+ optional HF upload)
  prepare_subset_labels.py     Subset-aware label CSV prep
camchex/             Legacy paper code (kept for reference/comparison)
data/                Datasets — mostly gitignored, lives at data/<subset_name>/
output/              Run outputs: output/<model_name>/runs/<run_id>-<run_name>/
```

Model assemblies stay in `src/model/`; generic encoders/decoders stay in
their shared packages. Training scripts compose those pieces from
`training/` without Lightning.

## End-to-end workflow

```bash
# 1. (optional) build a small subset of MIMIC for local experimentation.
#    Default = 10% by patient, written to data/subset/.
python scripts/build_mimic_subset.py --fraction 0.1 --skip-archive --skip-upload

# 2. produce label CSVs for the subset.
#    Writes data/<subset>/labels/{train,val,test}.csv with 26 CXR-LT labels,
#    clinical_indication, and ED vitals.
python scripts/prepare_subset_labels.py --subset-name subset

# 3. train.
python training/camchex/camchex_train.py \
    --train-df-path data/subset/labels/train.csv \
    --val-df-path data/subset/labels/val.csv \
    --test-df-path data/subset/labels/test.csv \
    --batch-size 4 --max-epochs 30 --lr 1e-4
```

Run all commands from the project root.

## Subset naming

`build_mimic_subset.py` auto-names its output:

| Command | Subset dir |
|---|---|
| `--fraction 0.1 --seed 42` (defaults) | `data/subset/` |
| `--fraction 0.01 --seed 42` | `data/subset_seed42_1pct/` |
| `--fraction 0.05 --seed 7` | `data/subset_seed7_5pct/` |
| `--subset-name foo` | `data/foo/` |

Pass the generated label CSV paths to the training/eval scripts with
`--train-df-path`, `--val-df-path`, and `--test-df-path`.

## Outputs

```text
output/camchex/runs/<run_id>-<run_name>/
  checkpoints/last.pt           latest torch checkpoint
  checkpoints/best.pt           best validation AP torch checkpoint
  logs/metrics.csv              train/validation metrics
  config.resolved.json          argparse values for the run
```

`<run_id>` defaults to a timestamp; override with `--run-id` to resume into
the same directory. `<run_name>` defaults to `baseline`; pass `--run-name`
to label sweeps.

Eval scripts write predictions and metrics to `--predictions-path` and
`--metrics-path`.

### Report-ablation eval (leakage probe)

Every text model is evaluated **twice** by default: once with the full inputs
(image + clinical indication [+ vitals]) and once with the **current study's
clinical indication blanked** to the in-distribution `"No clinical history
available."` placeholder — the same string the datasets emit for genuinely
missing indications, so the probe stays on the training distribution. The second
pass is written alongside the first as `*.no_report.csv` / `*.no_report.json`,
and the console prints a head/medium/tail AP+AUROC delta:

- **Δ (full − no_report) > 0** ⇒ the indication text was *helping* that metric
  (and a large drop is what you'd expect if it leaks label information).
- **Δ ≈ 0** ⇒ the report text adds little; the model is leaning on the image/vitals.

Only the *current* study's indication is dropped — for prior-aware models the
**prior study's report/indication is kept** (it was authored before this exam,
so it is legitimate prior information, not leakage). Pass `--skip-report-ablation`
to run only the full pass.

## Useful training flags

```bash
python training/camchex/camchex_train.py \
    --backbone-name convnext_small.fb_in22k_ft_in1k_384 \
    --image-size 384 --batch-size 4 --num-workers 4 \
    --accumulate-grad-batches 16 \
    --precision 16-mixed --accelerator auto --devices auto \
    --val-check-interval 0.25
```

Full list: `python training/camchex/camchex_train.py --help`.

### LR schedule and weight averaging (EMA)

- **`model.scheduler_init_args.schedule`** selects the cosine shape:
  `warm_restarts` (default — per-epoch SGDR sawtooth, faithful to the original
  CaMCheX code; **keep the best checkpoint, the converged tail can collapse on a
  restart**) or `single_cosine` (one monotone decay over the whole run, no restarts —
  recommended for a stable fine-tuning tail).
- **`--ema` / `--no-ema`** (or `trainer.ema`, default **off**) keeps an exponential
  moving average of the weights and uses it as the *evaluated and saved* model;
  `--ema-decay` (default `0.999`) controls the smoothing. Best paired with
  `single_cosine`. Note: EMA checkpoints are eval-ready but not meant for
  `--resume-from` (use `--checkpoint-path` for weights-only warm-start). Switching
  `schedule` between a checkpoint and a `--resume-from` is blocked for the same reason.
- **Discriminative (per-component) LR** — `model.optimizer_init_args.param_group_lrs`
  maps a parameter-name *prefix* to its own LR; unmatched params use the base `lr`
  (longest prefix wins). Default empty → **all components share one LR** (unchanged
  behavior, 2 param groups, checkpoint-compatible). Example: give a pretrained text
  encoder a smaller LR while the rest trains faster:

  ```yaml
  model:
    lr: 3.0e-5
    optimizer_init_args:
      param_group_lrs:
        "text_encoder.": 1.0e-5   # prefixes match model.named_parameters()
  ```

  Per-group LRs compose with the schedules (each group is scaled relative to its own
  initial LR). Caveat: the cosine `eta_min` is a single scalar (`base_lr *
  eta_min_factor`), so all groups converge to the *same* floor — the LR ratio holds
  early/mid-training and shrinks toward 1.0 in the tail.

### Loss function

The training criterion is selectable from a registry (`src/loss/__init__.py`,
`LOSS_REGISTRY`): `ASL` (Asymmetric Loss, default) and `FC` (multi-label sigmoid focal).

- **`--loss FC`** (or `model.loss: FC`) overrides the default ASL with focal.
- **`--loss FC ASL`** trains on their **weighted sum** (a `CompositeLoss`). Set weights
  with `--loss-weights 1.0 0.5` (positional) or `model.loss_weights: {FC: 1.0, ASL: 0.5}`
  — they matter because the losses differ in scale.
- Per-loss kwargs come from `model.loss_kwargs.<NAME>` (e.g. `FC: {gamma: 2.0, alpha: 0.25}`);
  `ASL` also inherits the existing flat `model.loss_init_args` for backward compatibility.
- Default (no flag, no `model.loss`) = `ASL` from `model.loss_init_args` — unchanged.

```yaml
model:
  loss: [FC, ASL]            # or a single name; CLI --loss overrides
  loss_weights: {FC: 1.0, ASL: 0.5}
  loss_kwargs:
    FC: {gamma: 2.0, alpha: 0.25}
  loss_init_args: { ... }    # ASL's class counts (also used when ASL is in the mix)
```

Add a new loss by registering it in `LOSS_REGISTRY` with the `forward(logits, float_labels)
-> scalar` convention.

## Image preprocessing, caching, and flash attention

These behaviors are shared across every training/eval script (wired in the
`training/utils/` helpers, re-exported via `training/common.py`) and default-on; no
model-specific code is involved.

**3-channel CXR build (`--third-channel-mode`).** Each CXR is turned into a deterministic
3-channel image. **ch0 = raw** and **ch1 = mild CLAHE** (clip 2.0 / 8×8) are always
pinned; only the **third channel** is a degree of freedom, so the flag names just that
channel. Channels are normalized with precomputed per-channel stats. Override per run:

```bash
--third-channel-mode histeq   # default: ch2 = global histogram equalization (raw_clahe_histeq)
--third-channel-mode clahe    # ch2 = strong CLAHE (clip 4.0 / 16x16) — same signal, higher contrast
--third-channel-mode lbp      # ch2 = uniform Local Binary Pattern (local micro-texture)
--third-channel-mode none     # legacy ImageNet RGB (plain grayscale-duplicated decode)
```

The internal/config name is still the full mode (`data.datamodule_cfg.channel_mode:
raw_clahe_histeq`); the short flag maps to it. At every train start the resolved channel
composition (per-channel filter, params, and normalization stats) is printed under
`[channels]` so a wrong mode is caught before the first epoch.

`lbp`'s precomputed stats assume scikit-image's *uniform* LBP (in `requirements.txt`);
without it the build falls back to a plain 8-neighbour LBP whose distribution won't match
those stats. `clahe`'s ch2 stats are currently **provisional** (estimated, not yet measured
on the training split — see `compute_channel_statistics.ipynb`).

Prior-aware models support this too; `--third-channel-mode none` reverts them to the plain
decode. The CLI options are gated by `ENABLED_THIRD_CHANNELS` in
`training/utils/constants.py` — append a short name there (a key of
`THIRD_CHANNEL_TO_MODE`) to expose it.

**Shared channel cache + automatic prebuild.** Built channels are cached as uint8 `.npy`
under `image_channel_cache_dir` (default `../cache/channels`), keyed by
`(raw path string, mode, preprocessing fingerprint)` — shared across models/runs and
self-invalidating. Before training, every channel-using script **prebuilds the whole
cache up front** so the first epoch reads a warm cache; the build runs in parallel and
skips already-cached images. The key is the raw path *string*, so cache hits never stat
the source filesystem — essential when images live on a slow mount (e.g. WSL
`/mnt/<drive>`). Resolution + decode happen only on a cache miss.

```bash
--cpu-fraction 0.5   # default: fraction of cores used for the prebuild
--cpu-fraction 2.0   # oversubscribe when the build is I/O-bound on a slow mount (idle CPU)
--skip-precompute    # skip the upfront prebuild; channels build lazily on first access
                     # (cache config untouched) — use when the cache is already warm
```

**Shared text-embedding cache.** With `--use-precomputed-text-embeddings`, the frozen
text encoder's CLS embeddings are cached under `text_embedding_cache_dir` (default
`../cache/text_embeddings`), content-addressed per model, and never recomputed for texts
already present. Both caches live under one `../cache/` tree (native disk, outside the
repo), reused across runs.

**Flash attention (automatic).** Text encoders load with the best available attention
backend: `flash_attention_2` when the `flash-attn` package and an Ampere+ GPU are both
present, otherwise PyTorch SDPA (which dispatches to the flash kernel on compatible CUDA
under fp16/bf16), falling back to eager on CPU/MPS. No flag needed.

## ConvNeXtV2 Nano + Numeric Vitals Variant

The isolated variant in `training/camchex_v2nano_vitals/` uses ConvNeXtV2 Nano,
frozen `microsoft/BiomedVLP-CXR-BERT-specialized`, and numeric ED vitals
projected as 8x8 tokens. See
`training/camchex_v2nano_vitals/README.md` for architecture notes, train/eval
commands, and optional clinical/image cache workflows.

## Adding a new model

1. Add reusable components under `src/{encoder,decoder,dataloader,model,loss}`.
2. Build a model assembly in `src/model/` wiring the pieces you need.
3. Copy an existing script under `training/` and adjust imports + argparse.
4. Train with `python training/<name>/<name>_train.py ...`.
   Outputs land in `output/<name>/runs/...` automatically.

## Device portability

The training stack runs on CUDA, ROCm (presents as CUDA in PyTorch), MPS,
or CPU — pick via `--accelerator`. Note: 16-mixed precision is flaky on MPS;
fall back to `--precision 32` there.

## Legacy

The `camchex/` directory contains the original paper code: a LightningCLI
+ YAML config layout. The data pipeline that used to live alongside it
now lives at `src/prepare/0{1,2,3,4}_*.py` and is shared with the refactored
training path. The legacy `camchex/` training entry is no longer used but
is kept for reference and comparison.

## TCIA download helper

```text
https://github.com/ygidtu/NBIA_data_retriever_CLI
```

```bash
./NBIA_data_retriever_CLI -i your_manifest.tcia -s ./output -p 4
```
