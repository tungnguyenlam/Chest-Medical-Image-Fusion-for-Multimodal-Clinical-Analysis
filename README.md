# Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis

Reproduces and extends CaMCheX — a multimodal chest X-ray classifier that
fuses up to 4 image views with ED clinical text and vital signs for 26-class
multi-label disease classification.

Active training code lives under `src/modules/`. The original paper code
remains in `camchex/` as legacy / reference and is no longer used by the
new training entry.

## Setup

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source miniforge3/bin/activate

conda create -n camchex python=3.13 -y
conda activate camchex
conda install -c conda-forge p7zip -y       # only needed to (un)bundle subsets

pip install uv

git clone https://github.com/tungnguyenlam/Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis.git
cd Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis
uv pip install -r requirements.txt
```

Copy the env template if you'll bundle/upload subsets or pull a private HF dataset:

```bash
cp .env.example .env
# fill in HF_TOKEN and DATA_PASSWORD
```

## Repository layout

```text
src/modules/
  encoders/          TimmImageEncoder, BioBertTextEncoder      (shared)
  decoders/          MLDecoder                                  (shared)
  dataloaders/       MimicMultiViewDataset + DataModule         (shared)
  callbacks/         EMA, PredictionWriter, RunLogger          (shared)
  models/
    camchex/         camchex.py (assembly), fusion.py,
                     lightning_module.py, loss.py,
                     train_camchex.py                          (per-model)
scripts/
  build_mimic_subset.py        Patient-level subset bundler (+ optional HF upload)
  prepare_subset_labels.py     Subset-aware label CSV prep
camchex/             Legacy paper code (kept verbatim, no longer used by training)
data/                Datasets — mostly gitignored, lives at data/<subset_name>/
output/              Run outputs: output/<model_name>/runs/<run_id>-<run_name>/
```

Each architecture you experiment with lives in its own folder under
`src/modules/models/<name>/` with its own assembly, Lightning module, and
`train_<name>.py` argparse entry. Encoders, decoders, dataloaders, and
callbacks under `src/modules/` are shared building blocks reused across
models.

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
python -m src.modules.models.camchex.train_camchex \
    --subset-name subset \
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

Pass the same `--subset-name` to `prepare_subset_labels.py` and to
`train_camchex.py`.

## Outputs

```text
output/camchex/runs/<run_id>-<run_name>/
  checkpoints/                  ModelCheckpoint
  logs/                         CSVLogger
  predictions/predictions.csv   PredictionWriter (test-set inference)
  metadata/                     RunLogger: git diff, pip freeze, env, command
  config.resolved.json          argparse values for the run
```

`<run_id>` defaults to a timestamp; override with `--run-id` to resume into
the same directory. `<run_name>` defaults to `baseline`; pass `--run-name`
to label sweeps.

## Useful training flags

```bash
python -m src.modules.models.camchex.train_camchex \
    --backbone-name convnext_small.fb_in22k_ft_in1k_384 \
    --image-size 384 --batch-size 4 --num-workers 4 \
    --fusion-num-layers 2 --fusion-nhead 8 \
    --accumulate-grad-batches 16 \
    --precision 16-mixed --accelerator auto --devices auto \
    --val-check-interval 0.25 \
    --no-ema --no-prediction-writer    # disable optional callbacks
```

Full list: `python -m src.modules.models.camchex.train_camchex --help`.

## Adding a new model

1. Create `src/modules/models/<name>/`.
2. Build an assembly (`<name>.py`) wiring whichever encoders/decoders/fusion
   you want from `src/modules/`.
3. Subclass `pl.LightningModule` in `lightning_module.py`.
4. Copy `train_camchex.py` to `train_<name>.py` and adjust imports + argparse.
5. Train with `python -m src.modules.models.<name>.train_<name> ...`.
   Outputs land in `output/<name>/runs/...` automatically.

Encoder, decoder, fusion, and head modules all accept a `name=` kwarg.
The `RunLoggerCallback` groups grad norms by `component_name`, so logs stay
stable across architecture swaps (e.g. `grad_norm/image_encoder_frontal`
regardless of which timm backbone is wired in).

## Device portability

The training stack runs on CUDA, ROCm (presents as CUDA in PyTorch), MPS,
or CPU — pick via `--accelerator`. Note: 16-mixed precision is flaky on MPS;
fall back to `--precision 32` there.

## Legacy

The `camchex/` directory contains the original paper code: a LightningCLI
+ YAML config layout with its own data pipeline (`camchex/data/01,02,03_*.py`)
against the full MIMIC tree. It is no longer used by the new training
entry but kept for reference and comparison.

## TCIA download helper

```text
https://github.com/ygidtu/NBIA_data_retriever_CLI
```

```bash
./NBIA_data_retriever_CLI -i your_manifest.tcia -s ./output -p 4
```
