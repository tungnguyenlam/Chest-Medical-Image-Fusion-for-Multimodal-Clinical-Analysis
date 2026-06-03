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
                     ConvNeXtV2 Nano + frozen CXR-BERT + numeric vitals variant
  singleview/        train/eval entrypoints for single-view image models
  common.py          plain PyTorch train/eval helpers
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
