# Chest-Medical-Image-Fusion-for-Multimodal-Clinical-Analysis

This repository reproduces and extends CaMCheX, a multimodal chest X-ray
classification model that combines images, clinical text, and ED observations.

## Setup

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
source miniforge3/bin/activate

conda create -n camchex python=3.13 -y
conda activate camchex
conda install -c conda-forge p7zip -y

pip install uv
uv pip install -r requirements.txt
```

Training commands run from `camchex/`. The baseline config lives at
`configs/baseline.yaml`; machine-local overrides live at
`camchex/config.local.yaml`.

```bash
cp camchex/config.local.yaml.example camchex/config.local.yaml
```

Edit `camchex/config.local.yaml` for local GPU count, batch size, workers, and
optional run settings.

## Configs And Run Names

The current baseline model uses:

```yaml
run:
  model_name: camchex
  name: baseline
  output_root: null
```

`run.model_name` is the model family and output namespace. With the baseline
settings, runs write to:

```text
output/camchex/runs/<timestamp>-baseline/
```

For a new model family, such as an encoder-swapped model, create a model config
with a different name:

```yaml
run:
  model_name: camchex-swin
  name: swapped-image-encoder
```

That writes to:

```text
output/camchex-swin/runs/<timestamp>-swapped-image-encoder/
```

Keep model identity in model configs, and keep machine/runtime details in
`camchex/config.local.yaml`.

## Readiness Checks

Run these from the repo root before a real training attempt:

```bash
python -m py_compile \
  camchex/main.py \
  camchex/callbacks/run_logger.py \
  camchex/model/wrapper.py \
  camchex/model/models.py \
  camchex/model/loss.py \
  camchex/dataset/datamodule.py \
  camchex/dataset/dataset.py

python -c "import yaml; yaml.safe_load(open('configs/baseline.yaml')); print('config ok')"
bash -n camchex/train.sh
python camchex/main.py fit --config configs/baseline.yaml --print_config
```

`--print_config` only checks LightningCLI parsing. It prints the static config
before the run directory is created, so it will not show the final rewritten
checkpoint/log paths.

## Smoke Test

Use a tiny local override first:

```yaml
# camchex/config.local.yaml
trainer:
  devices: 1
  fast_dev_run: true

data:
  dataloader_init_args:
    batch_size: 1
    num_workers: 0
    persistent_workers: false
```

Then run:

```bash
cd camchex
python main.py fit --config ../configs/baseline.yaml --config config.local.yaml
```

This verifies model construction, dataloading, forward pass, loss, validation,
logging, and checkpoint path wiring. It still requires the dataset symlinks and
CSV files described in `camchex/AGENTS.md`.

## Full Training

```bash
cd camchex
bash train.sh
```

To run a different model config without editing `train.sh`:

```bash
cd camchex
CAMCHEX_CONFIG=../configs/models/camchex-swin.yaml bash train.sh
```

Each run gets a fresh directory containing:

```text
config.resolved.yaml
checkpoints/
logs/
metadata/
predictions/
```

Metadata includes the command line, package versions, git commit/status/diff,
environment details, and `pip-freeze.txt`. Metrics include train loss,
validation loss, validation AP/AUROC summaries, per-class validation AP/AUROC,
learning rate, global grad norm, and per-module grad norms.

## Resume, Validate, Predict

Resume is explicit so new runs do not overwrite old ones:

```bash
cd camchex
python main.py fit --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/last.ckpt
```

Validate:

```bash
cd camchex
python main.py validate --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/last.ckpt
```

Predict:

```bash
cd camchex
python main.py predict --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/<best>.ckpt
```

## TCIA Download Helper

Install the NBIA Data Retriever CLI from:

```text
https://github.com/ygidtu/NBIA_data_retriever_CLI
```

Example:

```bash
./NBIA_data_retriever_CLI -i your_manifest.tcia -s ./output -p 4
```
