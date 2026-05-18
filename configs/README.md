# Experiment configs

`baseline.yaml` is the canonical config for the original CaMCheX experiment.
Active training code lives in `src/camchex/` and is launched from the repo root.
For compatibility with the original CSV/image paths, the training CLI changes
its working directory to the legacy `camchex/` folder before data/model
construction. Paths inside these configs therefore still behave like the old
CaMCheX training paths unless the code explicitly resolves them otherwise.

Typical use:

```bash
bash train.sh
```

To run a different config without editing `train.sh`:

```bash
CAMCHEX_CONFIG=configs/experiments/my_experiment.yaml bash train.sh
```

Machine-specific overrides still live in `camchex/config.local.yaml`, which is
gitignored and excluded from rsync. Override that path with
`CAMCHEX_LOCAL_CONFIG` only when you need a temporary local config.

`run.model_name` is the artifact namespace for a model family. The original
paper baseline uses `camchex`, so its default output root is
`../output/camchex/runs`. If you train a related model with a swapped encoder,
give it a new model name:

```yaml
run:
  model_name: camchex-swin
  name: baseline-swapped-encoder
```

Each training run gets a fresh directory under `../output/<model_name>/runs`
unless `run.output_root` is set explicitly, for example:

```text
../output/camchex/runs/20260518-113000-baseline/
  config.resolved.yaml
  checkpoints/
  logs/
  metadata/
  predictions/
```

Set `run.output_root` in `config.local.yaml` only if you want to override that
derived location entirely. Set `run.log_every_n_steps` to control step-level
loss, learning-rate logger cadence, and module gradient norm logging.
