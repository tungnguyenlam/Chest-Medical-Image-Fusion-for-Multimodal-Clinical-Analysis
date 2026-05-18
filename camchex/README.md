# CaMCheX

This is the official PyTorch Implementation of "Clinically-aligned Multi-modal Chest X-ray Classification". Accepted at ML4H 2025 in San Diego, USA.

<img src="images/CaMCheX.png" width="400">

Model weights to be released soon.

## Running on a shared server (avoiding OOM kills)

On a shared server with other users, the Linux OOM killer can pick your training process when someone else's job spikes memory. Linux doesn't let you truly "reserve" RAM, but you can make yourself a less attractive victim — lower your `oom_score_adj` so the noisy neighbour is picked instead:

```bash
# Launch training with a lowered OOM score (no root required for moderate values).
choom -n -500 -- python main.py fit --config ../configs/baseline.yaml --config config.local.yaml
```

`choom` ships with `util-linux`. The score range is -1000 (immune, root-only) to +1000 (kill me first); non-root users can practically go down to about -500. Children inherit the score, so this also covers dataloader workers.

If `choom` isn't installed:

```bash
echo -500 | sudo tee /proc/$$/oom_score_adj
python main.py fit --config ../configs/baseline.yaml --config config.local.yaml
```

Two complementary mitigations worth knowing about, in case score-lowering alone isn't enough:

- **Add swap.** Without swap the OOM killer fires the instant any allocation can't be served; with swap on NVMe, the kernel pages cold memory out first and training slows under pressure instead of dying. `sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`.
- **cgroups v2 (if available).** `systemd-run --user --scope -p MemoryMin=12G -p MemoryMax=14G python main.py fit ...` reserves a memory floor against other cgroups and caps your own usage. Needs user-level cgroup delegation enabled by the admin.

## After `python main.py fit`

Each training run writes to a unique directory under
`../output/<run.model_name>/runs/`. The original model uses
`run.model_name: camchex`, so its runs go under `../output/camchex/runs/`.
Set a new `run.model_name` for architecture variants such as encoder-swapped
models. This prevents a new training run from overwriting checkpoints,
predictions, logs, or resolved configs from an older run.

The training run produces two checkpoints under the run's `checkpoints/` folder:

- `last.ckpt` — most recent, regardless of metric.
- `epoch=XX-val_loss=…-val_ap=….ckpt` — best by `val_ap`.

Typical follow-up commands (run from `camchex/`):

```bash
# Validate a saved checkpoint on the development split.
python main.py validate --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/last.ckpt

# Generate predictions on the test split. Writes into the new run's predictions/ folder.
# (sigmoid probabilities, with horizontal-flip TTA averaged in).
python main.py predict --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/<best>.ckpt

# Resume training from the best checkpoint after a crash or to extend epochs.
python main.py fit --config ../configs/baseline.yaml --config config.local.yaml \
  --ckpt_path ../output/camchex/runs/<run>/checkpoints/last.ckpt
```

`predictions.csv` is a study-level CSV with one column per class. Scoring (per-class AP, mAP, head/medium/tail breakdowns) is not yet wrapped into a script — join `predictions.csv` against the labels in `../data/data-camchex/03_mimic_test.csv` and compute `sklearn.metrics.average_precision_score` per class.

For first-attempt runs on modest hardware, override `trainer.max_epochs` (e.g. to 5) in `config.local.yaml` rather than running the paper-grade 1000 epochs — enough to confirm the pipeline works and val_ap is climbing above random.
