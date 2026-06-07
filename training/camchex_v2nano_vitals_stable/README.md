# camchex_v2nano_vitals_stable

## Quick start

```bash
# train (single-cosine + EMA tail; CXR-BERT frozen via the text-embedding cache)
python training/camchex_v2nano_vitals_stable/camchex_v2nano_vitals_stable_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# eval (two passes: full vs. clinical-indication dropped -> *.no_report.{csv,json})
python training/camchex_v2nano_vitals_stable/camchex_v2nano_vitals_stable_eval.py \
  --checkpoint-path <ckpt> --use-precomputed-text-embeddings
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0. Don't
`--resume-from` an EMA checkpoint.

Stable fine-tuning-tail variant of [`camchex_v2nano_vitals`](../camchex_v2nano_vitals/).
**Same model** (`CaMCheXV2NanoVitalsModel`) — this directory differs only in the
**learning-rate schedule** and **weight averaging**. It exists to push the converged
baseline toward the best achievable score and to serve as a clean scheduler ablation.

## Motivation: the warm-restart tail collapse

The original CaMCheX code ([`camchex/model/wrapper.py`](../../camchex/model/wrapper.py))
and our faithful port ([`src/scheduler/WarmupCosineScheduler.py`](../../src/scheduler/WarmupCosineScheduler.py))
use `CosineAnnealingWarmRestarts(T_0=steps_per_epoch, T_mult=1)`. That makes the LR
sweep `lr → eta_min` **and restart to `lr` every epoch** — a per-epoch sawtooth
(visible in any LR-schedule plot of a baseline run).

While the model is still improving, each restart lands back in a good basin. But once
the model has **converged** (here, ~epoch 7, val_ap ≈ 0.486), the loss landscape is
flat and the full-amplitude restart at the next epoch boundary kicks it out of the good
basin into an **equal-loss but worse-AP** minimum. Because the optimizer minimizes ASL
(not AP) and ASL is dominated by easy negatives, the scalar loss barely moves (~0.053)
while AP collapses (0.486 → 0.318) and **does not recover** — every subsequent restart
re-kicks it. The standard SGDR remedy is to keep the *best* checkpoint, not the last.

## What this variant changes

- **`schedule: single_cosine`** — one `CosineAnnealingLR` over the whole run (after
  warmup): a single monotone `lr → eta_min` decay, no restarts, so the model settles
  into its minimum in the tail.
- **`ema: true` (decay 0.999)** — an exponential moving average of the weights is the
  *evaluated and saved* model (see `ModelEMA` in [`training/common.py`](../../training/common.py)).
  Weight averaging buys a small, free gain and specifically needs the stable
  (non-restarting) tail — averaging across sawtooth snapshots would mix unrelated basins.
- Modest **`lr: 1.0e-5`** and **`max_epochs: 10`** — tuned for *warm-starting* from the
  baseline's best checkpoint rather than training from scratch.

## How to run (recommended: warm-started tail)

Warm-start from the baseline's best checkpoint as a **weights-only init** (fresh
optimizer/scheduler/epoch — note `--checkpoint-path`, **not** `--resume-from`):

```bash
python training/camchex_v2nano_vitals_stable/camchex_v2nano_vitals_stable_train.py \
  --checkpoint-path output/camchex_v2nano_vitals/runs/<baseline-run>/checkpoints/epoch_007.pt \
  --use-precomputed-text-embeddings \
  --batch-size 16 --num-workers 4 --prefetch-factor 1 --val-batch-size 16
```

Evaluate the best checkpoint (its `model_state_dict` is already the EMA weights):

```bash
python training/camchex_v2nano_vitals_stable/camchex_v2nano_vitals_stable_eval.py \
  --checkpoint-path output/camchex_v2nano_vitals_stable/runs/<run>/checkpoints/epoch_XXX.pt
```

### From-scratch run instead

Set `lr: 3.0e-5` and `trainer.max_epochs: 30` in `config.yaml` and launch without
`--checkpoint-path`. This is the clean apples-to-apples scheduler ablation vs. the
warm-restart baseline (same init, same step 0, only the schedule differs).

## Caveats

- **Do not `--resume-from` an EMA checkpoint.** The saved weights are the EMA snapshot,
  while the optimizer state belongs to the raw weights — the two are inconsistent for a
  full resume. The training guard also blocks `--resume-from` across a schedule change
  (`warm_restarts` ↔ `single_cosine`); use `--checkpoint-path` for weights-only init.
- The scheduler change is shared infrastructure, but defaults preserve the baseline:
  `schedule` defaults to `warm_restarts` and `ema` defaults to off everywhere else.
