# EMA (Exponential Moving Average of weights) — what it is and why our checkpoints are the way they are

## The idea in one sentence

While the model trains, we keep a **second, shadow copy of the weights** that is a
slowly-moving average of the weights the optimizer actually updates — and *that*
averaged copy is what we evaluate and save, because it's usually a bit better and
a lot less jittery than any single training step.

This note explains exactly what the average is, what it buys us, and — most
importantly for the checkpointing work — **why an EMA checkpoint is eval-ready
but not safe to resume training from.** No prior background needed.

---

## 1. The two sets of weights

During an EMA run there are always *two* copies of every parameter:

- **Raw weights** $\theta_t$ — the ones the optimizer steps on, batch after batch.
  These are what backprop updates; they bounce around a lot.
- **Shadow weights** $\tilde\theta_t$ — the EMA. After every optimizer step we nudge
  the shadow a tiny fraction of the way toward the raw weights:

$$
\tilde\theta_t = d\,\tilde\theta_{t-1} + (1-d)\,\theta_t
$$

with a decay $d$ close to 1 (e.g. $0.999$). So the shadow is a low-pass-filtered
trajectory of training: it ignores the per-step noise and tracks where the raw
weights are *on average* heading.

In code this is exactly [`ModelEMA.update`](../training/utils/model.py):

```python
s.mul_(d).add_(v.detach(), alpha=1.0 - d)   # s = shadow param, v = raw param
```

Floating-point params get the moving average; integer/bool buffers (e.g.
`num_batches_tracked`) just copy the latest value, since averaging an integer
counter is meaningless.

### Why average at all?

SGD/AdamW near convergence doesn't sit at the minimum — it rattles around it in a
noisy ball whose size is set by the learning rate. Any single snapshot is a random
point in that ball. Averaging recent snapshots lands you **closer to the center of
the ball** than almost any individual point, which typically generalizes better.
It's the cheap, standard "free accuracy" trick (a.k.a. Polyak/weight averaging).

### The one precondition: a stable LR tail

Averaging only helps if the recent snapshots are all rattling around the **same**
basin. If the learning rate keeps restarting (warm restarts / cosine sawtooth), the
weights jump between *different* regions, and averaging across them mixes unrelated
solutions — worse than either. This is why EMA in this repo is paired with the
**single-cosine** schedule, not warm restarts (see
[`project_scheduler_warmrestart_collapse`] in memory and the
[`camchex_v2nano_vitals_stable`] variant). The `ModelEMA` docstring spells this
out too.

---

## 2. How EMA is used at eval and save time

The decision in this repo: **the EMA weights are the ones we validate and save.**
The raw weights only exist to keep training.

So around every validation / checkpoint, the trainer does a three-step swap
([`_save_training_checkpoint`](../training/utils/train.py), and `apply_to` /
`restore` in [`ModelEMA`](../training/utils/model.py)):

1. `ema.apply_to(model)` — stash the raw weights in a backup, load the **shadow**
   into the live model.
2. validate / `torch.save(...)` — so the metrics, the early-stop/best tracking, and
   the saved `model_state_dict` all reflect the **smoothed** model.
3. `ema.restore(model)` — put the **raw** weights back so training continues from
   where it was.

That's why `weights_are_ema: True` is stamped into the checkpoint payload — a flag
saying "the weights in this file are the smoothed snapshot, not the live training
weights."

---

## 3. Why an EMA checkpoint is eval-ready but **not** resume-safe

This is the crux for the checkpoint redesign.

A full training resume (`--resume-from`) restores three things and expects them to
**belong to each other**:

- `model_state_dict`  — the weights
- `optimizer_state_dict` — AdamW's first/second moment estimates ($m$, $v$) **for those exact weights**
- scheduler / scaler state

AdamW's moments are statistics *about the raw-weight trajectory* — they encode "how
has each raw parameter been moving recently." But in an EMA checkpoint we saved the
**shadow** weights as `model_state_dict` while the optimizer state still describes
the **raw** weights. They don't match. Resuming would apply momentum/variance
computed for one set of weights on top of a different set of weights — corrupting
the optimization.

On top of that, **the EMA shadow itself is never saved** today. So even if we
wanted to resume "correctly," the checkpoint is missing the shadow we'd need to keep
averaging into.

That's exactly why the recent guard
([`load_training_checkpoint`](../training/utils/train.py), commit `21e8f69`)
**refuses** `--resume-from` on a `weights_are_ema` checkpoint and tells you to use
`--checkpoint-path` (weights-only init, fresh optimizer/scheduler) instead.

### What each mode is good for

| Want to…                              | Use                | Works with EMA checkpoint? |
|---------------------------------------|--------------------|----------------------------|
| Evaluate / Grad-CAM / report numbers  | the saved weights  | ✅ yes — they're the point  |
| **Re-train** starting from these weights (fresh optimizer) | `--checkpoint-path` | ✅ yes |
| **Resume** mid-run after a crash (keep optimizer + schedule) | `--resume-from` | ❌ refused — moments mismatch |

---

## 4. What a *truly* resume-safe EMA checkpoint would need

For "continue training after a crash" to actually work under EMA, the resume
checkpoint must contain a self-consistent set:

- **raw weights** $\theta_t$ (i.e. saved *without* `ema.apply_to` — the live training weights),
- `optimizer_state_dict` — its moments match those raw weights,
- scheduler + scaler state,
- **the EMA shadow** $\tilde\theta_t$ — so averaging continues instead of restarting from the raw weights.

That requires giving `ModelEMA` a `state_dict()` / `load_state_dict()` for its
shadow, saving raw-not-EMA weights in the resume file, and relaxing the resume guard
for *that* flavor of checkpoint. The eval/`best` checkpoint can stay as it is today
(EMA snapshot, `weights_are_ema: True`).

This is the design split behind the planned checkpoint change:

- **`best.pt`** → EMA snapshot, eval-ready (what you report and Grad-CAM).
- **`last.pt`** → raw weights + optimizer + scheduler + scaler + EMA shadow → genuinely resumable.
- **`epoch_NNN.pt`** → weights-only (EMA snapshot, no optimizer) → cheap, for post-hoc per-epoch Grad-CAM.
