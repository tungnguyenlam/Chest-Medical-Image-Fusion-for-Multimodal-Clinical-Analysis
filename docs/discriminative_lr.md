# Discriminative learning rate — backbone vs. decoder

## The idea in one sentence

The image backbone arrives **pretrained** (its features are already useful), while
the fusion transformer and the MLDecoder head are **freshly initialised** (random).
Training them at one shared learning rate forces a bad compromise — high enough for
the fresh head destabilises the backbone, low enough for the backbone starves the
head — so we give the backbone a **lower LR** than everything else.

## The global default

Every model gets this automatically, no per-config setup required:

$$
\text{lr}_{\text{backbone}} = \text{mult} \times \text{lr}_{\text{base}}, \qquad \text{mult} = 0.3
$$

So with the repo's base `lr: 3.0e-5`, the image backbone trains at `9.0e-6` and
everything else (head/decoder, fusion transformer, projections, vitals, text
encoder if unfrozen) trains at `3.0e-5`.

"Backbone" = any parameter whose name starts with `image_encoder.` — the universal
prefix across the model variants (the per-view `frontal_encoder` / `lateral_encoder`
live *under* `image_encoder`). Frozen params (`requires_grad=False`) are excluded
from every group, so the lower LR only touches the trainable backbone params.

The default lives in one place,
[`DEFAULT_BACKBONE_LR_MULT`](../src/optimizer/AdamWOptimizer.py), and is applied in
`build_adamw_optimizer`.

### Why 0.3 and not 0.1?

ImageNet→chest-X-ray is a real domain shift, so the backbone genuinely needs to
*adapt*, not just be nudged. `0.1×` (the common "barely move it" default) tends to
under-train the backbone for medical transfer; `0.3×` protects the pretrained
structure while still letting it learn. It's one knob — tune it if a run says otherwise.

## How to change it

In precedence order (first match wins):

1. **CLI, per run:** `--backbone-lr-mult 0.5` (or `1.0` to disable and train everything
   at the base LR). Best for sweeps — no config edit.
2. **Config, per model:** under `model.optimizer_init_args`:
   ```yaml
   model:
     optimizer_init_args:
       backbone_lr_mult: 0.5        # or 1.0 to disable
   ```
3. **Full manual control:** give explicit absolute per-prefix LRs and the backbone
   default is skipped entirely (longest-prefix wins):
   ```yaml
   model:
     optimizer_init_args:
       param_group_lrs:
         "image_encoder.": 1.0e-5
         "image_encoder.lateral_encoder.": 5.0e-6   # finer-grained override
   ```

## What you'll see at startup

```
[optimizer] discriminative LR groups -> image_encoder. lr=9.00e-06/wd=0.01: <N> params, ... base lr=3.00e-05/wd=0.01: <M> params, ...
```

If a model's backbone is named something other than `image_encoder.` (or is fully
frozen), you'll instead see a one-line warning that nothing matched and all trainable
params train at the base LR — so a silently-uniform LR is never mistaken for the
intended split.

## How it interacts with the LR schedule

Each param group's **initial** LR is captured as its `base_lr` when the scheduler is
built, and the scheduler scales every group by the same factor relative to its own
`base_lr`. So the backbone stays proportionally lower (≈0.3×) through warmup, cosine
decay, and any warm restarts — the ratio is preserved end to end. Per-group LRs are
also logged to their own `train/lr/image_encoder.` columns in `train_steps.csv`.

## Checkpoint compatibility (important)

Turning this on changes the optimizer's **param-group layout** (now a base group and
an `image_encoder.` group, each further split decay / no-decay — up to 4 groups
instead of 2). Consequences:

- **Weights-only init (`--checkpoint-path`)** is unaffected — it never loads optimizer state.
- **Full resume (`--resume-from`)** requires a *matching* group layout. A checkpoint
  written **before** this change (2 groups) cannot be `--resume-from`'d into a run
  that now builds 4 groups — PyTorch rejects the mismatched group count. New runs
  write the new layout and resume cleanly within it. (EMA checkpoints are already
  blocked from full resume for a separate reason — see [docs/ema.md](ema.md).)

See also [training/FLAGS.md](../training/FLAGS.md) for the `--backbone-lr-mult` flag.
