# Discriminative learning rate — pretrained encoders vs. fresh head

## The idea in one sentence

The image backbone and the text encoder arrive **pretrained** (their features are
already useful), while the fusion transformer and the MLDecoder head are **freshly
initialised** (random). Training them at one shared learning rate forces a bad
compromise — high enough for the fresh head destabilises the pretrained encoders, low
enough for them starves the head — so we give each pretrained encoder a **lower LR**
than everything else.

## The global defaults

Every model gets this automatically, no per-config setup required:

$$
\text{lr}_{\text{backbone}} = 0.3 \times \text{lr}_{\text{base}}, \qquad
\text{lr}_{\text{text}} = 0.1 \times \text{lr}_{\text{base}}
$$

So with the repo's base `lr: 3.0e-5`, the image backbone trains at `9.0e-6`, the text
encoder trains at `3.0e-6`, and everything else (head/decoder, fusion transformer,
projections including `text_proj`, vitals) trains at `3.0e-5`.

The two prefixes are the universal attribute names across the model variants:

- **Backbone** = any parameter under `image_encoder.` (the per-view `frontal_encoder` /
  `lateral_encoder` live *under* `image_encoder`).
- **Text** = any parameter under `text_encoder.` (the shared `CaMCheXTextEncoder`, which
  wraps BioBERT/CXR-BERT). The freshly-initialised `text_proj` head sits *outside* this
  prefix and stays at the base LR.

Frozen params (`requires_grad=False`, e.g. a frozen text encoder) are excluded from
every group, so the lower LR only touches *trainable* encoder params.

The defaults live in one place,
[`DEFAULT_BACKBONE_LR_MULT` / `DEFAULT_TEXT_LR_MULT`](../src/optimizer/AdamWOptimizer.py),
and are applied together in `build_adamw_optimizer`.

### Why 0.3 for the backbone and 0.1 for the text encoder?

ImageNet→chest-X-ray is a real domain shift, so the backbone genuinely needs to
*adapt*, not just be nudged. `0.1×` (the common "barely move it" default) tends to
under-train the backbone for medical transfer; `0.3×` protects the pretrained structure
while still letting it learn.

The text encoder is a much larger, more delicate language model and the clinical text is
closer to its pretraining domain than chest X-rays are to ImageNet, so it needs less
adaptation and is easier to wreck with a high LR — hence the gentler `0.1×`. Both are
single knobs — tune them if a run says otherwise.

## How to change it

In precedence order (first match wins):

1. **CLI, per run:** `--backbone-lr-mult 0.5` and/or `--text-lr-mult 0.2` (set either to
   `1.0` to disable just that component and train it at the base LR). The two compose
   independently. Best for sweeps — no config edit.
2. **Config, per model:** under `model.optimizer_init_args`:
   ```yaml
   model:
     optimizer_init_args:
       backbone_lr_mult: 0.5        # or 1.0 to disable
       text_lr_mult: 0.2            # or 1.0 to disable
   ```
3. **Full manual control:** give explicit absolute per-prefix LRs and **both** the
   backbone and text defaults are skipped entirely — so re-state every prefix you want
   lowered (longest-prefix wins):
   ```yaml
   model:
     optimizer_init_args:
       param_group_lrs:
         "image_encoder.": 1.0e-5
         "image_encoder.lateral_encoder.": 5.0e-6   # finer-grained override
         "text_encoder.": 3.0e-6
   ```

## What you'll see at startup

```
[optimizer] discriminative LR groups -> image_encoder. lr=9.00e-06/wd=0.01: <N> params, ... text_encoder. lr=3.00e-06/wd=0.01: <K> params, ... base lr=3.00e-05/wd=0.01: <M> params, ...
```

If *neither* prefix matches any trainable param (e.g. a backbone renamed away from
`image_encoder.` and a fully-frozen text encoder), you'll instead see a one-line warning
that nothing matched and all trainable params train at the base LR — so a
silently-uniform LR is never mistaken for the intended split.

## How it interacts with the LR schedule

Each param group's **initial** LR is captured as its `base_lr` when the scheduler is
built, and the scheduler scales every group by the same factor relative to its own
`base_lr`. So the backbone stays proportionally lower (≈0.3×) and the text encoder
(≈0.1×) through warmup, cosine decay, and any warm restarts — the ratios are preserved
end to end. Per-group LRs are also logged to their own `train/lr/image_encoder.` and
`train/lr/text_encoder.` columns in `train_steps.csv`.

## Checkpoint compatibility (important)

Turning this on changes the optimizer's **param-group layout** (now a base group, an
`image_encoder.` group, and a `text_encoder.` group, each further split decay / no-decay
— up to 6 groups instead of 2; fewer if a component is frozen or disabled). Consequences:

- **Weights-only init (`--checkpoint-path`)** is unaffected — it never loads optimizer state.
- **Full resume (`--resume-from`)** requires a *matching* group layout. A checkpoint
  written **before** this change (2 groups, or 4 before the text split was added) cannot
  be `--resume-from`'d into a run that now builds more groups — PyTorch rejects the
  mismatched group count. New runs write the new layout and resume cleanly within it.
  (EMA checkpoints are already blocked from full resume for a separate reason — see
  [docs/ema.md](ema.md).)

See also [training/FLAGS.md](../training/FLAGS.md) for the `--backbone-lr-mult` and
`--text-lr-mult` flags.
