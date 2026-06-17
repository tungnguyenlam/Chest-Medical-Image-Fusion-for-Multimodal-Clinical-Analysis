# prior_aware_v5nano_bgpenalty

`prior_aware_v5nano` + a **background-attention penalty**: a small auxiliary loss
that discourages the current-image encoder from drawing evidence from the
*outside-patient* image margins (collimation bars, corners, burned-in text) — the
"Clever Hans" shortcut we saw lighting up some Grad-CAMs.

The model architecture is unchanged — this reuses `PriorAwareV5NanoModel` with one
new constructor knob, `background_penalty_lambda`. With the lambda at `0.0` it is
*bit-for-bit* the base v5; this family just sets it `> 0` and turns on the mask
emission in the dataloader.

## How it works (full math: [docs/background_attention_penalty.md](../../docs/background_attention_penalty.md))

1. **Mask.** For each current view the dataset computes a *confident-background*
   weight `M ∈ [0,1]` via `src/dataloader/body_mask.py` — a deliberately
   conservative margin/letterbox detector (border-connected, extreme-valued pixels
   inside a thin outer band). It is computed from the raw channel and pushed
   through the **same augmentation transform** as the image, so it stays aligned
   under rotation/crop/flip. Stored at 32×32 per view in `data["bg_mask"]`.
2. **Penalty.** The model average-pools `M` to its encoder grid (so each cell gets
   the *fraction* of it that is background), then penalizes the pre-LayerNorm
   feature energy there:
   `L_bg = Σ M̃·‖F‖² / Σ M̃`. Anatomy cells have `M̃ = 0`, so they are
   mathematically untouchable.
3. **Loss.** `L = L_cls + λ·L_bg`. `train_step` adds the (already λ-scaled) term
   the model returns alongside its logits.

## Why λ is small (default `0.01`)

The mask is near-perfect on the black-bar/collimation side but has **rare false
positives** on saturated edge tissue (chin/neck/upper abdomen at the frame). A
gentle λ steadily discourages the shortcut where the mask is right, without
punishing the model hard enough to learn "ignore the lower chest" where it's
wrong. Tune by watching: (a) val AUC holds — especially off-lung classes
(Support Devices, Pneumomediastinum, Subcutaneous Emphysema, Fracture) — and
(b) Grad-CAMs stop firing on the corners.

## Config knobs

- `model.model_init_args.background_penalty_lambda` — penalty strength (`0.0` = off).
- `data.datamodule_cfg.compute_bg_mask` — must be `true` when lambda `> 0`.
- `data.datamodule_cfg.bg_mask_cfg` — optional `BodyMaskConfig` overrides
  (e.g. `{bright_frac: 0.99}` to tighten the white gate).

## Run

```bash
# train
python -m training.prior_aware_v5nano_bgpenalty.prior_aware_train \
    --config training/prior_aware_v5nano_bgpenalty/config.yaml

# eval (penalty auto-disabled; identical inference path to base v5)
python -m training.prior_aware_v5nano_bgpenalty.prior_aware_eval \
    --config training/prior_aware_v5nano_bgpenalty/config.yaml \
    --checkpoint-path <run>/checkpoints/best.pt
```

## Sanity-check the mask first

```bash
python -m scripts.visualize_body_mask \
    --config training/prior_aware_v5nano_bgpenalty/config.yaml \
    --split val --num 16 --out output/body_mask_check
```
