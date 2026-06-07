# Single-view image baseline

Image-only CaMCheX baseline: one CXR through a single timm backbone + ML-Decoder, no
text / vitals / prior. Useful as the bottom-of-the-table reference for the multimodal
and prior-aware variants. `--view-position` filters to `all` / `frontal` / `lateral`.

## Quick start

```bash
# train (image only; EMA + single-cosine)
python training/singleview/singleview_train.py --ema --batch-size 8 --num-workers 4

# eval (single pass -- no text, so no report-ablation)
python training/singleview/singleview_eval.py --checkpoint-path <ckpt> --view-position all
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0. Don't
`--resume-from` an EMA checkpoint.
