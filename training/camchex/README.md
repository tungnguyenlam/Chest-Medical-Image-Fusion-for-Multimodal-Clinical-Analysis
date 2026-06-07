# CaMCheX (BioBERT multimodal)

The original multimodal CaMCheX: up to 4 CXR views + clinical indication + vitals
(rendered as text) fused via a shared **BioBERT** text encoder and a segment-aware
transformer, with an ML-Decoder head over the 26 CXR-LT classes.

## Quick start

```bash
# train (live BioBERT text encoder; EMA + single-cosine)
python training/camchex/camchex_train.py --ema --batch-size 4 --num-workers 4

# eval (two passes: full vs. clinical-indication dropped -> *.no_report.{csv,json})
python training/camchex/camchex_eval.py --checkpoint-path <ckpt>
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0. Don't
`--resume-from` an EMA checkpoint. The CXR-BERT-text variant lives in
[`training/camchex_cxrbert/`](../camchex_cxrbert/); the lighter ConvNeXtV2-Nano +
numeric-vitals successors are [`camchex_v2nano_vitals`](../camchex_v2nano_vitals/) and
[`camchex_v3nano`](../camchex_v3nano/).
