# CaMCheX (CXR-BERT text encoder)

Same model and data contract as [`training/camchex/`](../camchex/) (4 views +
clinical indication + vitals-as-text → segment-aware fusion → ML-Decoder), but the
text encoder is **`microsoft/BiomedVLP-CXR-BERT-specialized`** instead of BioBERT
(set in this directory's `config.yaml`).

## Quick start

```bash
# train (live CXR-BERT text encoder; EMA + single-cosine)
python training/camchex_cxrbert/camchex_train.py --ema --batch-size 4 --num-workers 4

# eval (two passes: full vs. clinical-indication dropped -> *.no_report.{csv,json})
python training/camchex_cxrbert/camchex_eval.py --checkpoint-path <ckpt>
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0. Don't
`--resume-from` an EMA checkpoint.
