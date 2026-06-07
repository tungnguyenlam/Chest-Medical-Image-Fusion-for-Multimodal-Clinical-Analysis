# Prior-Aware CaMCheX (CXR-BERT)

Same model and data contract as [`training/prior_aware/`](../prior_aware/) (current +
nearest-prior study, shared encoders, time-delta, prior-label token, prior report
token), but the text encoder is **`microsoft/BiomedVLP-CXR-BERT-specialized`** instead
of BioBERT. See the [`prior_aware` README](../prior_aware/README.md) for the full
architecture and per-token breakdown.

## Quick start

```bash
# 0. build the shared prior-aware parquet once (no --tokenizer; tokenized at load)
python src/prepare/04_build_prior_aware_dataset.py

# 1. train (CXR-BERT frozen via the text-embedding cache; EMA + single-cosine)
python training/prior_aware_cxrbert/prior_aware_train.py \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# 2. eval (two passes: full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_cxrbert/prior_aware_eval.py \
  --checkpoint-path <ckpt> --use-precomputed-text-embeddings
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0; don't
`--resume-from` an EMA checkpoint.

## Grad-CAM / Attribution

Same base `PriorAwareCaMCheXModel` as `training/prior_aware/`, so attribution works
identically — it declares `gradcam_runner_module = "src.interpret.run_prior_gradcam"` and
emits per-class current + prior panels (image, clinical/observation text, prior report,
prior-label per-class bar, time-delta token, current-vs-prior modality share). This variant
uses observation text rather than numeric vitals, so it writes `cur_obs`/`prv_obs` panels in
place of `vitals`. See [`training/prior_aware/README.md`](../prior_aware/README.md#grad-cam--attribution)
for the full description. Standalone:

```bash
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_cxrbert/config.yaml \
  --checkpoint-path output/prior_aware_cxrbert/runs/<run>/checkpoints/best.pt \
  --split val --scan-limit 800
```
