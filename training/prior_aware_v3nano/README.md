# Prior-Aware CaMCheX v3 Nano

## Quick start

```bash
# 0. build the shared prior-aware parquet once (no --tokenizer; tokenized at load)
python src/prepare/04_build_prior_aware_dataset.py

# 1. train, warm-started from a trained camchex_v3nano (577/581 tensors transfer;
#    prior-only params init fresh). Drop --checkpoint-path to train standalone.
python training/prior_aware_v3nano/prior_aware_train.py \
  --checkpoint-path output/camchex_v3nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings --ema --batch-size 4 --num-workers 4

# 2. eval (two passes: full vs. CURRENT clinical-indication dropped; prior text kept)
python training/prior_aware_v3nano/prior_aware_eval.py \
  --checkpoint-path output/prior_aware_v3nano/runs/<RUN_ID>/checkpoints/<BEST>.pt \
  --use-precomputed-text-embeddings
```

Tune `--batch-size` / `--num-workers` to your GPU; leave `val_num_workers` at 0; don't
`--resume-from` an EMA checkpoint.

### Low host-RAM mode (`--text-embeddings-gpu-resident`)

With `--use-precomputed-text-embeddings` the embedding cache is a dict-of-numpy held in
RAM, and under `persistent_workers` every dataloader worker inherits a copy (~0.7 GB
each) — on a memory-tight box (e.g. 16–32 GB already under pressure) this can trigger the
Linux OOM killer (`Killed`, while VRAM is nearly idle). Add `--text-embeddings-gpu-resident`
(opt-in; requires `--use-precomputed-text-embeddings`) to instead keep the embeddings as a
single frozen `[N, 768]` table inside the model — moved to the training device once — while
the dataset emits int64 row indices that the model gathers on-device.

It also resolves every row→table-index up front (in the parent) and drops the raw-text
columns from the worker DataFrames. This matters under `persistent_workers`: otherwise each
`__getitem__` reads the raw clinical/report strings to hash them, and touching those Python
`str` objects forces copy-on-write page copies into every worker, so per-worker RAM climbs
across the epoch. With indices precomputed, workers touch only fork-stable numeric arrays,
so RAM stays flat and you can raise `--num-workers` without multiplying it. The table is a
non-persistent buffer, so checkpoints are unchanged and eval still runs without the flag.

Related RAM measures apply to every v3nano run, not just this flag:
- **The parquet loads with the pyarrow dtype backend.** `PriorAwareDataset` reads with
  `dtype_backend="pyarrow"`, so string/list columns (image paths, labels, vitals) are stored
  as immutable Arrow buffers outside the Python object graph rather than object columns of
  per-element `str`/`list`/`int`. Reading an Arrow cell under fork doesn't bump a per-element
  refcount, so the shared parquet pages stay clean and per-worker RSS no longer climbs across
  the epoch — the same copy-on-write fix as the text-index precompute above, generalized to
  every remaining column. (Null cells then surface as `pd.NA`, not `None`; the dataset's
  null-checks handle both.)
- **Dead `obs` streams are skipped.** v3nano feeds vitals numerically (`VitalsTokenProjector`),
  so the vitals-as-text `obs` / `prior_obs` streams are never read by the model. Whenever the
  embedding cache is active the dataset emits only `text_embedding_streams`
  (`clin`, `prior_clin`, `prior_report`) and skips `obs`, so those raw-text columns are never
  touched. (Token-mode / attribution runs, cache off, still emit all streams — the base
  `PriorAwareCaMCheXModel` consumes `obs`.)
- **`gc.freeze()` before fork.** The training entry point freezes the long-lived heap before
  the dataloader workers fork, so the cyclic GC doesn't dirty shared object pages (another
  copy-on-write RAM-growth source) during the run.
- **glibc malloc arenas are capped** (`--malloc-arena-max`, default 2) before the workers fork,
  so each worker doesn't grow its own arena set (up to ~8×ncpu) and fragment RSS.
- **`--uint8-image-pipeline`** (opt-in) ships images as uint8 [0,255] and dequantizes +
  normalizes on-device, cutting the per-batch host buffer, pinned staging, and H2D copy ~4×.
  Requires a channel mode; train-time augs then run on uint8 (a numerics change worth an
  ablation). Eval stays on the numerically identical float path. See
  [`TRAINING_FLAGS.md`](../TRAINING_FLAGS.md#model-specific-flags).

> **Most of the list above only helps when `--num-workers > 0`.** The pyarrow backend,
> text-index precompute, `gc.freeze()`, and arena capping all exist to stop *per-worker* RSS
> from climbing via copy-on-write fork. At `--num-workers 0` there are no workers, so they do
> nothing — you pay the full main-process footprint with none of the worker-side savings.
> Lowering `--num-workers` is therefore **not** a host-RAM lever; it just removes the
> optimizations' reason to exist. At `num_workers 0` the dominant consumers are the CUDA
> context + cuDNN/cuBLAS, the `torch.compile`/Inductor working set (a transient spike — hence
> `TORCHINDUCTOR_COMPILE_THREADS=1`), the two parquet DataFrames, and the brief embedding-table
> build (`np.stack` doubles the 748 MB table momentarily, freed after `model.to(device)`).

**Seeing where the RAM goes.** Every run prints `[mem]` host-RSS milestones at the points that
move the footprint — `loaders built`, `text embeddings loaded to RAM`, `after build_index_table`,
`model -> device`, and `after compile setup` — each as `RSS=X MB (peak Y MB)`, where peak is the
process high-water mark (`VmHWM`) so it captures transient spikes even after they're freed. The
training progress bar also carries a live `ram=RSS/peak` (GB) field next to `vram=`, so growth
across an epoch (or the compile spike on the first batch) is visible without extra tooling.

Single-token variant of [`prior_aware_v2nano`](../prior_aware_v2nano/). Same
prior-aware design (current + prior branches sharing the ConvNeXtV2-Nano image
router, CXR-BERT text encoder, numeric vitals projector, time-delta embedding, and
`Linear(26→768)` prior-label token) — implemented in
[`src/model/PriorAwareV3NanoModel.py`](../../src/model/PriorAwareV3NanoModel.py).

## The one change: single tokens instead of 64-wide blocks

`prior_aware_v2nano` repeats each text CLS vector across the 8×8 image grid (64
**identical** tokens per text stream) and emits 64 vitals tokens. v3nano collapses
each non-image signal to a **single token** — the prior-aware analogue of how
`camchex_v3nano` slims `camchex_v2nano_vitals`. Images are unchanged (still the
4-view × 8×8 = 256-token grid).

| stream | v2nano | v3nano |
|---|---|---|
| current clinical | 64 (1 CLS × 64 copies) | **1** |
| current vitals | 64 (`grid_size=8`) | **1** (`grid_size=1`) |
| prior clinical | 64 | **1** |
| prior report | 64 | **1** |
| prior vitals | 64 | **1** |
| prior label | 1 | 1 |

Sequence length at 512px drops from ~833 to **518** tokens:

```
current: 256 image + 1 clinical + 1 vitals                       = 258
prior:   256 image + 1 clinical + 1 report + 1 vitals + 1 label  = 260
total                                                            = 518
```

This is an ablation of whether the 64× text/vitals duplication in v2nano carries
any signal, at ~⅜ the fusion-sequence length (attention is O(n²), so cheaper).

## Two modeling questions, answered

- **Multiple prior labels.** The prior label is a 26-dim **multi-hot** vector (a
  study can be positive for several diseases at once). The whole vector is projected
  to one token by `Linear(26→768)` — so "multiple labels" is encoded by construction;
  there are no per-label tokens.
- **Time between the two studies.** `days_since_prior` is bucketed into 8 ranges
  (`bucket_days`: ≤1d, 2–7d, 8–30d, 1–6mo, 6–12mo, 1–3y, >3y, plus "no prior") and
  embedded via `nn.Embedding(8, 768)`; that vector is added to **every** prior token
  (image, clinical, report, vitals, label). Current-study tokens get no delta.

## Segment slots (14)

```
0-3 cur views | 4 cur clin | 5 cur vitals
6-9 prv views | 10 prv clin | 11 prv vitals | 12 prv label | 13 prv report
```

## Build the parquet (shared across all prior-aware variants)

The parquet stores raw text only and is tokenized at load, so one build serves
every text model — no `--tokenizer`:

```bash
python src/prepare/04_build_prior_aware_dataset.py
```

## Train / eval

```bash
python training/prior_aware_v3nano/prior_aware_train.py
python training/prior_aware_v3nano/prior_aware_eval.py --checkpoint-path <ckpt>
```

Eval runs the two-pass report-ablation (full vs. current clinical-indication
dropped → `predictions.no_report.csv` / `metrics.no_report.json`); the prior
report/indication is kept in both passes. Pass `--skip-report-ablation` to run only
the full pass.

## Not checkpoint-compatible with v2nano

`vitals_projector.proj`'s last layer outputs `1·768` here vs `64·768` in v2nano, and
the clinical/report tokens aren't grid-expanded — so the two cannot share weights.
Train v3nano fresh.

## Grad-CAM / Attribution

This model is a *superset* of `camchex_v3nano`, so its attribution reuses the same
machinery (`src/interpret/{attribution,visualize}.py`) through a prior-aware wrapper
(`src/interpret/{prior_attribution,run_prior_gradcam}.py`) and adds the prior branch. One
`logit.backward()` per class yields:

- **current**: image Grad-CAM per CXR view, clinical-indication token grad×embedding,
  numeric vitals grad×value (single-token vitals here, but the attribution is per-vital),
- **prior**: prior-image Grad-CAM, prior clinical text, and the **prior radiology report**
  (findings + impression — legitimate prior context, fed only for the prior study),
- **prior label**: per-class grad×value over the projected 26-dim prior CheXpert vector
  (`Linear(26→768)` token) — *which prior findings drove the current prediction*,
- **time delta**: grad×embedding on the time-gap bucket token (signed contribution + bucket),
- **modality**: current-vs-prior contribution breakdown across every token group.

Each class writes one folder of inspect-by-hand PNGs:
`<Class>/{image,prior_image,text,prv_clin,prv_report,vitals,prior_vitals,prior_label,time_delta,modality}.png`.

The model declares `gradcam_runner_module = "src.interpret.run_prior_gradcam"`, so panels
are dumped automatically after each epoch's validation (controlled by `--gradcam-epochs`
/ `--gradcam-device`, like the other models), reusing the validation logits to pick two
representative studies per class (`best/` and `first/`). Studies without a prior render
placeholder prior panels and a masked time-delta. The dump forces the live CXR-BERT path
(cache off, grads on) so per-token attribution works even with cached-embedding training;
predictions are identical. The subprocess defaults to the **training device** (override
with `--gradcam-device`); attribution host-copies any device tensors, so CPU/CUDA/MPS all
work.

Standalone (run any checkpoint by hand). `--device` defaults to auto-detect; pass
`--studies-json` to reuse the trainer's per-class picks and skip the selection scan:

```bash
# scan the split for the best true-positive study per class
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v3nano/config.yaml \
  --checkpoint-path output/prior_aware_v3nano/runs/<run>/checkpoints/epoch_000.pt \
  --split val --scan-limit 800 --device cuda

# or reuse the studies the trainer already selected (no scan)
python -m src.interpret.run_prior_gradcam \
  --config training/prior_aware_v3nano/config.yaml \
  --checkpoint-path output/prior_aware_v3nano/runs/<run>/checkpoints/epoch_000.pt \
  --split val --device cuda \
  --studies-json output/prior_aware_v3nano/runs/<run>/gradcam/epoch_0/selection.json \
  --gradcam-out output/prior_aware_v3nano/runs/<run>/gradcam/epoch_0
```
