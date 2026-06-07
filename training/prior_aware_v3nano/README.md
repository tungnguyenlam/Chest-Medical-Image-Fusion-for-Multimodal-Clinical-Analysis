# Prior-Aware CaMCheX v3 Nano

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
