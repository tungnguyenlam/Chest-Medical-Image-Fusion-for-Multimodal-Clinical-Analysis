# v9 — Modality balance and temporal prior modeling (proposal)

**Status:** design proposal only — not implemented. When v9 is built, this file is
the canonical rationale; a thin `training/prior_aware_v9nano/PROPOSAL.md` can point
here the way v8 does for [`prior_aware_v8_label_graph.md`](prior_aware_v8_label_graph.md).

Successor line to [`prior_aware_v8nano`](../training/prior_aware_v8nano/). v9 keeps the
v6/v8 encoder + asymmetric fusion + (optional) v8 label-graph head **unchanged in
spirit** and addresses the two contributions deliberately deferred from v7 and v8:

1. **Image-over-text / image-over-prior modality dominance** — the fusion stack lets
   strong text and prior channels drown out the current image, so headline mAP can look
   good while the model is mostly copying priors rather than reading pixels.
2. **Multi-prior / temporal-decay modeling** — today we use exactly one chronologically
   previous study with a coarse `days_since_prior` bucket; we do not weight recency,
   filter same-episode priors, or aggregate multiple priors.

v7 (learned-query image pooler) and v8 (noise-aware label graph head) are **orthogonal**
contributions kept separate for clean thesis attribution. v9 composes with both later.

---

## 0. The one-sentence reframe

> A prior-aware CXR classifier should read the **current image first** and treat prior
> information as **time-weighted evidence**, not a copyable label channel — so v9
> regularizes modality balance and replaces the single nearest prior with a recency-aware,
> multi-study memory that down-weights same-episode copy-forward text.

---

## 1. Why now (the evidence v7/v8 left on the table)

### 1.1 Modality dominance

The prior-aware stack feeds the current study: image tokens + clinical indication +
vitals. The prior study contributes image + clinical + **full report** + vitals +
**ground-truth label vector** (see [`PriorAwareDataset.py`](../src/dataloader/PriorAwareDataset.py)
and [`PriorAwareV8NanoModel.py`](../src/model/PriorAwareV8NanoModel.py)).

Grad-CAM / modality-attribution runs on v2–v6 variants already show the classifier
leaning on `prior_label` and `prior_report` for chronic/structural classes. v5's
**prior latents** ([`docs/prior_latents.md`](prior_latents.md)) were a first fix: pool
261 prior tokens down to K=16 so the label vector cannot route cleanly to logits. v6
kept that pooler but did not add an explicit **image-side** counterweight or a
modality-balance training objective.

Honest consequence: absolute mAP on the prior-aware line can look "too good" relative
to image-only baselines even when the image encoder contributes little. That undermines
claims about small-finding detection and makes v8 graph-head gains hard to interpret.

### 1.2 Single prior + weak time model

`PreviousStudy` is the chronologically previous study per subject
(`src/prepare/01_make_dataset.py`, `groupby('subject_id')['study_id'].shift(1)`). Stage 04
joins one prior row; the model gets one `days_since_prior` scalar bucketed into
`delta_embedding` (8 buckets, index 0 = no prior).

Problems this creates:

- **Same-episode "priors".** Within one ED visit or admission, the "prior" can be
  hours apart and describe the *same* acute finding. The prior report often copy-forwards
  current findings → metrics inflate and temporal generalization is untested.
- **One prior wastes history.** Many subjects have multiple prior CXRs; the nearest
  prior is not always the most informative (e.g. a 3-year-old baseline vs a 1-week
  follow-up).
- **No explicit decay.** A 2-day prior and a 2-year prior share the same machinery
  aside from a coarse bucket; there is no smooth recency kernel or learned decay.

The [v8 leakage analysis](prior_aware_v8_leakage_analysis.md) flagged (1) and (2) as
follow-ups. v9 is the natural place to address them in the model, not just in eval
post-hoc filters.

### 1.3 What v9 is *not*

| topic | owner | v9 role |
|---|---|---|
| Label-correlation graph head | **v8** | optional compose (`graph_prior_label`, graph queries) |
| Learned-query image pooler / resolution | **v7** | unchanged; v9 does not replace the image path |
| Prior latents (K-query pooler on prior memory) | **v5/v6** | kept; v9 may extend, not remove |
| New label space / CXR-LT 2024 tasks | config variants | orthogonal |

---

## 2. Pillar A — Modality balance (current image vs text/prior)

**Goal.** Make the current-image tokens necessary for good performance, especially on
acute and small-finding classes, without removing clinically legitimate prior inputs.

### 2.1 Mechanisms to consider (pick 1–2 for a minimal v9, not all at once)

| mechanism | idea | provenance / note |
|---|---|---|
| **Modality dropout** | During training, stochastically drop entire prior blocks (image / report / label / all-prior) or current text — generalizes existing `label_dropout_p` | cheap; already partially there for prior block |
| **Gradient balancing / GradNorm** | Equalize gradient norms across modality parameter groups (image encoder vs text vs prior projections) | used in multimodal literature; needs careful grouping |
| **Auxiliary image-only head** | Small detached BCE on image tokens only (or high-res skip path); main loss unchanged | forces image path to carry signal; doubles head compute |
| **Modality-consistency penalty** | Penalize high confidence when image tokens are masked/zeroed but prior is present | related to v8 graph-consistency aux pattern |
| **Tighter prior bottleneck** | Lower `n_prior_latents`, raise `prior_latent_dropout`, or **drop `prior_label` token** at train time with high probability | extends v5; targets the copy shortcut directly |
| **Report ablation at train time** | Randomly blank `prior_report_text` while keeping prior image/labels | breaks report-copy path; eval must report prior-off |

**Recommended minimal arm:** extend training-time **prior-path dropout** (separate knobs
for `prior_label`, `prior_report`, `prior_image`) + report **modality-share logging** each
epoch (already partially available in interpret scripts). Add one explicit objective only
if dropout alone does not move small-finding mAP.

### 2.2 Readouts (not overall mAP alone)

- **ΔmAP (full − prior-zeroed)** at eval: `has_prior=False` or null all prior fields —
  the ablation the leakage audit called for; not the same as `drop_report` (which only
  blanks current indication).
- **ΔmAP (full − no prior report)** and **(full − no prior label)** separately.
- **Small-finding-subset mAP** and **tail-mAP** (same registration as v7/v8).
- **Modality attribution**: prior_label / prior_report share should drop on acute classes
  after v9 without collapsing chronic-class performance.

---

## 3. Pillar B — Multi-prior and temporal decay

**Goal.** Replace "one nearest prior" with a **recency-weighted memory** over the last
M studies per subject, and down-weight (or exclude) priors that are too close in time to
be independent evidence.

### 3.1 Data pipeline changes

Today: `04_build_prior_aware_dataset.py` resolves one `PreviousStudy` per row.

v9 needs (offline, same split boundaries as today):

1. For each current study, list up to **M prior study_ids** for the same `subject_id`
   with `StudyDateTime < current`, sorted descending (most recent first).
2. Store per-prior: `days_since`, image paths, vitals, clinical text, report text,
   label vector (same contract as today).
3. Persist **M** as a fixed cap (e.g. M=3 or M=5) with padding + masks — mirror how
   `MAX_VIEWS=4` is handled for images.
4. Optional **minimum gap** field: `prior_valid_mask` false when
   `days_since < min_gap_days` (configurable; e.g. 7 or 30) to drop same-episode priors
   from training and from "clean" eval splits.

New artifact: `prior_aware_{split}.parquet` v2 schema **or** sidecar
`prior_history_{split}.parquet` joined by `study_id` — decision deferred until we
measure parquet size and loader complexity.

### 3.2 Model changes

| component | v6/v8 today | v9 direction |
|---|---|---|
| Prior memory | one prior study → 261 tokens → K latents | **M studies**, each encoded then **aggregated** |
| Time signal | one `delta_embedding` bucket | **per-prior** delta + **global recency weights** |
| Aggregation | single `prior_pooler` | options: (a) concat + pool, (b) weighted sum of per-prior latent banks, (c) temporal Transformer over M prior summary tokens |

**Recency weighting (simple default):**

```
w_m = exp(-days_since_m / tau)   or   w_m = (1 + days_since_m / tau)^(-alpha)
```

with `tau` or `alpha` a config hyperparameter or a single learned scalar. Weights
normalize over valid priors; all-zero → fall back to `no_prior_token` sentinel (unchanged).

**Composition with v8:** if `graph_prior_label` is on, apply `prior_label @ Z` **per
prior** before weighting, so graph-aware label embeddings enter the temporal sum.

### 3.3 Leakage hygiene

- Priors must remain **strictly earlier** than the current study (already true for
  `shift(1)`; multi-prior extends the same rule).
- **Same-split only** when building lookups (unchanged from stage 04).
- **Same-episode filter** is a modeling choice, not just eval: training on 0-day priors
  teaches copy-forward. Document which eval slices use `min_gap_days > 0`.
- Never feed the **current** study's report (unchanged).

---

## 4. What changes vs v8 baseline (summary table)

| component | v8 | v9 (proposed) |
|---|---|---|
| current image / text / vitals encoders | — | **unchanged** |
| asymmetric fusion (current=tgt, prior=memory) | — | **unchanged** |
| label-graph head (optional) | v8 contribution | **unchanged** (compose) |
| prior latents (K pooler) | — | **unchanged** or tighter K / dropout |
| number of prior studies | 1 | **M (e.g. 3–5)** |
| temporal weighting | 8 coarse buckets | **per-prior decay + valid mask** |
| modality balance | implicit (LayerNorm only) | **explicit dropout and/or aux** |
| parquet / loader | single prior columns | **multi-prior columns or sidecar** |

---

## 5. Risks

| risk | mitigation |
|---|---|
| Modality regularization hurts chronic classes | per-class or head/tail grouped readouts; don't optimize only acute subset |
| Multi-prior blows up memory (M × 261 tokens) | per-prior pool to K latents *before* cross-attention; cap M low (3) first |
| Same-episode filter drops too much coverage | report prior coverage vs `min_gap_days` curve before picking default |
| Gains confounded with v8 graph | freeze v8 head; ablate v9 knobs on v6 head first |
| Parquet rebuild + cache invalidation | version schema; document rebuild command in README |

---

## 6. Ablation grid (config-only where possible)

| variant | isolates |
|---|---|
| v8 baseline (no v9 knobs) | — |
| `prior_dropout_{label,report,image}` sweeps | which prior channel drives copy shortcut |
| `n_prior_latents` ↓ and `prior_latent_dropout` ↑ | bottleneck-only vs full v9 |
| `min_gap_days` ∈ {0, 7, 30} | same-episode leakage |
| M ∈ {1, 3, 5} with fixed decay | multi-prior value vs noise |
| learned vs fixed `tau` | flexibility vs overfit |
| v9 + `head_mode: graph` vs `independent` | compose with v8 without conflating claims |

**Primary metrics:** tail-mAP, small-finding mAP, **ΔmAP under prior-zeroed eval**,
bootstrap CIs (test labels are noisy — same rule as v8).

---

## 7. Open questions (settle before coding)

1. **Default M and `min_gap_days`** — run `prior-study-viability` / parquet stats:
   distribution of inter-study gaps, count of subjects with ≥2/≥3 priors in-split.
2. **Minimal v9 scope** — modality-only v9a vs temporal-only v9b vs both in one
   model class? Recommendation: **v9a modality dropout + eval protocol first** (no parquet
   rebuild), then **v9b multi-prior** once gaps are characterized.
3. **Drop `prior_label` at train time?** Strong regularizer but changes deployment
   assumption (label often available in production). Prefer train-time dropout with
   eval-time label on unless product says otherwise.
4. **Subject-disjoint split audit** — one-off check that train/dev/test `subject_id`
   sets are disjoint (leakage audit follow-up).

---

## 8. Files to add / edit (when we build it)

- `docs/prior_aware_v9.md` — this file.
- `src/prepare/06_build_prior_history.py` (or extend `04_`) — multi-prior parquet builder.
- `src/dataloader/PriorAwareDataset.py` — load M priors + masks + recency fields.
- `src/model/PriorAwareV9NanoModel.py` — subclass v8 (or v6 if graph off); temporal
  aggregation + modality dropout hooks.
- `training/prior_aware_v9nano/{config.yaml, prior_aware_train.py, prior_aware_eval.py,
  README.md}` — thin entrypoints; eval adds **prior-zeroed** and **min_gap** slices.
- `training/utils/evaluation.py` — optional `evaluate_prior_ablation` alongside
  `evaluate_report_ablation`.

---

## 9. Related docs

- [`prior_aware_v8_label_graph.md`](prior_aware_v8_label_graph.md) — v8 graph head; defers
  modality + temporal work here.
- [`prior_latents.md`](prior_latents.md) — v5 prior bottleneck; v9 pillar A extends this.
- [`learned_query_image_pooling.md`](learned_query_image_pooling.md) — v7 image path; orthogonal.
- WORKLOG `2026-06-25 - v8nano leakage audit` — prior-shortcut and same-episode follow-ups.
