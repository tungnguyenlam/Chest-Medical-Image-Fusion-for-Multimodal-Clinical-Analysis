# v8 / prior-aware — data leakage and inflated metrics (analysis)

**Status:** investigation notes (2026-06-25). No code changes were made as part of this
audit. Use this doc when interpreting prior-aware v6–v8 metrics, designing eval
ablations, or scoping v9.

**Scope:** the full prior-aware data + model path that v8 inherits (`src/prepare/0{1,2,4}_*.py`,
`src/prepare/05_build_label_graph.py`, `src/dataloader/PriorAwareDataset.py`,
`src/model/PriorAwareV8NanoModel.py`, `training/prior_aware_v8nano/`,
`training/utils/{data,evaluation}.py`). The v8 **label-graph head** is audited separately
from the **informative-prior shortcut** shared with v6.

---

## 0. Executive summary

| question | answer |
|---|---|
| Does the v8 label graph leak test labels into training? | **No** — graph stats are train-split only; node features are class names only. |
| Does the current study's radiology report leak labels? | **No** — findings/impression are never fed for the current study. |
| Why do metrics look "so good"? | Mostly the **prior study's ground-truth labels + full report**, not the v8 graph head. Chronic classes are near-trivial to copy from the prior. |
| Is there still leakage risk? | **Yes, softer forms:** same-episode priors (copy-forward text), possible `subject_id` overlap across splits (unverified), mild hints in `clinical_indication`. |
| Does `drop_report` eval isolate leakage? | **No** — it only blanks the *current* clinical indication; prior label + prior report stay on. |

---

## 1. What each input actually is

Understanding leakage requires knowing exactly which fields cross the train/eval boundary
and what semantic content they carry.

### 1.1 Current study (prediction target)

| field | source column | content | leakage risk |
|---|---|---|---|
| `img` | current study JPEGs | pixels | none |
| `clin_text` | `clinical_indication` | indication + history only (`04_build_prior_aware_dataset.py` `_clin_text`) | **mild** — can state the answer ("known cardiomegaly") |
| `vital_values` | ED vitals at exam time | numeric vitals | low |
| `label` | CXR-LT 26-class vector | supervision only | n/a |

**Explicitly excluded:** the current study's `report` (findings + impression). Stage 01
builds `report` from findings/impression but stage 04 never exposes it for the current
study. Comment in `04_build_prior_aware_dataset.py`: *"Used only for the PRIOR study
… legitimate prior information rather than label leakage (unlike the current study's
report, which directly states the labels and is never used)."*

Config text streams (`training/prior_aware_v8nano/config.yaml`):

```yaml
text_embedding_streams: [clin_text, prior_clin_text, prior_report_text]
```

No current `report` stream exists.

### 1.2 Prior study (conditional context)

| field | content | leakage risk |
|---|---|---|
| `prior_img` | prior study CXRs | none (earlier exam) |
| `prior_clin_text` | prior indication/history | low |
| `prior_report_text` | prior findings + impression | **informative** — names findings; legitimate if prior is truly independent |
| `prior_label` | **26-dim ground-truth label vector** of the prior study | **strong shortcut** — near-copy for chronic classes |
| `prior_vital_values` | prior vitals | low |
| `days_since_prior` | time delta → `delta_embedding` | used weakly (8 coarse buckets) |

Wiring in the model (`PriorAwareV8NanoModel.py`):

- `prior_label` → `prior_label_proj` (or `prior_label @ Z` when `graph_prior_label: true`)
- `prior_report_text` → text encoder → fusion memory

Dataset emission (`PriorAwareDataset.py`):

```python
"prior_label": _to_fixed_array(row["prior_label"], ...) if has_prior else zeros
```

---

## 2. v8 label graph — leakage-safe

The v8-specific contribution is a **train-split-only** label-co-occurrence graph
(`src/prepare/05_build_label_graph.py`).

**Construction (offline):**

1. Read **only** `prior_aware_train.parquet`, column `label`.
2. Compute co-occurrence, shrunk `P(j|i)`, lift, BH-significance, curated hierarchy mask.
3. Encode **class names** (not labels) with CXR-BERT → `node_features` Z0.
4. Save frozen `label_graph.pt`. Sparsifier knobs (`lift_threshold`, `top_k`, …) apply
   at train time in `LabelGraphHead` — no rebuild needed to sweep them.

**Why this does not leak test labels:**

- Dev/test parquets never enter step 1.
- Node features are the strings `"Cardiomegaly"`, `"Pneumothorax"`, etc. — no per-sample data.
- The graph is identical for every forward pass regardless of split.

The v8 proposal risk table entry *"graph built train-split only, frozen before training"*
holds for classic label leakage.

---

## 3. Why metrics are inflated (main mechanism)

This is **not** v8-specific. v8 keeps the entire v6 encoder + fusion stack; only the
MLDecoder query source changes (graph-produced vs frozen-random).

### 3.1 The informative-prior shortcut

For many CXR-LT classes, **P(current positive | prior positive) ≈ 1**, especially
chronic/structural findings:

- Cardiomegaly, Tortuous Aorta, Calcification of the Aorta
- Emphysema, Fibrosis
- Support Devices, Atelectasis

The model receives:

1. **`prior_label`** — the exact multi-label vector of the previous study.
2. **`prior_report_text`** — the radiologist's findings/impression for that study.

A fusion model can achieve high mAP by **copying or paraphrasing the prior** with
minimal contribution from the current image. That is often **clinically realistic**
(a prior is available at deployment) but **methodologically misleading** when:

- attributing gains to the v8 graph head vs the image encoder;
- comparing to image-only or non-prior baselines;
- claiming small-finding or acute-disease detection.

v5 **prior latents** (`docs/prior_latents.md`) partially mitigate this by bottlenecking
261 prior tokens to K=16 before fusion — the label vector cannot route cleanly. v6/v8
still feed `prior_label` as an explicit token; the bottleneck mixes it but does not
remove the shortcut.

### 3.2 What this means for v8 ablations

The **only clean v8 attribution** is:

```yaml
head_mode: independent   # v6 frozen-random queries
head_mode: graph           # v8 graph queries
```

with **all else held constant** (same prior path, same data). Gains on top of an already
prior-inflated baseline measure the graph head, not overall "model quality."

Report **tail-mAP** and **small-finding-subset mAP** with bootstrap CIs (v8 README) —
overall mAP is dominated by chronic classes that the prior solves easily.

---

## 4. Softer leakage risks (not yet verified empirically)

### 4.1 Same-episode / non-independent prior

`PreviousStudy` is assigned in stage 01 as the chronologically previous study per
`subject_id`:

```python
cxr_df['PreviousStudy'] = cxr_df.groupby('subject_id')['study_id'].shift(1)
```

Within a single ED visit or admission, "prior" and "current" can be **hours apart**
describing the **same acute episode**. Radiologists copy-forward prior findings into
new reports. In that regime:

- `prior_report_text` may **restate current findings** → closer to true label leakage.
- `days_since_prior` can be **≈ 0** — the time bucket does not exclude these pairs.

**Mitigation (eval):** restrict to `days_since_prior > N` (e.g. 7 or 30 days) and
compare mAP. **Mitigation (model):** v9 `min_gap_days` proposal
([`prior_aware_v9.md`](prior_aware_v9.md)).

### 4.2 Patient overlap across splits

Splits are assigned per **`dicom_id`** from CXR-LT 2023 (`02_split_dataset.py`). Prior
lookup in stage 04 builds `lookup` from **one split only** — a test study's prior is
never a train study's row. So **cross-split prior content** does not leak.

**Still worth checking:** whether `subject_id` sets are **disjoint** across
train/dev/test. If the same patient appears in train and test with different dicoms,
the image encoder could memorize appearance. CXR-LT/MIMIC official splits are intended
to be patient-level; assert once on the prepared parquets.

### 4.3 Current clinical indication

`clin_text` is not the diagnostic report, but indication/history can hint at labels
("rule out pneumothorax", "known CHF"). Clinically available; mild inflation.

---

## 5. Eval ablations — what exists vs what is needed

### 5.1 `drop_report` (implemented)

`evaluate_report_ablation` runs two passes. `drop_report=True` calls
`_blank_prior_aware_current_indication` — empties **`clin_text` only**.

| pass | current `clin_text` | `prior_report_text` | `prior_label` |
|---|---|---|---|
| full | on | on | on |
| `no_report` | **blanked** | on | on |

**Gotcha:** `metrics.no_report.json` still includes the full prior signal. The name
"no_report" means *no current clinical indication*, not *no radiology report anywhere*.

### 5.2 Recommended ablations (not yet implemented)

| ablation | how | isolates |
|---|---|---|
| **Prior-zeroed** | `has_prior=False` or zero/null all prior fields at eval | image + current clin/vitals only |
| **No prior label** | zero `prior_label` token only | report-copy vs label-copy |
| **No prior report** | blank `prior_report_text` only | label-copy vs report-copy |
| **`min_gap` slice** | eval subset with `days_since_prior > N` | same-episode leakage |
| **v8 head** | `independent` vs `graph`, prior held constant | graph head only |

Prior-zeroed eval is the single most important missing slice for interpreting headline
mAP. See follow-ups in [`prior_aware_v9.md`](prior_aware_v9.md) §2.2.

---

## 6. Data-flow diagram

```text
                    TRAIN SPLIT ONLY
                    ┌─────────────────────┐
                    │ 05_build_label_graph │
                    │  counts, lift, Z0    │
                    └──────────┬──────────┘
                               │ frozen label_graph.pt
                               v
CURRENT STUDY ──► image, clin_text, vitals ──► encoders ──┐
                                                          ├──► fusion ──► MLDecoder
PRIOR STUDY   ──► prior_img, prior_clin,                  │      (queries from
                  prior_REPORT, prior_LABEL,               │       graph or random)
                  prior_vitals, delta_emb ────────────────┘
                               ▲
                               │ same-split lookup only (stage 04)
                               │ PreviousStudy = prev study_id per subject

NEVER FED: current study report (findings/impression)
```

---

## 7. Follow-ups (quantitative — not run yet)

Script or notebook to run on `data/data-camchex/prior_aware_{train,development,test}.parquet`:

1. **`subject_id` overlap** — intersection sizes across splits (expect empty).
2. **`days_since_prior` distribution** — histogram, fraction with gap < 1d / 7d / 30d.
3. **Per-class P(current=1 | prior=1)** — quantifies copyability by class (train or test).
4. **Prior coverage** — fraction of studies with `has_prior=True` per split.

Implement prior-zeroed and `min_gap` eval passes in `training/utils/evaluation.py` when
revisiting.

---

## 8. Related docs

- [`prior_aware_v8_label_graph.md`](prior_aware_v8_label_graph.md) — v8 graph design; defers
  modality/temporal issues to v9.
- [`prior_aware_v9.md`](prior_aware_v9.md) — proposed fixes (modality balance, multi-prior,
  `min_gap_days`).
- [`prior_latents.md`](prior_latents.md) — v5 prior bottleneck vs label-copy shortcut.
- [`training/prior_aware_v8nano/README.md`](../training/prior_aware_v8nano/README.md) — v8
  train/eval entrypoints and headline ablation (`head_mode`).
