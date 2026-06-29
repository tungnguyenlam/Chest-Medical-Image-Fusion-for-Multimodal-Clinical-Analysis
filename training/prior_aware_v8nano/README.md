# Prior-Aware v8 Nano — noise-aware label-correlation graph head

Successor to [`prior_aware_v6nano`](../prior_aware_v6nano/). v8 keeps the **entire v6
encoder + asymmetric fusion stack unchanged** and changes one thing: the 26 `MLDecoder`
classifier queries are no longer frozen-random `nn.Embedding` rows — they are produced by a
**directed, noise-aware label-correlation graph** so correlated classes share representation.
This is a *head-only* contribution, for clean attribution against the v6 baseline.

- Design rationale (noise model, shrinkage, significance, denoiser framing):
  [`docs/prior_aware_v8_label_graph.md`](../../docs/prior_aware_v8_label_graph.md) — canonical.
- Elevator pitch + ablation grid: [`PROPOSAL.md`](PROPOSAL.md).
- Decision-gate result (2026-06-25): docs §10 and `output/prior_aware_v8nano/`.

## Why this design (the gate result)

The cheap decision gate (`data/00-examine-data/v8_label_graph_gate.ipynb`) ran before any
training and found: **all 10 tail classes are connected** (only `No Finding` is isolated),
and the strong tail edges are clinically sound (air-leak family, Mass→Lung Lesion,
Emphysema→Fibrosis). But **significance alone passes 42% of pairs** at N=128k, and the
deterministic prune never fires (shrinkage already tames it). So the operative sparsifier
here is **`lift_threshold` + `graph_top_k`**, not significance — that's why those knobs lead.

Honest expectation: the structure is real and reaches the tail, but the graph largely
re-encodes co-occurrence that per-sample BCE already sees, so treat a within-CI tail-mAP
result as a legitimate outcome. Run the **minimal arm first** (`independent` vs
`graph` baseline), not the full grid.

## Build the graph artifact (once, before training)

Train-split only — leakage-safe. Produces `data/data-camchex/label_graph.pt`
(shrunk lift / conditional / significance / curated matrices + CXR-BERT class-name node
features). Construction knobs like `lift_threshold`/`top_k` are applied **in the model**, so
you do not rebuild the artifact to sweep them.

```bash
python src/prepare/05_build_label_graph.py \
    --train-parquet data/data-camchex/prior_aware_train.parquet \
    --out data/data-camchex/label_graph.pt
```

## Train

```bash
python training/prior_aware_v8nano/prior_aware_train.py \
    --config training/prior_aware_v8nano/config.yaml \
    --run-name prior_aware_v8nano_baseline
```

Same flags as v6 (`--use-precomputed-text-embeddings`, `--text-embeddings-gpu-resident`,
`--uint8-image-pipeline`, …). The graph head adds ~no image/text cost (26-node GNN).

## The head-only knobs (`model_init_args`)

| knob | meaning |
|---|---|
| `head_mode` | `graph` (v8) or `independent` (== v6 frozen-random queries; the baseline arm, same class) |
| `gnn` | `gcn` (fixed lift-weighted propagation) or `gat` (learned edge attention) |
| `graph_layers` | GNN depth (≤2 — over-smoothing guard) |
| `lift_threshold`, `graph_top_k` | the operative sparsifier (per the gate) |
| `use_significance` | also AND the BH-significant mask onto the lift threshold |
| `use_hierarchy_edges` | union the curated clinical-hierarchy edges (doc §3.3) |
| `graph_dir` | `directed` or `symmetrized` (edge *survival* is symmetric; only *weights* differ) |
| `reweight_p` | ML-GCN self-mass: node keeps `1-p`, spreads `p` over neighbours |
| `graph_mode` | `joint` (recommended) or `pretrain_freeze` (margin-ranking pretrain of `Z`, then freeze — structure-vs-capacity, §4C) |
| `graph_prior_label` | route the prior label vector through `Z` (secondary injection) |
| `graph_consistency_lambda` | `>0` enables the soft co-occurrence consistency aux loss (§4B), detached-coupled like the bg penalty |

## Ablation grid (config-only)

The headline comparison is **`head_mode: independent` vs `head_mode: graph`** (with the
default `lift+top_k+curated` construction), reporting **tail-mAP and small-finding-subset
mAP with bootstrap CIs** (per docs §5 — test labels are noisy, so report the CI not just the
point). Further arms: `gnn: gcn|gat`, `graph_dir: directed|symmetrized`,
`use_significance`/`use_hierarchy_edges`/`graph_prior_label`/`graph_consistency_lambda`
on/off, `graph_mode: joint|pretrain_freeze`.

## Eval — ablation grid + Grad-CAM assets (per checkpoint)

`eval_no_cache.py` is a single, deterministic, arch-generic driver for **one** checkpoint
(`best.pt`; `last.pt` is ignored). It writes everything into the checkpoint's own run dir
under `<run>/ablation/`. The on-disk channel cache is disabled (channels decode fresh from
JPEG) and CXR-BERT loads with `low_cpu_mem_usage=False` (avoids the MPS meta-tensor crash).

```bash
python training/prior_aware_v8nano/eval_no_cache.py \
    --checkpoint-path output/prior_aware_v8nano/runs/<run>/checkpoints/best.pt
```

It runs two things:

1. **Ablation grid** (eval-time input ablations, same checkpoint): `full`, `gray3`
   (1-channel grayscale replicated x3 — the old `eval_no_cache_gray` behaviour, a
   train/eval *mismatch* probe, not a channel-design proof), `drop_vitals`, `drop_report`
   (current clinical indication blanked), `drop_prior` (whole prior branch masked). Each arm
   writes `predictions_<arm>.csv` + `metrics_<arm>.json`, prints a delta vs `full`, and all
   arms roll up into `ablation_summary.csv`. Pick a subset with `--ablations` (`full` is
   always included as the delta baseline); `--skip-ablations` runs Grad-CAM only.
2. **Grad-CAM raw assets** (full pipeline): one best (highest-confidence TP) study per class
   into `ablation/gradcam/<Class>/` — the 3 input planes split out as individual raw PNGs +
   the Grad-CAM heatmap (current and prior), plus `indication-current.txt` /
   `indication-prior.txt` / `report-prior.txt` / `vitals-*.txt` and `meta.json`. Skip with
   `--skip-gradcam`.

There is **no graph-head arm**: the v6-vs-v8 (no-graph-vs-graph) comparison comes from
running this same script on each checkpoint separately (it reads the arch from the
checkpoint's config, so it runs on a v6 checkpoint too).

## Grad-CAM

`prior_aware_v8nano` is registered in `src/interpret/run_prior_gradcam.py`; the graph
consistency aux is stripped for attribution so the forward stays logits-only (encoder/fusion/
`delta_embedding` names are preserved, so the existing hooks survive).

## Not addressed by v8

Image-over-text modality dominance and multi-prior / temporal-decay — those are the v9 line,
kept separate for clean thesis attribution.
