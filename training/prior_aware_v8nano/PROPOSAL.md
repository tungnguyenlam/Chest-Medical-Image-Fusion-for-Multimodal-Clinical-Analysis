# Prior-Aware v8 Nano — Label-Graph-Aware Head (proposal)

Successor to [`prior_aware_v6nano`](../prior_aware_v6nano/). v8 keeps the entire v6
encoder + asymmetric fusion stack **unchanged** and adds one thing: a **directed
label-correlation graph** that conditions the classifier head, in the spirit of
ML-GCN / CheXGCN (Chen et al., CVPR 2019) and the GAT-margin-ranking idea from Duy Anh's
EHR thesis.

This is a *head-only* contribution. It does **not** touch the image/text/prior modality
balance and does **not** add multi-prior/temporal-decay modeling — those are deliberately
deferred to a separate v9 line so the two thesis contributions get clean attribution.
The earlier-deferred modality-path idea — a learned-query (Perceiver) pooler on the
**current** image that decouples fusion token count from input resolution — has been
split out as its own v7 line; see
[`training/prior_aware_v7nano/PROPOSAL.md`](../prior_aware_v7nano/PROPOSAL.md) (to be
written) and the parked design note in
[`docs/learned_query_image_pooling.md`](../../docs/learned_query_image_pooling.md).

> **The labels are noisy.** This short proposal assumes clean co-occurrence; the real design
> must defend against report-derived label noise (spurious co-mention edges, high-variance
> rare-class estimates). The full noise-aware design — Bayesian shrinkage, significance-tested
> edges, labeler-vs-clinical edge separation, and the graph-as-denoiser framing — lives in
> [`docs/prior_aware_v8_label_graph.md`](../../docs/prior_aware_v8_label_graph.md), which
> supersedes this file. Read that before implementing.

## One-line thesis

> The 26 CXR-LT classes are not independent. A directed graph built from train-split
> conditional co-occurrence `P(j|i)` lets each class's classifier vector borrow strength
> from its clinical neighbours — and because the structure is **asymmetric and reaches the
> long tail**, this is exactly the regime where a label graph helps and an independent-class
> head cannot.

## Why now: the evidence (notebook §8 / §8b)

The Pearson/phi correlation matrix (§8) looked thin — most off-diagonal cells near zero,
structure confined to the cardiogenic and air-leak clusters. **That was the wrong
statistic.** Correlation is symmetric and mean-centered, so it is dominated by the huge
*No Finding* negative mass and collapses the rare-class signal.

The conditional matrix `P(j|i)` (§8b) tells the real story:

- **The tail is not isolated.** Pneumomediastinum → Enlarged Cardiomediastinum ≈ 1.0;
  Subcutaneous Emphysema → {Support Devices, Pneumothorax}; Pneumoperitoneum →
  {Support Devices, Pleural Effusion}; Nodule → Lung Opacity; Tortuous Aorta →
  Cardiomegaly; Fibrosis → Lung Opacity. The rare classes a graph is meant to rescue
  have sharp conditional profiles.
- **The structure is directed.** `P(j|i) ≠ P(i|j)` for the strongest edges (e.g.
  Pneumomediastinum→Enlarged Cardiomediastinum is ~1.0 but the reverse is faint). A
  symmetric correlation/co-occurrence adjacency literally cannot represent this; a
  **directed** graph can. This is the modeling reason v8 isn't redundant with a plain head.

Two honest caveats this proposal has to respect:

1. **Base-rate edges.** Much of the warm left block in the `P(j|i)` heatmap (Support
   Devices, Lung Opacity, …) is just "these classes are common," not real association.
   The graph must be built on **lift `P(j|i)/P(j)`**, not raw `P(j|i)`, or it learns
   nothing but prevalence. (See the lift cell in §8b.)
2. **Ontology-artifact edges.** `P(Enlarged Cardiomediastinum | Pneumomediastinum) ≈ 1.0`
   is suspiciously deterministic — likely a labeling-hierarchy overlap (both mediastinal
   terms) rather than independent clinical co-occurrence. Such edges inflate apparent graph
   value without teaching the model anything new. Flag and optionally prune them.

## What changes vs v6 (and what does not)

| component | v6 | v8 |
|---|---|---|
| image / text / prior encoders | — | **unchanged** |
| asymmetric cross-attention fusion (current=tgt, prior=memory) | — | **unchanged** |
| prior latents, high-res skip, context bottleneck, per-modality LayerNorms, sentinel | — | **unchanged** |
| classifier head | `MLDecoder`, 26 **frozen-random** query vectors | `MLDecoder`, 26 query vectors **produced by a label-graph module** |
| `prior_label_proj` (`Linear(n_classes→d_model)`) | bare independent-class projection | optional: `prior_label_vec @ Z` (graph-aware prior-label embedding) |

Everything that made v6 the regularized, lower-capacity geometry stays. v8 is strictly
additive at the head.

## The label graph

**Construction (offline, train split only — leakage-safe):**

1. From `prior_aware_train.parquet`, compute `N(i)`, `N(i,j)`, and `P(j|i)=N(i,j)/N(i)`
   (the §8b matrix). Dev/test never enter the graph.
2. Convert to **lift** `L(i,j)=P(j|i)/P(j)` and binarize at a lift threshold (not a raw-prob
   threshold) so base-rate-only edges are dropped. Keep the edge *direction*.
3. **ML-GCN re-weighting** to fight over-smoothing: each node keeps mass `1−p` on itself and
   spreads `p` over its neighbours (`p≈0.25`, per the paper). Yields the directed adjacency
   `A ∈ R^{26×26}` the graph module consumes.
4. (optional) **clinical-hierarchy edges**: add hand-curated parent/child edges for the
   ontology pairs (e.g. Enlarged Cardiomediastinum ⊃ Cardiomegaly) instead of letting the
   ≈1.0 co-occurrence stand as a learned edge.
5. Persist `A` and the node features as a `.pt` artifact built by a small prepare script,
   so training is deterministic and the graph is inspectable.

**Node features:** the 26 class *names* encoded once by the frozen CXR-BERT
(`microsoft/BiomedVLP-CXR-BERT-specialized`) → `Z0 ∈ R^{26×768}`. 768 matches MLDecoder's
`decoder_embedding`, so the graph output drops straight into the query slot with no
projection. (ML-GCN uses GloVe; CXR-BERT class-name embeddings are the domain-matched
upgrade and we already have the encoder loaded.)

**Graph module:** 2 layers. Because the structure is directed, use a **directed GCN** (apply
`A` as-is, not symmetrized) or a lightweight **GAT** (Duy Anh's angle — attention over
neighbours, no fixed edge weights). Output `Z ∈ R^{26×768}`, one vector per class.

## Injection points

- **Primary — MLDecoder query embeddings.** Today `query_embed` is `nn.Embedding(26, 768)`
  with `requires_grad_(False)` — 26 *frozen random* per-class queries. v8 replaces these with
  the graph output `Z`. This is the exact ML-GCN injection slot, dimension-matched.
- **Secondary (bonus) — prior-label projection.** Replace the bare `prior_label_proj` with
  `prior_label_vec @ Z`, so the prior study's labels enter fusion through graph-aware
  embeddings too. Cheap, and it composes the graph with the existing prior path.

## Two training modes (config switch)

| mode | how | provenance |
|---|---|---|
| `joint` (recommended) | graph module trained end-to-end with the rest; `Z` recomputed each forward from fixed `Z0` + fixed `A` + learned GCN/GAT weights | ML-GCN / CheXGCN |
| `pretrain_freeze` | pretrain `Z` with a **margin-ranking** objective on the graph (pull connected classes together, push unconnected apart), then freeze `Z` as the query init | Duy Anh's GAT-margin-ranking |

Start with `joint` — it's the standard ML-GCN setup and lets the head adapt the graph output
to the fusion features. `pretrain_freeze` is the cleaner ablation for "did the graph
*structure* help, independent of extra trainable capacity."

## What it should help (and what it won't)

- **Should help:** the long tail and the directed clusters — Pneumomediastinum,
  Pneumoperitoneum, Subcutaneous Emphysema, Pleural Other, Nodule, Fibrosis, Tortuous Aorta.
  These have neighbours to borrow from.
- **Won't move much:** isolated classes (whatever the §8b isolated-classes line reports at
  the chosen threshold) and the already-easy common classes.
- **Not addressed by v8 at all:** image-over-text modality dominance, and multi-prior /
  recency-decay. Those are v9. Saying so up front keeps the thesis claims honest.

## Risks

| risk | mitigation |
|---|---|
| over-smoothing (all classes collapse to one vector) | 2 layers max; ML-GCN re-weight with self-mass `1−p`; residual `Z = Z0 + GCN(Z0)` |
| base-rate edges teach nothing | build on **lift**, not raw `P(j|i)` |
| ontology-artifact edges (≈1.0 deterministic) | flag, prune, or replace with curated hierarchy edges |
| leakage (graph sees labels) | graph built **train-split only**, frozen before training; report it explicitly |
| modest / tail-concentrated gains | pre-register tail-mAP and small-finding-subset mAP as the primary readouts, not overall mAP |

## Ablation grid (config-only where possible)

| variant | isolates |
|---|---|
| `head: independent` (v6 frozen-random queries) | the whole graph contribution (baseline) |
| `graph_source: correlation` vs `lift` | whether base-rate correction matters |
| `graph_dir: directed` vs `symmetrized` | whether edge direction carries signal |
| `gnn: gcn` vs `gat` | fixed vs learned edge weights (Duy Anh's GAT angle) |
| `mode: joint` vs `pretrain_freeze` | structure vs extra trainable capacity |
| `+ prior_label_proj graph` on/off | the secondary injection point |
| `+ hierarchy_edges` on/off | curated ontology edges vs pure co-occurrence |

As with v5/v6, **track tail-mAP and a small-finding-subset mAP separately** — that's where
this lives or dies.

## Files to add / edit (when we build it)

- `src/model/PriorAwareV7NanoModel.py` — subclass/fork of v6; adds the graph module and the
  two injection points; keeps `delta_embedding`, encoder, and fusion names intact so
  Grad-CAM hooks survive.
- `src/prepare/05_build_label_graph.py` — offline graph builder (train-split `P(j|i)`→lift→
  re-weight→`A`; CXR-BERT class-name node features; saves a `.pt` artifact).
- `training/prior_aware_v8nano/{config.yaml, prior_aware_train.py, prior_aware_eval.py, README.md}`
  — mirror the v6 dir; new knobs: `graph_path`, `gnn` (gcn|gat), `graph_layers`,
  `graph_mode` (joint|pretrain_freeze), `lift_threshold`, `reweight_p`, `use_hierarchy_edges`,
  `graph_prior_label` (bool).
- `src/interpret/run_prior_gradcam.py` — register `prior_aware_v8nano` in `_MODEL_CLASSES`.

## Open question to settle before coding

Confirm `lift_threshold` and `reweight_p` from the §8b output: read the **tail-class lift
table** and the **isolated-classes line**. If too many tail classes go isolated at the
chosen threshold, lower it (more edges) or switch to a top-k-neighbours-per-node graph
instead of a global threshold. That single decision sets how much of the tail the graph can
actually reach.
