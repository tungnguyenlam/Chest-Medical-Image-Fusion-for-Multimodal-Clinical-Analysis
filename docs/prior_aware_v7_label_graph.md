# v7 — Noise-aware label-correlation graph head

This note refines the [v7 proposal](../training/prior_aware_v7nano/PROPOSAL.md) for the
single fact that reshapes the whole design: **the CXR-LT labels are noisy.** It explains
where the noise comes from, how it touches every part of a label-graph head, and how to
build v7 so the graph *survives* the noise instead of amplifying it. The shorter PROPOSAL.md
is the elevator pitch; this is the design rationale and the thesis-chapter source.

---

## 0. The one-sentence reframe

> Per-sample labels are unreliable, so don't trust them one image at a time — trust the
> **population-level co-occurrence structure**, which is far more stable than any single
> label, and inject it as a structured prior on the head. But that structure is *itself*
> estimated from the noisy labels, so the graph must be built with the noise modeled
> explicitly: confidence-weighted edges, shrinkage for rare classes, and a hard separation
> between *clinical* co-occurrence and *labeler* co-occurrence.

This makes v7 a **noise-robustness contribution**, not just a long-tail trick. That is a
stronger and more defensible thesis claim.

---

## 1. Where the noise comes from

CXR-LT 2023 labels are **derived from free-text radiology reports**, not from pixel-level
annotation. That provenance produces several distinct, non-random noise channels:

1. **NLP-labeler error.** The labels are mined from reports by automated labelers
   (CheXpert/NegBio-style for the original findings, weaker report text-mining for the
   rare classes CXR-LT added). These miss negation ("no pneumothorax" → false positive),
   misread uncertainty ("possible nodule"), and miss findings phrased unusually
   (false negatives). **The rare classes the graph is meant to help have the *weakest*
   labelers and therefore the *most* noise.** This is the central tension.

2. **Uncertainty collapse.** Reports hedge ("likely", "cannot exclude"). Binarizing a
   3-way (positive / uncertain / negative) signal into {0,1} injects systematic bias whose
   sign depends on the u→0 vs u→1 policy. Either way the positive set is contaminated.

3. **Report→image broadcast (view mismatch).** One report describes a *study*; its labels
   are copied to *every image* in that study, including the lateral. A finding visible only
   on the frontal is still tagged "present" on the lateral. So an individual image's label
   can be unsupported by *that image's* pixels even when the report is correct. This is
   noise the model sees as "the image says nothing but the label is positive."

4. **Co-mention artifacts → spurious graph edges.** When two findings are described in the
   same sentence or routinely co-reported, the labeler tags them together regardless of
   whether they co-occur *clinically*. This is the most dangerous channel **for a label
   graph specifically**, because it manufactures exactly the high-`P(j|i)` edges the graph
   feeds on. The notebook §8b `P(Enlarged Cardiomediastinum | Pneumomediastinum) ≈ 1.0` is
   the textbook symptom: a near-deterministic edge that is far more likely *labeler
   ontology overlap* than independent biology.

The takeaways that drive the design:

- Noise is **heteroscedastic** — worst on the tail, worst on the lateral views, worst on
  hedged findings.
- Noise is **structured, not random** — co-mention artifacts create *correlated* errors,
  which a co-occurrence graph will faithfully encode if not defended against.

---

## 2. How noise hits a label-graph head, part by part

| stage | what noise does | why it matters for v7 |
|---|---|---|
| **graph construction** | `P(j|i)` is a noisy estimate; for rare `i` (small `N(i)`) it is *high-variance*; co-mention artifacts inflate specific edges | the graph can encode labeler behavior instead of disease structure |
| **graph propagation** | a GCN/GAT spreads a class's signal to neighbours; if the source label is wrong, the error propagates to correlated classes | noise can be *amplified*, not averaged out, if edges are over-trusted |
| **head training** | per-sample BCE on noisy targets is the supervision; the graph fights it | this is the *opportunity*: a stable structural prior regularizes against per-sample noise |
| **evaluation** | the test labels are noisy too → measured mAP has its own error bars, concentrated on the tail | small tail-mAP gains may be inside the noise floor; need CIs |

The middle two rows are the whole story: **a well-built graph is a denoiser (averages out
independent per-sample noise via population structure); a badly-built graph is a noise
amplifier (propagates correlated labeler errors).** Every design choice below is about
staying on the right side of that line.

---

## 3. Noise-aware graph construction

The §8b pipeline (train-split-only `P(j|i)` → lift → ML-GCN reweight) is the skeleton. v7
adds three noise defenses **before** the adjacency is frozen.

### 3.1 Shrink rare-class estimates toward base rate

A point estimate `P(j|i)=N(i,j)/N(i)` from `N(i)=50` positives is almost worthless. Apply
**Bayesian shrinkage** toward the marginal `P(j)`:

```
P̂(j|i) = (N(i,j) + α·P(j)) / (N(i) + α)
```

with a small pseudocount `α` (e.g. 5–20). Rare classes (small `N(i)`) get pulled toward base
rate — i.e. "we don't have enough evidence, assume no special association" — while common
classes are barely touched. This is the single most important defense: it stops the tail's
high-variance estimates from minting confident-looking spurious edges.

### 3.2 Keep only *statistically significant* edges (confidence, not point value)

Threshold on a **lower confidence bound on the lift**, not the lift point estimate. For each
directed pair run a one-sided test that `P(j|i) > P(j)` (Fisher exact / a Wilson lower bound
on the conditional rate), with **Benjamini–Hochberg** multiple-testing correction across the
`26×25` pairs. An edge survives only if the association is unlikely to be a small-sample
fluke. This naturally lets *real* tail edges through (Pneumoperitoneum→Support Devices has
few but consistent co-occurrences) while killing noise-born ones.

### 3.3 Separate *clinical* co-occurrence from *labeler* co-occurrence

The ≈1.0 deterministic edges are the tell. Two mitigations, used together:

- **Prune near-deterministic edges** above a cap (e.g. drop `P̂(j|i) > 0.95`) unless they are
  on a curated whitelist — these carry almost no information beyond "same label" and are the
  likeliest artifacts.
- **Curated clinical-hierarchy edges.** Hand-encode the handful of ontology relations
  (Enlarged Cardiomediastinum ⊃ Cardiomegaly; the air-leak family) as explicit typed edges
  instead of inferring them from co-mention. This replaces the noisiest learned edges with
  domain truth and is a clean, defensible thesis decision.

### 3.4 Edge weights carry confidence into the model

Don't binarize to {0,1}. Keep each surviving edge's weight = its **shrunk lift** (or the
confidence-bound value), so the GNN trusts a well-evidenced edge more than a marginal one.
The ML-GCN self-mass reweight (`1−p` on self, `p` spread over neighbours) is applied on top,
so even a confident neighbourhood can't wash out a node's own identity.

> **Provenance/leakage:** the graph is built from `prior_aware_train.parquet` only, frozen to
> a `.pt` artifact before training, and inspected in the notebook. Dev/test labels never
> enter it. Report this explicitly — a graph that saw test labels would be leakage.

---

## 4. The graph as a denoising prior (training)

Three composable mechanisms, in increasing ambition. Start with (A); (B)/(C) are ablations.

**(A) Structural prior on the head (baseline v7).** Replace MLDecoder's 26 frozen-random
`query_embed` vectors with graph-produced per-class vectors `Z ∈ R^{26×768}` (CXR-BERT
class-name node features `Z0`, 2-layer directed GCN/GAT, residual `Z = Z0 + GNN(Z0)`). The
head can no longer treat classes as independent; correlated classes share representation, so
a single noisy positive label can't push one class arbitrarily without consequences for its
neighbours. This alone is a mild denoiser.

**(B) Co-occurrence consistency regularizer (soft).** Add a small auxiliary loss that
penalizes predicted probability vectors that violate the graph — e.g. if the model predicts
Pneumomediastinum present, its Enlarged-Cardiomediastinum logit shouldn't be near zero. Keep
the weight small and **detached-coupled** like the existing background penalty
(`loss + loss.detach()·aux`) so it shapes without dominating. This directly turns the graph
into a noise-correcting prior on outputs. Risk: it also propagates *artifact* edges, so it
must run on the cleaned graph from §3.

**(C) Margin-ranking pretrain (Duy Anh's mechanism, naturally noise-robust).** Pretrain `Z`
with a ranking objective on the graph (connected classes closer than unconnected) and freeze
it. Ranking over aggregate structure is inherently more robust to individual-label noise than
per-sample BCE, because it never looks at a single image's label — it looks at the pooled
co-occurrence. This is the cleanest "did the *structure* help, independent of added trainable
capacity" ablation.

**Loss interplay.** Whatever the head, the per-sample loss should already be noise-tolerant —
the asymmetric loss (ASL) the family uses down-weights easy negatives and is somewhat robust;
**add mild label smoothing** so no single noisy target is trusted at full confidence. The
graph mechanisms above sit *on top* of this, not instead of it.

---

## 5. Evaluation under noisy ground truth

- **Primary readouts:** tail-mAP (rarest ~10 classes) and a small-finding-subset mAP, each
  with **bootstrap confidence intervals** — because the *test* labels are noisy, a +0.3 mAP
  on a 543-positive class may be inside the noise floor. Report the CI, not just the point.
- **Per-class deltas vs the v6 independent-head baseline**, broken out by where §8b said the
  graph reaches (connected tail classes) vs where it doesn't (isolated classes). The graph
  should move the former and leave the latter flat; if it moves *isolated* classes, something
  is leaking.
- **Artifact audit:** check that confirmed gains aren't concentrated on the pruned
  near-deterministic pairs (which would mean we're scoring the labeler's consistency, not
  clinical skill).
- If a small curated clean-label subset is feasible, report on it as a noise-controlled
  secondary check. (Optional — flag if not available.)

---

## 6. Risks (noise-updated)

| risk | mitigation |
|---|---|
| spurious edges from co-mention artifacts | shrinkage (§3.1), significance test (§3.2), deterministic-edge prune + curated hierarchy (§3.3) |
| rare-class estimates are high-variance | Bayesian shrinkage toward base rate; significance threshold, not point value |
| GNN amplifies correlated label errors | 2 layers max, residual `Z0 + GNN`, ML-GCN self-mass, confidence-weighted edges |
| consistency loss (B) propagates artifacts | run it only on the cleaned graph; small detached-coupled weight |
| gains inside the test-label noise floor | bootstrap CIs on tail-mAP; per-class deltas, not aggregate only |
| over-smoothing (classes collapse) | residual connection; cap layers; report rank/spread of `Z` |
| leakage (graph sees labels) | train-split-only, frozen artifact, explicitly reported |

---

## 7. Ablation grid

| variant | isolates |
|---|---|
| `head: independent` (v6 frozen-random) | the entire graph contribution (baseline) |
| `edges: lift` vs `edges: shrunk-significant` | whether noise-aware construction (§3.1–3.2) matters |
| `+ deterministic-prune / hierarchy edges` on/off | the labeler-artifact defense (§3.3) |
| `weights: binary` vs `confidence-weighted` | whether edge confidence helps (§3.4) |
| `graph_dir: directed` vs `symmetrized` | whether edge direction carries signal |
| `gnn: gcn` vs `gat` | fixed vs learned edge weights |
| `mode: joint` vs `pretrain_freeze` (margin-ranking) | structure vs capacity; noise-robustness of ranking (§4C) |
| `+ consistency loss (B)` on/off | the graph-as-output-prior denoiser |
| `+ label smoothing` on/off | per-sample noise tolerance interplay (§4) |

The headline comparison for the thesis is **`independent` vs `shrunk-significant + curated`**:
does a *noise-aware* graph beat both no-graph and a *naive* graph? If naive ≈ no-graph but
noise-aware > both, that's the contribution.

---

## 8. What v7 is *not*

v7 is a head-only, noise-aware label-graph contribution. It does **not** address
image-over-text modality dominance, and it does **not** add multi-prior / temporal-decay
modeling. Those are the **v8** line, kept separate so the two thesis contributions get clean
attribution. The graph composes with v8 later via `prior_label_proj` (graph-aware
prior-label embedding).

---

## 9. Build order (when we commit)

1. `src/prepare/05_build_label_graph.py` — train-split counts → shrinkage → significance
   test (BH-corrected) → deterministic prune + curated hierarchy → confidence-weighted
   directed adjacency `A`; CXR-BERT class-name node features `Z0`; save inspectable `.pt`.
2. Extend notebook §8b to dump the **shrunk-significant** adjacency alongside the raw one, so
   we can eyeball how many edges (and which tail classes) survive the noise defenses.
3. `src/model/PriorAwareV7NanoModel.py` — fork v6; add GNN module + the two injection points;
   preserve encoder/fusion/`delta_embedding` names so Grad-CAM hooks survive.
4. `training/prior_aware_v7nano/{config.yaml, prior_aware_train.py, prior_aware_eval.py,
   README.md}` — mirror v6; new knobs per §7.
5. Register `prior_aware_v7nano` in `src/interpret/run_prior_gradcam.py`.

**Decision gate before step 3:** read the surviving-edge count and the isolated-tail-class
list from step 2. That sets `α`, the significance threshold, and whether to use a global
cutoff or top-k-neighbours-per-node.
