# Prior-Aware v8 Nano — Graph Head Architecture

This document describes the architecture of the **graph head** (`LabelGraphHead`) in
the `prior_aware_v8nano` model variant and how it integrates into the main model.

- Graph head: [src/model/graph_head.py](../src/model/graph_head.py)
- Main model: [src/model/PriorAwareV8NanoModel.py](../src/model/PriorAwareV8NanoModel.py)
- Classification head: [src/decoder/MLDecoder.py](../src/decoder/MLDecoder.py)
- Config: [training/prior_aware_v8nano/config.yaml](../training/prior_aware_v8nano/config.yaml)

## The idea behind it (motivation)

For the full design rationale see [prior_aware_v8_label_graph.md](prior_aware_v8_label_graph.md);
this is the short version.

**The problem.** CXR-LT labels are mined from free-text radiology reports, not from
pixel-level annotation, so they are **noisy** — NLP-labeler errors, uncertainty collapse
("possible nodule" → 1/0), report→image broadcast (a study's labels copied to every view),
and co-mention artifacts. Crucially the noise is *heteroscedastic* (worst on the rare tail
classes the model most needs help with) and *structured* (correlated errors, not random).

**The reframe.** Don't trust the labels one image at a time — trust the **population-level
co-occurrence structure**, which is far more stable than any single label. Inject that
structure as a *structured prior on the classification head* so the head can no longer treat
the 26 classes as independent. Clinically correlated classes then share representation, and a
single noisy positive can't push one class around without consequences for its neighbors.
This makes v8 a **noise-robustness** contribution, not just a long-tail trick.

**The catch, and the defenses.** The co-occurrence structure is *itself* estimated from the
noisy labels, so a naively-built graph would faithfully encode labeler behavior (it would be
a noise *amplifier* rather than a *denoiser*). The graph is therefore built with the noise
modeled explicitly, all before the adjacency is frozen:

- **Bayesian shrinkage** of `P(j|i)` toward the base rate `P(j)` — kills high-variance
  spurious edges minted from a handful of rare-class positives.
- **BH-corrected significance** + **lift threshold / top-k** sparsification — keeps only
  well-evidenced edges and stops the genuine high-lift tail edges from being diluted in a
  dense "everything mildly co-occurs" blob.
- **Curated clinical-hierarchy edges** — hand-encode the few ontology relations (e.g. the
  air-leak family) instead of inferring them from co-mention artifacts.
- **Confidence-weighted edges + ML-GCN self-mass reweight + ≤2 layers + residual** — so a
  confident neighborhood can't wash out a node's own identity, and correlated errors aren't
  over-propagated (the over-smoothing / amplification guard).

**Why it lives on the head queries.** The MLDecoder uses one learnable query vector per
class. v8 replaces those frozen-random queries with `z_graph` — class embeddings produced by
propagating CXR-BERT class-name node features over the cleaned graph. The structure is thus
applied exactly where the per-class decision is made, leaving the encoder/fusion (and the
Grad-CAM hooks) untouched. An optional soft *consistency loss* (off by default) can further
push predicted probabilities to respect the graph.

**Leakage note.** The graph is built from the train split only, frozen to a `.pt` artifact
before training; dev/test labels never enter it.

## Key idea

The graph head runs independently of the per-sample forward pass. It propagates the
frozen 26-class node embeddings (`Z0`) over a learned, clinically-pruned
label-correlation graph to produce `z_graph [26, 768]`. These vectors **replace the
MLDecoder's frozen random query vectors**, so each class query carries shared structure
from its clinically correlated neighbors. The fused image/text/vitals/prior tokens
become the decoder memory, and the decoder attends those graph-aware queries against
them to emit the 26 logits.

Defaults from config: `head_mode: graph`, `gnn: gcn` (GAT path shown dashed),
`graph_consistency_lambda: 0.0` (aux loss wired but off by default).

## Diagram 1 — Inside the Graph Head (`LabelGraphHead`)

```mermaid
flowchart TD
    subgraph ARTIFACT["Frozen artifact: label_graph.pt (train-split only)"]
        Z0["Z0 — node features [26, 768]<br/>CXR-BERT class-name embeddings"]
        LIFT["lift P̂(j|i)/P(j) [26,26]"]
        SIG["sig — BH-Fisher mask [26,26]"]
        CUR["curated clinical edges [26,26]"]
        PCOND["pcond P(j|i) [26,26]"]
    end

    subgraph BUILD["build_adjacency()"]
        K1["keep = lift &gt; 1.5"]
        K2["& significance mask"]
        K3["| curated hierarchy edges"]
        TOPK["top-k=6 strongest out-edges/node"]
        RW["ML-GCN reweight (p=0.25)<br/>self-mass + neighbor spread"]
        A["A — row-stochastic adjacency [26,26]"]
        EM["edge_mask [26,26] (for GAT)"]
    end

    LIFT --> K1 --> K2
    SIG --> K2 --> K3
    CUR --> K3 --> TOPK --> RW --> A
    K3 --> EM

    subgraph GNN["GNN propagation (≤2 layers)"]
        L1["Layer 1: GCN  h' = GELU(A·(hW)+b)<br/>or GATv2 multi-head (4 heads)"]
        L2["Layer 2: → out_dim 768"]
        L1 --> L2
    end

    Z0 --> L1
    A --> L1
    A --> L2
    EM -.GAT path.-> L1

    RES["res_proj(Z0)  [26,768]"]
    Z0 --> RES
    ADD(("＋ residual"))
    L2 --> ADD
    RES --> ADD
    ADD --> LN["LayerNorm"]
    LN --> Z["Z — graph-aware queries [26, 768]"]

    PCOND -. "consistency loss: P(j∣i)·p_i·(1−p_j)" .-> CONS["aux graph_consistency_loss"]

    style Z fill:#cfe8ff,stroke:#1f6feb
    style Z0 fill:#e8ffd9
    style A fill:#fff2cc
```

## Diagram 2 — Graph Head in the Main v8 Nano Model

```mermaid
flowchart TD
    subgraph INPUTS["Inputs"]
        IMG["Frontal + Lateral CXR 512×512"]
        TXT["Clinical text + prior report"]
        VIT["Vitals [7]"]
        PL["Prior study label [26]"]
    end

    subgraph BACKBONE["Encoders → fused tokens (d_model=640)"]
        CNX["ConvNeXtV2-Nano (×2 views)<br/>→ [B,640,16,16] → pool [B,640,8,8]"]
        BERT["BiomedVLP-CXR-BERT → [B,768]<br/>→ text proj [B,2,640]"]
        VP["Vitals projector → [B,1,640]"]
    end

    IMG --> CNX
    TXT --> BERT
    VIT --> VP

    TGT["target tokens tgt [B, S, 640]"]
    CNX --> TGT
    BERT --> TGT
    VP --> TGT

    PMEM["Prior memory → latent pooler<br/>(16 queries) [B,16,640]"]
    PL --> PMEM
    CNX -.prior view.-> PMEM

    FUSION["Asymmetric Fusion Transformer (decoder)<br/>tgt cross-attends prior memory → [B,S,640]"]
    TGT --> FUSION
    PMEM --> FUSION

    subgraph GH["Graph Head (Diagram 1)"]
        ZG["z_graph [26, 768]"]
    end

    subgraph HEAD["MLDecoder classification head"]
        EMB["embed_standart: [B,S,640] → [B,S,768] (memory)"]
        DEC["TransformerDecoder<br/>tgt = z_graph queries [26,B,768]"]
        GFC["group FC duplicate_pooling → logits [B,26]"]
        EMB --> DEC --> GFC
    end

    FUSION --> EMB
    ZG ==>|query_embed replaces<br/>random queries| DEC

    GFC --> LOGITS["logits [B, 26]"]
    ZG -. "consistency loss on sigmoid logits" .-> AUX["+ graph_consistency_loss<br/>+ background penalty"]
    LOGITS -.-> AUX

    style ZG fill:#cfe8ff,stroke:#1f6feb
    style FUSION fill:#ffe0cc
    style LOGITS fill:#d9ffd9
```
