"""Noise-aware label-correlation graph head for Prior-Aware v8.

This is the single thing v8 adds on top of v6: a small graph module that turns the
26 class-name node features ``Z0`` into 26 *correlation-aware* per-class query vectors
``Z`` that drop straight into ``MLDecoder``'s query slot (replacing its 26 frozen-random
``query_embed`` vectors). See ``docs/prior_aware_v8_label_graph.md`` for the design
rationale and ``training/prior_aware_v8nano/PROPOSAL.md`` for the elevator pitch.

Design choices that come straight from the noise-aware doc:

* **The adjacency is built here, from the frozen artifact, using config knobs** -- the
  artifact (``label_graph.pt``) ships the *shrunk* lift / conditional / significance /
  curated matrices and the node features, and this module turns them into the operative
  directed adjacency ``A``. That makes the §7 ablation grid (``lift_threshold``, ``top_k``,
  ``graph_dir``, ``reweight_p``, ``use_hierarchy_edges``) **config-only** -- no artifact
  rebuild. The 2026-06-25 gate (``docs §10``) found significance alone passes 42% of pairs,
  so the operative sparsifier here is ``top_k`` / ``lift_threshold``, *not* significance.
* **ML-GCN self-mass reweight** (``reweight_p``): each node keeps mass ``1 - p`` on itself
  and spreads ``p`` over its neighbours, so a confident neighbourhood can't wash out a
  node's own identity (anti-over-smoothing). Isolated nodes keep full self-mass.
* **Residual** ``Z = Z0 + GNN(Z0)`` and **<=2 layers** -- the other two over-smoothing guards.
* **Directed by default.** Edge *survival* is symmetric (Fisher OR>1 is transpose-
  invariant), so direction only bites through the asymmetric edge *weights*
  (``P(j|i) != P(i|j)``); ``graph_dir: symmetrized`` collapses that.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacency(
    lift: torch.Tensor,
    sig: torch.Tensor,
    curated: torch.Tensor,
    *,
    lift_threshold: float,
    top_k: int,
    use_significance: bool,
    use_hierarchy_edges: bool,
    symmetrize: bool,
    reweight_p: float,
) -> torch.Tensor:
    """Turn the frozen artifact matrices into a row-stochastic directed adjacency ``A``.

    Edge ``i -> j`` is kept when its shrunk ``lift(i,j) > lift_threshold`` (and, if
    ``use_significance``, it also passed the BH-corrected Fisher test), optionally unioned
    with curated clinical-hierarchy edges. Per the gate finding, ``top_k`` keeps only each
    node's strongest out-edges so the ~16 genuine high-lift tail edges aren't diluted in a
    42%-dense blob. Weights are the shrunk lift; ML-GCN self-mass reweight is applied last.

    Returns ``A`` (``[K, K]``, row-stochastic) used as ``H' = A @ (H W)``.
    """
    k = lift.shape[0]
    eye = torch.eye(k, dtype=torch.bool, device=lift.device)

    keep = lift > lift_threshold
    if use_significance:
        keep = keep & sig.bool()
    keep = keep & ~eye  # never a self-edge here; self-mass is added by the reweight
    if use_hierarchy_edges:
        keep = keep | (curated.bool() & ~eye)

    weight = lift * keep.to(lift.dtype)  # confidence-weighted (§3.4)

    if top_k and top_k > 0 and top_k < k:
        # keep each source node's top_k strongest out-edges, drop the rest
        kept = torch.zeros_like(weight)
        vals, idx = weight.topk(min(top_k, k), dim=1)
        kept.scatter_(1, idx, vals)
        weight = kept * (kept > 0).to(weight.dtype)

    if symmetrize:
        weight = torch.maximum(weight, weight.t())

    row_sum = weight.sum(dim=1, keepdim=True)
    has_nb = (row_sum > 0).to(weight.dtype)            # 1 where the node has >=1 neighbour
    neighbour = weight / row_sum.clamp(min=1e-12)      # rows with neighbours sum to 1
    a = reweight_p * neighbour                          # off-diagonal mass = p
    # self-mass: (1-p) if the node has neighbours, else 1.0 (isolated node = identity row)
    self_mass = (1.0 - reweight_p) * has_nb + (1.0 - has_nb)
    a = a + torch.diag_embed(self_mass.squeeze(1))
    return a


class _GCNLayer(nn.Module):
    """Directed GCN propagation ``H' = dropout(GELU(A @ (H W) + b))`` with fixed ``A``."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.gelu(a @ self.lin(h)))


class _GATLayer(nn.Module):
    """Masked single-/multi-head graph attention (GATv2-style) over the edge set.

    Attention is permitted only where ``edge_mask`` is true (plus self-loops); everywhere
    else the score is ``-inf`` so softmax ignores it. Unlike the GCN this learns the edge
    weights rather than trusting the fixed lift -- the ``gnn: gat`` ablation arm.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float, nheads: int = 4):
        super().__init__()
        assert out_dim % nheads == 0, (out_dim, nheads)
        self.nheads = nheads
        self.head_dim = out_dim // nheads
        self.w = nn.Linear(in_dim, out_dim)
        # GATv2: score = a . LeakyReLU(W h_i + W h_j); one `a` vector per head
        self.attn = nn.Parameter(torch.empty(nheads, self.head_dim))
        nn.init.xavier_normal_(self.attn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        k = h.shape[0]
        wh = self.w(h).view(k, self.nheads, self.head_dim)             # (K, He, d)
        # pairwise sums (K query, K key, He, d)
        pair = wh.unsqueeze(1) + wh.unsqueeze(0)                        # (Kq, Kk, He, d)
        e = (F.leaky_relu(pair, 0.2) * self.attn).sum(-1)              # (Kq, Kk, He)
        mask = edge_mask.unsqueeze(-1)                                  # (Kq, Kk, 1)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = self.dropout(torch.softmax(e, dim=1))                  # attention over keys
        out = torch.einsum("qkh,khd->qhd", alpha, wh)                 # (Kq, He, d)
        return F.gelu(out.reshape(k, -1))


class LabelGraphHead(nn.Module):
    """Produces the 26 graph-aware MLDecoder query vectors ``Z`` from frozen ``Z0`` + ``A``.

    ``artifact`` is the dict loaded from ``label_graph.pt`` (keys ``lift``, ``pcond``,
    ``sig``, ``curated_mask``, ``node_features``, ``classes``). ``out_dim`` must equal the
    decoder embedding width (768) so ``Z`` drops into the query slot with no projection.
    """

    def __init__(
        self,
        artifact: dict,
        *,
        out_dim: int = 768,
        gnn: str = "gcn",
        layers: int = 2,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        lift_threshold: float = 1.5,
        top_k: int = 6,
        use_significance: bool = True,
        use_hierarchy_edges: bool = True,
        symmetrize: bool = False,
        reweight_p: float = 0.25,
        gat_heads: int = 4,
    ):
        super().__init__()
        z0 = artifact["node_features"].float()
        k, feat_dim = z0.shape
        self.num_classes = k
        self.out_dim = out_dim
        hidden_dim = hidden_dim or out_dim

        a = build_adjacency(
            artifact["lift"].float(), artifact["sig"], artifact["curated_mask"],
            lift_threshold=lift_threshold, top_k=top_k, use_significance=use_significance,
            use_hierarchy_edges=use_hierarchy_edges, symmetrize=symmetrize, reweight_p=reweight_p,
        )
        # buffers: frozen graph state travels with the checkpoint and to-device moves.
        self.register_buffer("Z0", z0, persistent=True)
        self.register_buffer("A", a, persistent=True)
        # edge_mask for GAT = wherever A has off-diagonal mass, plus self-loops
        eye = torch.eye(k, dtype=torch.bool)
        self.register_buffer("edge_mask", (a > 0) | eye, persistent=True)
        # pcond drives the consistency regularizer (probability-space, not lift)
        self.register_buffer("pcond", artifact["pcond"].float(), persistent=True)
        self.n_edges = int(((a > 0) & ~eye).sum().item())

        # input projection Z0(feat_dim) -> hidden, then `layers` graph layers, then -> out_dim
        self.gnn_type = gnn
        dims = [feat_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        mods = []
        for i in range(layers):
            if gnn == "gcn":
                mods.append(_GCNLayer(dims[i], dims[i + 1], dropout))
            elif gnn == "gat":
                mods.append(_GATLayer(dims[i], dims[i + 1], dropout, nheads=gat_heads))
            else:
                raise ValueError(f"gnn must be 'gcn' or 'gat', got {gnn!r}")
        self.layers = nn.ModuleList(mods)
        # residual needs matching width; project Z0 to out_dim once for the skip
        self.res_proj = nn.Identity() if feat_dim == out_dim else nn.Linear(feat_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self._frozen = False  # set by pretrain_and_freeze (graph_mode: pretrain_freeze)

    def _propagate(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, self.A) if self.gnn_type == "gcn" else layer(h, self.edge_mask)
        return h

    def _compute_z(self) -> torch.Tensor:
        h = self._propagate(self.Z0)
        return self.out_norm(self.res_proj(self.Z0) + h)

    def forward(self) -> torch.Tensor:
        """Return ``Z`` (``[K, out_dim]``): graph-propagated, residual-connected node vectors.

        In ``pretrain_freeze`` mode the GNN was pre-optimized at init and ``Z`` is a frozen
        buffer (structure-only, no extra trainable head capacity -- the cleanest "did the
        *structure* help" ablation, doc §4C)."""
        if self._frozen:
            return self.Z_frozen
        return self._compute_z()

    def pretrain_and_freeze(self, *, steps: int = 400, margin: float = 0.5, lr: float = 1e-2,
                            seed: int = 0) -> None:
        """Margin-ranking pretrain of the GNN on the graph alone, then freeze ``Z`` (doc §4C).

        Objective: a connected pair ``(i, j)`` should be closer in ``Z`` than a random
        unconnected pair, by ``margin`` (cosine distance). This looks only at the pooled
        co-occurrence structure -- never a single image's label -- so it is inherently more
        noise-robust than per-sample BCE. After ``steps`` of Adam, ``Z`` is detached into a
        buffer and the GNN params are frozen; ``forward`` then returns the fixed ``Z``.
        """
        eye = torch.eye(self.num_classes, dtype=torch.bool, device=self.A.device)
        connected = ((self.A > 0) & ~eye)
        src, dst = connected.nonzero(as_tuple=True)
        if src.numel() == 0:  # empty graph -> nothing to pretrain; just freeze the residual
            self.register_buffer("Z_frozen", self._compute_z().detach(), persistent=True)
            self._frozen = True
            return
        gen = torch.Generator(device="cpu").manual_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            z = F.normalize(self._compute_z(), dim=1)
            pos = 1.0 - (z[src] * z[dst]).sum(1)                       # cosine distance, connected
            neg_dst = dst[torch.randperm(dst.numel(), generator=gen)]  # shuffled = mostly unconnected
            neg = 1.0 - (z[src] * z[neg_dst]).sum(1)
            loss = F.relu(margin + pos - neg).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            self.register_buffer("Z_frozen", self._compute_z().detach(), persistent=True)
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def consistency_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Soft co-occurrence consistency penalty (doc §4B), probability-space.

        For a strong directed edge ``i -> j`` (high ``P(j|i)``), predicting ``i`` present
        while ``j`` is near-absent violates the graph. Penalize ``mean_ij  P(j|i) * p_i *
        (1 - p_j)`` over the kept edges. ``probs`` is sigmoid(logits) ``[B, K]``. Returns a
        dimensionless scalar; the caller scales it by ``graph_consistency_lambda`` and
        train_step couples it to the detached criterion loss (same as the bg penalty).
        """
        edge = (self.A > 0).to(probs.dtype)
        eye = torch.eye(self.num_classes, device=probs.device, dtype=probs.dtype)
        w = self.pcond * edge * (1.0 - eye)                       # (K, K) weight on i->j
        pi = probs.unsqueeze(2)                                    # (B, K, 1)  source i
        pj = probs.unsqueeze(1)                                    # (B, 1, K)  target j
        viol = pi * (1.0 - pj)                                     # (B, K, K)
        return (w.unsqueeze(0) * viol).sum() / (probs.shape[0] * (w.sum() + 1e-6))
