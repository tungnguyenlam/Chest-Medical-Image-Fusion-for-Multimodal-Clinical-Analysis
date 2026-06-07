import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Weighted sum of several criteria sharing the ``forward(pred, label)`` convention.

    Used for ``--loss FC ASL``: ``total = sum(weight_i * loss_i(pred, label))``. Weights
    matter because different losses live on different scales (e.g. focal ~O(0.01) vs ASL
    ~O(0.05) here); tune them via ``model.loss_weights``.
    """

    def __init__(self, losses: list[nn.Module], weights: list[float] | None = None, names: list[str] | None = None):
        super().__init__()
        if not losses:
            raise ValueError("CompositeLoss needs at least one loss")
        self.losses = nn.ModuleList(losses)
        if weights is None:
            weights = [1.0] * len(losses)
        if len(weights) != len(losses):
            raise ValueError(f"got {len(weights)} weights for {len(losses)} losses")
        self.weights = [float(w) for w in weights]
        self.names = list(names) if names is not None else [f"loss{i}" for i in range(len(losses))]
        # Weighted per-term values from the most recent forward (for logging/plotting).
        self.last_terms: dict[str, float] = {}

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        total = None
        terms: dict[str, float] = {}
        for name, w, loss in zip(self.names, self.weights, self.losses):
            term = w * loss(pred, label)
            terms[name] = float(term.detach())
            total = term if total is None else total + term
        self.last_terms = terms
        return total
