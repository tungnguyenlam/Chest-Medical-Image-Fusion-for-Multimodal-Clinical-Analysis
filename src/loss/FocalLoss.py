import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label sigmoid focal loss (RetinaNet-style), matching the criterion
    convention used here: ``forward(pred, label)`` where ``pred`` is raw logits and
    ``label`` is a float {0,1} multi-label target; returns a scalar.

    ``gamma`` focuses training on hard examples (gamma=0 -> plain BCE). ``alpha``
    (in [0,1], or None to disable) re-weights the positive class to counter the
    heavy negative imbalance in CXR-LT. Defaults follow the original focal-loss paper.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        p = torch.sigmoid(pred)
        # p_t = p for positives, 1-p for negatives -> the modulating factor (1-p_t)^gamma
        p_t = p * label + (1.0 - p) * (1.0 - label)
        loss = ce * (1.0 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * label + (1.0 - self.alpha) * (1.0 - label)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
