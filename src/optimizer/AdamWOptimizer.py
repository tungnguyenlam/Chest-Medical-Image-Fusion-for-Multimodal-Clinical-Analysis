from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor
from torch.optim import AdamW, Optimizer


def build_adamw_optimizer(
    params: Iterable[Tensor],
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
