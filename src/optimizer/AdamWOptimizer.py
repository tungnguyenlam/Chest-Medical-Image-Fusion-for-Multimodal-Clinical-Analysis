from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW, Optimizer


def split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Bias and 1-D params (LayerNorm/BatchNorm gain, embeddings' scaling, etc.) should not decay.
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_adamw_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    return AdamW(
        split_decay_param_groups(model, weight_decay),
        lr=lr,
        betas=betas,
        eps=eps,
    )
