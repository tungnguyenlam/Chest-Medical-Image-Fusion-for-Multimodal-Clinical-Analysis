from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    lr: float,
    steps_per_epoch: int,
    warmup_steps: int,
    warmup_start_factor: float = 1e-3,
    eta_min_factor: float = 0.1,
) -> SequentialLR:
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch,
        T_mult=1,
        eta_min=lr * eta_min_factor,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
