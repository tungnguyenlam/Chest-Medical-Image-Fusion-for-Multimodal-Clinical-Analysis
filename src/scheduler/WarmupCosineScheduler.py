from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    lr: float,
    steps_per_epoch: int,
    warmup_steps: int,
    total_steps: int | None = None,
    schedule: str = "warm_restarts",
    warmup_start_factor: float = 1e-3,
    eta_min_factor: float = 0.1,
) -> SequentialLR:
    """Linear warmup, then a cosine phase. ``schedule`` selects the cosine shape.

    ``"warm_restarts"`` (default, faithful to the original CaMCheX code):
        ``CosineAnnealingWarmRestarts`` with period ``T_0=steps_per_epoch`` and
        ``T_mult=1`` -- the LR sweeps ``lr -> eta_min`` every epoch and then
        *restarts* to ``lr`` at each epoch boundary (a per-epoch sawtooth). This
        is SGDR; useful for snapshot ensembling but it re-kicks a converged
        model every epoch, which destabilises the fine-tuning tail. Keep the
        best checkpoint, not the last.

    ``"single_cosine"``:
        One ``CosineAnnealingLR`` over the whole run
        (``T_max = total_steps - warmup_steps``) -- a single monotone decay
        ``lr -> eta_min`` with no restarts, so the model settles into its
        minimum in the tail. Requires ``total_steps``.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    eta_min = lr * eta_min_factor

    if schedule == "single_cosine":
        if total_steps is None:
            raise ValueError("schedule='single_cosine' requires total_steps")
        t_max = max(1, int(total_steps) - int(warmup_steps))
        cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif schedule == "warm_restarts":
        cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,
            T_mult=1,
            eta_min=eta_min,
        )
    else:
        raise ValueError(
            f"unknown schedule={schedule!r}; expected 'warm_restarts' or 'single_cosine'"
        )

    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
