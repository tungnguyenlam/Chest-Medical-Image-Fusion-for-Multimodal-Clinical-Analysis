from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    LinearLR,
    SequentialLR,
)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    steps_per_epoch: int,
    warmup_steps: int,
    total_steps: int | None = None,
    schedule: str = "warm_restarts",
    warmup_start_factor: float = 1e-3,
    eta_min_factor: float = 0.1,
) -> SequentialLR:
    """Linear warmup, then a cosine phase. ``schedule`` selects the cosine shape.

    ``"warm_restarts"`` (default, faithful to the original CaMCheX code):
        A cosine that sweeps ``lr -> eta_min`` over ``steps_per_epoch`` steps and
        then *restarts* to ``lr`` at each epoch boundary (a per-epoch sawtooth),
        equivalent to SGDR with ``T_0=steps_per_epoch`` and ``T_mult=1``. Useful
        for snapshot ensembling but it re-kicks a converged model every epoch,
        which destabilises the fine-tuning tail. Keep the best checkpoint, not
        the last.

    ``"single_cosine"``:
        One cosine over the whole run (``T_max = total_steps - warmup_steps``) --
        a single monotone decay ``lr -> eta_min`` with no restarts, so the model
        settles into its minimum in the tail. Requires ``total_steps``.

    The cosine is implemented as a multiplicative ``LambdaLR`` factor in
    ``[eta_min_factor, 1.0]`` applied to *each param group's own* base LR, so the
    floor scales with the group's discriminative-LR multiplier (a group peaking
    at ``0.3 x`` also floors at ``0.3 x``). The absolute ``eta_min`` scalar that
    ``CosineAnnealingLR`` / ``CosineAnnealingWarmRestarts`` apply identically to
    every group would otherwise clamp all groups to the same floor and break the
    intended peak ratio at the trough.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    if schedule == "single_cosine":
        if total_steps is None:
            raise ValueError("schedule='single_cosine' requires total_steps")
        t_max = max(1, int(total_steps) - int(warmup_steps))

        def cosine_factor(step: int) -> float:
            t = min(step, t_max) / t_max
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return eta_min_factor + (1.0 - eta_min_factor) * cos

    elif schedule == "warm_restarts":
        period = max(1, int(steps_per_epoch))

        def cosine_factor(step: int) -> float:
            t = (step % period) / period
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return eta_min_factor + (1.0 - eta_min_factor) * cos

    else:
        raise ValueError(
            f"unknown schedule={schedule!r}; expected 'warm_restarts' or 'single_cosine'"
        )

    cosine = LambdaLR(optimizer, lr_lambda=cosine_factor)

    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
