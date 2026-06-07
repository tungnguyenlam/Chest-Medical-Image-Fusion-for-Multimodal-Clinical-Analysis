from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW, Optimizer


def _resolve_lr(name: str, param_group_lrs: dict[str, float], base_lr: float) -> float:
    """LR for parameter ``name``: the value of the longest matching prefix in
    ``param_group_lrs`` (e.g. ``"text_encoder."`` or ``"image_encoder.frontal_encoder."``),
    or ``base_lr`` if nothing matches. Longest-prefix wins so a more specific prefix
    overrides a broader one."""
    best_prefix: str | None = None
    for prefix in param_group_lrs:
        if name.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
            best_prefix = prefix
    return base_lr if best_prefix is None else float(param_group_lrs[best_prefix])


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    param_group_lrs: dict[str, float] | None = None,
) -> list[dict]:
    """Split trainable params into AdamW groups by (learning rate, decay-eligibility).

    Decay split: bias and 1-D params (LayerNorm/BatchNorm gain, embeddings' scaling, etc.)
    get weight_decay=0; everything else gets ``weight_decay``.

    Discriminative LR (optional): ``param_group_lrs`` maps a parameter-name *prefix* to a
    learning rate; matching params use that LR instead of ``base_lr`` (longest-prefix wins).
    When it is empty/None every param uses ``base_lr`` and exactly two groups are produced
    (decay, no_decay) -- identical to the historical behaviour, so existing optimizer
    checkpoints stay compatible.
    """
    param_group_lrs = param_group_lrs or {}
    # Insertion-ordered buckets keyed by (lr, no_decay); named_parameters() order is stable,
    # so group ordering is deterministic and reproducible across runs/resumes.
    buckets: dict[tuple[float, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = _resolve_lr(name, param_group_lrs, base_lr)
        no_decay = p.ndim <= 1 or name.endswith(".bias")
        buckets.setdefault((lr, no_decay), []).append(p)

    groups: list[dict] = []
    for (lr, no_decay), params in buckets.items():
        groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return groups


# Back-compat alias: the historical name, now a thin wrapper (no discriminative LR).
def split_decay_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    return build_param_groups(model, base_lr=0.0, weight_decay=weight_decay)


def build_adamw_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    param_group_lrs: dict[str, float] | None = None,
) -> Optimizer:
    """AdamW with decay/no-decay split and optional per-component (discriminative) LR.

    ``param_group_lrs`` (from ``optimizer_init_args.param_group_lrs`` in config) maps a
    parameter-name prefix to its LR; unmatched params use ``lr``. Default None -> all
    components share ``lr``. Per-group LRs compose with the LR schedulers, which scale
    each group relative to its own initial LR.
    """
    groups = build_param_groups(model, base_lr=lr, weight_decay=weight_decay, param_group_lrs=param_group_lrs)
    if param_group_lrs:
        summary = ", ".join(
            f"lr={g['lr']:.2e}/wd={g['weight_decay']:g}: {sum(p.numel() for p in g['params'])} params"
            for g in groups
        )
        print(f"[optimizer] discriminative LR groups -> {summary}")
    return AdamW(
        groups,
        lr=lr,
        betas=betas,
        eps=eps,
    )
